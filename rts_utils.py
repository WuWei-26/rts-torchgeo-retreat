import math
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import kornia.augmentation as K
import matplotlib.pyplot as plt
import rasterio
import torch
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.windows import window_index
from rtree import index
from torch import Tensor
from torch.nn import functional as F
from torchgeo.datasets import BoundingBox, RasterDataset, unbind_samples

AREA_K_SIZE = 100
AREA_STRIDE = 10


class AreaPoolLoss(torch.nn.Module):
    def __init__(self, k_size=100, stride=50):
        super().__init__()
        self.k_size = k_size
        self.stride = stride

    def forward(self, output, target):
        # pool of square window of size=3, stride=2
        area_avg = torch.nn.AvgPool2d(kernel_size=self.k_size, stride=self.stride)
        output_avg = area_avg(output)
        target_avg = area_avg(target)
        area_avg_diff = torch.square(output_avg - target_avg)
        loss_value = torch.mean(area_avg_diff)  # normalized by element number
        return loss_value


class LossWeightedMSE(torch.nn.Module):
    def __init__(self, W=10):
        super().__init__()
        self.W = float(W)

    def forward(self, output, target, weight_mask):
        M = weight_mask.float()
        diff = torch.square(output - target)
        weighted_diff = diff * (self.W * M + 1.0)
        loss_value = torch.mean(weighted_diff)
        return loss_value


class AWingLoss(torch.nn.Module):
    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.omega = float(omega)
        self.epsilon = float(epsilon)
        self.theta = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** (self.alpha - y)))
            * (self.alpha - y)
            * ((self.theta / self.epsilon) ** (self.alpha - y - 1))
            / self.epsilon
        )
        C = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** (self.alpha - y)
        )
        case1_ind = torch.abs(y - y_pred) < self.theta
        case2_ind = torch.abs(y - y_pred) >= self.theta
        lossMat[case1_ind] = self.omega * torch.log(
            1
            + torch.abs((y[case1_ind] - y_pred[case1_ind]) / self.epsilon)
            ** (self.alpha - y[case1_ind])
        )
        lossMat[case2_ind] = (
            A[case2_ind] * torch.abs(y[case2_ind] - y_pred[case2_ind]) - C[case2_ind]
        )
        return lossMat


class LossWeightedAWing(torch.nn.Module):
    def __init__(self, W=10, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.W = float(W)
        self.Awing = AWingLoss(alpha, omega, epsilon, theta)

    def forward(self, output, target, weight_mask):
        M = weight_mask.float()
        diff = self.Awing(output, target)
        # diff = torch.square(output - target)
        weighted_diff = diff * (self.W * M + 1.0)
        loss_value = torch.mean(weighted_diff)
        return loss_value


class BandNormalize(K.IntensityAugmentationBase2D):
    """Normalize channels using mean and variance."""

    def __init__(self, means: Tensor, stds: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {
            "means": means.view(1, -1, 1, 1),
            "stds": stds.view(1, -1, 1, 1),
        }
        self.band_num = len(means)

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        x = input[..., : self.band_num, :, :]
        x = (x - flags["means"]) / (flags["stds"] + 1e-10)
        input[..., : self.band_num, :, :] = x
        return input


def pad_HW(image, div=32):
    height = image.shape[-2]
    wdith = image.shape[-1]
    wdith_pad = 0
    height_pad = 0
    if height % div != 0:
        height_pad = (height // div + 1) * div - height
    if wdith % div != 0:
        wdith_pad = (wdith // div + 1) * div - wdith
    # image_pad = np.pad(image, ((0, 0), (0, height_pad), (0, wdith_pad)), 'constant')
    # to pad the last 2 dimensions of the input tensor, then use  (padding_left, padding_right, padding_top , padding_bottom)
    image_pad = F.pad(
        image, (0, wdith_pad, 0, height_pad), "constant"
    )  # , constant_values=params['mean']

    return image_pad, height_pad, wdith_pad


def mute_image(image, mean):
    """set the image values as mean"""
    mean = mean.view(1, -1, 1, 1)
    image[..., :, :, :] = mean

    return image


def calc_statistics_landsat(dset: RasterDataset, bands):
    """
    Calculate the statistics (mean and std) for the entire dataset
    Warning: This is an approximation. The correct value should take into account the
    mean for the whole dataset for computing individual stds.
    For correctness I suggest checking: http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """

    # To avoid loading the entire dataset in memory, we will loop through each img
    # The filenames will be retrieved from the dataset's rtree index

    if bands and dset.all_bands:
        band_indexes = [dset.all_bands.index(i) + 1 for i in bands]
    files = [
        item.object for item in dset.index.intersection(dset.index.bounds, objects=True)
    ]

    # Reseting statistics
    accum_mean = 0
    accum_std = 0

    for file in files:
        img = rasterio.open(file).read(indexes=band_indexes) * 0.0000275 - 0.2  # type: ignore (image*0.0000275-0.2)
        accum_mean += img.reshape((img.shape[0], -1)).mean(axis=1)
        accum_std += img.reshape((img.shape[0], -1)).std(axis=1)

    # at the end, we shall have 2 vectors with lenght n=chnls
    # we will average them considering the number of images
    return accum_mean / len(files), accum_std / len(files)


def get_time_bounds(index: index.Index):
    start_date = index.bounds[4]
    end_date = index.bounds[5]
    start_date_str = datetime.fromtimestamp(start_date).strftime(
        "%Y-%m-%d"
    )  # %H:%M:%S.%f
    end_date_str = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")  # %H:%M:%S.%f
    return start_date_str, end_date_str


def expand_to_bbox(image: Tensor, bbox: BoundingBox, t_bbox: BoundingBox, res: int):
    # out_width = round((t_bbox.maxx - t_bbox.minx) / res)
    # out_height = round((t_bbox.maxy - t_bbox.miny) / res)
    out_width = math.ceil((t_bbox.maxx - t_bbox.minx) / res)
    out_height = math.ceil((t_bbox.maxy - t_bbox.miny) / res)
    if image.shape[-2] == out_height and image.shape[-1] == out_width:
        # print('not expanded')
        return image
    # print(image.ndim)
    if image.ndim == 3:
        t_image = torch.zeros([image.shape[0], out_height, out_width])
    elif image.ndim == 2:
        t_image = torch.zeros([out_height, out_width])
    # print('input shape: ', image.shape)
    # print('target shape: ', t_image.shape)
    transform = transform_from_bounds(
        t_bbox.minx, t_bbox.miny, t_bbox.maxx, t_bbox.maxy, out_width, out_height
    )  # west, south, east, north, width, height
    bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
    q_window = win_from_bounds(*bounds, transform)
    # print(q_window)
    row_slice, col_slice = window_index(q_window)
    # row_slice = slice(*row_slice.indices(image.shape[-2])) # to match size
    # col_slice = slice(*col_slice.indices(image.shape[-1])) # S.indices(len) -> (start, stop, stride)
    # to ensure matching the size
    row_slice_2 = slice(row_slice.start, row_slice.start + image.shape[-2])
    col_slice_2 = slice(col_slice.start, col_slice.start + image.shape[-1])
    if image.ndim == 3:
        t_image[:, row_slice_2, col_slice_2] = image
    elif image.ndim == 2:
        t_image[row_slice_2, col_slice_2] = image
    return t_image


def plot_imgs(
    images: Iterable, axs: Iterable, chnls: List[int] = [2, 1, 0], bright: float = 3.0
):
    for img, ax in zip(images, axs):
        # print(img.max())
        arr = torch.clamp(img * bright, min=0, max=1).numpy()
        rgb = arr.transpose(1, 2, 0)[:, :, chnls]
        ax.imshow(rgb)
        ax.axis("off")


def plot_heatmap(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        # ax.imshow(mask.squeeze().numpy(), cmap='Blues')
        ax.imshow(mask.numpy(), cmap="jet")
        ax.axis("off")


def plot_msks(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow(mask.numpy(), cmap="gray")
        ax.axis("off")


def plot_dem(masks: Iterable, axs: Iterable):
    for mask, ax in zip(masks, axs):
        ax.imshow((mask.numpy() + 2) / 4, cmap="gray")
        ax.axis("off")


def plot_batch(
    batch: Dict,
    sample_num: Optional[int] = None,
    bright: float = 3.0,
    cols: int = 3,
    width: int = 6,
    chnls: List[int] = [3, 2, 1],
):
    # Get the samples and the number of items in the batch
    samples = unbind_samples(batch.copy())
    if sample_num:
        if sample_num > len(samples):
            raise ValueError(
                f"Input sample num {sample_num} is greater than the length of samples {len(samples)}!"
            )
        else:
            samples = samples[:sample_num]

    # if batch contains images and masks, the number of images will be doubled
    n = 3 * len(samples) if ("image" in batch) and ("mask" in batch) else len(samples)

    # calculate the number of rows in the grid
    rows = n // cols + (1 if n % cols != 0 else 0)

    # create a grid
    _, axs = plt.subplots(rows, cols, figsize=(cols * width, rows * width))

    if ("image" in batch) and ("mask" in batch):
        # plot the images on the first col
        plot_imgs(
            images=map(lambda x: x["image"], samples),
            axs=axs.reshape(-1)[::3],
            chnls=chnls,
            bright=bright,
        )

        # plot the masks on the second col
        plot_heatmap(
            masks=map(lambda x: x["mask"][0, :, :], samples), axs=axs.reshape(-1)[1::3]
        )

        # plot the masks on the third col
        if samples[0]["mask"].shape[0] == 2:
            plot_msks(
                masks=map(lambda x: x["mask"][1, :, :], samples),
                axs=axs.reshape(-1)[2::3],
            )
        else:
            plot_dem(
                masks=map(lambda x: x["mask"][2, :, :], samples),
                axs=axs.reshape(-1)[2::3],
            )

    else:
        if "image" in batch:
            plot_imgs(
                images=map(lambda x: x["image"], samples),
                axs=axs.reshape(-1),
                chnls=chnls,
                bright=bright,
            )

        elif "mask" in batch:
            plot_msks(masks=map(lambda x: x["mask"], samples), axs=axs.reshape(-1))

    plt.plot()
