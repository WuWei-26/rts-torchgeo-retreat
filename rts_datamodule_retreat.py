# rts_datamodule_retreat.py

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# from kornia import image_to_tensor, tensor_to_image
import kornia.augmentation as K
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import rasterio

from pathlib import Path

# from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
from kornia.constants import Resample
from kornia.enhance import denormalize, normalize
from rasterio.crs import CRS
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datasets import (
    BoundingBox,
    IntersectionDataset,
    UnionDataset,
    stack_samples,
)
from torchgeo.samplers import Units

# import torchvision
from torchvision import transforms

from rts_dataset_retreat import (
    Landsat8SR,
    Landsat57SR,
    MeanTPI,
    RtsMask,
    TestIntersectionDEM,
    TestLandsat8SR,
    TestLandsat57SR,
    TestMeanTPI,
    RTSTemporalPairDataset,
    RetreatMapDataset,
)
from rts_sampler_retreat import RandomGeoSamplerMultiRoiMultiYear, TestPreChippedGeoSampler
from rts_utils import plot_batch

# torchvision.models.resnet18

if os.cpu_count() is not None:
    N_WORKERS = int(os.cpu_count() / 2)  # type: ignore
else:
    N_WORKERS = 4
DEM_NULL = -7
# Asia_North_Albers_Equal_Area_Conic from QGIS esri:102025
CRS_AEA = rasterio.CRS.from_proj4(
    "+proj=aea +lat_0=30 +lon_0=95 +lat_1=15 +lat_2=65 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

# mask the ground truth mask corresponding to non-data (cloud) & convert to float

RETREAT_SCALE =  28.90128517150879 # global_max retreat = 28.90128517150879
# RETREAT_SCALE =  1.0 # dont scale

def collate_fn_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return stack_samples(batch)  # 用torchgeo的stack_samples保持一致

# TODO: mask portion of DEM pixels
# add noise, alter brightness
class DataAugmentation(torch.nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, patch_size=(256, 256), p=0.5) -> None:
        super().__init__()
        self.transforms = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            # range of degrees to select from (-degrees, +degrees)
            K.RandomRotation(
                degrees=45,
                resample=Resample.NEAREST.name,
                align_corners=None,  # can only be None for Nearest resampling # type: ignore
                p=p,
            ),
            # K.RandomResizedCrop(
            #     size=patch_size,
            #     scale=(0.5, 1.0),
            #     ratio=(0.9, 1.1),
            #     resample=Resample.NEAREST.name,
            #     align_corners=None,  # can only be None for Nearest resampling
            #     p=p
            # ),
            # < 1 darker, > 1 brighter, factor = bright_ness - 1, img_adjust: Tensor = image + factor,
            # img_adjust = img_adjust.clamp(min=0.0, max=1.0), need to do before normalize
            K.RandomBrightness(brightness=(0.95, 1.05), p=p),
            K.RandomGaussianNoise(mean=0.0, std=0.01, p=p),
            data_keys=["image", "mask"],
        )
        # self.aug_dem = K.RandomErasing(scale=(0.02, 0.33), value=DEM_NULL, p=1)
        self.aug_dem = K.RandomGaussianNoise(mean=0.0, std=0.1, p=p)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, sample_batch: dict) -> Dict[str, Tensor]:
        image = sample_batch["image"]
        mask = sample_batch["mask"]
        image, mask = self.transforms(image, mask)
        mask = mask.clone()
        dem_input = mask[:, 2, ...].clone() 
        mask[:, 2, ...] = self.aug_dem(dem_input)
        # mask[:, 2, ...] = self.aug_dem(mask[:, 2, ...])
        return {"image": image, "mask": mask}

class DataAugmentationPair(torch.nn.Module):
    def __init__(self, patch_size=(256, 256), p=0.5) -> None:
        super().__init__()
        self.geom = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            K.RandomRotation(degrees=45, resample=Resample.NEAREST.name, align_corners=None, p=p),
            K.RandomBrightness(brightness=(0.95, 1.05), p=p),
            K.RandomGaussianNoise(mean=0.0, std=0.01, p=p),
            data_keys=["input"],
            same_on_batch=True,
        )
        self.aug_dem = K.RandomGaussianNoise(mean=0.0, std=0.1, p=p)

    @torch.no_grad()
    def forward(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        img_t = sample["image_t"]   # [B, C, H, W]
        img_tm1 = sample["image_tm1"] # [B, C, H, W]
        B = img_t.shape[0]
        combined = torch.cat([img_t, img_tm1], dim=0)  # [2B, C, H, W]
        combined_aug = self.geom(combined)              # [2B, C, H, W]

        sample["image_t"] = combined_aug[:B]
        sample["image_tm1"] = combined_aug[B:]

        if "dem_t" in sample:
            sample["dem_t"] = self.aug_dem(sample["dem_t"])
        if "dem_tm1" in sample:
            sample["dem_tm1"] = self.aug_dem(sample["dem_tm1"])

        return sample

class BandsNormalize(torch.nn.Module):
    """Normalize a tensor image with mean and standard deviation."""

    def __init__(
        self,
        mean: Union[Tensor, Tuple[float], List[float], float],
        std: Union[Tensor, Tuple[float], List[float], float],
    ) -> None:
        super().__init__()

        if isinstance(mean, (int, float)):
            mean = torch.tensor([mean])

        if isinstance(std, (int, float)):
            std = torch.tensor([std])

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)[None]

        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)[None]

        self.mean = mean
        self.std = std

    def forward(self, sample: dict) -> dict:
        image = sample["image"]
        # mean = self.mean.view(-1, 1, 1)
        # std = self.std.view(-1, 1, 1)
        image = normalize(image, self.mean, self.std)
        # image = (image - mean) / std
        sample["image"] = image
        return sample

class BandsNormalizePair(torch.nn.Module):
    def __init__(self, mean: Union[Tensor, Tuple[float], List[float], float],
                 std: Union[Tensor, Tuple[float], List[float], float]) -> None:
        super().__init__()
        if isinstance(mean, (int, float)): mean = torch.tensor([mean])
        if isinstance(std, (int, float)):  std  = torch.tensor([std])
        if isinstance(mean, (tuple, list)): mean = torch.tensor(mean)[None]
        if isinstance(std, (tuple, list)):  std  = torch.tensor(std)[None]
        self.mean, self.std = mean, std

    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample["image_t"]   = normalize(sample["image_t"],   self.mean, self.std)
        sample["image_tm1"] = normalize(sample["image_tm1"], self.mean, self.std)
        return sample

class BandsDenormalize(BandsNormalize):
    """Denormalize a tensor image with mean and standard deviation."""

    def forward(self, sample: dict) -> dict:
        image = sample["image"]
        # mean = self.mean.view(1, -1, 1, 1)
        # std = self.std.view(1, -1, 1, 1)
        image = denormalize(image, self.mean, self.std)
        # image = (image * std) + mean
        sample["image"] = image
        return sample

class BandsDenormalizePair(BandsNormalizePair):
    def forward(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
        sample["image_t"]   = denormalize(sample["image_t"],   self.mean, self.std)
        sample["image_tm1"] = denormalize(sample["image_tm1"], self.mean, self.std)
        return sample

class Preprocess(torch.nn.Module):
    """Module to perform pre-process on sample dictionary of torch tensors."""

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, sample: dict) -> dict:
        image = sample["image"].float()
        image = torch.nan_to_num(image)
        # assert not torch.isnan(image).any()
        non_data_mask = image[1, :, :] == 0  # got mask before applying scale
        image = Landsat8SR.apply_scale(image)  # convert to 0-1
        sample["image"] = image

        if sample.get("mask") is not None:
            mask = sample["mask"].float()
            mask = torch.nan_to_num(mask)

            # TODO: separate the preprocess of dem from image dataset
            if mask.shape[0] >= 2:
                mask[0:2, non_data_mask] = 0  # mask first two bands
                # mask[2, non_data_mask] = -7  # for dem
                # if torch.isnan(mask[2, :, :]).any():
                #     mask[2, :, :] = torch.nan_to_num(
                #         mask[2, :, :], nan=DEM_NULL)  # replace nan as -7
                # assert not torch.isnan(mask[2, :, :]).any()
                # mask[2,:,:] = 0
                # mask[2,:,:] = mask[2,:,:]/1.5 # for segmentation to 0-1

                # for heatmap to 0-1 from 12 to 15 to 20 to 30 (no_gamma)
                mask[0, :, :] = mask[0, :, :] / 30
                mask[1, :, :] = mask[1, :, :] / 255  # for segmentation to 0-1

            # mask = np.nan_to_num(mask)

            sample["mask"] = mask

        # if sample.get("dem") is not None:
        #     dem = sample["dem"]
        #     dem[:,non_data_mask] = 0
        #     sample["dem"] = dem
        # del sample['crs']
        # del sample['bbox']
        return sample

class PreprocessPair(torch.nn.Module):
    @torch.no_grad()
    def forward(self, sample: dict) -> dict:
        for k in ["image_t", "image_tm1"]:
            if k in sample:
                x = sample[k].float()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                # Landsat SR scaling
                non_data_mask = (x[:,1] == 0) if x.dim()==4 else (x[1]==0)
                x = Landsat8SR.apply_scale(x)
                sample[k] = x

        # heatmap, mask, retreat_map
        if "heatmap" in sample:
            hm = sample["heatmap"].float()
            hm = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
            hm = torch.clamp(hm / 30.0, min=0.0, max=1.5)  # ~[0,1.5]
            sample["heatmap"] = hm
        if "mask" in sample:
            mk = sample["mask"].float()
            mk = torch.nan_to_num(mk, nan=0.0, posinf=0.0, neginf=0.0)
            mk = torch.clamp(mk / 255.0, min=0.0, max=1.0) # 0/255 -> 0/1
            sample["mask"] = mk
        if "retreat_map" in sample:
            rt = sample["retreat_map"].float()
            rt = torch.nan_to_num(rt, nan=0.0, posinf=0.0, neginf=0.0)
            rt = rt / RETREAT_SCALE  # 
            rt = torch.clamp(rt, min=0.0)
            sample["retreat_map"] = rt
        for k in ["dem_t","dem_tm1"]:
            if k in sample:
                d = sample[k].float()
                d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                sample[k] = d
        return sample

class LandsatDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir: str,
        dem_dir: str,
        mask_dir: str,
        train_length: int,
        val_length: int,
        test_length: int,
        year_dict_train: Dict[Any, Sequence[int | float]],
        year_dict_val: Dict[Any, Sequence[int | float]],
        year_dict_test: Dict[Any, Sequence[int | float]],
        band_num: int = 3,
        use_dem: bool = True,
        batch_size: int = 32,
        patch_size: int = 256,
        crs: Optional[CRS] = None,
        res: Optional[float] = 30,
        use_l7: bool = False,
        retreat_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.retreat_dir = retreat_dir

        if band_num == 3:
            self.landsat_bands_l89 = ["SR_B4", "SR_B3", "SR_B2"]
            self.landsat_bands_l57 = ["SR_B3", "SR_B2", "SR_B1"]
        elif band_num == 4:
            self.landsat_bands_l89 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]
            self.landsat_bands_l57 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4"]
        elif band_num == 6:
            self.landsat_bands_l89 = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            self.landsat_bands_l57 = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
        else:
            raise ValueError(f"Unsupported band_num={band_num}")

        self.ds_l89 = Landsat8SR(root=img_dir, crs=crs, bands=self.landsat_bands_l89, res=res, cache=False)

        if use_l7:
            self.ds_l57 = Landsat57SR(root=img_dir, crs=crs, bands=self.landsat_bands_l57, res=res, cache=False)
            self.img_dataset = UnionDataset(self.ds_l89, self.ds_l57)
            stats_ds = self.ds_l89
        else:
            self.img_dataset = self.ds_l89
            stats_ds = self.ds_l89

        self.bands = stats_ds.bands
        self.rgb_indexes = stats_ds.rgb_indexes

        self.preprocess = PreprocessPair()
        self.augmentation = DataAugmentationPair(patch_size=(patch_size, patch_size), p=0.5)
        self.normalize = BandsNormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.denormalize = BandsDenormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.transform = transforms.Compose([self.preprocess])

        mask_bands = ["heatmap", "segment", "retreat"]
        self.mask_dataset = RtsMask(
            root=mask_dir, crs=crs, bands=mask_bands, res=res, cache=False, debug=False
            )
        dem_bands = ["mean_tpi"]
        self.dem_dataset = MeanTPI(root=dem_dir, crs=crs, bands=dem_bands, res=res, cache=False)

        # retreat dataset ~/ww_cryostore/BLH_heatmap/retreat_map_YYYY.tif
        if self.retreat_dir:
            self.retreat_dataset = RetreatMapDataset(
                root=self.retreat_dir,
                crs=crs,
                res=res,
                cache=False
            )
        else:
            self.retreat_dataset = None


        self.dataset = RTSTemporalPairDataset(
            img_ds=self.img_dataset,
            mask_ds=self.mask_dataset,
            dem_ds=self.dem_dataset,
            retreat_ds=self.retreat_dataset,
            transforms=self.preprocess,
        )

        self.train_sampler = RandomGeoSamplerMultiRoiMultiYear(
            self.dataset,
            size=self.patch_size,
            stride=int(self.patch_size / 2),
            length=train_length,
            year_dict=year_dict_train,
            units=Units.PIXELS,
            pair=True,
            prev_delta=1,
        )
        self.val_sampler = RandomGeoSamplerMultiRoiMultiYear(
            self.dataset,
            size=self.patch_size,
            stride=int(self.patch_size / 2),
            length=val_length,
            year_dict=year_dict_val,
            units=Units.PIXELS,
            pair=True,
            prev_delta=1,
        )
        self.test_sampler = RandomGeoSamplerMultiRoiMultiYear(
            self.dataset,
            size=self.patch_size,
            stride=int(self.patch_size / 2),
            length=test_length,
            year_dict=year_dict_test,
            units=Units.PIXELS,
            pair=True,
            prev_delta=1,
        )
        self.demo_sampler = RandomGeoSamplerMultiRoiMultiYear(
            self.dataset,
            size=self.patch_size,
            stride=int(self.patch_size / 2),
            length=self.batch_size,
            year_dict=year_dict_test,
            units=Units.PIXELS,
            pair=True,
            prev_delta=1,
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        # if stage == "fit" or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True,
        #                        transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(
        #         mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(
        #         self.data_dir, train=False, transform=self.transform)
        pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.train_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=collate_fn_skip_none,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=collate_fn_skip_none,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.test_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=collate_fn_skip_none,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.demo_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=stack_samples,
        )

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     # only perform augmentation during training
    #     if self.trainer and self.trainer.training:
    #         batch = self.augmentation(batch)
    #     batch = self.normalize(batch)
    #     return batch
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer and self.trainer.training:
            batch = self.augmentation(batch)
        batch = self.normalize(batch)
        return batch

    def transfer_batch_to_device(
        self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int
    ) -> Dict[str, Tensor]:
        """Transfer batch to device.

        Defines how custom data types are moved to the target device.

        Args:
            batch: A batch of data that needs to be transferred to a new device.
            device: The target device as defined in PyTorch.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A reference to the data on the new device.
        """
        # Non-Tensor values cannot be moved to a device
        # TODO: convert bbox to tensor for later usage
        # if "crs" in batch.keys():
        #     del batch["crs"]
        # if "bbox" in batch.keys():
        #     del batch["bbox"]

        tensor_keys = ["image_t","image_tm1","dem_t","dem_tm1","mask","heatmap","retreat_map"]
        batch_tensors = {k: v for k, v in batch.items() if k in tensor_keys}
        batch_tensors = super().transfer_batch_to_device(batch_tensors, device, dataloader_idx)
        return batch_tensors

    def show_batch(
        self,
        data_loader,
        sample_num: Optional[int] = None,
        bright: float = 3,
        width: int = 6,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        def _ensure_loader(dl):
            return dl() if callable(dl) else dl

        def _batch_size_from_tensors(batch: dict) -> int:
            for v in batch.values():
                if torch.is_tensor(v) and v.dim() >= 1:
                    return v.shape[0]
            return 0

        def _safe_len(x):
            return len(x) if isinstance(x, (list, tuple)) else 0

        plt.style.use("default")
        loader = _ensure_loader(data_loader)
        batch = next(iter(loader))

        is_pair = all(k in batch for k in ["image_t", "image_tm1", "heatmap", "mask", "retreat_map"])

        B = _batch_size_from_tensors(batch)
        N = min(sample_num or 6, B) if B > 0 else 0

        if "path_t" in batch and "path_tm1" in batch and N > 0:
            pt = batch["path_t"]
            ptm = batch["path_tm1"]
            npt = _safe_len(pt)
            nptm = _safe_len(ptm)
            for i in range(N):
                if i < npt:
                    print(f"[paths] t={batch.get('year_t', ['?']*N)[i] if isinstance(batch.get('year_t'), (list, tuple, torch.Tensor)) else batch.get('year_t', '?')} -> {pt[i]}")
                if i < nptm:
                    print(f"[paths] t-1={batch.get('year_tm1', ['?']*N)[i] if isinstance(batch.get('year_tm1'), (list, tuple, torch.Tensor)) else batch.get('year_tm1', '?')} -> {ptm[i]}")

        if not is_pair:
            # batch_aug = self.augmentation(batch)
            batch_aug = batch # no rotation augmentation
            batch_aug = self.normalize(batch_aug)
            try:
                batch_vis = self.denormalize(batch_aug)
            except Exception:
                batch_vis = batch_aug

            print("Before augmentation:\n")
            print(batch["image"].shape)
            plot_batch(
                batch=batch,
                sample_num=sample_num,
                bright=bright,
                chnls=self.rgb_indexes,
                width=width,
            )
            plt.show()

            print("After augmentation:\n")
            plot_batch(
                batch=batch_vis,
                sample_num=sample_num,
                bright=bright,
                chnls=self.rgb_indexes,
                width=width,
            )
            plt.show()
            return

        batch_aug = self.augmentation(batch)
        batch_aug = self.normalize(batch_aug)
        try:
            batch_vis = self.denormalize(batch_aug)
        except Exception:
            batch_vis = batch_aug

        img_t = batch_vis["image_t"]      # [B, C, H, W]
        img_tm1 = batch_vis["image_tm1"]    # [B, C, H, W]
        heatmap = batch_vis["heatmap"]      # [B, 1, H, W]
        seg = batch_vis["mask"]         # [B, 1, H, W]
        retreat = batch_vis["retreat_map"]  # [B, 1, H, W]
        dem_t_bf = batch.get("dem_t", None)

        B = img_t.shape[0]
        N = min(sample_num or 6, B)
        def _nz(t):
            try:
                return int((t>0).sum().item())
            except Exception:
                return -1
        print("Shapes:",
            "img_t", tuple(img_t.shape),
            "img_tm1", tuple(img_tm1.shape),
            "heatmap", tuple(heatmap.shape),
            "seg", tuple(seg.shape),
            "retreat", tuple(retreat.shape))

        for i in range(N):
            hm_i = heatmap[i] if heatmap.ndim==4 else heatmap[i:i+1]
            sg_i = seg[i]     if seg.ndim==4     else seg[i:i+1]
            rt_i = retreat[i] if retreat.ndim==4 else retreat[i:i+1]
            print(f"[{i}] nz(heatmap)={_nz(hm_i)} nz(seg)={_nz(sg_i)} nz(retreat)={_nz(rt_i)}")

        def get_vis_ch(batch_tensor: torch.Tensor, i: int, fallback_hw=None) -> np.ndarray:
            if fallback_hw is None:
                fallback_hw = (img_t.shape[-2], img_t.shape[-1])
            fh, fw = fallback_hw
            if batch_tensor is None:
                return np.zeros((fh, fw), dtype=np.float32)
            if batch_tensor.ndim == 4:
                if batch_tensor.shape[1] >= 1:
                    return batch_tensor[i, 0].detach().cpu().numpy()
                else:
                    return np.zeros((fh, fw), dtype=np.float32)
            elif batch_tensor.ndim == 3:
                return batch_tensor[i].detach().cpu().numpy()
            else:
                return np.zeros((fh, fw), dtype=np.float32)

        # plot
        ncols = 5
        fig, axes = plt.subplots(N, ncols, figsize=(ncols * width, N * width / 2), squeeze=False)

        rgb_idx = self.rgb_indexes if hasattr(self, "rgb_indexes") else [2, 1, 0]
        def _rgb(img_chw: torch.Tensor) -> np.ndarray:
            arr = img_chw[rgb_idx,...].detach().cpu().numpy().transpose(1, 2, 0)
            return np.clip(arr * bright, 0, 1)
        
        def _get_year(batch, key, i):
            v = batch.get(key, None)
            if v is None:
                return "?"
            if isinstance(v, (list, tuple)):
                return v[i] if i < len(v) else v[0]
            if torch.is_tensor(v):
                if v.ndim == 1:
                    return int(v[i].item())
                return int(v.item())
            return v

        for i in range(N):
            y_t=_get_year(batch, "year_t", i)
            y_tm1=_get_year(batch, "year_tm1", i)
            # year t RGB
            axes[i, 0].imshow(_rgb(img_t[i])); axes[i, 0].set_title(f"Year {y_t} RGB"); axes[i, 0].axis("off")
            # year t heatmap GT
            hm_i = get_vis_ch(heatmap, i)
            axes[i, 1].imshow(np.clip(hm_i, 0, 1.3), cmap="jet", vmin=0, vmax=1.3)
            axes[i, 1].set_title(f"Year {y_t} Heatmap GT"); axes[i, 1].axis("off")
            # year t-1 RGB
            axes[i, 2].imshow(_rgb(img_tm1[i])); axes[i, 2].set_title(f"Year {y_tm1} RGB"); axes[i, 2].axis("off")
            # retreat GT
            ret_i = get_vis_ch(retreat, i)
            axes[i, 3].imshow(ret_i, cmap="turbo")
            axes[i, 3].set_title("Retreat GT"); axes[i, 3].axis("off")
            # TPI
            tpi_bf_i = get_vis_ch(dem_t_bf, i) if dem_t_bf is not None else np.zeros((img_t.shape[-2], img_t.shape[-1]), dtype=np.float32)
            import numpy as np
            if np.isfinite(tpi_bf_i).any():
                vmax_bf = max(abs(np.nanpercentile(tpi_bf_i, 99)), abs(np.nanpercentile(tpi_bf_i, 1)))
            else:
                vmax_bf = 2.0
            axes[i, 4].imshow(tpi_bf_i, cmap="gray", vmin=-vmax_bf, vmax=vmax_bf)
            axes[i, 4].set_title("TPI"); axes[i, 4].axis("off")

        plt.tight_layout()
        plt.show()

class LandsatPairInferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir: str,
        year_t: int,
        year_tm1: int,
        roi: BoundingBox,
        dem_dir: Optional[str] = None,
        band_num: int = 4,
        use_dem: bool = True,
        batch_size: int = 1,
        patch_size: int = 256,
        crs: Optional[CRS] = None,
        res: Optional[float] = 30,
        use_l7: bool = True,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.img_dir = img_dir
        self.dem_dir = dem_dir
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.use_dem = use_dem
        self.year_t = year_t
        self.year_tm1 = year_tm1
        self.roi = roi
        self.preprocess = PreprocessPair() #pairpreprocess

        if band_num == 3:
            bands_l89 = ["SR_B4","SR_B3","SR_B2"]; bands_l57 = ["SR_B3","SR_B2","SR_B1"]
        elif band_num == 4:
            bands_l89 = ["SR_B2","SR_B3","SR_B4","SR_B5"]; bands_l57 = ["SR_B1","SR_B2","SR_B3","SR_B4"]
        elif band_num == 6:
            bands_l89 = ["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]; bands_l57 = ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"]
        else:
            raise ValueError(f"Unsupported band_num={band_num}")
        
        self.img_dataset = self._create_multi_year_dataset(
            img_dir, [year_t, year_tm1], bands_l89, bands_l57, crs, res, use_l7
        )

        try:
            stats_ds = TestLandsat8SR(root=img_dir, crs=crs, bands=bands_l89, res=res, cache=False)
        except Exception:
            img_path = Path(img_dir)
            if img_path.name == 'test' and img_path.parent.name.isdigit():
                base_dir = img_path.parent.parent
            else:
                base_dir = img_path
            year_t_dir = base_dir / str(year_t) / "test"
            stats_ds = TestLandsat8SR(root=str(year_t_dir), crs=crs, bands=bands_l89, res=res, cache=False)

        self.bands = stats_ds.bands
        self.rgb_indexes = stats_ds.rgb_indexes

        # normalisation
        self.normalize = BandsNormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.denormalize = BandsDenormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.transform = self.preprocess

        # DEM/TPI
        if use_dem and dem_dir:
            dem_bands = ["mean_tpi"]
            self.dem_dataset = TestMeanTPI(root=dem_dir, crs=crs, bands=dem_bands, res=res, cache=False)
            ds_for_pair = TestIntersectionDEM(self.img_dataset, self.dem_dataset, transforms=self.transform)
        else:
            self.dem_dataset = None
            ds_for_pair = self.img_dataset

        self.dataset = RTSTemporalPairDataset(
            img_ds=self.img_dataset,
            mask_ds=None,
            dem_ds=self.dem_dataset,
            retreat_ds=None,
            transforms=self.preprocess,
            year_t=year_t,
            year_tm1=year_tm1,
        )

        roi_dict = {
            str(self.year_t): {
                "cluster_id": [0],
                "roi": [BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt)],
                "weight": [1.0],
            }
        }
        year_dict = {"year": [self.year_t], "weight": [1.0], "roi": [roi_dict[str(self.year_t)]]}
        # self.predict_sampler = TestPreChippedGeoSampler(
        #     self.dataset, 
        #     min_size=32 * 2, 
        #     roi=self.roi, 
        #     units=Units.PIXELS,
        #     year_t=year_t,
        #     year_tm1=year_tm1,
        # )
        self.predict_sampler = TestPreChippedGeoSampler(
            self.dataset,
            size=self.patch_size,
            stride=self.patch_size,   # 先不重叠；后面可改成 self.patch_size // 2
            roi=self.roi,
            units=Units.PIXELS,
            year_t=year_t,
            year_tm1=year_tm1,
        )
    
    def _create_multi_year_dataset(self, img_dir, years, bands_l89, bands_l57, crs, res, use_l7):
        
        print(f"creating multi-year dataset: {years}")
        img_path = Path(img_dir)
        base_dir = img_path
        print(f"inference image path: {base_dir}")
        
        l89_datasets = []
        l57_datasets = []
        
        for year in years:
            year_dir = base_dir / str(year) / "test"
            if year_dir.exists():
                try:
                    ds_l89 = TestLandsat8SR(
                        root=str(year_dir), 
                        crs=crs, 
                        bands=bands_l89, 
                        res=res, 
                        cache=False
                    )
                    l89_datasets.append(ds_l89)
                    print(f"successfully created L8/9 dataset for year {year}: {ds_l89.bounds}")
                except Exception as e:
                    print(f"creating dataset failed: {e}")
                
                # L5/7
                if use_l7:
                    try:
                        ds_l57 = TestLandsat57SR(
                            root=str(year_dir), 
                            crs=crs, 
                            bands=bands_l57, 
                            res=res, 
                            cache=False
                        )
                        l57_datasets.append(ds_l57)
                        print(f"successfully created L5/7 dataset for year {year}: {ds_l57.bounds}")
                    except Exception as e:
                        print(f"creating L5/7 dataset failed: {e}")
            else:
                print(f"year {year} directory does not exist: {year_dir}")
        
        if len(l89_datasets) == 1:
            combined_l89 = l89_datasets[0]
        else:
            combined_l89 = UnionDataset(*l89_datasets)
        
        #L5/7
        if use_l7 and l57_datasets:
            if len(l57_datasets) == 1:
                combined_l57 = l57_datasets[0]
            else:
                combined_l57 = UnionDataset(*l57_datasets)
            
            final_dataset = UnionDataset(combined_l89, combined_l57)
            print(f"final dataset (L8/9 + L5/7), bounds: {final_dataset.bounds}")
        else:
            final_dataset = combined_l89
            print(f"final dataset (L8/9 only), bounds: {final_dataset.bounds}")
        return final_dataset

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = self.normalize(batch)
        return batch

    def transfer_batch_to_device(self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int) -> Dict[str, Tensor]:
        tensor_keys = ["image_t", "image_tm1", "dem_t", "dem_tm1"]
        # metadata_keys = ["bbox", "path", "crs", "image"]
        metadata_keys = ["bbox", "path", "crs", "image",
                     "path_t", "path_tm1",
                     "transform_t", "transform_tm1",
                     "year_t", "year_tm1"]
        
        new_batch = {}
        for key in tensor_keys:
            if key in batch and torch.is_tensor(batch[key]):
                new_batch[key] = batch[key].to(device)
        for key in metadata_keys:
            if key in batch:
                new_batch[key] = batch[key]
        return new_batch
    
    def predict_dataloader(self):
        def collate_single_sample(batch):
            if len(batch) == 1:
                sample = batch[0]
                for key in ["image_t", "image_tm1", "dem_t", "dem_tm1"]:
                    if key in sample and torch.is_tensor(sample[key]):
                        sample[key] = sample[key].unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                
                # retain transform
                for key in ["transform_t", "transform_tm1", "bbox", "path_t", "path_tm1", "crs", "year_t", "year_tm1"]:
                    if key in sample:
                        if not isinstance(sample[key], (list, tuple)):
                            sample[key] = [sample[key]]
                return sample
            else:
                return stack_samples(batch)

        return DataLoader(
            self.dataset,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            batch_size=1,
            collate_fn=collate_single_sample,
        )