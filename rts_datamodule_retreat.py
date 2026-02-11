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
        # 1) 取两时相影像
        img_t = sample["image_t"]   # [B, C, H, W]
        img_tm1 = sample["image_tm1"] # [B, C, H, W]
        B = img_t.shape[0]

        # 2) 在 batch 维拼接 → 同一调用中使用相同随机参数
        combined = torch.cat([img_t, img_tm1], dim=0)  # [2B, C, H, W]
        combined_aug = self.geom(combined)              # [2B, C, H, W]

        # 3) 切分回两时相
        sample["image_t"] = combined_aug[:B]
        sample["image_tm1"] = combined_aug[B:]

        # 4) DEM/TPI 不做几何变换，仅加噪声（可选）
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
        # 影像（标准化比例你已有）
        for k in ["image_t", "image_tm1"]:
            if k in sample:
                x = sample[k].float()
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                # Landsat SR 缩放
                non_data_mask = (x[:,1] == 0) if x.dim()==4 else (x[1]==0)
                x = Landsat8SR.apply_scale(x)
                sample[k] = x

        # 标签：heatmap、mask、retreat_map
        if "heatmap" in sample:
            hm = sample["heatmap"].float()
            hm = torch.nan_to_num(hm, nan=0.0, posinf=0.0, neginf=0.0)
            hm = torch.clamp(hm / 30.0, min=0.0, max=1.5)  # 归一到 ~[0,1.5]
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
            rt = torch.clamp(rt, min=0.0)  # 距离/米 或 米/年，保持非负
            sample["retreat_map"] = rt

        # TPI/DEM：对称显示、但训练前先保证有限
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
            # 供 RTSTemporalPairDataset 用于空间/时间索引
            self.img_dataset = UnionDataset(self.ds_l89, self.ds_l57)
            stats_ds = self.ds_l89  # 归一化与 RGB 索引统一用 L8/9 的统计
        else:
            self.img_dataset = self.ds_l89
            stats_ds = self.ds_l89

        # 用“基准数据集”取统计与 RGB 索引（注意：不要从 UnionDataset 取）
        self.bands = stats_ds.bands
        self.rgb_indexes = stats_ds.rgb_indexes

        self.preprocess = PreprocessPair()  # 对 image_t & image_tm1 做 SR 缩放与标签缩放
        self.augmentation = DataAugmentationPair(patch_size=(patch_size, patch_size), p=0.5)
        self.normalize = BandsNormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.denormalize = BandsDenormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        # 保留 transform 管线接口（如果你在 Dataset 中还会用到）
        self.transform = transforms.Compose([self.preprocess])

        mask_bands = ["heatmap", "segment", "retreat"]
        self.mask_dataset = RtsMask(
            root=mask_dir, crs=crs, bands=mask_bands, res=res, cache=False, debug=False
            )

        if not use_dem:
            raise ValueError("RTSTemporalPairDataset 需要 dem_ds（标准化 TPI）。请设 use_dem=True。")
        dem_bands = ["mean_tpi"]
        self.dem_dataset = MeanTPI(root=dem_dir, crs=crs, bands=dem_bands, res=res, cache=False)

        # 独立的 retreat 数据集（可选：直接读取 ~/ww_cryostore/BLH_heatmap/retreat_map_YYYY.tif）
        # 注意：只要设置 crs/res 与目标网格一致，RasterDataset 会在线重投影到目标窗口。
        if self.retreat_dir:
            self.retreat_dataset = RetreatMapDataset(
                root=self.retreat_dir,
                crs=crs,   # 例如 "EPSG:32646"
                res=res,   # 例如 30
                cache=False
            )
        else:
            self.retreat_dataset = None


        self.dataset = RTSTemporalPairDataset(
            img_ds=self.img_dataset,
            mask_ds=self.mask_dataset,
            dem_ds=self.dem_dataset,
            retreat_ds=self.retreat_dataset,
            transforms=self.preprocess,  # 成对预处理在 Dataset 内生效
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
            collate_fn=stack_samples,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.val_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=stack_samples,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.test_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=stack_samples,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset,
            sampler=self.demo_sampler,
            num_workers=N_WORKERS,
            batch_size=self.batch_size,
            collate_fn=stack_samples,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # only perform augmentation during training
        if self.trainer and self.trainer.training:
            # perform GPU/Batched data augmentation
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
            # 从 batch 中任意张量获取 batch 维大小
            for v in batch.values():
                if torch.is_tensor(v) and v.dim() >= 1:
                    return v.shape[0]
            return 0

        def _safe_len(x):
            return len(x) if isinstance(x, (list, tuple)) else 0

        plt.style.use("default")
        loader = _ensure_loader(data_loader)
        batch = next(iter(loader))

        # 双时相判断
        is_pair = all(k in batch for k in ["image_t", "image_tm1", "heatmap", "mask", "retreat_map"])

        # 先计算 B/N，避免后续打印时出现 N 未定义
        B = _batch_size_from_tensors(batch)
        N = min(sample_num or 6, B) if B > 0 else 0

        # 若有路径信息，先打印核验（此时已知 N）
        if "path_t" in batch and "path_tm1" in batch and N > 0:
            pt  = batch["path_t"]
            ptm = batch["path_tm1"]
            # 兼容 list/tuple，避免越界
            npt  = _safe_len(pt)
            nptm = _safe_len(ptm)
            for i in range(N):
                if i < npt:
                    print(f"[paths] t={batch.get('year_t', ['?']*N)[i] if isinstance(batch.get('year_t'), (list, tuple, torch.Tensor)) else batch.get('year_t', '?')} -> {pt[i]}")
                if i < nptm:
                    print(f"[paths] t-1={batch.get('year_tm1', ['?']*N)[i] if isinstance(batch.get('year_tm1'), (list, tuple, torch.Tensor)) else batch.get('year_tm1', '?')} -> {ptm[i]}")

        if not is_pair:
            # 单时相回退
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

        # 双时相增广/归一化（成对）
        batch_aug = self.augmentation(batch)
        batch_aug = self.normalize(batch_aug)
        try:
            batch_vis = self.denormalize(batch_aug)
        except Exception:
            batch_vis = batch_aug

        img_t   = batch_vis["image_t"]      # [B, C, H, W]
        img_tm1 = batch_vis["image_tm1"]    # [B, C, H, W]
        heatmap = batch_vis["heatmap"]      # [B, 1, H, W]
        seg     = batch_vis["mask"]         # [B, 1, H, W]
        retreat = batch_vis["retreat_map"]  # [B, 1, H, W]
        dem_t_bf = batch.get("dem_t", None)

        # 重新计算 B/N（以防 batch augmentation 改变 batch 维；通常不会）
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

        # 安全取单通道（若缺通道则用零图占位）
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

        # 作图（后续逻辑保持不变）
        ncols = 5
        fig, axes = plt.subplots(N, ncols, figsize=(ncols * width, N * width / 2), squeeze=False)

        rgb_idx = self.rgb_indexes if hasattr(self, "rgb_indexes") else [2, 1, 0]
        def _rgb(img_chw: torch.Tensor) -> np.ndarray:
            arr = img_chw[rgb_idx,...].detach().cpu().numpy().transpose(1, 2, 0)
            return np.clip(arr * bright, 0, 1)

        for i in range(N):
            # year t RGB
            axes[i, 0].imshow(_rgb(img_t[i])); axes[i, 0].set_title("Year t RGB"); axes[i, 0].axis("off")
            # year t heatmap GT
            hm_i = get_vis_ch(heatmap, i)
            axes[i, 1].imshow(np.clip(hm_i, 0, 1.3), cmap="jet", vmin=0, vmax=1.3)
            axes[i, 1].set_title("Year t Heatmap GT"); axes[i, 1].axis("off")
            # year t-1 RGB
            axes[i, 2].imshow(_rgb(img_tm1[i])); axes[i, 2].set_title("Year t-1 RGB"); axes[i, 2].axis("off")
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
        num_workers: int = 0,       # 推理阶段建议单进程，避免 GDAL+多进程阻塞
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

        # 成对预处理/归一化（不做增强）
        self.preprocess = PreprocessPair()  # 对 image_t/image_tm1 做 SR 缩放与标签缩放
        # 用 L8/9 统计作为归一化基准
        # 影像数据集（L8/9；如 use_l7=True 则 Union 到 L57）
        if band_num == 3:
            bands_l89 = ["SR_B4","SR_B3","SR_B2"]; bands_l57 = ["SR_B3","SR_B2","SR_B1"]
        elif band_num == 4:
            bands_l89 = ["SR_B2","SR_B3","SR_B4","SR_B5"]; bands_l57 = ["SR_B1","SR_B2","SR_B3","SR_B4"]
        elif band_num == 6:
            bands_l89 = ["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]; bands_l57 = ["SR_B1","SR_B2","SR_B3","SR_B4","SR_B5","SR_B7"]
        else:
            raise ValueError(f"Unsupported band_num={band_num}")
        
        # 🔧 关键修改：支持多年份数据集
        self.img_dataset = self._create_multi_year_dataset(
            img_dir, [year_t, year_tm1], bands_l89, bands_l57, crs, res, use_l7
        )

        # ds_l89 = TestLandsat8SR(root=img_dir, crs=crs, bands=bands_l89, res=res, cache=False)
        # if use_l7:
        #     ds_l57 = TestLandsat57SR(root=img_dir, crs=crs, bands=bands_l57, res=res, cache=False)
        #     self.img_dataset = UnionDataset(ds_l89, ds_l57)
        #     stats_ds = ds_l89
        # else:
        #     self.img_dataset = ds_l89
        #     stats_ds = ds_l89

        # ✅ 只创建一个用于获取统计信息的数据集，不覆盖 self.img_dataset
        # stats_ds = TestLandsat8SR(root=img_dir, crs=crs, bands=bands_l89, res=res, cache=False)

        # ✅ 仅用于获取统计信息（均值/标准差），不覆盖 self.img_dataset
        try:
            stats_ds = TestLandsat8SR(root=img_dir, crs=crs, bands=bands_l89, res=res, cache=False)
        except Exception:
            # 如果 img_dir 本身没有数据，尝试从 year_t 目录获取
            img_path = Path(img_dir)
            if img_path.name == 'test' and img_path.parent.name.isdigit():
                base_dir = img_path.parent.parent
            else:
                base_dir = img_path
            year_t_dir = base_dir / str(year_t) / "test"
            stats_ds = TestLandsat8SR(root=str(year_t_dir), crs=crs, bands=bands_l89, res=res, cache=False)

        self.bands = stats_ds.bands
        self.rgb_indexes = stats_ds.rgb_indexes

        # 成对归一化（推理阶段不做增强，仅归一化）
        self.normalize = BandsNormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.denormalize = BandsDenormalizePair(mean=stats_ds.bands_mean, std=stats_ds.bands_std)
        self.transform = self.preprocess  # 只做预处理

        # DEM/TPI（可选）
        if use_dem and dem_dir:
            dem_bands = ["mean_tpi"]
            self.dem_dataset = TestMeanTPI(root=dem_dir, crs=crs, bands=dem_bands, res=res, cache=False)
            ds_for_pair = TestIntersectionDEM(self.img_dataset, self.dem_dataset, transforms=self.transform)
        else:
            self.dem_dataset = None
            ds_for_pair = self.img_dataset  # 仅影像

        # 成对推理数据集：只读影像与 DEM，不读标签
        # 复用训练版的成对数据集 RTSTemporalPairDataset 也可，但推理时通常不需要 heatmap/mask/retreat_map
        # 这里直接用训练版 RTSTemporalPairDataset，只取 image_t/image_tm1/dem_t/dem_tm1 键（Transforms 成对预处理）
        self.dataset = RTSTemporalPairDataset(
            img_ds=self.img_dataset,       # 用原影像数据集（TestLandsat* 支持 path）
            mask_ds=None,                  # 推理不需要标签；RTSTemporalPairDataset需改为允许 None（或用你前面提供的 PairTemporalDataset）
            dem_ds=self.dem_dataset,
            retreat_ds=None,
            transforms=self.preprocess,
            year_t=year_t,
            year_tm1=year_tm1,
        )

        # 构造一个 year_dict/pair=True 的采样器（只针对给定 ROI 和年份对）
        roi_dict = {
            # 采样器需要一个“形如训练的 ROI 字典”，这里构造单一条目
            str(self.year_t): {
                "cluster_id": [0],
                "roi": [BoundingBox(roi.minx, roi.maxx, roi.miny, roi.maxy, roi.mint, roi.maxt)],
                "weight": [1.0],
            }
        }
        year_dict = {"year": [self.year_t], "weight": [1.0], "roi": [roi_dict[str(self.year_t)]]}

        # self.predict_sampler = RandomGeoSamplerMultiRoiMultiYear(
        #     self.dataset,
        #     size=self.patch_size,
        #     stride=self.patch_size,
        #     length=200, # self.patch_size
        #     year_dict=year_dict,
        #     units=Units.PIXELS,
        #     pair=True,        # 关键：返回 {'bbox','year_t','year_tm1'}
        #     prev_delta=self.year_t - self.year_tm1 if (self.year_t - self.year_tm1) > 0 else 1,
        # )

        self.predict_sampler = TestPreChippedGeoSampler(
            self.dataset, 
            min_size=32 * 2, 
            roi=self.roi, 
            units=Units.PIXELS,
            year_t=year_t,
            year_tm1=year_tm1,
        )
    
    def _create_multi_year_dataset(self, img_dir, years, bands_l89, bands_l57, crs, res, use_l7):
        """创建支持多年份的数据集"""
        
        print(f"🔧 创建多年份数据集，年份: {years}")
        
        # 从 img_dir 推断基础目录结构
        img_path = Path(img_dir)
        
        # 如果 img_dir 是 /path/2016/test 格式，需要回到基础目录
        if img_path.name == 'test' and img_path.parent.name.isdigit():
            base_dir = img_path.parent.parent
            print(f"检测到年份目录结构，基础目录: {base_dir}")
        else:
            # 如果 img_dir 就是基础目录，直接使用
            base_dir = img_path
            print(f"使用基础目录: {base_dir}")
        
        # 收集所有年份的数据集
        l89_datasets = []
        l57_datasets = []
        
        for year in years:
            year_dir = base_dir / str(year) / "test"
            
            if year_dir.exists():
                print(f"  添加年份 {year}: {year_dir}")
                
                # L8/9 数据集
                try:
                    ds_l89 = TestLandsat8SR(
                        root=str(year_dir), 
                        crs=crs, 
                        bands=bands_l89, 
                        res=res, 
                        cache=False
                    )
                    l89_datasets.append(ds_l89)
                    print(f"    L8/9 数据集创建成功，边界: {ds_l89.bounds}")
                except Exception as e:
                    print(f"    ⚠️ L8/9 数据集创建失败: {e}")
                
                # L5/7 数据集（如果需要）
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
                        print(f"    L5/7 数据集创建成功")
                    except Exception as e:
                        print(f"    ⚠️ L5/7 数据集创建失败: {e}")
            else:
                print(f"  ❌ 年份 {year} 目录不存在: {year_dir}")
        
        # 合并数据集
        if not l89_datasets:
            raise ValueError(f"没有找到任何有效的L8/9数据集，年份: {years}")
        
        # 合并L8/9数据集
        if len(l89_datasets) == 1:
            combined_l89 = l89_datasets[0]
        else:
            combined_l89 = UnionDataset(*l89_datasets)
        
        # 如果需要L5/7，进一步合并
        if use_l7 and l57_datasets:
            if len(l57_datasets) == 1:
                combined_l57 = l57_datasets[0]
            else:
                combined_l57 = UnionDataset(*l57_datasets)
            
            final_dataset = UnionDataset(combined_l89, combined_l57)
            print(f"最终合并数据集 (L8/9 + L5/7)，边界: {final_dataset.bounds}")
        else:
            final_dataset = combined_l89
            print(f"最终数据集 (仅L8/9)，边界: {final_dataset.bounds}")
        
        return final_dataset

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # 推理阶段不做增广：仅归一化成对影像
        batch = self.normalize(batch)
        return batch

    def transfer_batch_to_device(self, batch: Dict[str, Tensor], device: torch.device, dataloader_idx: int) -> Dict[str, Tensor]:
        """只移动张量到设备，保留元数据在CPU上"""
        
        # 定义需要移动到设备的张量字段
        tensor_keys = ["image_t", "image_tm1", "dem_t", "dem_tm1"]
        
        # 定义需要保留在CPU的元数据字段
        metadata_keys = ["bbox", "path", "crs", "image"]
        
        # 创建新的批次字典
        new_batch = {}
        
        # 移动张量字段到设备
        for key in tensor_keys:
            if key in batch and torch.is_tensor(batch[key]):
                new_batch[key] = batch[key].to(device)
        
        # 保留元数据字段在CPU（不移动）
        for key in metadata_keys:
            if key in batch:
                new_batch[key] = batch[key]
        
        return new_batch
    
    def predict_dataloader(self):
        def collate_single_sample(batch):
            """每个样本单独处理，不堆叠（因为尺寸可能不同）"""
            # batch 是一个列表，每个元素是一个 sample dict
            # 由于 batch_size=1，直接返回第一个元素
            if len(batch) == 1:
                sample = batch[0]
                # 为张量添加 batch 维度
                for key in ["image_t", "image_tm1", "dem_t", "dem_tm1"]:
                    if key in sample and torch.is_tensor(sample[key]):
                        sample[key] = sample[key].unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                return sample
            else:
                # 如果 batch_size > 1，使用 stack_samples（但要求尺寸相同）
                return stack_samples(batch)

        return DataLoader(
            self.dataset,
            sampler=self.predict_sampler,
            num_workers=self.num_workers,
            batch_size=1,  # ✅ 强制 batch_size=1，每个影像单独处理
            collate_fn=collate_single_sample,
        )
    
    # def predict_dataloader(self):
    #     def collate_and_filter(batch):
    #         # 用 torchgeo 的 stack_samples 做默认聚合
    #         batch = stack_samples(batch)
    #         # 只保留张量字段
    #         keep = ["image_t", "image_tm1", "dem_t", "dem_tm1", "bbox", "path", "crs", "image"]
    #         batch_filtered={}

    #         for k in keep:
    #             if k in batch:
    #                 batch_filtered[k] = batch[k]
            
    #         return batch_filtered
    #         # batch_tensors = {k: v for k, v in batch.items() if k in keep and torch.is_tensor(v)}
    #         # return batch_tensors

    #     return DataLoader(
    #         self.dataset,
    #         sampler=self.predict_sampler,
    #         num_workers=self.num_workers,
    #         batch_size=self.batch_size,
    #         collate_fn=collate_and_filter,
    #     )