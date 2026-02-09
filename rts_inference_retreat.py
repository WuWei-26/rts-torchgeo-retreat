# rts_inference_retreat.py

from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from datetime import datetime
import pytorch_lightning as pl
import rasterio
import matplotlib.pyplot as plt
# from rts_dataset import expand_to_bbox
from rts_utils import expand_to_bbox
from torchgeo.datasets import BoundingBox
from rts_datamodule_retreat import LandsatPairInferenceDataModule
from rasterio.crs import CRS
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

class PairTemporalDataset(torch.utils.data.Dataset):
    @staticmethod
    def _year_bbox(bbox: BoundingBox, year: int) -> BoundingBox:
        # 按年时间窗过滤（与训练一致）
        mint = datetime(year - 1, 12, 31).timestamp()
        maxt = datetime(year + 1, 1, 1).timestamp()
        return BoundingBox(bbox.minx, bbox.maxx, bbox.miny, bbox.maxy, mint, maxt)

    def __init__(self, img_ds, dem_ds=None, transforms=None):
        self.img_ds = img_ds
        self.dem_ds = dem_ds
        self.transforms = transforms
        self.index = img_ds.index
        self.bounds = img_ds.bounds
        self.crs = img_ds.crs
        self.res = img_ds.res

    def __len__(self):
        return 10**9

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        query: {'bbox': BoundingBox, 'year_t': int, 'year_tm1': int}
        """
        bbox = query["bbox"]
        year_t = int(query["year_t"])
        year_tm1 = int(query["year_tm1"])
        q_t = self._year_bbox(bbox, year_t)
        q_tm1 = self._year_bbox(bbox, year_tm1)
        s_img_t = self.img_ds[q_t]
        try:
            s_img_tm1 = self.img_ds[q_tm1]
        except Exception:
            s_img_tm1 = self.img_ds[q_t]
        image_t = s_img_t["image"].float()
        image_tm1 = s_img_tm1["image"].float()

        if self.dem_ds is not None:
            s_dem_t = self.dem_ds[q_t]
            try:
                s_dem_tm1 = self.dem_ds[q_tm1]
            except Exception:
                s_dem_tm1 = self.dem_ds[q_t]
            dem_t = s_dem_t["mask"].float()
            dem_tm1 = s_dem_tm1["mask"].float()
        else:
            dem_t = torch.zeros((1, image_t.shape[-2], image_t.shape[-1]), dtype=torch.float32)
            dem_tm1 = torch.zeros_like(dem_t)

        sample = {
            "image_t": image_t,
            "image_tm1": image_tm1,
            "dem_t": dem_t,
            "dem_tm1": dem_tm1,
            "bbox": bbox,
            "crs": self.crs,
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

def expand_and_mask(image: Tensor, heatmap: Tensor, segment: Tensor, bbox: BoundingBox, target_bbox: BoundingBox, res: int = 30):
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0)  # [H,W] -> [1,H,W]
    if segment.dim() == 2:
        segment = segment.unsqueeze(0)  # [H,W] -> [1,H,W]
    
    # expand the input raster
    image = expand_to_bbox(image, bbox, target_bbox, res)
    heatmap = expand_to_bbox(heatmap, bbox, target_bbox, res)
    segment = expand_to_bbox(segment, bbox, target_bbox, res)

    # values of 0 are interpreted as False, everything else as True.
    image_mask = image[0:1, :, :] > 0
    pr_heatmap_ma = np.ma.array(heatmap.numpy(), mask=~image_mask.numpy())
    pr_segment_ma = np.ma.array(segment.numpy(), mask=~image_mask.numpy())
    image_mask_np = (1 * image_mask).numpy().astype(np.float32)  # convert bool to int

    return pr_heatmap_ma, pr_segment_ma, image_mask_np

def save_predictions(
    export_path: Path,
    bbox: BoundingBox,
    crs: CRS,
    pr_heatmap: np.ndarray,
    pr_segment: np.ndarray,
    pr_count: np.ndarray,
    image_count: np.ndarray,
    transform=None, 

) -> bool:

    def to_hw(x, name):
        x = np.asarray(x, dtype=np.float32)
        x = np.squeeze(x)
        if x.ndim != 2:
            raise ValueError(f"{name} 期望 squeeze 后为 (H,W)，但得到 {x.shape}")
        return x

    pr_heatmap = to_hw(pr_heatmap, "pr_heatmap")
    pr_segment = to_hw(pr_segment, "pr_segment")
    pr_count = to_hw(pr_count, "pr_count")
    image_count = to_hw(image_count, "image_count")

    print("save_predictions shapes (H,W):")
    print("pr_heatmap:", pr_heatmap.shape)
    print("pr_segment:", pr_segment.shape)
    print("pr_count:", pr_count.shape)
    print("image_count:", image_count.shape)

    height, width = pr_heatmap.shape

    if transform is None:
        raise ValueError("please pass a transform.")
        # rio_transform = rasterio.transform.from_bounds(
        #     bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
        # )
    rio_transform = transform

    # rio_transform = rasterio.transform.from_bounds(
    #     bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
    # )

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 4,
        "dtype": "float32",
        "crs": crs,
        "transform": rio_transform,
        "compress": "lzw",
    }

    all_bands = np.stack(
        [pr_heatmap, pr_segment, pr_count, image_count], axis=0
    )  # (4,H,W)

    with rasterio.open(export_path, "w", **profile) as dst:
        dst.write(all_bands)  # 直接写 (4,H,W)
        dst.set_band_description(1, "heatmap")
        dst.set_band_description(2, "segment")
        dst.set_band_description(3, "count")
        dst.set_band_description(4, "image_count")

    print("Prediction results saved:\n", export_path)
    return True

def show_results(image, heatmap, bright=3):
    # image -> RGB
    if torch.is_tensor(image):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = np.asarray(image)

    if img_np.ndim == 3 and img_np.shape[0] >= 3:
        rgb = img_np[:3].transpose(1, 2, 0)
    elif img_np.ndim == 3 and img_np.shape[0] == 1:
        rgb = img_np[0]
    else:
        rgb = img_np

    if rgb.ndim == 3:
        rgb = np.clip((rgb * bright + 2) / 4, 0, 1)

    # heatmap -> (H,W)
    if torch.is_tensor(heatmap):
        h_np = heatmap.detach().cpu().numpy()
    else:
        h_np = np.asarray(heatmap)

    if h_np.ndim == 3 and h_np.shape[0] == 1:
        h_np = h_np[0]
    elif h_np.ndim == 4 and h_np.shape[1] == 1:
        h_np = h_np[0, 0]

    print("heatmap max value:", float(h_np.max()))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb, cmap="gray" if rgb.ndim == 2 else None)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(h_np, 0.05, 1.2), cmap='jet')
    plt.title("Heatmap Prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def predict_and_export(
        rts_model,
        trainer: pl.Trainer,
        datamodule: LandsatPairInferenceDataModule,
        model_name: str,
        date_str: str,
        export_dir: Path,
):
    target_bbox = datamodule.roi

    predict_dataloader = datamodule.predict_dataloader()
    all_batches = list(predict_dataloader)  # 收集所有 batch
    
    predict_dataloader_for_predict = datamodule.predict_dataloader()
    predictions_list = trainer.predict(
        model=rts_model, dataloaders=predict_dataloader_for_predict
    )

    pr_heatmap_list = []
    pr_segment_list = []
    image_mask_list = []
    target_crs = None
    img_rgb = None

    def to_2d_numpy(arr):
        if hasattr(arr, 'filled'):
            arr = arr.filled(0)
        if hasattr(arr, 'numpy'):
            arr = arr.numpy()
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        arr = np.array(arr, dtype=np.float32)
        while arr.ndim > 2:
            arr = arr.squeeze(0)
        return arr

    # input_transform = None

    for data_batch, predictions in zip(all_batches, predictions_list):
        pr_heatmaps = predictions['heatmap']     # [B,1,H,W] or [B,H,W]
        pr_segments = predictions['prob_mask']   # [B,1,H,W] or [B,H,W]
        image_batch = data_batch['image_t']      # [B,C,H,W]

        # if "transform" in data_batch:
        input_transform = data_batch["transform"]
        if isinstance(input_transform, (list, tuple)):
            input_transform = input_transform[idx]

        idx = 0  # batch_size = 1
        image = image_batch[idx]           # [C,H,W]
        
        if pr_heatmaps.dim() == 4 and pr_heatmaps.shape[1] == 1:
            pr_heatmap = pr_heatmaps[idx, 0]   # [H,W]
            pr_segment = pr_segments[idx, 0]   # [H,W]
        elif pr_heatmaps.dim() == 3:
            pr_heatmap = pr_heatmaps[idx]      # [H,W]
            pr_segment = pr_segments[idx]      # [H,W]
        else:
            pr_heatmap = pr_heatmaps[idx, 0]
            pr_segment = pr_segments[idx, 0]
        
        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[idx]
        
        if target_crs is None:
            target_crs = data_batch['crs']
            if isinstance(target_crs, (list, tuple)):
                target_crs = target_crs[idx]
        
        if img_rgb is None and hasattr(datamodule, 'rgb_indexes'):
            img_rgb = image[datamodule.rgb_indexes, :, :]

        pr_heatmap_exp, pr_segment_exp, image_mask_exp = expand_and_mask(
            image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
        )

        pr_heatmap_list.append(pr_heatmap_exp)
        pr_segment_list.append(pr_segment_exp)
        image_mask_list.append(to_2d_numpy(image_mask_exp))

    pr_heatmap_concat = np.ma.concatenate(pr_heatmap_list, axis=0)  # [N,1,H,W]
    pr_segment_concat = np.ma.concatenate(pr_segment_list, axis=0)  # [N,1,H,W]
    
    pr_heatmap_median = np.ma.median(pr_heatmap_concat, axis=0).filled(fill_value=0)
    pr_segment_median = np.ma.median(pr_segment_concat, axis=0).filled(fill_value=0)
    
    pr_segment_count = np.ma.sum(pr_segment_concat, axis=0).filled(fill_value=0)
    
    image_mask_2d_list = []
    for mask in image_mask_list:
        mask_2d = to_2d_numpy(mask)
        image_mask_2d_list.append(mask_2d)

    image_mask_stack = np.stack(image_mask_2d_list, axis=0)  # [N,H,W]
    image_count = np.sum(image_mask_stack, axis=0)  # [H,W]

    pr_heatmap_median = to_2d_numpy(pr_heatmap_median)
    pr_segment_median = to_2d_numpy(pr_segment_median)
    pr_segment_count = to_2d_numpy(pr_segment_count)
    # image_count = to_2d_numpy(image_count)

    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"pr_heatmap_median shape: {pr_heatmap_median.shape}")
    print(f"pr_segment_median shape: {pr_segment_median.shape}")
    print(f"pr_segment_count shape: {pr_segment_count.shape}")
    print(f"image_count shape: {image_count.shape}")
    print(f"image_count range: [{image_count.min()}, {image_count.max()}]")
    print(f"pr_heatmap_median range: [{pr_heatmap_median.min():.4f}, {pr_heatmap_median.max():.4f}]")
    print(f"pr_segment_median range: [{pr_segment_median.min():.4f}, {pr_segment_median.max():.4f}]")

    if img_rgb is not None:
        show_results(image=img_rgb, heatmap=pr_heatmap_median, bright=3)

    pr_tiff_path = export_dir / f"pr_mask_{date_str}_median_{model_name}.tif"

    # print("Export shape:", pr_heatmap.shape)
    # print("Export transform:", rio_transform)
    # print("Export bbox:", bbox)

    save_predictions(
        export_path=pr_tiff_path,
        bbox=target_bbox,
        crs=target_crs,
        pr_heatmap=pr_heatmap_median,
        pr_segment=pr_segment_median,
        pr_count=pr_segment_count,
        image_count=image_count,
        transform=input_transform
    )
    
    print(f"Successfully exported to: {pr_tiff_path}")
    return True

def predict_and_export_mean(
        rts_model,
        trainer: pl.Trainer,
        datamodule,
        model_name: str,
        date_str: str,
        export_dir: Path,
):
    predict_dataloader = datamodule.predict_dataloader()
    target_bbox = datamodule.roi

    predictions_list = trainer.predict(
        model=rts_model, dataloaders=predict_dataloader
    )

    pr_heatmap_sum = None
    pr_segment_sum = None
    image_count = None
    target_crs = None

    for data_batch, predictions in zip(predict_dataloader, predictions_list):
        pr_heatmaps = predictions['heatmap']     # [B,1,H,W]
        pr_segments = predictions['prob_mask']   # [B,1,H,W]
        image_batch = data_batch['image_t']      # [B,C,H,W]

        idx = 0  # batch_size = 1
        image = image_batch[idx]           # [C,H,W]
        pr_heatmap = pr_heatmaps[idx, 0]   # [H,W]
        pr_segment = pr_segments[idx, 0]   # [H,W]
        
        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[idx]
        
        target_crs = data_batch['crs']
        if isinstance(target_crs, (list, tuple)):
            target_crs = target_crs[idx]

        pr_heatmap_exp, pr_segment_exp, image_mask_exp = expand_and_mask(
            image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
        )

        def to_2d_numpy(arr):
            """将数组转换为 2D numpy 数组"""
            if hasattr(arr, 'filled'):
                arr = arr.filled(0)
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            arr = np.array(arr, dtype=np.float32)
            while arr.ndim > 2:
                arr = arr.squeeze(0)
            return arr
        
        pr_heatmap_np = to_2d_numpy(pr_heatmap_exp)
        pr_segment_np = to_2d_numpy(pr_segment_exp)
        image_mask_np = to_2d_numpy(image_mask_exp)

        if pr_heatmap_sum is None:
            pr_heatmap_sum = np.zeros_like(pr_heatmap_np, dtype=np.float32)
            pr_segment_sum = np.zeros_like(pr_segment_np, dtype=np.float32)
            image_count = np.zeros_like(pr_heatmap_np, dtype=np.float32)

        valid_mask = image_mask_np > 0
        pr_heatmap_sum[valid_mask] += pr_heatmap_np[valid_mask]
        pr_segment_sum[valid_mask] += pr_segment_np[valid_mask]
        image_count[valid_mask] += 1

    image_count_safe = np.maximum(image_count, 1)
    pr_heatmap_mean = pr_heatmap_sum / image_count_safe
    pr_segment_mean = pr_segment_sum / image_count_safe

    pr_heatmap_mean[image_count == 0] = 0
    pr_segment_mean[image_count == 0] = 0

    export_dir.mkdir(parents=True, exist_ok=True)

    print("pr_heatmap_mean shape:", pr_heatmap_mean.shape)
    print(f"image_count: [{image_count.min()}, {image_count.max()}]")

    pr_tiff_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

    save_predictions(
        export_path=pr_tiff_path,
        bbox=target_bbox,
        crs=target_crs,
        pr_heatmap=pr_heatmap_mean,
        pr_segment=pr_segment_mean,
        pr_count=pr_segment_sum,
        image_count=image_count,
    )
    
    print(f"Successfully exported to: {pr_tiff_path}")
    return True
