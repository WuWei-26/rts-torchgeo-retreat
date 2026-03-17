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

import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds

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

        # 复用影像索引边界
        self.index = img_ds.index
        self.bounds = img_ds.bounds
        self.crs = img_ds.crs
        self.res = img_ds.res

    def __len__(self):
        # 交给外部采样器控制
        return 10**9

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Tensor]:
        """
        期望 query: {'bbox': BoundingBox, 'year_t': int, 'year_tm1': int}
        """
        bbox = query["bbox"]
        year_t = int(query["year_t"])
        year_tm1 = int(query["year_tm1"])

        q_t = self._year_bbox(bbox, year_t)
        q_tm1 = self._year_bbox(bbox, year_tm1)

        # 读取影像（t、t-1）
        s_img_t = self.img_ds[q_t]
        try:
            s_img_tm1 = self.img_ds[q_tm1]
        except Exception:
            s_img_tm1 = self.img_ds[q_t]  # 回退到 t（可按需改为 raise）

        image_t = s_img_t["image"].float()
        image_tm1 = s_img_tm1["image"].float()

        # 读取 DEM/TPI（可选）
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
    # expand the input raster
    image = expand_to_bbox(image, bbox, target_bbox, res)
    heatmap = expand_to_bbox(heatmap, bbox, target_bbox, res)
    segment = expand_to_bbox(segment, bbox, target_bbox, res)

    # values of 0 are interpreted as False, everything else as True.
    image_mask = image[0:1, :, :] > 0
    heatmap = heatmap.unsqueeze(dim=0)
    segment = segment.unsqueeze(dim=0)

    # heatmap_mask =  heatmap > 0 # 0.1
    # heatmap_mask = image_mask & heatmap_mask # mask both input and output
    segment = (segment > 0.5).float()
    # segment_mask = segment_mask & image_mask


    pr_heatmap_ma = np.ma.array(heatmap.numpy(), mask=~image_mask.numpy())
    pr_segment_ma = np.ma.array(segment.numpy(), mask=~image_mask.numpy())
    image_mask_np = 1*image_mask.numpy()  # convert bool to int

    # 不用 segment 掩膜，只用 image_mask（或完全不用 mask）
    # image_mask = image[0:1, :, :] > 0
    # heatmap = heatmap.unsqueeze(0)         # [1,H,W]

    # pr_heatmap_ma = np.ma.array(heatmap.numpy(), mask=~image_mask)
    # pr_segment_ma = np.ma.array(image_mask.numpy(), mask=False)
    # image_mask_np = image_mask.numpy()

    return pr_heatmap_ma, pr_segment_ma, image_mask_np

def save_predictions(
    export_path: Path,
    # bbox: BoundingBox, rs,
    res: int,
    transform,
    crs: CRS,
    pr_heatmap: np.ndarray,
    pr_segment: np.ndarray,
    pr_count: np.ndarray,
    image_count: np.ndarray,
) -> bool:
    
    import rasterio
    import numpy as np

    def to_hw(x, name):
        x = np.asarray(x, dtype=np.float32)
        x = np.squeeze(x)          # 去掉所有长度为1的维度
        if x.ndim != 2:
            raise ValueError(f"{name} 期望 squeeze 后为 (H,W)，但得到 {x.shape}")
        return x

    pr_heatmap  = to_hw(pr_heatmap,  "pr_heatmap")
    pr_segment  = to_hw(pr_segment,  "pr_segment")
    pr_count    = to_hw(pr_count,    "pr_count")
    image_count = to_hw(image_count, "image_count")

    print("save_predictions shapes (H,W):")
    print("  pr_heatmap :", pr_heatmap.shape)
    print("  pr_segment :", pr_segment.shape)
    print("  pr_count   :", pr_count.shape)
    print("  image_count:", image_count.shape)

    height, width = pr_heatmap.shape

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
        "transform": transform,
        "compress": "lzw",
    }

    # ---- 2. 组合成 (4,H,W) 的 3D 数组一次写入 ----
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

# def predict_and_export(
#         rts_model,
#         trainer: pl.Trainer,
#         datamodule,
#         model_name: str,
#         date_str: str,
#         export_dir: Path,
# ):
#     predict_dataloader = datamodule.predict_dataloader()
#     target_bbox = datamodule.roi
#     res=30

#     predictions_list = trainer.predict(
#         model=rts_model, dataloaders=predict_dataloader
#     )

#     pr_heatmap_sum = None
#     pr_segment_sum = None
#     image_count = None
#     target_crs = None
#     target_transform = None

#     for data_batch, predictions in zip(predict_dataloader, predictions_list):
#         pr_heatmaps = predictions['heatmap']     # [B,1,H,W]
#         pr_segments = predictions['prob_mask']   # [B,1,H,W]
#         image_batch = data_batch['image_t']      # [B,C,H,W]

#         idx = 0  # batch_size = 1
#         image = image_batch[idx]           # [C,H,W]
#         pr_heatmap = pr_heatmaps[idx, 0]   # [H,W]
#         pr_segment = pr_segments[idx, 0]   # [H,W]
        
#         data_bbox = data_batch['bbox']
#         if isinstance(data_bbox, (list, tuple)):
#             data_bbox = data_bbox[idx]
        
#         target_crs = data_batch['crs']
#         if isinstance(target_crs, (list, tuple)):
#             target_crs = target_crs[idx]

#         if target_transform is None:
#             from rasterio.transform import from_origin
#             import math
#             # snap target_bbox origin 到数据网格
#             snap_minx = math.floor(target_bbox.minx / res) * res
#             snap_maxy = math.ceil(target_bbox.maxy / res) * res
#             target_transform = from_origin(snap_minx, snap_maxy, res, res)

#         # 展开到整图坐标
#         pr_heatmap_exp, pr_segment_exp, image_mask_exp = expand_and_mask(
#             image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
#         )

#         def to_2d_numpy(arr):
#             """将数组转换为 2D numpy 数组"""
#             if hasattr(arr, 'filled'):
#                 arr = arr.filled(0)
#             if hasattr(arr, 'numpy'):
#                 arr = arr.numpy()
#             arr = np.array(arr, dtype=np.float32)
#             while arr.ndim > 2:
#                 arr = arr.squeeze(0)
#             return arr
        
#         pr_heatmap_np = to_2d_numpy(pr_heatmap_exp)
#         pr_segment_np = to_2d_numpy(pr_segment_exp)
#         image_mask_np = to_2d_numpy(image_mask_exp)

#         if pr_heatmap_sum is None:
#             pr_heatmap_sum = np.zeros_like(pr_heatmap_np, dtype=np.float32)
#             pr_segment_sum = np.zeros_like(pr_segment_np, dtype=np.float32)
#             image_count = np.zeros_like(pr_heatmap_np, dtype=np.float32)

#         valid_mask = image_mask_np > 0
#         pr_heatmap_sum[valid_mask] += pr_heatmap_np[valid_mask]
#         pr_segment_sum[valid_mask] += pr_segment_np[valid_mask]
#         image_count[valid_mask] += 1

#     image_count_safe = np.maximum(image_count, 1)
#     pr_heatmap_mean = pr_heatmap_sum / image_count_safe
#     pr_segment_mean = pr_segment_sum / image_count_safe

#     pr_heatmap_mean[image_count == 0] = 0
#     pr_segment_mean[image_count == 0] = 0

#     export_dir.mkdir(parents=True, exist_ok=True)

#     print("pr_heatmap_mean shape:", pr_heatmap_mean.shape)
#     print(f"image_count 范围: [{image_count.min()}, {image_count.max()}]")

#     pr_tiff_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

#     save_predictions(
#         export_path=pr_tiff_path,
#         res=res,
#         transform=target_transform,
#         crs=target_crs,
#         pr_heatmap=pr_heatmap_mean,
#         pr_segment=pr_segment_mean,
#         pr_count=pr_segment_sum,
#         image_count=image_count,
#     )
    
#     print(f"Successfully exported to: {pr_tiff_path}")
#     return True

def predict_and_export_georef(
    rts_model, trainer, datamodule,
    model_name, date_str, export_dir
):
    predict_dataloader = datamodule.predict_dataloader()
    target_bbox = datamodule.roi
    res = 30  # ✅ 明确定义

    H = int(round((target_bbox.maxy - target_bbox.miny) / res))
    W = int(round((target_bbox.maxx - target_bbox.minx) / res))

    heatmap_sum = np.zeros((H, W), dtype=np.float32)
    segment_sum = np.zeros((H, W), dtype=np.float32)
    count_arr   = np.zeros((H, W), dtype=np.float32)

    from rasterio.transform import from_origin
    global_transform = from_origin(
        target_bbox.minx, target_bbox.maxy, res, res
    )
    target_crs = None

    predictions_list = trainer.predict(
        model=rts_model, dataloaders=predict_dataloader
    )

    for data_batch, predictions in zip(predict_dataloader, predictions_list):
        pr_heatmap = predictions['heatmap'][0, 0].numpy()
        pr_segment = predictions['prob_mask'][0, 0].numpy()
        image_t    = data_batch['image_t'][0]

        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[0]

        if target_crs is None:
            target_crs = data_batch['crs']
            if isinstance(target_crs, (list, tuple)):
                target_crs = target_crs[0]

        col_off = int(round((data_bbox.minx - target_bbox.minx) / res))
        row_off = int(round((target_bbox.maxy - data_bbox.maxy) / res))
        patch_h, patch_w = pr_heatmap.shape

        row_end = min(row_off + patch_h, H)
        col_end = min(col_off + patch_w, W)
        patch_h_clip = row_end - row_off
        patch_w_clip = col_end - col_off

        if row_off < 0 or col_off < 0 or patch_h_clip <= 0 or patch_w_clip <= 0:
            continue

        valid = (image_t[0, :patch_h_clip, :patch_w_clip].numpy() != 0)
        heatmap_sum[row_off:row_end, col_off:col_end][valid] += pr_heatmap[:patch_h_clip, :patch_w_clip][valid]
        segment_sum[row_off:row_end, col_off:col_end][valid] += pr_segment[:patch_h_clip, :patch_w_clip][valid]
        count_arr  [row_off:row_end, col_off:col_end][valid] += 1

    safe_count = np.maximum(count_arr, 1)
    heatmap_mean = heatmap_sum / safe_count
    segment_mean = segment_sum / safe_count
    heatmap_mean[count_arr == 0] = 0
    segment_mean[count_arr == 0] = 0

    segment_binary = (segment_mean > 0.5).astype(np.float32)
    heatmap_mean = heatmap_mean * segment_binary
    
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

    save_predictions(
        export_path=out_path,
        res=res,
        transform=global_transform,
        crs=target_crs,
        pr_heatmap=heatmap_mean,
        pr_segment=segment_mean,
        pr_count=count_arr,
        image_count=count_arr,
    )
    return True