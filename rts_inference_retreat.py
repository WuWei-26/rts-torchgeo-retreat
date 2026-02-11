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
    """
    推理用的成对数据集：只返回影像与 DEM：
      keys: image_t, image_tm1, dem_t, dem_tm1
    - img_ds: Landsat8SR 或 Union(Landsat8SR, Landsat57SR)
    - dem_ds: MeanTPI（或其它 DEM/TPI 数据集）
    """

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
    # segment_mask = segment > 0.5
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
    bbox: BoundingBox,
    crs: CRS,
    pr_heatmap: np.ndarray,
    pr_segment: np.ndarray,
    pr_count: np.ndarray,
    image_count: np.ndarray,
) -> bool:
    """
    将整图预测写出为 4 波段的 GeoTIFF:
      band1: pr_heatmap
      band2: pr_segment
      band3: pr_count
      band4: image_count

    所有输入可以是 (H,W) 或 (1,H,W) 或 (B,1,H,W)，函数内部统一为 (4,H,W) 再写入。
    """
    import rasterio
    import numpy as np

    # ---- 1. 转为 numpy & squeeze 到 (H,W) ----
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
    rio_transform = rasterio.transform.from_bounds(
        bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height
    )

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

# def save_predictions(
#         export_path: Path,
#         bbox: BoundingBox,
#         crs: CRS,
#         pr_heatmap: np.ndarray,
#         pr_segment: np.ndarray,
#         pr_count: np.ndarray,
#         image_count: np.ndarray
# ):
#     height = pr_heatmap.shape[-2]
#     width = pr_heatmap.shape[-1]
#     rio_transform = rasterio.transform.from_bounds(
#         bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height)  # west, south, east, north, width, height

#     # .format(test_year=test_year, composite=composite, stem=filepath.stem, model=model_name, suffix=filepath.suffix)
#     with rasterio.open(
#             export_path,
#             mode="w",
#             driver="GTiff",
#             height=height,
#             width=width,
#             count=4,
#             dtype=np.dtype('float32'),
#             crs=crs,
#             transform=rio_transform
#     ) as pr_mask_dataset:
#         pr_mask_dataset.write(pr_heatmap, 1)  # index start from 1
#         pr_mask_dataset.set_band_description(1, 'heatmap')
#         pr_mask_dataset.write(pr_segment, 2)  # index start from 1
#         pr_mask_dataset.set_band_description(2, 'segment')
#         pr_mask_dataset.write(pr_count, 3)  # index start from 1
#         pr_mask_dataset.set_band_description(3, 'count')
#         pr_mask_dataset.write(image_count, 4)  # index start from 1
#         pr_mask_dataset.set_band_description(4, 'image_count')
#     print('Prediction results saved: \n', export_path)
#     return True


# def show_results(image, heatmap, bright):
#     plt.figure(figsize=(16, 8))

#     plt_col = 2
#     plt.subplot(1, plt_col, 1)
#     # print(image.max())
#     # convert CHW -> HWC
#     plt.imshow((image*bright).clamp(min=0, max=1).permute(1, 2, 0))
#     plt.title("Input Image (R G B)")
#     plt.axis("off")

#     plt.subplot(1, plt_col, 2)
#     print('heatmap max value:', heatmap.max())
#     # just squeeze classes dim, because we have only one class
#     plt.imshow(heatmap.clip(min=0.05, max=1.2), cmap='jet')
#     plt.title("Heatmap Prediction")
#     plt.axis("off")

#     # plt.subplot(1, plt_col, 2)
#     # plt.imshow(count, cmap='gray') # just squeeze classes dim, because we have only one class
#     # plt.title("Heatmap GT")
#     # plt.axis("off")
#     plt.show()

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

    # 使用累加数组和计数数组来计算平均值
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
        
        # ✅ 修复：正确获取 bbox
        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[idx]
        
        # ✅ 修复：正确获取 crs
        target_crs = data_batch['crs']
        if isinstance(target_crs, (list, tuple)):
            target_crs = target_crs[idx]

        # 展开到整图坐标
        pr_heatmap_exp, pr_segment_exp, image_mask_exp = expand_and_mask(
            image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
        )

        # ✅ 修复：确保转换为 2D numpy 数组
        def to_2d_numpy(arr):
            """将数组转换为 2D numpy 数组"""
            if hasattr(arr, 'filled'):
                arr = arr.filled(0)
            if hasattr(arr, 'numpy'):
                arr = arr.numpy()
            arr = np.array(arr, dtype=np.float32)
            # 去除多余维度，确保是 2D
            while arr.ndim > 2:
                arr = arr.squeeze(0)
            return arr
        
        pr_heatmap_np = to_2d_numpy(pr_heatmap_exp)
        pr_segment_np = to_2d_numpy(pr_segment_exp)
        image_mask_np = to_2d_numpy(image_mask_exp)

        # 初始化累加数组
        if pr_heatmap_sum is None:
            pr_heatmap_sum = np.zeros_like(pr_heatmap_np, dtype=np.float32)
            pr_segment_sum = np.zeros_like(pr_segment_np, dtype=np.float32)
            image_count = np.zeros_like(pr_heatmap_np, dtype=np.float32)

        # 只在有效区域累加
        valid_mask = image_mask_np > 0
        pr_heatmap_sum[valid_mask] += pr_heatmap_np[valid_mask]
        pr_segment_sum[valid_mask] += pr_segment_np[valid_mask]
        image_count[valid_mask] += 1

    # 计算平均值（避免除以0）
    image_count_safe = np.maximum(image_count, 1)
    pr_heatmap_mean = pr_heatmap_sum / image_count_safe
    pr_segment_mean = pr_segment_sum / image_count_safe

    # 无覆盖区域设为0
    pr_heatmap_mean[image_count == 0] = 0
    pr_segment_mean[image_count == 0] = 0

    export_dir.mkdir(parents=True, exist_ok=True)

    print("pr_heatmap_mean shape:", pr_heatmap_mean.shape)
    print(f"image_count 范围: [{image_count.min()}, {image_count.max()}]")

    # 保存结果
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

# def predict_and_export(
#         rts_model,
#         trainer: pl.Trainer,
#         datamodule: LandsatPairInferenceDataModule,
#         model_name: str,
#         date_str: str,
#         export_dir: Path,
# ):
#     predict_dataloader = datamodule.predict_dataloader()
#     target_bbox = datamodule.roi

#     predictions_list = trainer.predict(
#         model=rts_model, dataloaders=predict_dataloader
#     )

#     # 使用累加数组和计数数组来计算平均值
#     pr_heatmap_sum = None
#     pr_segment_sum = None
#     image_count = None
#     target_crs = None

#     for data_batch, predictions in zip(predict_dataloader, predictions_list):
#         pr_heatmaps = predictions['heatmap']     # [B,1,H,W]
#         pr_segments = predictions['prob_mask']   # [B,1,H,W]
#         image_batch = data_batch['image_t']      # [B,C,H,W]

#         idx = 0  # batch_size = 1
#         image = image_batch[idx]
#         pr_heatmap = pr_heatmaps[idx, 0]
#         pr_segment = pr_segments[idx, 0]
#         data_bbox = data_batch['bbox'][idx]

#         img_rgb = data_batch["image_t"][0, datamodule.rgb_indexes, :, :]
#         target_crs = data_batch['crs'][0]

#         # 展开到整图坐标
#         pr_heatmap_exp, pr_segment_exp, image_mask_exp = expand_and_mask(
#             image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
#         )

#         # 转换为 numpy（如果还是 tensor）
#         if hasattr(pr_heatmap_exp, 'numpy'):
#             pr_heatmap_np = pr_heatmap_exp.numpy() if not hasattr(pr_heatmap_exp, 'filled') else pr_heatmap_exp.filled(0)
#         elif hasattr(pr_heatmap_exp, 'filled'):
#             pr_heatmap_np = pr_heatmap_exp.filled(0)
#         else:
#             pr_heatmap_np = np.array(pr_heatmap_exp)
            
#         if hasattr(pr_segment_exp, 'numpy'):
#             pr_segment_np = pr_segment_exp.numpy() if not hasattr(pr_segment_exp, 'filled') else pr_segment_exp.filled(0)
#         elif hasattr(pr_segment_exp, 'filled'):
#             pr_segment_np = pr_segment_exp.filled(0)
#         else:
#             pr_segment_np = np.array(pr_segment_exp)
            
#         if hasattr(image_mask_exp, 'numpy'):
#             image_mask_np = image_mask_exp.numpy() if hasattr(image_mask_exp, 'numpy') else np.array(image_mask_exp)
#         else:
#             image_mask_np = np.array(image_mask_exp)

#         # 初始化累加数组
#         if pr_heatmap_sum is None:
#             pr_heatmap_sum = np.zeros_like(pr_heatmap_np, dtype=np.float32)
#             pr_segment_sum = np.zeros_like(pr_segment_np, dtype=np.float32)
#             image_count = np.zeros_like(image_mask_np, dtype=np.float32)

#         # 只在有效区域累加
#         valid_mask = image_mask_np > 0
#         pr_heatmap_sum[valid_mask] += pr_heatmap_np[valid_mask]
#         pr_segment_sum[valid_mask] += pr_segment_np[valid_mask]
#         image_count[valid_mask] += 1

#     # 计算平均值（避免除以0）
#     image_count_safe = np.maximum(image_count, 1)
#     pr_heatmap_mean = pr_heatmap_sum / image_count_safe
#     pr_segment_mean = pr_segment_sum / image_count_safe

#     # 无覆盖区域设为0
#     pr_heatmap_mean[image_count == 0] = 0
#     pr_segment_mean[image_count == 0] = 0

#     export_dir.mkdir(parents=True, exist_ok=True)

#     print("pr_heatmap_mean shape:", pr_heatmap_mean.shape)
#     print(f"image_count 范围: [{image_count.min()}, {image_count.max()}]")

#     # 可视化
#     try:
#         show_results(image=img_rgb, heatmap=pr_heatmap_mean, bright=3)
#     except Exception as e:
#         print("show_results ERROR", e)

#     # 保存结果 - 使用 mean 而非 median
#     pr_tiff_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

#     save_predictions(
#         export_path=pr_tiff_path,
#         bbox=target_bbox,
#         crs=target_crs,
#         pr_heatmap=pr_heatmap_mean,
#         pr_segment=pr_segment_mean,
#         pr_count=pr_segment_sum,
#         image_count=image_count,
#     )
    
#     print(f"Prediction exported to: {pr_tiff_path}")
#     return True

# def predict_and_export(
#         rts_model,
#         trainer: pl.Trainer,
#         datamodule: LandsatPairInferenceDataModule,
#         model_name: str,
#         date_str: str,
#         export_dir: Path,
# ):

#     predict_dataloader = datamodule.predict_dataloader()
#     target_bbox = datamodule.roi

#     predictions_list = trainer.predict(
#         model=rts_model, dataloaders=predict_dataloader
#     )

#     pr_heatmap_list_np = []
#     pr_segment_list_np = []
#     image_mask_list_np = []

#     # pr_heatmap_sum = None
#     # pr_segment_sum = None
#     # image_count = None
#     # target_crs = None
    
#     for data_batch, predictions in zip(predict_dataloader, predictions_list):
#         pr_heatmaps = predictions['heatmap']     # [B,1,H,W]
#         pr_segments = predictions['prob_mask']   # [B,1,H,W] 或 segment prob
#         image_batch = data_batch['image_t']      # [B,C,H,W]

#         idx = 0  # batch_size = 1
#         image = image_batch[idx]                  # [C,H,W]
#         pr_heatmap = pr_heatmaps[idx, 0]              # [H,W]
#         pr_segment = pr_segments[idx, 0]              # [H,W]
#         data_bbox = data_batch['bbox'][idx]

#         # 直接从数据批次获取必要信息（只为了 show / profile）
#         img_rgb = data_batch["image_t"][0, datamodule.rgb_indexes, :, :]
#         target_crs = data_batch['crs'][0]

#         # 展开到整图坐标 & 掩膜
#         pr_heatmap_ma, pr_segment_ma, image_mask_np = expand_and_mask(
#             image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30
#         )
#         pr_heatmap_list_np.append(pr_heatmap_ma)
#         pr_segment_list_np.append(pr_segment_ma)
#         image_mask_list_np.append(image_mask_np)

#     # ---- 拼整图：对所有 patch 沿 axis=0 做 median ----
#     pr_segment_concat_np = np.ma.concatenate(pr_segment_list_np, axis=0)   # [N,1,H,W] masked
#     pr_heatmap_concat_np = np.ma.concatenate(pr_heatmap_list_np, axis=0)   # [N,1,H,W] masked

#     pr_segment_median = np.ma.median(pr_segment_concat_np, axis=0).filled(fill_value=0)   # (1,H,W)
#     pr_heatmap_median = np.ma.median(pr_heatmap_concat_np, axis=0).filled(fill_value=0)   # (1,H,W)

#     # 计数/覆盖次数
#     pr_heatmap_count = np.ma.sum(pr_segment_concat_np, axis=0).filled(fill_value=0)       # (1,H,W)
#     image_mask_concat_np = np.concatenate(image_mask_list_np, axis=0)                     # (N,1,H,W)
#     image_count = np.sum(image_mask_concat_np, axis=0)                                    # (1,H,W)

#     export_dir.mkdir(parents=True, exist_ok=True)

#     print("pr_heatmap_median shape:", pr_heatmap_median.shape)

#     # 可视化看一眼整图（只取 band 和 heatmap）
#     try:
#         show_results(image=img_rgb,  # [C,H_patch,W_patch]；对整图可以接受整体色调
#                      heatmap=pr_heatmap_median,
#                      bright=3)
#     except Exception as e:
#         print("show_results ERROR", e)

#     composites = ['median']
#     for composite in composites:
#         pr_tiff_path = export_dir / f"pr_mask_{date_str}_{composite}_{model_name}.tif"

#         save_predictions(
#             export_path=pr_tiff_path,
#             bbox=target_bbox,
#             crs=target_crs,
#             pr_heatmap=pr_heatmap_median,    # (1,H,W)
#             pr_segment=pr_segment_median,    # (1,H,W)
#             pr_count=pr_heatmap_count,       # (1,H,W)
#             image_count=image_count,         # (1,H,W)
#         )
#     return True

# def predict_and_export(
#         rts_model,
#         trainer: pl.Trainer,
#         datamodule: LandsatPairInferenceDataModule,
#         model_name: str,
#         date_str: str,
#         export_dir: Path,
# ):

#     predict_dataloader = datamodule.predict_dataloader()
#     target_bbox = datamodule.roi
#     # test_year = datamodule.year

#     predictions_list = trainer.predict(
#         model=rts_model, dataloaders=datamodule.predict_dataloader())

#     pr_heatmap_list_np = []
#     pr_segment_list_np = []
#     image_mask_list_np = []

#     raster_filepath_set = False
#     # TODO: avoid call dataloader twice here, find a way to save the meta data information
#     for data_batch, predictions in zip(predict_dataloader, predictions_list):
#         # pr_heatmaps = predictions['pred_heat']
#         # pr_segments = predictions['prob_mask']
#         pr_heatmaps = predictions['heatmap']
#         pr_segments = predictions['prob_mask']
#         image_batch = data_batch['image_t']

#         idx = 0  # batch_size = 1
#         image = image_batch[idx, :, :, :]
#         pr_heatmap = pr_heatmaps[idx, :, :]
#         pr_segment = pr_segments[idx, :, :]
#         data_bbox = data_batch['bbox'][idx]
#         # filepath = data_batch['path'][idx]

#         # if 'median' in filepath and 'geometric' not in filepath:
#     #     if 'mosaic' in filepath:
#     #         # data_batch_median = data_batch
#     #         img_rgb = data_batch["image"][0, datamodule.rgb_indexes, :, :]
#     #         target_crs = data_batch['crs'][0]
#     #         raster_filepath = Path(filepath)
#     #         raster_filepath_set = True
#     #         print(f'Raster for visualizing:\n{filepath}')
#     #     else:
#     #         print(filepath)

#     #     pr_heatmap_ma, pr_segment_ma, image_mask_np = expand_and_mask(
#     #         image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30)
#     #     pr_heatmap_list_np.append(pr_heatmap_ma)
#     #     pr_segment_list_np.append(pr_segment_ma)
#     #     image_mask_list_np.append(image_mask_np)

#     # if not raster_filepath_set:
#     #     img_rgb = data_batch["image"][0, datamodule.rgb_indexes, :, :]
#     #     target_crs = data_batch['crs'][0]
#     #     raster_filepath = Path(filepath)
#     #     raster_filepath_set = True
#     #     print(f'Raster for visualizing:\n{filepath}')

#         # 直接从数据批次获取必要信息
#         img_rgb = data_batch["image_t"][0, datamodule.rgb_indexes, :, :]
#         target_crs = data_batch['crs'][0]

#         pr_heatmap_ma, pr_segment_ma, image_mask_np = expand_and_mask(
#             image, pr_heatmap, pr_segment, data_bbox, target_bbox, res=30)
#         pr_heatmap_list_np.append(pr_heatmap_ma)
#         pr_segment_list_np.append(pr_segment_ma)
#         image_mask_list_np.append(image_mask_np)

#     pr_segment_concat_np = np.ma.concatenate(pr_segment_list_np, axis=0)
#     pr_heatmap_concat_np = np.ma.concatenate(pr_heatmap_list_np, axis=0)
#     pr_segment_median = np.ma.median(
#         pr_segment_concat_np, axis=0).filled(fill_value=0)
#     pr_heatmap_median = np.ma.median(
#         pr_heatmap_concat_np, axis=0).filled(fill_value=0)
#     # pr_segment_mean = np.ma.mean(
#     #     pr_segment_concat_np, axis=0).filled(fill_value=0)
#     # pr_heatmap_mean = np.ma.mean(
#     #     pr_heatmap_concat_np, axis=0).filled(fill_value=0)

#     # pr_heatmap_count = np.ma.count(pr_heatmap_concat_np, axis=0) # Count the non-masked elements of the array along the given axis.
#     pr_heatmap_count = np.ma.sum(pr_segment_concat_np, axis=0).filled(
#         fill_value=0)  # sum the segment values
#     image_mask_concat_np = np.concatenate(image_mask_list_np, axis=0)
#     image_count = np.sum(image_mask_concat_np, axis=0)
#     # count_mask = pr_heatmap_count >= 2
#     # pr_heatmap_median[~count_mask] = 0
#     # pr_segment_median[~count_mask] = 0
#     # pr_heatmap_mean[~count_mask] = 0
#     # pr_segment_mean[~count_mask] = 0
#     # mask heatmap and segment with the count mask

#     # export_dir = Path(f'/DATA/DATA1/joey/pr_mask_rts_aea_{suffix}')
#     export_dir.mkdir(parents=True, exist_ok=True)
    
#     # DEBUG
#     print("pr_heatmap_median shape:", pr_heatmap_median.shape) # DEBUG
#     # squeeze 通道维 -> (H,W)
#     # if pr_heatmap_median.ndim == 3 and pr_heatmap_median.shape[0] == 1:
#     #     pr_heatmap_median = pr_heatmap_median[0]   # (H,W)
#     # if pr_segment_median.ndim == 3 and pr_segment_median.shape[0] == 1:
#     #     pr_segment_median = pr_segment_median[0]   # (H,W)
#     # #

#     show_results(image=img_rgb,
    
#                  heatmap=pr_heatmap_median,
#                  bright=3)
#     composites = ['median']
#     # pr_heatmap_list = [pr_heatmap_median, pr_heatmap_mean]
#     # pr_segment_list = [pr_segment_median, pr_segment_mean]
#     for composite in composites:
#         # pr_tiff_path = export_dir / \
#             # f"pr_mask_{date_str}_{composite}_{model_name}_{raster_filepath.stem}{raster_filepath.suffix}"
#             # 使用固定的文件名格式，移除原始文件名相关部分
#         pr_tiff_path = export_dir / f"pr_mask_{date_str}_{composite}_{model_name}.tif"
 
#         save_predictions(
#             export_path=pr_tiff_path,
#             bbox=target_bbox,
#             crs=target_crs,
#             pr_heatmap=pr_heatmap_median,
#             pr_segment=pr_segment_median,
#             pr_count=pr_heatmap_count,
#             image_count=image_count
#         )
#     return True
