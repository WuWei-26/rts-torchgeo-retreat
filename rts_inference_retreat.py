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

# def predict_and_export_georef(
#     rts_model, trainer, datamodule,
#     model_name, date_str, export_dir,
# ):
#     from rasterio.transform import from_origin

#     predict_dataloader = datamodule.predict_dataloader()
#     target_bbox = datamodule.roi

#     # ✅ 从推理影像文件直接获取标准网格
#     landsat_path = datamodule.dataset.img_ds.index.intersection(
#         datamodule.dataset.img_ds.index.bounds, objects=True
#     )
#     # 更简单：从第一个batch获取path_t然后打开文件
#     first_batch = next(iter(predict_dataloader))
#     path_t = first_batch.get("path_t")
#     if isinstance(path_t, (list, tuple)):
#         path_t = path_t[0]

#     with rasterio.open(path_t) as src:
#         ls_transform = src.transform   # origin=(438075, 3930075), res=30
#         ls_crs = src.crs
#         print(f"Landsat transform: {ls_transform}")

#     res_x = ls_transform.a
#     res_y = abs(ls_transform.e)
#     roi_origin_x = ls_transform.c
#     roi_origin_y = ls_transform.f

#     H = int(round((target_bbox.maxy - target_bbox.miny) / res_y))
#     W = int(round((target_bbox.maxx - target_bbox.minx) / res_x))
#     global_transform = rasterio.transform.Affine(
#         res_x, 0, roi_origin_x,
#         0, -res_y, roi_origin_y
#     )
#     print(f"整图尺寸: {H}x{W}, transform: {global_transform}")

#     heatmap_sum = np.zeros((H, W), dtype=np.float32)
#     segment_sum = np.zeros((H, W), dtype=np.float32)
#     count_arr   = np.zeros((H, W), dtype=np.float32)
#     target_crs  = ls_crs

#     predictions_list = trainer.predict(
#         model=rts_model, dataloaders=predict_dataloader
#     )

#     for data_batch, predictions in zip(predict_dataloader, predictions_list):
#         pr_heatmap = predictions['heatmap'][0, 0].numpy()
#         pr_segment = predictions['prob_mask'][0, 0].numpy()
#         image_t    = data_batch['image_t'][0]

#         data_bbox = data_batch['bbox']
#         if isinstance(data_bbox, (list, tuple)):
#             data_bbox = data_bbox[0]

#         # ✅ snap到Landsat网格
#         col_off = int(round((data_bbox.minx - roi_origin_x) / res_x))
#         row_off = int(round((roi_origin_y - data_bbox.maxy) / res_y))

#         patch_h, patch_w = pr_heatmap.shape
#         row_end = min(row_off + patch_h, H)
#         col_end = min(col_off + patch_w, W)
#         patch_h_clip = row_end - row_off
#         patch_w_clip = col_end - col_off

#         if row_off < 0 or col_off < 0 or patch_h_clip <= 0 or patch_w_clip <= 0:
#             continue

#         valid = (image_t[0, :patch_h_clip, :patch_w_clip].numpy() != 0)
#         heatmap_sum[row_off:row_end, col_off:col_end][valid] += pr_heatmap[:patch_h_clip, :patch_w_clip][valid]
#         segment_sum[row_off:row_end, col_off:col_end][valid] += pr_segment[:patch_h_clip, :patch_w_clip][valid]
#         count_arr  [row_off:row_end, col_off:col_end][valid] += 1

#     safe_count = np.maximum(count_arr, 1)
#     heatmap_mean = heatmap_sum / safe_count
#     segment_mean = segment_sum / safe_count
#     heatmap_mean[count_arr == 0] = 0
#     segment_mean[count_arr == 0] = 0

#     segment_binary = (segment_mean > 0.5).astype(np.float32)
#     heatmap_mean   = heatmap_mean * segment_binary

#     export_dir.mkdir(parents=True, exist_ok=True)
#     out_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

#     save_predictions(
#         export_path=out_path,
#         res=int(res_x),
#         transform=global_transform,
#         crs=target_crs,
#         pr_heatmap=heatmap_mean,
#         pr_segment=segment_mean,
#         pr_count=count_arr,
#         image_count=count_arr,
#     )
#     return True

def predict_and_export_georef(
    rts_model, trainer, datamodule,
    model_name, date_str, export_dir,
    aggregation_method='median',
):
    from rasterio.transform import from_origin

    predict_dataloader = datamodule.predict_dataloader()
    target_bbox = datamodule.roi

    # ✅ 直接从Landsat影像文件获取标准网格，不依赖第一个batch
    # 找year_t对应的影像文件
    from datetime import datetime
    year_t = datamodule.year_t
    year_mint = datetime(year_t, 1, 1).timestamp()
    year_maxt = datetime(year_t + 1, 1, 1).timestamp()
    bbox_year_t = BoundingBox(
        target_bbox.minx, target_bbox.maxx,
        target_bbox.miny, target_bbox.maxy,
        year_mint, year_maxt
    )

    # 从img_dataset里找year_t的文件
    def collect_leaf_datasets(ds):
        from torchgeo.datasets import UnionDataset
        if isinstance(ds, UnionDataset):
            result = []
            for sub in ds.datasets:
                result.extend(collect_leaf_datasets(sub))
            return result
        return [ds]

    all_ds = collect_leaf_datasets(datamodule.img_dataset)
    year_t_ds = [
        ds for ds in all_ds
        if ds.bounds.mint <= year_mint and ds.bounds.maxt >= year_maxt
    ]

    landsat_path = None
    for ds in year_t_ds:
        hits = list(ds.index.intersection(tuple(bbox_year_t), objects=True))
        if hits:
            landsat_path = hits[0].object
            break

    if landsat_path is None:
        raise ValueError(f"找不到year_t={year_t}对应的Landsat影像文件")

    with rasterio.open(landsat_path) as src:
        ls_transform = src.transform
        ls_crs = src.crs
        print(f"Landsat文件: {landsat_path}")
        print(f"Landsat transform: {ls_transform}")

    res_x = ls_transform.a
    res_y = abs(ls_transform.e)
    roi_origin_x = ls_transform.c  # 整个影像文件的origin
    roi_origin_y = ls_transform.f
    target_crs = ls_crs

    H = int(round((target_bbox.maxy - target_bbox.miny) / res_y))
    W = int(round((target_bbox.maxx - target_bbox.minx) / res_x))
    
    # ✅ target_bbox相对于Landsat文件origin的偏移
    col_start = int(round((target_bbox.minx - roi_origin_x) / res_x))
    row_start = int(round((roi_origin_y - target_bbox.maxy) / res_y))
    
    # 输出文件的origin = 文件origin + ROI偏移
    out_origin_x = roi_origin_x + col_start * res_x
    out_origin_y = roi_origin_y - row_start * res_y
    global_transform = rasterio.transform.Affine(
        res_x, 0, out_origin_x,
        0, -res_y, out_origin_y
    )
    print(f"image size: {H}x{W}")
    print(f"output transform: {global_transform}")

    heatmap_sum = np.zeros((H, W), dtype=np.float32)
    segment_sum = np.zeros((H, W), dtype=np.float32)
    count_arr   = np.zeros((H, W), dtype=np.float32)

    # 🔧 使用字典存储多个预测值（更高效）
    heatmap_dict = {}  # (i, j) -> [值1, 值2, ...]
    segment_dict = {}
    count_arr = np.zeros((H, W), dtype=np.float32)

    predictions_list = trainer.predict(
        model=rts_model, dataloaders=predict_dataloader
    )

    batch_idx = 0
    for data_batch, predictions in zip(predict_dataloader, predictions_list):
        pr_heatmap = predictions['heatmap'][0, 0].numpy()
        pr_segment = predictions['prob_mask'][0, 0].numpy()
        image_t    = data_batch['image_t'][0]

        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[0]

        col_off = int(round((data_bbox.minx - out_origin_x) / res_x))
        row_off = int(round((out_origin_y - data_bbox.maxy) / res_y))

        patch_h, patch_w = pr_heatmap.shape
        row_end = min(row_off + patch_h, H)
        col_end = min(col_off + patch_w, W)
        patch_h_clip = row_end - row_off
        patch_w_clip = col_end - col_off

        if row_off < 0 or col_off < 0 or patch_h_clip <= 0 or patch_w_clip <= 0:
            batch_idx += 1
            continue

        valid = (image_t[0, :patch_h_clip, :patch_w_clip].numpy() != 0)
        
        # 🔧 将有效像素存储到字典中
        valid_indices = np.where(valid)
        for local_i, local_j in zip(valid_indices[0], valid_indices[1]):
            global_i = row_off + local_i
            global_j = col_off + local_j
            
            key = (global_i, global_j)
            if key not in heatmap_dict:
                heatmap_dict[key] = []
                segment_dict[key] = []
            
            heatmap_dict[key].append(pr_heatmap[local_i, local_j])
            segment_dict[key].append(pr_segment[local_i, local_j])
            count_arr[global_i, global_j] += 1
        
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f" {batch_idx} batches processed, currently storing {len(heatmap_dict)} pixels")
    
    heatmap_result = np.zeros((H, W), dtype=np.float32)
    segment_result = np.zeros((H, W), dtype=np.float32)
    
    if aggregation_method == 'median':
        print("using median aggregation")
        for (i, j), values in heatmap_dict.items():
            heatmap_result[i, j] = np.median(values)
        for (i, j), values in segment_dict.items():
            segment_result[i, j] = np.median(values)
            
    elif aggregation_method == 'mean':
        print("using mean aggregation")
        for (i, j), values in heatmap_dict.items():
            heatmap_result[i, j] = np.mean(values)
        for (i, j), values in segment_dict.items():
            segment_result[i, j] = np.mean(values)
    else:
        raise ValueError(f"Unknown aggregation_method: {aggregation_method}")

    heatmap_mean = heatmap_result
    segment_mean = segment_result

    # 掩膜无数据区域
    heatmap_mean[count_arr == 0] = 0
    segment_mean[count_arr == 0] = 0

    segment_binary = (segment_mean > 0.5).astype(np.float32)
    heatmap_mean   = heatmap_mean * segment_binary

    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"pr_mask_{date_str}_{aggregation_method}_{model_name}.tif"

    print(f"  - heatmap range: [{heatmap_mean.min():.4f}, {heatmap_mean.max():.4f}]")
    print(f"  - segment range: [{segment_mean.min():.4f}, {segment_mean.max():.4f}]")

    save_predictions(
        export_path=out_path,
        res=int(res_x),
        transform=global_transform,
        crs=target_crs,
        pr_heatmap=heatmap_mean,
        pr_segment=segment_mean,
        pr_count=count_arr,
        image_count=count_arr,
    )
    return True

    # batch_idx = 0
    # for data_batch, predictions in zip(predict_dataloader, predictions_list):
    #     pr_heatmap = predictions['heatmap'][0, 0].numpy()
    #     pr_segment = predictions['prob_mask'][0, 0].numpy()
    #     image_t    = data_batch['image_t'][0]

    #     data_bbox = data_batch['bbox']
    #     if isinstance(data_bbox, (list, tuple)):
    #         data_bbox = data_bbox[0]

    #     # ✅ 相对于输出图像origin计算偏移
    #     col_off = int(round((data_bbox.minx - out_origin_x) / res_x))
    #     row_off = int(round((out_origin_y - data_bbox.maxy) / res_y))

    #     patch_h, patch_w = pr_heatmap.shape
    #     row_end = min(row_off + patch_h, H)
    #     col_end = min(col_off + patch_w, W)
    #     patch_h_clip = row_end - row_off
    #     patch_w_clip = col_end - col_off

    #     if row_off < 0 or col_off < 0 or patch_h_clip <= 0 or patch_w_clip <= 0:
    #         continue

    #     valid = (image_t[0, :patch_h_clip, :patch_w_clip].numpy() != 0)
    #     heatmap_sum[row_off:row_end, col_off:col_end][valid] += pr_heatmap[:patch_h_clip, :patch_w_clip][valid]
    #     segment_sum[row_off:row_end, col_off:col_end][valid] += pr_segment[:patch_h_clip, :patch_w_clip][valid]
    #     count_arr  [row_off:row_end, col_off:col_end][valid] += 1

    # safe_count = np.maximum(count_arr, 1)
    # heatmap_mean = heatmap_sum / safe_count
    # segment_mean = segment_sum / safe_count
    # heatmap_mean[count_arr == 0] = 0
    # segment_mean[count_arr == 0] = 0

    # segment_binary = (segment_mean > 0.5).astype(np.float32)
    # heatmap_mean   = heatmap_mean * segment_binary

    # export_dir.mkdir(parents=True, exist_ok=True)
    # out_path = export_dir / f"pr_mask_{date_str}_mean_{model_name}.tif"

    # save_predictions(
    #     export_path=out_path,
    #     res=int(res_x),
    #     transform=global_transform,
    #     crs=target_crs,
    #     pr_heatmap=heatmap_mean,
    #     pr_segment=segment_mean,
    #     pr_count=count_arr,
    #     image_count=count_arr,
    # )
    # return True