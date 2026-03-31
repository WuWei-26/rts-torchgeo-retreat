from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from datetime import datetime
import pytorch_lightning as pl
import rasterio
import matplotlib.pyplot as plt
from rts_utils import expand_to_bbox
from torchgeo.datasets import BoundingBox
from rts_datamodule_retreat import LandsatPairInferenceDataModule
from rasterio.crs import CRS
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import rasterio
from rasterio.transform import from_bounds, from_origin
from rasterio.windows import from_bounds as window_from_bounds

class PairTemporalDataset(torch.utils.data.Dataset):

    @staticmethod
    def _year_bbox(bbox: BoundingBox, year: int) -> BoundingBox:
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
    # expand the input raster
    image = expand_to_bbox(image, bbox, target_bbox, res)
    heatmap = expand_to_bbox(heatmap, bbox, target_bbox, res)
    segment = expand_to_bbox(segment, bbox, target_bbox, res)

    # values of 0 are interpreted as False, everything else as True.
    image_mask = image[0:1, :, :] > 0
    heatmap = heatmap.unsqueeze(dim=0)
    segment = segment.unsqueeze(dim=0)

    segment_mask = segment > 0.8
    segment_mask = segment_mask & image_mask

    pr_heatmap_ma = np.ma.array(heatmap.numpy(), mask=~segment_mask.numpy())
    pr_segment_ma = np.ma.array(segment.numpy(), mask=~segment_mask.numpy())
    image_mask_np = 1*image_mask.numpy()  # convert bool to int

    return pr_heatmap_ma, pr_segment_ma, image_mask_np

def save_predictions(
    export_path: Path,
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
        x = np.squeeze(x)
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

    all_bands = np.stack(
        [pr_heatmap, pr_segment, pr_count, image_count], axis=0
    )

    with rasterio.open(export_path, "w", **profile) as dst:
        dst.write(all_bands)
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

def predict_and_export_georef(
    rts_model, trainer, datamodule,
    model_name, date_str, export_dir,
    aggregation_method='median',
):

    predict_dataloader = datamodule.predict_dataloader()
    target_bbox = datamodule.roi

    # get transform and crs from a reference Landsat image
    from datetime import datetime
    year_t = datamodule.year_t
    year_mint = datetime(year_t, 1, 1).timestamp()
    year_maxt = datetime(year_t + 1, 1, 1).timestamp()
    bbox_year_t = BoundingBox(
        target_bbox.minx, target_bbox.maxx,
        target_bbox.miny, target_bbox.maxy,
        year_mint, year_maxt
    )

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
        raise ValueError(f"Could not find a reference Landsat image at year {year_t}.")

    with rasterio.open(landsat_path) as src:
        ref_transform = src.transform
        ref_crs = src.crs
        print(f"\n reference Landsat image: {landsat_path}")
        print(f"   Transform: {ref_transform}")
        print(f"   CRS: {ref_crs}")

    # rereference pred to the same grid as the reference image
    res_x = ref_transform.a
    res_y = abs(ref_transform.e)
    
    # origins of the reference image
    ref_origin_x = ref_transform.c  # x coordinate of the upper left
    ref_origin_y = ref_transform.f  # y coordinate of the upper left
    
    print(f"\n reference grid origin: ({ref_origin_x}, {ref_origin_y})")
    print(f"ROI bbox: ({target_bbox.minx}, {target_bbox.miny}) -> ({target_bbox.maxx}, {target_bbox.maxy})")
    
    col_float = (target_bbox.minx - ref_origin_x) / res_x
    row_float = (ref_origin_y - target_bbox.maxy) / res_y
    
    # align
    col_aligned = int(round(col_float))
    row_aligned = int(round(row_float))
    
    output_origin_x = ref_origin_x + col_aligned * res_x
    output_origin_y = ref_origin_y - row_aligned * res_y
    
    col_end_float = (target_bbox.maxx - ref_origin_x) / res_x
    row_end_float = (ref_origin_y - target_bbox.miny) / res_y
    col_end_aligned = int(round(col_end_float))
    row_end_aligned = int(round(row_end_float))
    
    W = col_end_aligned - col_aligned
    H = row_end_aligned - row_aligned
    
    global_transform = rasterio.transform.Affine(
        res_x, 0, output_origin_x,
        0, -res_y, output_origin_y
    )
    
    print(f"roi coordinates: col={col_float:.2f}, row={row_float:.2f}")
    print(f"aligned pixel coordinates: col={col_aligned}, row={row_aligned}")
    print(f"\n output image parameters:")
    print(f"size: {H} x {W}")
    print(f"pixel range: [{col_aligned}:{col_end_aligned}, {row_aligned}:{row_end_aligned}]")
    print(f"aligned upper left corner: ({output_origin_x}, {output_origin_y})")
    print(f"Transform: {global_transform}")
    print(f"Resolution: {res_x} x {res_y}")
    
    heatmap_dict = {}
    segment_dict = {}
    count_arr = np.zeros((H, W), dtype=np.float32)

    # predict
    predictions_list = trainer.predict(model=rts_model, dataloaders=predict_dataloader)

    batch_idx = 0
    
    for data_batch, predictions in zip(predict_dataloader, predictions_list):
        pr_heatmap = predictions['heatmap'][0, 0].numpy()
        pr_segment = predictions['prob_mask'][0, 0].numpy()
        image_t = data_batch['image_t'][0]
        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[0]

        col_off = int(round((data_bbox.minx - output_origin_x) / res_x))
        row_off = int(round((output_origin_y - data_bbox.maxy) / res_y))

        patch_h, patch_w = pr_heatmap.shape
        row_end = min(row_off+patch_h, H)
        col_end = min(col_off+patch_w, W)
        patch_h_clip = row_end-row_off
        patch_w_clip = col_end-col_off

        # skip patch outside the output image
        if row_off < 0 or col_off < 0 or patch_h_clip <= 0 or patch_w_clip <= 0:
            batch_idx += 1
            continue

        # valid pixels only
        valid = (image_t[0, :patch_h_clip, :patch_w_clip].numpy() != 0)
        valid_indices = np.where(valid)
        for local_i, local_j in zip(valid_indices[0], valid_indices[1]):
            global_i = row_off + local_i
            global_j = col_off + local_j
            
            if global_i >= H or global_j >= W:
                continue
            
            key = (global_i, global_j)
            if key not in heatmap_dict:
                heatmap_dict[key] = []
                segment_dict[key] = []
            
            heatmap_dict[key].append(pr_heatmap[local_i, local_j])
            segment_dict[key].append(pr_segment[local_i, local_j])
            count_arr[global_i, global_j] += 1
        
        batch_idx += 1

    print(f"\n total batches processed: {batch_idx}")
    
    # aggregation
    heatmap_result = np.zeros((H, W), dtype=np.float32)
    segment_result = np.zeros((H, W), dtype=np.float32)
    
    if aggregation_method == 'median':
        print(f"\n median aggregation")
        for (i, j), values in heatmap_dict.items():
            heatmap_result[i, j] = np.median(values)
        for (i, j), values in segment_dict.items():
            segment_result[i, j] = np.median(values)

    heatmap_mean = heatmap_result
    segment_mean = segment_result

    heatmap_mean[count_arr == 0] = 0
    segment_mean[count_arr == 0] = 0

    # segment_binary = (segment_mean > 0.5).astype(np.float32)
    # heatmap_mean = heatmap_mean * segment_binary

    print(f"heatmap range: [{heatmap_mean.min():.4f}, {heatmap_mean.max():.4f}]")
    print(f"segment range: [{segment_mean.min():.4f}, {segment_mean.max():.4f}]")
    print(f"number of pixels: {(count_arr > 0).sum()} / {H * W} ({(count_arr > 0).sum() / (H * W) * 100:.1f}%)")

    # save
    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / f"pr_mask_{date_str}_{aggregation_method}_{model_name}.tif"

    save_predictions(
        export_path=out_path,
        res=int(res_x),
        transform=global_transform,
        crs=ref_crs,
        pr_heatmap=heatmap_mean,
        pr_segment=segment_mean,
        pr_count=count_arr,
        image_count=count_arr,
    )
    
    print(f"\ninference finished.")
    return True