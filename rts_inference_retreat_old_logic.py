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

def crop_image_to_bbox(image, bbox, res):
    """
    根据 bbox 尺寸裁剪 image，使其和 bbox 对齐
    """
    expected_h = int(round((bbox.maxy - bbox.miny) / res))
    expected_w = int(round((bbox.maxx - bbox.minx) / res))

    return image[:, :expected_h, :expected_w]

def expand_and_mask(image: Tensor, heatmap: Tensor, segment: Tensor, bbox: BoundingBox, target_bbox: BoundingBox, res: int = 30):
    # expand the input raster
    image = expand_to_bbox(image, bbox, target_bbox, res)
    heatmap = expand_to_bbox(heatmap, bbox, target_bbox, res)
    segment = expand_to_bbox(segment, bbox, target_bbox, res)

    # values of 0 are interpreted as False, everything else as True.
    image_mask = image[0:1, :, :] > 0
    heatmap = heatmap.unsqueeze(dim=0)
    segment = segment.unsqueeze(dim=0)
    
    # segment = (segment > 0.5).float()
    segment_mask = segment > 0.8
    segment_mask = segment_mask & image_mask
    pr_heatmap_ma = np.ma.array(heatmap.numpy(), mask=~segment_mask.numpy())
    pr_segment_ma = np.ma.array(segment.numpy(), mask=~segment_mask.numpy())
    image_mask_np = 1*image_mask.numpy() # convert bool to int

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
    
    def to_hw(x, name):
        x = np.asarray(x, dtype=np.float32)
        x = np.squeeze(x)
        if x.ndim != 2:
            raise ValueError(f"{name} expected (H,W) after squeeze, but got {x.shape}.")
        return x

    pr_heatmap = to_hw(pr_heatmap, "pr_heatmap")
    pr_segment = to_hw(pr_segment, "pr_segment")
    pr_count = to_hw(pr_count, "pr_count")
    image_count = to_hw(image_count, "image_count")

    height, width = pr_heatmap.shape

    rio_transform = rasterio.transform.from_bounds(
        bbox.minx, bbox.miny, bbox.maxx, bbox.maxy, width, height) # west, south, east, north, width, height

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
        rgb = np.clip((rgb * bright), 0, 1)

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
    plt.title("Input Image (Year t)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(h_np, 0.05, 1.2), cmap='jet')
    plt.title("Heatmap Prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def predict_and_export_OLD_LOGIC(
    rts_model,
    trainer: pl.Trainer,
    datamodule: LandsatPairInferenceDataModule,
    model_name: str,
    date_str: str,
    export_dir: Path,
    aggregation_method='median',
):
    
    predict_dataloader = datamodule.predict_dataloader()
    target_bbox = datamodule.roi
    
    predictions_list = trainer.predict(
        model=rts_model, dataloaders=datamodule
    )
    
    pr_heatmap_list_np = []
    pr_segment_list_np = []
    image_mask_list_np = []
    
    raster_filepath_set = False
    target_crs = None
    img_rgb_for_viz = None
    
    batch_idx = 0
    for data_batch, predictions in zip(predict_dataloader, predictions_list):
        pr_heatmaps = predictions['heatmap']
        pr_segments = predictions['prob_mask']
        image_t_batch = data_batch['image_t']
        
        idx = 0
        image_t = image_t_batch[idx, :, :, :]
        pr_heatmap = pr_heatmaps[idx, 0, :, :]
        pr_segment = pr_segments[idx, 0, :, :]
        data_bbox = data_batch['bbox']
        if isinstance(data_bbox, (list, tuple)):
            data_bbox = data_bbox[0]
        
        if 'path_t' in data_batch:
            filepath = data_batch['path_t']
            if isinstance(filepath, (list, tuple)):
                filepath = filepath[0]
        else:
            filepath = "unknown"
        
        if target_crs is None and 'crs' in data_batch:
            crs_list = data_batch['crs']
            if isinstance(crs_list, (list, tuple)):
                target_crs = crs_list[0]
            else:
                target_crs = crs_list
        
        pr_heatmap_ma, pr_segment_ma, image_mask_np = expand_and_mask(
            image_t, pr_heatmap, pr_segment, 
            data_bbox, target_bbox, res=30
        )
        pr_heatmap_list_np.append(pr_heatmap_ma)
        pr_segment_list_np.append(pr_segment_ma)
        image_mask_list_np.append(image_mask_np)
        
        if not raster_filepath_set:
            if hasattr(datamodule, 'rgb_indexes'):
                img_rgb_for_viz = image_t_batch[0, datamodule.rgb_indexes, :, :]
            else:
                img_rgb_for_viz = image_t_batch[0, [2, 1, 0], :, :]
            raster_filepath_set = True
        
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"{batch_idx} batches processed")
    
    print(f"\n {batch_idx} batches processed in total.")
    
    # Concatenate
    pr_segment_concat_np = np.ma.concatenate([s[np.newaxis] for s in pr_segment_list_np], axis=0)
    pr_heatmap_concat_np = np.ma.concatenate([h[np.newaxis] for h in pr_heatmap_list_np], axis=0)
    
    print(f"Stacked shape: {pr_heatmap_concat_np.shape}")
    print(f"Number of valid pixels: {np.ma.count(pr_heatmap_concat_np) / pr_heatmap_concat_np.size * 100:.1f}%")
    
    # aggregation
    if aggregation_method == 'median':
        pr_segment_result = np.ma.median(pr_segment_concat_np, axis=0).filled(fill_value=0)
        pr_heatmap_result = np.ma.median(pr_heatmap_concat_np, axis=0).filled(fill_value=0)
    else:  # mean
        pr_segment_result = np.ma.mean(pr_segment_concat_np, axis=0).filled(fill_value=0)
        pr_heatmap_result = np.ma.mean(pr_heatmap_concat_np, axis=0).filled(fill_value=0)
    
    pr_heatmap_count = np.ma.sum(pr_segment_concat_np, axis=0).filled(fill_value=0)
    image_mask_concat_np = np.concatenate([m for m in image_mask_list_np], axis=0)
    image_count = np.sum(image_mask_concat_np, axis=0)
    
    print(f"Heatmap range: [{pr_heatmap_result.min():.4f}, {pr_heatmap_result.max():.4f}]")
    print(f"Segment range: [{pr_segment_result.min():.4f}, {pr_segment_result.max():.4f}]")
    
    export_dir.mkdir(parents=True, exist_ok=True)
    
    if img_rgb_for_viz is not None:
        show_results(image=img_rgb_for_viz, heatmap=pr_heatmap_result, bright=3)
    
    pr_tiff_path = export_dir / f"pr_mask_{date_str}_{aggregation_method}_{model_name}_OLD_LOGIC.tif"
    
    save_predictions(
        export_path=pr_tiff_path,
        bbox=target_bbox,
        crs=target_crs,
        pr_heatmap=pr_heatmap_result,
        pr_segment=pr_segment_result,
        pr_count=pr_heatmap_count,
        image_count=image_count,
    )
    
    print(f"\nInference finished.")
    return True