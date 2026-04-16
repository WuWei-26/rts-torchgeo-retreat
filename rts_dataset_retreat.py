# rts_dataset_retreat.py

import glob
import math
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from rasterio.crs import CRS
from rasterio.windows import from_bounds
from torch import Tensor
from torchgeo.datasets import (
    IntersectionDataset,
    RasterDataset,
    UnionDataset,
    # unbind_samples,
)
from torchgeo.datasets import BoundingBox
from rtree.index import Index, Property

# L89_REGEX = r"^LANDSAT_LC0[89].*_C02_T1_L2_LC0[89]_\d{6}_(?P<date>\d{4}).+(?:clip|hohxil).tif"
L89_REGEX = r"^LANDSAT_(?:LC08|LC09|LC08_LC09)_C02_T1_L2_(?:LC08|LC09|LC08_LC09)_\d{6}_(?P<date>\d{4}).+(?:clip|clip_m|hohxil|hohxil_utm).tif"  # |\d{2}
# L57_REGEX = r"^LANDSAT_\S{19,20}_\d{6}_(?P<date>\d{4}).+.tif"
L57_REGEX = r"^LANDSAT_(?:LT05|LE07|LT05_LE07)_C02_T1_L2_(?:LT05|LE07|LT05_LE07)_\d{6}_(?P<date>\d{4}).+.tif"
L5_REGEX = r"^LANDSAT_LT05_C02_T1_L2_LT05_\d{6}_(?P<date>\d{4}).+.tif"
MASK_REGEX = r"^gt_(?P<band>\S{7})_30m_v3_.*_(?P<date>\d{4})\d{4}.tif"
TPI_REGEX = r"^NASADEM_HGT_001_tpi_.+"

class Landsat8SR(RasterDataset):
    # sample: LANDSAT_LC08_C02_T1_L2_LC08_137036_20160810_Cloud_04.tif
    """按目录年份严格索引的 Landsat 8/9 数据集"""
    filename_glob = "LANDSAT_LC08*.tif"
    # filename_regex = L89_REGEX
    filename_regex = r".*LANDSAT_LC0[89].*\.tif$"
    # 只匹配 geometric_median 合成影像
    # filename_regex = r"^LANDSAT_.*geometric_median.*\.tif$"
    date_format = "%Y%m%d"
    # date_format = "%Y"  # only acquire year information
    is_image = True
    # is_dem = False
    separate_files = False
    all_bands = ("SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7")
    rgb_bands = ("SR_B4", "SR_B3", "SR_B2")
    all_bands_mean = torch.asarray(
        [
            0.05852928,
            0.07153611,
            0.11833665,
            0.15907602,
            0.27192301,
            0.25811666,
            0.21108147,
        ],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [
            0.02198817,
            0.02448466,
            0.03277782,
            0.04925627,
            0.05472392,
            0.06181966,
            0.06714131,
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)
        # self.band_indexes start from 1 for rasterio read
        # print("Available bands: ", self.bands)
        if not hasattr(self, "band_indexes"):
            if bands is None:
                self.band_indexes = list(range(1, len(self.all_bands) + 1))
            else:
                self.band_indexes = [self.all_bands.index(band) + 1 for band in bands]
        # print("Band indexes: ", self.band_indexes)
        self.bands_mean = self.all_bands_mean[np.array(self.band_indexes) - 1]
        self.bands_std = self.all_bands_std[np.array(self.band_indexes) - 1]
        # print(self.all_bands)
        # print(self.rgb_bands)
        # for band in self.rgb_bands:
        #     self.rgb_indices.append(self.bands.index(band))
        self.rgb_indexes = [self.bands.index(band) for band in self.rgb_bands]

        self._reindex_by_directory_year()

    def _reindex_by_directory_year(self):
        new_index = Index(interleaved=False, properties=Property(dimension=3))
        
        for item in self.index.intersection(self.index.bounds, objects=True):
            filepath = item.object

            dir_year = self._extract_dir_year(filepath)
            if '/test/' in filepath:
                continue  # exclude test data
            if dir_year is None:
                new_index.insert(item.id, item.bounds, filepath)
                continue
            old_bounds = item.bounds
            minx, maxx, miny, maxy = old_bounds[0], old_bounds[1], old_bounds[2], old_bounds[3]
            mint = datetime(dir_year, 1, 1, 0, 0, 0).timestamp()
            maxt = datetime(dir_year+1, 1, 1, 0, 0, 0).timestamp()
            new_index.insert(item.id, (minx, maxx, miny, maxy, mint, maxt), filepath)
        
        self.index = new_index
        print(f"[{self.__class__.__name__}] Reindexed {new_index.get_size()} files by directory year")
    
    def _extract_dir_year(self, filepath: str) -> Optional[int]:
        """从文件路径提取目录年份
        支持两种格式：
        - /path/2019/file.tif -> 2019
        - /path/20190930/file.tif -> 2019
        """
        match = re.search(r'/(\d{4})(\d{2})(\d{2})/', filepath)
        if match:
            return int(match.group(1))
        match = re.search(r'/(\d{4})/', filepath)
        if match:
            return int(match.group(1))
        return None

    def __getitem__(self, query):
        if isinstance(query, dict):
            bbox: BoundingBox = query["bbox"]
        elif isinstance(query, BoundingBox):
            bbox = query
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        hits = list(self.index.intersection(tuple(bbox), objects=True))
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(f"bbox: {bbox} not found in index bounds: {self.bounds}")

        query_year = datetime.fromtimestamp(bbox.mint).year
        matched_files = [
            fp for fp in filepaths
            if self._extract_dir_year(fp) == query_year
        ]
        if matched_files:
            COMPOSITE_KEYWORDS = ["geometric_median", "median", "medoid", "mosaic"]
            composite_files = [
                fp for fp in matched_files
                if any(kw in os.path.basename(fp) for kw in COMPOSITE_KEYWORDS)
            ]
            filepath = composite_files[0] if composite_files else matched_files[0]
        else:
            available_years = sorted(set(self._extract_dir_year(fp) for fp in filepaths if self._extract_dir_year(fp)))
            raise IndexError(
                f"No file found for year {query_year}. "
                f"Query bbox: {bbox}, Available years: {available_years}"
            )

        src = self._cached_load_warp_file(filepath) if self.cache else self._load_warp_file(filepath)
        if isinstance(self.res, (int, float)):
            res_x = res_y = float(self.res)
        else:
            res_x, res_y = self.res
        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        out_width = math.ceil((bbox.maxx - bbox.minx) / res_x)
        out_height = math.ceil((bbox.maxy - bbox.miny) / res_y)

        band_indexes = self.band_indexes
        if band_indexes is not None:
            max_band = src.count
            bad = [i for i in band_indexes if i < 1 or i > max_band]
            if bad:
                raise IndexError(f"Requested band indexes {bad} out of range 1..{max_band}")

        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)

        dest = src.read(indexes=band_indexes, out_shape=out_shape, 
                       window=from_bounds(*bounds, src.transform))
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        window = from_bounds(*bounds, src.transform)
        transform = src.window_transform(window)

        tensor = torch.tensor(dest)
        sample = {"crs": self.crs, "bbox": bbox, "path": filepath, "transform": transform}
        
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor
            
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _pick_file(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)

        if len(vrt_fhs) == 1:
            src = vrt_fhs[0]
        else:  # modified by zyzhao, just random choose one
            # src_idx = np.random.randint(len(vrt_fhs))
            src_idx = torch.randint(len(vrt_fhs), (1,))
            src = vrt_fhs[src_idx]

        if isinstance(self.res, (int, float)):
            res_x = self.res
            res_y = self.res
        else:
            res_x, res_y = self.res
        out_width = round((query.maxx - query.minx) / res_x)
        out_height = round((query.maxy - query.miny) / res_y)
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)
        
        # dest = src.read(
        #     indexes=band_indexes,
        #     out_shape=out_shape,
        #     window=from_bounds(*bounds, src.transform),
        # )
        try:
            dest = src.read(
                indexes=band_indexes,
                out_shape=out_shape,
                window=from_bounds(*bounds, src.transform),
            )
        except Exception:
            dest = np.zeros(out_shape, dtype=np.float32)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        return tensor

    def plot(self, sample, bright=3):
        check_rgb = all(item in self.bands for item in self.rgb_bands)
        if not check_rgb:
            raise Exception("Need R G B bands to visualize")

        # Find the correct bands
        # rgb_indices = []
        # for band in self.rgb_bands:
        #     rgb_indices.append(self.bands.index(band))

        # Reorder and rescale the image
        if sample["image"].ndim == 4:
            image = sample["image"][0, self.rgb_indexes, :, :].permute(1, 2, 0)
        else:
            image = sample["image"][self.rgb_indexes, :, :].permute(1, 2, 0)

        if image.max() > 10:
            image = self.apply_scale(image)
        image = torch.clamp(image * bright, min=0, max=1)

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.axis("off")

        return fig

    @staticmethod
    def apply_scale(image):
        return image * 0.0000275 - 0.2
class Landsat5SR(Landsat8SR):
    # sample: LANDSAT_LC08_C02_T1_L2_LC08_137036_20160810_Cloud_04.tif
    filename_glob = "LANDSAT_LT05_C02_T1_L2_*.tif"
    filename_regex = L5_REGEX
    date_format = "%Y%m%d"
    # date_format = "%Y"  # only acquire year information
    is_image = True
    # is_dem = False
    separate_files = False
    all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    rgb_bands = ["SR_B3", "SR_B2", "SR_B1"]
    all_bands_mean = torch.asarray(
        [0.07153611, 0.11833665, 0.15907602, 0.27192301, 0.25811666, 0.21108147],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [0.02448466, 0.03277782, 0.04925627, 0.05472392, 0.06181966, 0.06714131],
        dtype=torch.float32,
    )
class Landsat57SR(Landsat8SR):
    # sample: LANDSAT_LC08_C02_T1_L2_LC08_137036_20160810_Cloud_04.tif
    filename_glob = "LANDSAT_L*.tif"
    filename_regex = L57_REGEX
    date_format = "%Y"  # only acquire year information
    is_image = True
    # is_dem = False
    separate_files = False
    all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    rgb_bands = ["SR_B3", "SR_B2", "SR_B1"]
    all_bands_mean = torch.asarray(
        [0.07153611, 0.11833665, 0.15907602, 0.27192301, 0.25811666, 0.21108147],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [0.02448466, 0.03277782, 0.04925627, 0.05472392, 0.06181966, 0.06714131],
        dtype=torch.float32,
    )
class RtsMask(Landsat8SR):
    filename_glob = "gt*.tif"
    filename_regex = MASK_REGEX
    date_format = "%Y"
    is_image = False
    separate_files = True
    all_bands = ["heatmap", "segment", "retreat"]
    rgb_bands = []

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)
        self.debug = debug

    def plot(self, sample):
        import matplotlib.pyplot as plt
        mask = sample["mask"]
        n = mask.shape[0]
        cols = 3 if n >= 3 else 2
        fig, axs = plt.subplots(1, cols)
        axs[0].imshow(mask[0] / 30, cmap="jet");   axs[0].axis("off"); axs[0].set_title("heatmap")
        axs[1].imshow(mask[1] / 255, cmap="gray"); axs[1].axis("off"); axs[1].set_title("segment")
        if n >= 3:
            axs[2].imshow(mask[2], cmap="turbo");  axs[2].axis("off"); axs[2].set_title("retreat")
        return fig

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(query, dict):
            bbox: BoundingBox = query["bbox"]
            year = query.get("year", None)
            path = query.get("path", None)
        elif isinstance(query, BoundingBox):
            bbox = query
            year, path = None, None
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])

        rgx = re.compile(self.filename_regex, re.VERBOSE)
        if year is not None:
            fps = []
            for fp in filepaths:
                m = rgx.match(os.path.basename(fp))
                if m and "date" in m.groupdict() and m.group("date") == str(year):
                    fps.append(fp)
            filepaths = fps

        if self.debug:
            print("\n[RtsMask debug] bbox:", bbox)
            print("  year:", year)
            print("  hit files:", [os.path.basename(fp) for fp in filepaths])
            bands_found = []
            for fp in filepaths:
                m = rgx.match(os.path.basename(fp))
                bands_found.append(m.group("band") if (m and "band" in m.groupdict()) else "N/A")
            print("  bands_found:", bands_found)

        if not filepaths:
            raise IndexError(f"bbox: {bbox} (year={year}) not found in index bounds: {self.bounds}")

        ref = self._pick_file([filepaths[0]], bbox)
        H, W = ref.shape[-2], ref.shape[-1]
        data_list: List[Tensor] = []
        for band in self.bands:
            candidates = []
            for fp in filepaths:
                m = rgx.match(os.path.basename(fp))
                if m and "band" in m.groupdict() and m.group("band") == band:
                    candidates.append(fp)
            if candidates:

                band_tensor = self._pick_file([candidates[0]], bbox)
                if band_tensor.ndim == 3 and band_tensor.shape[0] > 1:
                    band_tensor = band_tensor[:1,...]
                if self.debug:
                    print(f"  -> use {os.path.basename(candidates[0])} for band={band}, shape={tuple(band_tensor.shape)}")
            else:
                band_tensor = torch.zeros((1, H, W), dtype=torch.float32)
                if self.debug:
                    print(f"  -> MISSING band={band}, fill zeros with shape={(1,H,W)}")
            data_list.append(band_tensor)

        data = torch.cat(data_list, dim=0)  # [3,H,W]
        if self.debug:
            nz = lambda t: int((t > 0).sum().item())
            print(f"  output shape={tuple(data.shape)}, nz heat/seg/ret=({nz(data[0])}, {nz(data[1])}, {nz(data[2])})")

        sample = {"crs": self.crs, "bbox": bbox, "mask": data, "transform": self._cached_load_warp_file(filepaths[0]).transform}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
class RetreatMapDataset(RasterDataset):
    # retreat_map_YYYY.tif
    filename_glob = "retreat_map_*.tif"
    # filename_regex = r"^retreat_map_(?P<date>\d{4})\.tif$" # for 1-year window
    filename_regex = r"^retreat_map_(?P<year_t>\d{4})_(?P<year_tm1>\d{4})\.tif$" # for multi-year window
    date_format = "%Y"
    is_image = False
    separate_files = False
    all_bands = ["retreat"]
    rgb_bands = []

    def __init__(
        self,
        root: str,
        crs: Optional[str] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms=None,
        cache: bool = True,
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)
    
    def __getitem__(self, query):
        if isinstance(query, dict):
            bbox = query["bbox"]
            # year = query.get("year", None)
            year_t=query.get("year_t", query.get("year", None))
            year_tm1=query.get("year_tm1", None)
        elif isinstance(query, BoundingBox):
            # bbox, year = query, None
            bbox, year_t, year_tm1 = query, None, None
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = [hit.object for hit in hits]

        if year_t is not None:
            rgx = re.compile(self.filename_regex)
            year_t_str = str(int(year_t))
            filtered = []
            for fp in filepaths:
                m = rgx.match(os.path.basename(fp))
                if m and m.group("year_t") == year_t_str:
                    if year_tm1 is None or m.group("year_tm1") == str(int(year_tm1)):
                        filtered.append(fp)
            filepaths = filtered

            # filepaths = [
            #     fp for fp in filepaths
            #     if (m := rgx.match(os.path.basename(fp))) and m.group("date") == year_str
            # ]
        if not filepaths:
            raise IndexError(f"bbox: {bbox} (year_t={year_t}, year_tm1={year_tm1}) "
                f"not found in index bounds: {self.bounds}")
        filepath = filepaths[0]

        src = self._cached_load_warp_file(filepath) if self.cache else self._load_warp_file(filepath)

        if isinstance(self.res, (tuple, list)):
            res_x, res_y = self.res[0], self.res[1]
        else:
            res_x = res_y = float(self.res)

        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        # out_width = math.ceil((bbox.maxx - bbox.minx) / self.res)
        # out_height = math.ceil((bbox.maxy - bbox.miny) / self.res)
        out_width = math.ceil((bbox.maxx - bbox.minx) / res_x)
        out_height = math.ceil((bbox.maxy - bbox.miny) / res_y)
        out_shape = (src.count, out_height, out_width)

        dest = src.read(out_shape=out_shape, window=from_bounds(*bounds, src.transform))

        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        sample = {"crs": self.crs, "bbox": bbox, "mask": tensor, "path": filepath}

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

class RTSTemporalPairDataset(torch.utils.data.Dataset):
    """
      image_t, image_tm1, dem_t, dem_tm1, mask, heatmap, retreat_map
    """
    @staticmethod
    def _year_bbox(bbox: BoundingBox, year: int) -> BoundingBox:
        from datetime import datetime
        mint = datetime(year, 1, 1).timestamp()
        maxt = datetime(year + 1, 1, 1).timestamp()
        return BoundingBox(bbox.minx, bbox.maxx, bbox.miny, bbox.maxy, mint, maxt)

    def __init__(self, img_ds, mask_ds, dem_ds, retreat_ds=None, transforms=None, year_t=None, year_tm1=None):
        self.img_ds = img_ds
        self.mask_ds = mask_ds
        self.dem_ds = dem_ds
        self.retreat_ds = retreat_ds
        self.transforms = transforms
        self.index = img_ds.index
        self.bounds = img_ds.bounds
        self.crs = img_ds.crs
        self.res = img_ds.res
        self.year_t = year_t
        self.year_tm1 = year_tm1

    def _get_image_by_year(self, bbox: BoundingBox, year: int) -> dict:
        if not isinstance(self.img_ds, UnionDataset):
            return self.img_ds[bbox]

        def collect_leaf_datasets(ds):
            if isinstance(ds, UnionDataset):
                result = []
                for sub in ds.datasets:
                    result.extend(collect_leaf_datasets(sub))
                return result
            return [ds]

        all_datasets = collect_leaf_datasets(self.img_ds)

        year_mint = datetime(year, 1, 1).timestamp()
        year_maxt = datetime(year + 1, 1, 1).timestamp()
        year_filtered = [
            ds for ds in all_datasets
            if ds.bounds.mint <= year_mint and ds.bounds.maxt >= year_maxt
        ]

        if not year_filtered:
            year_filtered = all_datasets

        if year > 2012:
            preferred_types = (Landsat8SR, Landsat57SR)
        else:
            preferred_types = (Landsat57SR, Landsat8SR)

        last_error = None
        for preferred_type in preferred_types:
            for ds in year_filtered:
                if isinstance(ds, preferred_type):
                    try:
                        return ds[bbox]
                    except IndexError as e:
                        last_error = e
                        continue

        for ds in year_filtered:
            try:
                return ds[bbox]
            except IndexError as e:
                last_error = e
                continue

        raise IndexError(
            f"No image found for year {year}, bbox: {bbox}. Last error: {last_error}"
        )

    def __len__(self):
        return 10**9
    
    def __getitem__(self, query: dict):
        """
        支持两种查询格式:
        1. dict: {'bbox': BoundingBox, 'year_t': int, 'year_tm1': int}
        2. BoundingBox: 直接传入 bbox（年份从初始化获取）
        """
        if isinstance(query, dict):
            bbox = query["bbox"]
            year_t = query.get("year_t", self.year_t)
            year_tm1 = query.get("year_tm1", self.year_tm1)
        elif isinstance(query, BoundingBox):
            bbox = query
            year_t = self.year_t
            year_tm1 = self.year_tm1
        else:
            raise TypeError(f"query 必须是 dict 或 BoundingBox，得到 {type(query)}")
        
        if year_t is None or year_tm1 is None:
            raise ValueError(
                f"year_t 和 year_tm1 必须提供。"
                f"当前: year_t={year_t}, year_tm1={year_tm1}"
            )

        bbox_t = self._year_bbox(bbox, year_t)
        bbox_tm1 = self._year_bbox(bbox, year_tm1)

        # t
        # s_img_t = self.img_ds[bbox_t]
        s_img_t = self._get_image_by_year(bbox_t, year_t)
        
        try:
            s_img_tm1 = self._get_image_by_year(bbox_tm1, year_tm1)
        except Exception:
            print(f"[WARNING] lack year_tm1={year_tm1} image, using year_t={year_t}")
            s_img_tm1 = self._get_image_by_year(bbox_t, year_t)

        # DEM
        if self.dem_ds is not None:
            # s_dem_t = self.dem_ds[q_t]
            s_dem_t = self.dem_ds[bbox_t]
            try:
                # s_dem_tm1 = self.dem_ds[q_tm1]
                s_dem_tm1 = self.dem_ds[bbox_tm1]
            except Exception:
                # s_dem_tm1 = self.dem_ds[q_t]
                s_dem_tm1 = self.dem_ds[bbox_t]
        else:
            s_dem_t = None
            s_dem_tm1 = None

        sample = {
            "image_t": s_img_t["image"].float(),
            "image_tm1": s_img_tm1["image"].float(),
            "bbox": bbox,
            "crs": self.crs,
            "path_t": s_img_t.get("path", ""),
            "path_tm1": s_img_tm1.get("path", ""),
            "year_t": year_t,
            "year_tm1": year_tm1,
            "transform_t":   s_img_t.get("transform",   None),
            "transform_tm1": s_img_tm1.get("transform", None),
        }

        if s_dem_t is not None:
            sample["dem_t"] = s_dem_t["mask"].float()
            sample["dem_tm1"] = s_dem_tm1["mask"].float()

        if self.mask_ds is not None:
            # q_mask_t = {"bbox": bbox_t, "year": year_t}
            # s_mask_t = self.mask_ds[bbox_t]
            s_mask_t = self.mask_ds[{"bbox": bbox_t, "year": year_t}]
            labels = s_mask_t["mask"].float()
            H, W = labels.shape[-2], labels.shape[-1]
            heatmap = labels[0:1,...]
            mask = labels[1:2,...]
            sample["heatmap"] = heatmap
            sample["mask"] = mask

            # retreat
            if self.retreat_ds is not None:
                try:
                    # q_ret_t = {"bbox": bbox_t, "year": year_t}
                    # s_ret = self.retreat_ds[{"bbox": bbox_t, "year": year_t}]
                    s_ret = self.retreat_ds[{"bbox": bbox_t, "year_t": year_t, "year_tm1": year_tm1}]
                    if s_ret["mask"].ndim == 3:
                        retreat = s_ret["mask"][:1,...].float()
                    else:
                        retreat = s_ret["mask"].unsqueeze(0).float()
                except Exception:
                    retreat = torch.zeros((1, H, W), dtype=labels.dtype, device=labels.device)
            else:
                retreat = torch.zeros((1, H, W), dtype=labels.dtype, device=labels.device)

            sample["retreat_map"] = retreat

        if self.mask_ds is not None:
            if sample["heatmap"].sum() == 0 and sample["mask"].sum() == 0:
                return None

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


# TODO: merge dem tiles


class MeanTPI(RasterDataset):
    filename_glob = "*.tif"
    filename_regex = TPI_REGEX
    # date_format = "%Y"
    is_image = False  # treat it as mask
    # is_dem = False
    separate_files = False
    all_bands = ["mean_tpi"]
    rgb_bands = []
    
    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,    
    ) -> None:
        super().__init__(root, crs, res, bands, transforms, cache)
        
        # 初始化 band_indexes
        if not hasattr(self, "band_indexes"):
            if bands is None:
                self.band_indexes = list(range(1, len(self.all_bands) + 1))
            else:
                self.band_indexes = [self.all_bands.index(band) + 1 for band in bands]
    
    def __getitem__(self, query):
        if isinstance(query, dict):
            bbox: BoundingBox = query["bbox"]
        elif isinstance(query, BoundingBox):
            bbox = query
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")
        query_bbox = BoundingBox(
            bbox.minx, bbox.maxx, bbox.miny, bbox.maxy,
            self.bounds.mint, self.bounds.maxt
        )
        
        hits = list(self.index.intersection(tuple(query_bbox), objects=True))
        filepaths = [hit.object for hit in hits]
        
        if not filepaths:
            raise IndexError(f"bbox: {bbox} not found in index bounds: {self.bounds}")
        
        filepath = filepaths[0]
        src = self._cached_load_warp_file(filepath) if self.cache else self._load_warp_file(filepath)
        
        if isinstance(self.res, (int, float)):
            res_x = res_y = float(self.res)
        else:
            res_x, res_y = self.res
        
        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        out_width = math.ceil((bbox.maxx - bbox.minx) / res_x)
        out_height = math.ceil((bbox.maxy - bbox.miny) / res_y)
        
        band_indexes = self.band_indexes
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)
        
        dest = src.read(indexes=band_indexes, out_shape=out_shape,
                       window=from_bounds(*bounds, src.transform))
        
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        window = from_bounds(*bounds, src.transform)
        transform = src.window_transform(window)
        
        tensor = torch.tensor(dest)
        sample = {"crs": self.crs, "bbox": bbox, "mask": tensor.float(), "path": filepath, "transform": transform}
        
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def plot(self, sample):
        # squeeze the mask
        dem = sample["mask"]  # .squeeze().numpy()

        # Plot the mask
        fig, ax = plt.subplots()  # , figsize=(15, 7)
        ax.imshow((dem[0, :, :] + 2) / 4, cmap="gray")
        ax.axis("off")

        return fig


class TestLandsat8SR(Landsat8SR):
    # sample: S2B_L2A_20190225_N0211_R117_6Bands_S1.tif
    filename_glob = "LANDSAT_*.tif"
    filename_regex = r"^LANDSAT_(?:LC08|LC09|LC08_LC09)_C02_T1_L2_(?:LC08|LC09|LC08_LC09)_\d{6}_(?P<date>\d{8}).*\.tif$"
    # filename_regex = r"^S2.{5}_(?P<date>\d{8})_N\d{4}_R\d{3}_6Bands_S\d{1}"
    # filename_regex = r"^LANDSAT_LC08_C02_T1_L2_LC08_\d{6}_(?P<date>\d{4}).+"
    # filename_regex = L89_REGEX
    date_format = "%Y%m%d"
    is_image = True
    separate_files = False
    all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    rgb_bands = ["SR_B4", "SR_B3", "SR_B2"]

    all_bands_mean = torch.asarray(
        [
            0.05852928,
            0.07153611,
            0.11833665,
            0.15907602,
            0.27192301,
            0.25811666,
            0.21108147,
        ],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [
            0.02198817,
            0.02448466,
            0.03277782,
            0.04925627,
            0.05472392,
            0.06181966,
            0.06714131,
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
    ) -> None:
        if self.separate_files:
            raise NotImplementedError(
                "Testing for separated files are not supported yet"
            )
        super().__init__(root, crs, res, bands, transforms, cache)
    
    def _reindex_by_directory_year(self):
        new_index = Index(interleaved=False, properties=Property(dimension=3))
        for item in self.index.intersection(self.index.bounds, objects=True):
            filepath = item.object
            dir_year = self._extract_dir_year(filepath)
            if dir_year is None:
                new_index.insert(item.id, item.bounds, filepath)
                continue
            old_bounds = item.bounds
            minx, maxx, miny, maxy = old_bounds[0], old_bounds[1], old_bounds[2], old_bounds[3]
            mint = datetime(dir_year, 1, 1, 0, 0, 0).timestamp()
            maxt = datetime(dir_year+1, 1, 1, 0, 0, 0).timestamp()
            new_index.insert(item.id, (minx, maxx, miny, maxy, mint, maxt), filepath)
        self.index = new_index
        print(f"[{self.__class__.__name__}] Reindexed {new_index.get_size()} files by directory year")

    def __getitem__(self, query):
        if isinstance(query, dict):
            bbox = query["bbox"]
            filepath = query.get("path", None)
        elif isinstance(query, BoundingBox):
            bbox = query
            filepath = None
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])
        #  filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(f"bbox: {bbox} not found in index bounds: {self.bounds}")

        filepath = filepath or filepaths[0]

        src = self._cached_load_warp_file(filepath) if self.cache else self._load_warp_file(filepath)

        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        out_width  = math.ceil((bbox.maxx - bbox.minx) / self.res[0])
        out_height = math.ceil((bbox.maxy - bbox.miny) / self.res[1])

        band_indexes = self.band_indexes
        if band_indexes is not None:
            max_band = src.count
            bad = [i for i in band_indexes if i < 1 or i > max_band]
            if bad:
                raise IndexError(f"Requested band indexes {bad} out of range 1..{max_band} for {os.path.basename(filepath)}")

        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)

        dest = src.read(
            indexes=band_indexes,
            out_shape=out_shape,
            window=from_bounds(*bounds, src.transform),
        )

        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        window = from_bounds(*bounds, src.transform)
        transform = src.window_transform(window)

        sample = {"crs": self.crs, "bbox": bbox, "path": filepath, "transform": transform}

        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

class TestLandsat5SR(TestLandsat8SR):
    # sample: LANDSAT_LC08_C02_T1_L2_LC08_137036_20160810_Cloud_04.tif
    filename_glob = "LANDSAT_LT05_C02_T1_L2_*.tif"
    # filename_regex = L5_REGEX
    filename_regex = r"^LANDSAT_(?:LT05|LE07|LT05_LE07)_C02_T1_L2_(?:LT05|LE07|LT05_LE07)_\d{6}_(?P<date>\d{8}).*\.tif$"
    date_format = "%Y%m%d"  # only acquire year information
    is_image = True
    separate_files = False
    all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    rgb_bands = ["SR_B3", "SR_B2", "SR_B1"]
    all_bands_mean = torch.asarray(
        [0.07153611, 0.11833665, 0.15907602, 0.27192301, 0.25811666, 0.21108147],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [0.02448466, 0.03277782, 0.04925627, 0.05472392, 0.06181966, 0.06714131],
        dtype=torch.float32,
    )

class TestLandsat57SR(TestLandsat8SR):
    # sample: LANDSAT_LC08_C02_T1_L2_LC08_137036_20160810_Cloud_04.tif
    filename_glob = "LANDSAT_L*.tif"
    filename_regex = r"^LANDSAT_(?:LT05|LE07|LT05_LE07)_C02_T1_L2_(?:LT05|LE07|LT05_LE07)_\d{6}_(?P<date>\d{4})\d{4}.*\.tif$"
    # filename_regex = L57_REGEX
    date_format = "%Y"  # only acquire year information
    is_image = True
    separate_files = False
    all_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
    rgb_bands = ["SR_B3", "SR_B2", "SR_B1"]
    all_bands_mean = torch.asarray(
        [0.07153611, 0.11833665, 0.15907602, 0.27192301, 0.25811666, 0.21108147],
        dtype=torch.float32,
    )
    all_bands_std = torch.asarray(
        [0.02448466, 0.03277782, 0.04925627, 0.05472392, 0.06181966, 0.06714131],
        dtype=torch.float32,
    )

# TODO: merge dem tiles
class TestMeanTPI(MeanTPI):
    filename_glob = "*.tif"
    # filename_regex = r'^NASADEM_HGT_001_tpi_mean_.+'
    filename_regex = TPI_REGEX
    # date_format = "%Y"
    is_image = False  # treat it as mask
    # is_dem = False
    separate_files = False
    all_bands = ["mean_tpi"]

    def __getitem__(self, query):
        if isinstance(query, BoundingBox):
            bbox = query
        elif isinstance(query, dict) and "bbox" in query:
            bbox = query["bbox"]
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        hits = self.index.intersection(tuple(bbox), objects=True)
        filepaths = [hit.object for hit in hits]
        if not filepaths:
            raise IndexError(f"query: {bbox} not found in index with bounds: {self.bounds}")
        filepath = filepaths[0]

        src = self._cached_load_warp_file(filepath) if self.cache else self._load_warp_file(filepath)

        bounds = (bbox.minx, bbox.miny, bbox.maxx, bbox.maxy)
        out_width  = math.ceil((bbox.maxx - bbox.minx) / self.res[0])
        out_height = math.ceil((bbox.maxy - bbox.miny) / self.res[1])

        band_indexes = self.band_indexes
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)

        dest = src.read(
            indexes=band_indexes,
            out_shape=out_shape,
            window=from_bounds(*bounds, src.transform),
        )

        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)
        sample = {"crs": self.crs, "bbox": bbox, "path": filepath}
        if self.is_image:
            sample["image"] = tensor.float()
        else:
            sample["mask"] = tensor
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

class TestIntersectionDEM(IntersectionDataset):
    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        # include objects, ds1 for raster image, ds2 for dem
        i = 0
        ds1, ds2 = self.datasets
        for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
            for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
                box1 = BoundingBox(*hit1.bounds)  # type: ignore
                box2 = BoundingBox(*hit2.bounds)  # type: ignore
                self.index.insert(i, tuple(box1 & box2), hit1.object)
                i += 1

    def __getitem__(self, query: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(query, BoundingBox):
            bbox = query
        elif isinstance(query, dict) and "bbox" in query:
            bbox = query["bbox"]
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        if not bbox.intersects(self.bounds):
            raise IndexError(f"query: {query} not found in index with bounds: {self.bounds}")

        hits = list(self.index.intersection(tuple(bbox), objects=True))
        if not hits:
            raise IndexError(f"query: {bbox} not found in merged index with bounds: {self.bounds}")
        path_ds1 = hits[0].object

        q1 = {"bbox": bbox, "path": path_ds1}
        q2 = {"bbox": bbox}

        samples = [self.datasets[0][q1], self.datasets[1][q2]]
        sample = self.collate_fn(samples)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample