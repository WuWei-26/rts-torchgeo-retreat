# rts_sampler.py

import math
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import torch
from rtree import index
from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import GeoSampler, Units
from torchgeo.samplers.utils import _to_tuple, get_random_bounding_box, tile_to_chips


class RandomGeoSamplerMultiROI(GeoSampler):
    """Samples elements from multiple regions of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible. Note that
    randomly sampled chips may overlap.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[Tuple[float, float], float],
        length: Optional[int],
        # roi: Optional[BoundingBox] = None,
        roi_dict: Dict[Any, Sequence[BoundingBox | float]],
        units: Units = Units.PIXELS,
    ) -> None:
        """Initialize a new Sampler instance.

        Args:
            dataset: dataset to index from
            size: dimensions of each :term:`patch`
            length: number of random samples to draw per epoch
                (defaults to approximately the maximal number of non-overlapping
                :term:`chips <chip>` of size ``size`` that could be sampled from
                the dataset)
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            units: defines if ``size`` is in pixel or CRS units
        """

        self.res = dataset.res
        self.roi_dict = roi_dict

        self.size = _to_tuple(size)

        if units == Units.PIXELS:
            # self.size = (self.size[0] * self.res, self.size[1] * self.res)
            self.size = (self.size[0] * self.res[1], self.size[1] * self.res[0])

        self.length = 0
        # self.hits = []
        self.bboxs = []
        areas = []
        # rewrite
        self.index = index.Index(
            interleaved=False, properties=index.Property(dimension=3)
        )
        for roi, weight in zip(roi_dict["roi"], roi_dict["weight"]):
            hits = dataset.index.intersection(tuple(roi), objects=True)
            for hit in hits:
                # bbox in type of BoundingBox
                bbox: BoundingBox = BoundingBox(*hit.bounds) & roi
                # hit.object is the filepath
                self.index.insert(hit.id, tuple(bbox), hit.object)
                if (
                    bbox.maxx - bbox.minx >= self.size[1]
                    and bbox.maxy - bbox.miny >= self.size[0]
                ):
                    if bbox.area > 0:
                        # TODO: change stride value to increase sample number
                        rows, cols = tile_to_chips(bbox, self.size)
                        self.length += rows * cols
                    else:
                        self.length += 1
                    self.bboxs.append(bbox)
                    areas.append(bbox.area * weight)  # weighted areas
                    # areas.append(weight) # weighted areas

        if length is not None:
            self.length = length

        # torch.multinomial requires float probabilities > 0
        self.areas = torch.tensor(areas, dtype=torch.float)
        if torch.sum(self.areas) == 0:
            self.areas += 1

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile, weighted by area
            # print('areas: ', self.areas) # add by zyzhao
            idx = torch.multinomial(self.areas, 1)
            # hit = self.hits[idx]
            bbox = self.bboxs[idx]
            # bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bbox, self.size, self.res)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class RandomGeoSamplerMultiRoiMultiYear:
    def __init__(self, dataset: GeoDataset,
                 size: Union[Tuple[float, float], float],
                 stride: Union[Tuple[float, float], float],
                 length: Optional[int],
                 year_dict: Dict[Any, Sequence[int | float]],
                 units: Units = Units.PIXELS,
                 pair: bool = False,
                 prev_delta: int = 1) -> None:

        self.dataset = dataset
        self.pair = pair
        self.prev_delta = int(prev_delta)
        self.year_dict = year_dict

        # 归一化 size/stride
        def to_tuple(v): return (float(v), float(v)) if not isinstance(v, (tuple, list)) else (float(v[0]), float(v[1]))
        size   = to_tuple(size)
        stride = to_tuple(stride)

        # 分辨率与单位换算
        if isinstance(dataset.res, (tuple, list)):
            rx, ry = abs(float(dataset.res[0])), abs(float(dataset.res[1]))
        else:
            rx = ry = abs(float(dataset.res))
        if units == Units.PIXELS:
            self.size   = (size[0] * ry,   size[1] * rx)     # (H,W)
            self.stride = (stride[0] * ry, stride[1] * rx)
        else:
            self.size, self.stride = size, stride

        # R-tree（仅调试用，可选）
        self.index = index.Index(interleaved=False, properties=index.Property(dimension=3))

        # 每年的 chip 与概率
        self.year_bboxs: Dict[int, List[BoundingBox]] = {}
        self.year_areas: Dict[int, torch.Tensor] = {}
        self.year_lengths: List[int] = []

        # 年窗与切片函数
        def year_bbox(b: BoundingBox, y: int) -> BoundingBox:
            mint = datetime(y, 1, 1).timestamp()
            maxt = datetime(y+1, 1, 1).timestamp()
            return BoundingBox(b.minx, b.maxx, b.miny, b.maxy, mint, maxt)

        def chips_from_bbox(bbox: BoundingBox) -> List[BoundingBox]:
            chips = []
            H = bbox.maxy - bbox.miny
            W = bbox.maxx - bbox.minx
            if H < self.size[0] or W < self.size[1]:
                return chips
            y = bbox.miny
            while y + self.size[0] <= bbox.maxy + 1e-9:
                x = bbox.minx
                while x + self.size[1] <= bbox.maxx + 1e-9:
                    chips.append(BoundingBox(x, x + self.size[1], y, y + self.size[0], bbox.mint, bbox.maxt))
                    x += self.stride[1]
                y += self.stride[0]
            return chips

        # 针对 year_dict 中的每个 t，分别构建 t 与 t-1 的 chip
        for y, roi_entry in zip(self.year_dict["year"], self.year_dict["roi"]):
            y = int(y)
            ym1 = y - self.prev_delta
            # 初始化容器（如果以前未构建）
            if y   not in self.year_bboxs: self.year_bboxs[y]   = []
            if ym1 not in self.year_bboxs: self.year_bboxs[ym1] = []
            areas_y   = []  # 仅用于 t 年的抽样
            areas_ym1 = []  # 不用于抽样，只用于 active_years 判定
            year_len = 0

            for roi, roi_w in zip(roi_entry["roi"], roi_entry["weight"]):
                # t 年：按年窗命中数据集，切 chip
                qb_t = year_bbox(roi, y)
                for hit in self.dataset.index.intersection(tuple(qb_t), objects=True):
                    big = BoundingBox(*hit.bounds) & qb_t
                    chip_list = chips_from_bbox(big)
                    if not chip_list: continue
                    self.year_bboxs[y].extend(chip_list)
                    areas_y.extend([max(ch.area, 1e-9) * float(roi_w) for ch in chip_list])
                    year_len += len(chip_list)
                    # 可选：登记到 R-tree 便于调试
                    # self.index.insert(hit.id, tuple(big), hit.object)

                # t-1 年：同样构建（只用于 active_years 判定）
                qb_tm1 = year_bbox(roi, ym1)
                for hit in self.dataset.index.intersection(tuple(qb_tm1), objects=True):
                    big = BoundingBox(*hit.bounds) & qb_tm1
                    chip_list = chips_from_bbox(big)
                    if not chip_list: continue
                    self.year_bboxs[ym1].extend(chip_list)
                    areas_ym1.extend([max(ch.area, 1e-9) for ch in chip_list])

            # 概率向量（t 年用于抽样；tm1 年不抽样也不需要存概率，但可存一个占位）
            prob_t = torch.tensor(areas_y, dtype=torch.float)
            if prob_t.numel() == 0:
                prob_t = torch.zeros(0, dtype=torch.float)
            elif float(prob_t.sum()) == 0:
                prob_t = torch.ones_like(prob_t)

            self.year_areas[y] = prob_t
            self.year_lengths.append(year_len)

            # tm1 的概率占位（不用于抽样）
            prob_tm1 = torch.tensor(areas_ym1, dtype=torch.float)
            if prob_tm1.numel() == 0:
                prob_tm1 = torch.zeros(0, dtype=torch.float)
            self.year_areas[ym1] = self.year_areas.get(ym1, prob_tm1)

        # 总长度
        if length is not None:
            self.length = int(length)
        else:
            # 所有 t 年 chip 总数（也可用 max(...)）
            self.length = sum(len(self.year_bboxs.get(int(y), [])) for y in self.year_dict["year"])
        if self.length <= 0:
            raise RuntimeError("No tiles found. Check year_dict and ROI coverage.")

        # 激活年份：t 与 t-1 都有 chip 且 t 年概率非空
        self.active_years = []
        self.active_weights = []
        for y, w in zip(self.year_dict["year"], self.year_dict["weight"]):
            y = int(y); ym1 = y - self.prev_delta
            ok_y   = len(self.year_bboxs.get(y, []))   > 0 and self.year_areas.get(y, torch.tensor([])).numel()   > 0 and float(self.year_areas[y].sum())   > 0
            ok_ym1 = len(self.year_bboxs.get(ym1, [])) > 0 and self.year_areas.get(ym1, torch.tensor([])).numel() > 0
            if ok_y and ok_ym1:
                self.active_years.append(y)
                self.active_weights.append(float(w))
        if len(self.active_years) == 0:
            raise RuntimeError("No active years with valid (t,t-1) chips. Verify chip building for t and t-1.")

        yw = torch.tensor(self.active_weights, dtype=torch.float)
        if float(yw.sum()) == 0:
            yw = torch.ones_like(yw)
        self.year_weights_tensor = yw / yw.sum()

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _ in range(self.length):
            yi = torch.multinomial(self.year_weights_tensor, 1).item()
            year_t = self.active_years[yi]
            year_tm1 = year_t - self.prev_delta

            areas = self.year_areas[year_t]
            if areas.numel() == 0 or float(areas.sum()) == 0:
                continue
            idx = torch.multinomial(areas, 1).item()
            bbox = self.year_bboxs[year_t][idx]

            yield {"bbox": bbox, "year_t": year_t, "year_tm1": year_tm1}


class TestPreChippedGeoSampler(GeoSampler):
    """Samples entire files at a time.

    This is particularly useful for datasets that contain geospatial metadata
    and subclass :class:`~torchgeo.datasets.GeoDataset` but have already been
    pre-processed into :term:`chips <chip>`.

    This sampler should not be used with :class:`~torchgeo.datasets.NonGeoDataset`.
    You may encounter problems when using an :term:`ROI <region of interest (ROI)>`
    that partially intersects with one of the file bounding boxes, when using an
    :class:`~torchgeo.datasets.IntersectionDataset`, or when each file is in a
    different CRS. These issues can be solved by adding padding.
    """

    def __init__(
        self,
        dataset: GeoDataset,
        min_size: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        shuffle: bool = False,
        units: Units = Units.PIXELS,
        year_t: Optional[int] = None,  # 添加这两个参数
        year_tm1: Optional[int] = None
    ) -> None:
        """Initialize a new Sampler instance.

        .. versionadded:: 0.3

        Args:
            dataset: dataset to index from
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``dataset.index``)
            shuffle: if True, reshuffle data at every epoch
        """
        super().__init__(dataset, roi)
        self.shuffle = shuffle

        self.year_t = year_t
        self.year_tm1 = year_tm1

        # self.min_size = min_size
        self.min_size = _to_tuple(min_size)

        if units == Units.PIXELS:
            self.min_size = (self.min_size[0] * self.res[1] , self.min_size[1] * self.res[0])

        self.hits = []
        for hit in self.index.intersection(tuple(self.roi), objects=True):
            bounds = BoundingBox(*hit.bounds)  # type: ignore # no & roi
            if (
                bounds.maxx - bounds.minx >= self.min_size[1]
                and bounds.maxy - bounds.miny >= self.min_size[0]
            ):
                # self.length += 1
                self.hits.append(hit)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
            and the exact single file filepath
        """
        generator: Callable[[int], Iterable[int]] = range
        if self.shuffle:
            generator = torch.randperm # type: ignore

        for idx in generator(len(self)):
            query = {
                "bbox": BoundingBox(*self.hits[idx].bounds),
                "path": cast(str, self.hits[idx].object),
                "year_t": self.year_t,  # 添加这两行
                "year_tm1": self.year_tm1
            }
            yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.hits)
