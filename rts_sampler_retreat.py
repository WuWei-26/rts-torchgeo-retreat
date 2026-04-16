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
import random
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

        def to_tuple(v): return (float(v), float(v)) if not isinstance(v, (tuple, list)) else (float(v[0]), float(v[1]))
        size = to_tuple(size)
        stride = to_tuple(stride)

        if isinstance(dataset.res, (tuple, list)):
            rx, ry = abs(float(dataset.res[0])), abs(float(dataset.res[1]))
        else:
            rx = ry = abs(float(dataset.res))
        if units == Units.PIXELS:
            self.size   = (size[0] * ry,   size[1] * rx)
            self.stride = (stride[0] * ry, stride[1] * rx)
        else:
            self.size, self.stride = size, stride

        self.index = index.Index(interleaved=False, properties=index.Property(dimension=3))

        # (year_t, year_tm1) -> roi_idx -> chips
        self.year_roi_chips: Dict[Tuple[int,int], List[List[BoundingBox]]] = {}
        self.year_roi_weights: Dict[Tuple[int,int], torch.Tensor] = {}
        self.year_lengths: List[int] = []
        self.year_bboxs: Dict[Tuple[int,int], List[BoundingBox]] = {}
        self.year_areas: Dict[Tuple[int,int], torch.Tensor] = {}

        def year_bbox(b: BoundingBox, y: int) -> BoundingBox:
            mint = datetime(y, 1, 1).timestamp()
            maxt = datetime(y+1, 1, 1).timestamp()
            return BoundingBox(b.minx, b.maxx, b.miny, b.maxy, mint, maxt)

        def chips_from_bbox(bbox: BoundingBox) -> List[BoundingBox]:
            chips = []
            H = bbox.maxy - bbox.miny
            W = bbox.maxx - bbox.minx
            if H < self.size[0] or W < self.size[1] or bbox.mint >= bbox.maxt:
                return chips
            y = bbox.miny
            while y + self.size[0] <= bbox.maxy + 1e-9:
                x = bbox.minx
                while x + self.size[1] <= bbox.maxx + 1e-9:
                    chips.append(BoundingBox(x, x + self.size[1], y, y + self.size[0], bbox.mint, bbox.maxt))
                    x += self.stride[1]
                y += self.stride[0]
            return chips

        for y, ym1, roi_entry in zip(self.year_dict["year"], self.year_dict["year_tm1"], self.year_dict["roi"]):
            y   = int(y)
            ym1 = int(ym1)
            pair_key = (y, ym1)  # ← tuple key，避免同一 year_t 多配对互相覆盖

            self.year_bboxs[pair_key] = []

            roi_chips_list: List[List[BoundingBox]] = []
            roi_weights_list: List[float] = []
            all_areas_flat: List[float] = []

            for roi, roi_w in zip(roi_entry["roi"], roi_entry["weight"]):
                qb_t = year_bbox(roi, y)

                chips_this_roi: List[BoundingBox] = []
                areas_this_roi: List[float] = []

                for hit in self.dataset.index.intersection(tuple(qb_t), objects=True):
                    big = BoundingBox(*hit.bounds) & qb_t
                    for ch in chips_from_bbox(big):
                        ch_as_tm1 = BoundingBox(ch.minx, ch.maxx, ch.miny, ch.maxy,
                                                datetime(ym1, 1, 1).timestamp(),
                                                datetime(ym1+1, 1, 1).timestamp())
                        hits_tm1 = list(self.dataset.index.intersection(tuple(ch_as_tm1), objects=False))
                        if True:
                            self.year_bboxs[pair_key].append(ch)
                            areas_this_roi.append(max(ch.area, 1e-9) * float(roi_w))
                            chips_this_roi.append(ch)

                roi_chips_list.append(chips_this_roi)
                roi_weights_list.append(float(roi_w) if chips_this_roi else 0.0)
                all_areas_flat.extend(areas_this_roi)

            self.year_roi_chips[pair_key] = roi_chips_list
            rw = torch.tensor(roi_weights_list, dtype=torch.float)
            if float(rw.sum()) == 0:
                rw = torch.ones_like(rw)
            self.year_roi_weights[pair_key] = rw / rw.sum()

            prob = torch.tensor(all_areas_flat, dtype=torch.float)
            if prob.numel() > 0 and float(prob.sum()) == 0:
                prob = torch.ones_like(prob)
            self.year_areas[pair_key] = prob
            self.year_lengths.append(len(self.year_bboxs[pair_key]))
            print(f"[Sampler] year={y} tm1={ym1}: {len(self.year_bboxs[pair_key])} valid chips")
            for ri, (chips, w) in enumerate(zip(roi_chips_list, roi_weights_list)):
                cid = roi_entry.get("cluster_id", list(range(len(roi_chips_list))))[ri]
                print(f"  ROI {ri} (cluster={cid}): {len(chips)} chips, weight={w:.4f}")

        if length is not None:
            self.length = int(length)
        else:
            self.length = sum(len(v) for v in self.year_bboxs.values())
        if self.length <= 0:
            raise RuntimeError("No tiles found. Check year_dict and ROI coverage.")

        self.active_years: List[Tuple[int,int]] = []
        self.active_weights: List[float] = []
        for y, ym1, w in zip(self.year_dict["year"], self.year_dict["year_tm1"], self.year_dict["weight"]):
            y, ym1 = int(y), int(ym1)
            pair_key = (y, ym1)
            ok = (len(self.year_bboxs.get(pair_key, [])) > 0
                  and self.year_areas.get(pair_key, torch.tensor([])).numel() > 0
                  and float(self.year_areas[pair_key].sum()) > 0)
            if ok:
                self.active_years.append(pair_key)
                self.active_weights.append(float(w))

        yw = torch.tensor(self.active_weights, dtype=torch.float)
        if float(yw.sum()) == 0:
            yw = torch.ones_like(yw)
        self.year_weights_tensor = yw / yw.sum()

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _ in range(self.length):
            yi = torch.multinomial(self.year_weights_tensor, 1, replacement=True).item()
            year_t, year_tm1 = self.active_years[yi]
            pair_key = (year_t, year_tm1)

            roi_weights   = self.year_roi_weights[pair_key]
            roi_chips_list = self.year_roi_chips[pair_key]

            valid_roi_idx = [i for i, chips in enumerate(roi_chips_list) if len(chips) > 0]
            if not valid_roi_idx:
                continue

            valid_weights = roi_weights[valid_roi_idx]
            if float(valid_weights.sum()) == 0:
                valid_weights = torch.ones_like(valid_weights)
            valid_weights = valid_weights / valid_weights.sum()

            ri = valid_roi_idx[torch.multinomial(valid_weights, 1, replacement=True).item()]
            chips_in_roi = roi_chips_list[ri]

            bbox = random.choice(chips_in_roi)
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
        year_t: Optional[int] = None,
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
                "year_t": self.year_t,
                "year_tm1": self.year_tm1
            }
            yield query

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return len(self.hits)
