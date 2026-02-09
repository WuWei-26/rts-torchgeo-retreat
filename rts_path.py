import os
from pathlib import Path

host_name = os.uname().nodename

if host_name == "birkeland":
    print("Running on birkeland")
    base_dir = Path("/media/tempData/zhzh5009/Data/RTS")
    img_dir = base_dir / "landsat/blh_hxm"
    dem_dir = base_dir / "NASADEM"
    mask_dir = base_dir / "heatmap/gt_mask_rts_v3_aea_updated_r4_4_no_gamma_32646"
    polygon_dir = base_dir / "polygons"

elif host_name == "cryo-A100":
    print("Running on cryo-A100")
    base_dir = Path("/DATA/DATA1/joey")
    img_dir = base_dir / "LANDSAT_LC08_MIX"
    dem_dir = base_dir / "NASADEM"
    mask_dir = base_dir / "gt_mask_rts_v3_aea_updated_r4_4_no_gamma_32646"
    polygon_dir = base_dir / "rts_polygons"

assert img_dir.exists()
assert dem_dir.exists()
assert mask_dir.exists()
assert polygon_dir.exists()
