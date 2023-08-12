"""
Adapted from: https://github.com/pgr-me/rasmussen-705.603/blob/main/FinalProject/download.py
"""
# Standard library imports
import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List

# Third-party imports
import dask.distributed
import dask.utils
from datacube.utils.cog import write_cog
from dotenv import load_dotenv
import geopandas as gpd
import mgrs
import numpy as np
from odc.stac import stac_load
import pandas as pd
import planetary_computer as pc
from pyproj import CRS, Proj
from pystac_client import Client
from shapely.geometry import box
from xarray.core.dataset import Dataset

# Local imports
from utils import DirectoryHelper, to_float


print("[Download]: Load environment variables from .env file.")
load_dotenv()
USGS_API_KEY = os.environ["USGS_API_KEY"]
USGS_TOKEN_NAME = os.environ["USGS_TOKEN_NAME"]
USGS_USERNAME = os.environ["USGS_USERNAME"]
USGS_PASSWORD = os.environ["USGS_PASSWORD"]
AWS_ACCESS_KEY = os.environ["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = os.environ["AWS_SECRET_KEY"]
NASA_EARTHDATA_S3_ACCESS_KEY = os.environ["NASA_EARTHDATA_S3_ACCESS_KEY"]
NASA_EARTHDATA_S3_SECRET_KEY = os.environ["NASA_EARTHDATA_S3_SECRET_KEY"]
NASA_EARTHDATA_S3_SESSION = os.environ["NASA_EARTHDATA_S3_SESSION"]
NASA_EARTHDATA_USERNAME = os.environ["NASA_EARTHDATA_USERNAME"]
NASA_EARTHDATA_PASSWORD = os.environ["NASA_EARTHDATA_PASSWORD"]

RES = 10
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
COLLECTION_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa"]
OUTPUT_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "ndvi", "qa"]
BOX_LENGTH = 1000  # meters
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"


def argparser():
    parser = argparse.ArgumentParser(description="Download STAC data for selected regions.")
    parser.add_argument(
        "--data_dir", "-d",
        type=Path,
        help="Path to data directory."
    )
    parser.add_argument(
        "--box_length", "-box",
        type=int,
        default=BOX_LENGTH,
        help="Width of region."
    )
    parser.add_argument(
        "--cores", "-c",
        type=int,
        default=1,
        help="Number of cores to use."
    )
    parser.add_argument(
        "--skip_download", "-s",
        action="store_true",
        help="True to skip downloading COGs."
    )
    parser.add_argument(
        "--stac_endpoint", "-e",
        type=str,
        default=STAC_ENDPOINT,
        help="STAC enpoint URL."
    )
    parser.add_argument(
        "--collection", "-cl",
        type=str,
        default=COLLECTION,
        help="List of STAC collections to download from.",
    )
    parser.add_argument(
        "--config_src", "-cfg",
        type=Path,
        help="Path to STAC config JSON.",
    )
    parser.add_argument(
        "--output_bands", "-b",
        nargs="+",
        default=OUTPUT_BANDS,
        help="List of bands to retain.",
    )
    parser.add_argument(
        "--regions_src", "-r",
        type=Path,
        help="Path to regions JSON.",
    )
    parser.add_argument(
        "--resolution", "-res",
        type=int,
        default=RES,
        help="Path to regions JSON.",
    )

    return parser.parse_args()



def download_cogs(
    region: str,
    timerange: str,
    bbox: List[float],
    resolution: int,
    cfg: Dict[str, Any],
    stac_endpoint: str = STAC_ENDPOINT,
    collection: str = COLLECTION,
    bands: List[str] = COLLECTION_BANDS,
    compute_ndvi: bool = True,
):
    """
    Download COGs for regions.
    Arguments:
        region: Region-specific dictionary that provides ID, time-range and bounds.
        timerange: Start and end date in YYYY-MM-DD/YYYY-MM-DD format.
        bbox: Bounding box coordinates in [minx, miny, maxx, maxy] format.
        resolution: Resolution in meters.
        cfg: Datacube configuration.
        stac_endpoint: STAC endpoint URL.
        collections: STAC collections list.
        output_bands: List of output bands.
        compute_ndvi: True to compute NDVI.
    Returns: xarray dataset.
    """
    bands = [x for x in bands if x.lower() != "ndvi"]
    print(f"[{region}][{timerange}]: Search catalog.")
    catalog = Client.open(stac_endpoint)
    query = catalog.search(
        collections=collection,
        datetime=timerange,
        bbox=bbox,
    )
    items = list(query.get_items())
    print(f"[{region}][{timerange}]: Found {len(items)} items.")

    items = [item for item in items if item.properties["eo:cloud_cover"] < 30]
    print(f"[{region}][{timerange}]: Selected {len(items)} items.")

    print(f"[{region}][{timerange}]: Load items into data cube.")
    xx = stac_load(
        items,
        bands=bands,
        resolution=resolution,
        chunks={"x": 1028, "y": 1028},
        stac_cfg=cfg,
        patch_url=pc.sign,
        crs="utm",
        bbox=bbox,
        fail_on_error=False,
    )
    output_bands = bands
    if compute_ndvi:
        nir08 = to_float(xx.nir08)
        red = to_float(xx.red)
        ndvi = ((nir08 - red) / (nir08 + red)).fillna(0) * 10000
        xx["ndvi"] = ndvi
        output_bands = bands + ["ndvi"]

    print(f"[{region}]: Re-order bands.")
    xx = xx[output_bands].astype(np.int32)

    return xx


def make_region_geojson(name: str, mgrs_id: str, box_length: int, output_dir: Path):
    
    mgrs_obj = mgrs.MGRS()
    
    utm_zone, hemisphere, easting, northing = mgrs_obj.MGRSToUTM(mgrs_id)
    xmin, ymin, xmax, ymax = (
        easting - box_length // 2,
        northing - box_length // 2,
        easting + box_length // 2,
        northing + box_length // 2,
    )
    hemisphere = True if hemisphere == "S" else False
    di = dict(proj="utm", zone=utm_zone, south=hemisphere)
    crs = CRS.from_dict(di)

    geo = box(xmin, ymin, xmax, ymax)
    gseries = gpd.GeoSeries([geo], crs=crs).to_crs("EPSG:4326")
    gseries.to_file(output_dir / f"{name}-{mgrs_id}.geojson", driver="GeoJSON")
    return gseries


def write_cogs(region: str, xx: Dataset, cogs_dir: Path, overwrite=False):
    """
    Write COGs for one region.
    Arguments:
        region: Region name.
        xx: Dataset to write.
        cogs_dir: Directory to write COGs to.
    """
    n_files = len(xx.time.data)
    print(f"[{region}]: Write {n_files} TIFs.")

    for i in range(n_files):
        date = xx.isel(time=i).time.dt.strftime("%Y-%m-%d").data
        dst = cogs_dir / f"{region}_{date}.tif"
        if not dst.exists() or overwrite:
            try:
                arr = xx.isel(time=i).to_array()
                write_cog(geo_im=arr, fname=dst, overwrite=False).compute()
                print(f"[{region}]: Wrote {dst.name}.")
            except Exception as e:
                print(f"[{region}]: Failed to write {dst.name}.")
                print(f"[{region}]: {e}")
        else:
            print(f"[{region}]: {dst.name} exists.")


if __name__ == "__main__":
    args = argparser()
    
    # TODO: REMOVE
    args.data_dir = Path("C:\\Users\\Peter\\gh\\st-hc\\data")
    args.regions_src = Path("C:\\Users\\Peter\\gh\\st-hc\\regions.json")
    args.config_src = Path("C:\\Users\\Peter\\gh\\st-hc\\cfg.json")
    
    dir_helper = DirectoryHelper(args.data_dir)
    with open(args.regions_src) as f:
        regions = json.load(f)
    shutil.copyfile(args.regions_src, dir_helper.raw_dir / "regions.json")
    for region, di in regions.items():
        print(f"[{region}]: Collect inputs.")
        mgrs_id = di["mgrs"]
        geoseries = make_region_geojson(region, mgrs_id, args.box_length, dir_helper.regions_dir)
        bbox = geoseries.bounds.loc[0].values.tolist()
        with open(args.config_src)  as f:
            cfg = json.load(f)
        
        print(f"[{region}]: Download COGs.")
        xx = download_cogs(
            region,
            di["timerange"],
            bbox,
            args.resolution,
            cfg,
            args.stac_endpoint,
            args.collection,
            args.output_bands,
            compute_ndvi=True,
        )
        print(f"[{region}]: Write COGs.")
        write_cogs(region, xx, dir_helper.s2_cogs_dir)
        print(f"[{region}]: Done.")