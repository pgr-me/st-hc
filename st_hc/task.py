# Standard library imports
import argparse
from datetime import datetime as dt
import json
import os
import multiprocessing
from pathlib import Path
import warnings

# Third party imports
import geopandas as gpd
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
import rasterio as rio
from scipy.spatial.distance import mahalanobis
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from download import download_cogs

# Data dir for users of Docker
DATA_DIR = Path("/work/data")
# Parallelization params
CORES = multiprocessing.cpu_count() - 1
# Download constants
RES = 10
STAC_ENDPOINT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = ["sentinel-2-l2a"]
COLLECTION_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "qa"]
OUTPUT_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22", "ndvi", "qa"]
# Train MLP constants
BATCH_SIZE = 1028
EPOCHS = 30
GAMMA = 0.1
LAT_SPACE_SIZES = [2, 4, 16, 64, 256,]
LRS = [0.5, 0.1]
STEP_SIZES = [5, 10, 15]
TR_VAL_SPLIT = 0.9
WEIGHT_DECAY = 5e-4
# KMeans constants
K = 4
region_sample_size = 5000
# Change detection constants
BOCD_PARAMS = dict(hazard=1/100, mean0=1, var0=2, varx=1,)
WINDOW = 10
START_INDEX = 10
N_STDS = 1.75



def parser():
    parser = argparse.ArgumentParser(description="Run the pipeline.")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR, help="Path to data directory.")
    parser.add_argument("--cores", type=int, default=1, help="Number of cores to use.")
    # Download args
    parser.add_argument("--skip_download", action="store_true", help="True to skip downloading COGs.")
    parser.add_argument("--stac_endpoint", type=str, default=STAC_ENDPOINT, help="STAC enpoint URL.")
    parser.add_argument("--collections", nargs="+", default=COLLECTIONS, help="List of STAC collections to download from.")
    parser.add_argument("--output_bands", nargs="+", default=OUTPUT_BANDS, help="List of bands to retain.")
    # Train MLP args
    parser.add_argument("--skip_train_mlp", action="store_true", help="True to skip training of MLP.")
    parser.add_argument("--tr_val_split", type=float, default=TR_VAL_SPLIT, help="Fraction of training v. validation set.")
    parser.add_argument("--batch_size", nargs="+", default=BATCH_SIZE, help="MLP batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="MLP epochs.")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="MLP gamma.")
    # MLP inference args
    parser.add_argument("--skip_mlp_inference", action="store_true", help="True to skip MLP inference.")
    # KMeans args
    parser.add_argument("--skip_clustering", action="store_true", help="True to skip KMeans task.")
    # Change detection args
    parser.add_argument("--skip_change_detection", action="store_true", help="True to skip change detection task.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    data_dir = args.data_dir
    cores = args.cores
    # Primary data dirs
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"
    # Download dir
    cogs_dir = interim_dir / "cogs"
    # MLP training dirs
    models_dir = interim_dir / "models"
    scores_dir = interim_dir / "scores"
    # MLP inference dirs
    encoded_dir = interim_dir / "encoded"
    meta_dir = interim_dir / "meta"
    # Cluster dirs
    cluster_dir = interim_dir / "cluster"
    # Make dirs
    for dir_ in [cogs_dir, models_dir, scores_dir, encoded_dir, meta_dir, cluster_dir]:
        dir_.mkdir(parents=True, exist_ok=True)
        
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Download task
    if not args.skip_download:
        os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "FALSE"
        print("[Download]: Load input region geojson and config.")
        with open(raw_dir / "cfg.json")  as f:
            cfg = json.load(f)
        regions = gpd.read_file(raw_dir / "regions.geojson")
        regions["time_range"] = regions["s2_start"] + "/" + regions["s2_end"]
        download_cogs(
            regions, cogs_dir, cfg,
            stac_endpoint=args.stac_endpoint,
            collections=args.collections,
            output_bands=args.output_bands,
        )
  