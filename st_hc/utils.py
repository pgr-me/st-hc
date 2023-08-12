#!/usr/bin/python3
# Standard-library imports
from pathlib import Path

# Third-party imports
from xarray.core.dataset import Dataset
import matplotlib.pyplot as plt


class DirectoryHelper:
    """
    Helper class for managing data directories.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

        # Primary data dirs
        self.raw_dir = data_dir / "raw"
        self.interim_dir = data_dir / "interim"
        self.processed_dir = data_dir / "processed"
        # Regions dir
        self.raw_dir / "regions"
        # Download dirs
        self.s2_cogs_dir = self.interim_dir / "s2_cogs"
        self.ls_cogs_dir = self.interim_dir / "ls_cogs"
        self.s1_cogs_dir = self.interim_dir / "s1_cogs"
        self.regions_dir = self.interim_dir / "regions"
        # Phase dirs
        self.raw_phase1 = self.raw_dir / "phase1"
        self.raw_phase2 = self.raw_dir / "phase2"
        self.raw_phase3 = self.raw_dir / "phase3"
        self.raw_phase4 = self.raw_dir / "phase4"
        self.interim_phase1 = self.interim_dir / "phase1"
        self.interim_phase2 = self.interim_dir / "phase2"
        self.interim_phase3 = self.interim_dir / "phase3"
        self.interim_phase4 = self.interim_dir / "phase4"
        self.processed_phase1 = self.processed_dir / "phase1"
        self.processed_phase2 = self.processed_dir / "phase2"
        self.processed_phase3 = self.processed_dir / "phase3"
        self.processed_phase4 = self.processed_dir / "phase4"
        # Make dirs
        for dir_ in [
            self.s2_cogs_dir,
            self.ls_cogs_dir,
            self.s1_cogs_dir,
            self.regions_dir,
            self.raw_phase1,
            self.raw_phase2,
            self.raw_phase3,
            self.raw_phase4,
            self.interim_phase1,
            self.interim_phase2,
            self.interim_phase3,
            self.interim_phase4,
            self.processed_phase1,
            self.processed_phase2,
            self.processed_phase3,
            self.processed_phase4,
        ]:
            dir_.mkdir(parents=True, exist_ok=True)


def to_float(xx: Dataset) -> Dataset:
    """
    Converts the data type of the dataset to float32.
    Arguments:
        xx {Dataset} -- Dataset to convert.
    Returns: Converted dataset.
    """
    _xx = xx.astype("float32")
    nodata = _xx.attrs.pop("nodata", None)
    if nodata is None:
        return _xx
    return _xx.where(xx != nodata)
