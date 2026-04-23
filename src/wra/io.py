from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import xarray as xr


WIND_VARIABLES = ("u10", "v10", "u100", "v100")
TIME_CANDIDATES = ("time", "valid_time")
LAT_CANDIDATES = ("latitude", "lat")
LON_CANDIDATES = ("longitude", "lon")

REQUIRED_POWER_COLUMNS = ("Wind Speed [m/s]", "Power [kW]")


def _to_path_list(paths: Iterable[str | Path], suffix: str) -> list[Path]:
    """Convert an iterable of paths to a validated list of Path objects."""
    path_list = [Path(path) for path in paths]

    if not path_list:
        raise FileNotFoundError(f"No files with suffix '{suffix}' were provided.")

    for path in path_list:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected '{suffix}' file, got: {path}")

    return sorted(path_list)


def _find_name(candidates: tuple[str, ...], names: Iterable[str]) -> str:
    """Return the first matching name from a list of candidates."""
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"Could not find any of {candidates} in {list(names)}")


def _standardize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize dimension/coordinate names to:
    - time
    - latitude
    - longitude
    """
    rename_map: dict[str, str] = {}

    all_names = set(ds.dims) | set(ds.coords)

    time_name = _find_name(TIME_CANDIDATES, all_names)
    lat_name = _find_name(LAT_CANDIDATES, all_names)
    lon_name = _find_name(LON_CANDIDATES, all_names)

    if time_name != "time":
        rename_map[time_name] = "time"
    if lat_name != "latitude":
        rename_map[lat_name] = "latitude"
    if lon_name != "longitude":
        rename_map[lon_name] = "longitude"

    ds = ds.rename(rename_map)

    if "time" not in ds.coords:
        raise KeyError("Dataset does not contain a valid time coordinate.")

    ds = ds.sortby("time")
    ds = ds.sortby("latitude")
    ds = ds.sortby("longitude")

    return ds


def load_wind_dataset(nc_files: Iterable[str | Path]) -> xr.Dataset:
    """
    Load and concatenate multiple ERA5 NetCDF files into one xarray Dataset.
    """
    file_list = _to_path_list(nc_files, ".nc")

    datasets: list[xr.Dataset] = []
    for file_path in file_list:
        with xr.open_dataset(file_path) as ds:
            ds_std = _standardize_dataset(ds)
            datasets.append(ds_std.load())

    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")

    time_index = combined.indexes["time"]
    combined = combined.isel(time=~time_index.duplicated())

    missing_variables = [var for var in WIND_VARIABLES if var not in combined.data_vars]
    if missing_variables:
        raise KeyError(
            "Dataset is missing required wind variables: "
            + ", ".join(missing_variables)
        )

    return combined


def load_wind_dataset_from_directory(inputs_dir: str | Path) -> xr.Dataset:
    """Load all NetCDF wind files found in a directory."""
    inputs_path = Path(inputs_dir)
    nc_files = sorted(inputs_path.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"No NetCDF files found in: {inputs_path}")

    return load_wind_dataset(nc_files)

def infer_turbine_name(csv_path: str | Path) -> str:
    """Infer a clean turbine name from the CSV file name."""
    stem = Path(csv_path).stem.lower()

    if "15mw" in stem:
        return "nrel_15mw"
    if "5mw" in stem:
        return "nrel_5mw"

    return Path(csv_path).stem.lower().replace(" ", "_")

def load_power_curve(csv_path: str | Path) -> pd.DataFrame:
    """Load one turbine power-curve CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Power curve file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_columns = [col for col in REQUIRED_POWER_COLUMNS if col not in df.columns]
    if missing_columns:
        raise KeyError(
            "Power curve file is missing required columns: "
            + ", ".join(missing_columns)
        )

    df = df.sort_values("Wind Speed [m/s]").reset_index(drop=True)
    return df


def load_all_power_curves(inputs_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all turbine power-curve CSV files found in a directory."""
    inputs_path = Path(inputs_dir)
    csv_files = sorted(inputs_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV power-curve files found in: {inputs_path}")

    curves: dict[str, pd.DataFrame] = {}
    for csv_file in csv_files:
        turbine_name = infer_turbine_name(csv_file)
        curves[turbine_name] = load_power_curve(csv_file)

    return curves