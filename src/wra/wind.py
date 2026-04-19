from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def wind_speed_from_uv(u: Any, v: Any) -> Any:
    """
    Compute wind speed from eastward (u) and northward (v) components.

    Parameters
    ----------
    u : array-like
        Eastward wind component.
    v : array-like
        Northward wind component.

    Returns
    -------
    array-like
        Wind speed with the same broadcasted shape as u and v.
    """
    return np.hypot(u, v)


def wind_direction_from_uv(u: Any, v: Any, calm_threshold: float = 0.0) -> Any:
    """
    Compute meteorological wind direction from u and v components.

    Meteorological convention:
    - 0°   = wind coming from North
    - 90°  = wind coming from East
    - 180° = wind coming from South
    - 270° = wind coming from West

    Parameters
    ----------
    u : array-like
        Eastward wind component.
    v : array-like
        Northward wind component.
    calm_threshold : float, optional
        Wind speeds less than or equal to this threshold are treated as calm
        winds and assigned NaN direction.

    Returns
    -------
    array-like
        Wind direction in degrees within [0, 360), with NaN for calm winds.
    """
    speed = wind_speed_from_uv(u, v)
    direction = (270.0 - np.rad2deg(np.arctan2(v, u))) % 360.0

    if isinstance(direction, xr.DataArray):
        return direction.where(speed > calm_threshold)

    return np.where(speed > calm_threshold, direction, np.nan)


def add_wind_variables_to_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Add wind speed and wind direction variables to an ERA5 dataset.

    Required variables in the dataset:
    - u10, v10
    - u100, v100

    Added variables:
    - ws10, wd10
    - ws100, wd100
    """
    required_variables = ("u10", "v10", "u100", "v100")
    missing_variables = [var for var in required_variables if var not in ds.data_vars]

    if missing_variables:
        raise KeyError(
            "Dataset is missing required variables: " + ", ".join(missing_variables)
        )

    ds = ds.copy()

    ds["ws10"] = wind_speed_from_uv(ds["u10"], ds["v10"])
    ds["wd10"] = wind_direction_from_uv(ds["u10"], ds["v10"])

    ds["ws100"] = wind_speed_from_uv(ds["u100"], ds["v100"])
    ds["wd100"] = wind_direction_from_uv(ds["u100"], ds["v100"])

    ds["ws10"].attrs.update({"units": "m/s", "long_name": "10 m wind speed"})
    ds["wd10"].attrs.update(
        {"units": "degrees", "long_name": "10 m wind direction"}
    )
    ds["ws100"].attrs.update({"units": "m/s", "long_name": "100 m wind speed"})
    ds["wd100"].attrs.update(
        {"units": "degrees", "long_name": "100 m wind direction"}
    )

    return ds