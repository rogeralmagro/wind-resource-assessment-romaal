from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TURBINE_HUB_HEIGHTS = {
    "nrel_5mw": 90.0,
    "nrel_15mw": 150.0,
}


def get_turbine_hub_height(turbine_name: str) -> float:
    """
    Return the hub height for a supported turbine.
    """
    turbine_key = turbine_name.lower()
    if turbine_key not in TURBINE_HUB_HEIGHTS:
        raise KeyError(
            f"Unsupported turbine '{turbine_name}'. "
            f"Supported turbines: {list(TURBINE_HUB_HEIGHTS.keys())}"
        )
    return TURBINE_HUB_HEIGHTS[turbine_key]


def validate_power_curve(power_curve: pd.DataFrame) -> None:
    """
    Validate that a power curve contains the required columns.
    """
    required_columns = ("Wind Speed [m/s]", "Power [kW]")
    missing = [col for col in required_columns if col not in power_curve.columns]

    if missing:
        raise KeyError(
            "Power curve is missing required columns: " + ", ".join(missing)
        )


def interpolate_power_curve(
    wind_speed: xr.DataArray | np.ndarray,
    power_curve: pd.DataFrame,
) -> xr.DataArray | np.ndarray:
    """
    Interpolate turbine power output [kW] from wind speed using the power curve.

    Values below the minimum wind speed or above the maximum wind speed
    are set to 0 kW.
    """
    validate_power_curve(power_curve)

    curve_ws = power_curve["Wind Speed [m/s]"].to_numpy(dtype=float)
    curve_power = power_curve["Power [kW]"].to_numpy(dtype=float)

    if isinstance(wind_speed, xr.DataArray):
        values = np.interp(
            wind_speed.values,
            curve_ws,
            curve_power,
            left=0.0,
            right=0.0,
        )
        power = xr.DataArray(
            values,
            coords=wind_speed.coords,
            dims=wind_speed.dims,
            attrs={
                "units": "kW",
                "long_name": "Interpolated turbine power output",
            },
        )
        return power

    wind_speed = np.asarray(wind_speed, dtype=float)
    return np.interp(
        wind_speed,
        curve_ws,
        curve_power,
        left=0.0,
        right=0.0,
    )


def compute_power_output_series(
    wind_speed: xr.DataArray,
    power_curve: pd.DataFrame,
    turbine_name: str,
) -> xr.DataArray:
    """
    Compute the turbine power output time series [kW] from wind speed.
    """
    power = interpolate_power_curve(wind_speed, power_curve)

    if not isinstance(power, xr.DataArray):
        raise TypeError("Expected an xarray.DataArray power series.")

    power.attrs["turbine_name"] = turbine_name
    power.attrs["units"] = "kW"
    power.attrs["long_name"] = f"Power output for {turbine_name}"

    return power


def compute_aep_from_power_series(power_series: xr.DataArray) -> dict[str, float]:
    """
    Compute AEP from an hourly power time series.

    Assumes each time step represents 1 hour.

    Returns
    -------
    dict
        Dictionary with:
        - aep_kwh
        - aep_mwh
        - aep_gwh
        - mean_power_kw
        - n_hours
    """
    values = np.asarray(power_series.values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Power series contains no valid values.")

    aep_kwh = float(np.sum(values))
    aep_mwh = aep_kwh / 1_000.0
    aep_gwh = aep_kwh / 1_000_000.0
    mean_power_kw = float(np.mean(values))
    n_hours = float(values.size)

    return {
        "aep_kwh": aep_kwh,
        "aep_mwh": aep_mwh,
        "aep_gwh": aep_gwh,
        "mean_power_kw": mean_power_kw,
        "n_hours": n_hours,
    }