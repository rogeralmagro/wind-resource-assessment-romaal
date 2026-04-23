from __future__ import annotations

import numpy as np
import xarray as xr


def estimate_power_law_exponent(
    ws_low: xr.DataArray,
    ws_high: xr.DataArray,
    z_low: float = 10.0,
    z_high: float = 100.0,
) -> xr.DataArray:
    """
    Estimate the power-law exponent alpha from wind speeds at two heights.

    Parameters
    ----------
    ws_low : xr.DataArray
        Wind speed at the lower height.
    ws_high : xr.DataArray
        Wind speed at the higher height.
    z_low : float, optional
        Lower height in meters.
    z_high : float, optional
        Higher height in meters.

    Returns
    -------
    xr.DataArray
        Time series of estimated alpha values.
    """
    if z_low <= 0 or z_high <= 0:
        raise ValueError("Heights must be positive.")
    if z_low == z_high:
        raise ValueError("z_low and z_high must be different.")

    valid = (ws_low > 0) & (ws_high > 0)

    alpha = xr.where(
        valid,
        np.log(ws_high / ws_low) / np.log(z_high / z_low),
        np.nan,
    )

    alpha.attrs["long_name"] = "Power-law exponent"
    alpha.attrs["units"] = "-"
    return alpha


def power_law_wind_speed(
    ws_ref: xr.DataArray,
    z: float,
    z_ref: float = 100.0,
    alpha: float | xr.DataArray = 0.14,
) -> xr.DataArray:
    """
    Compute wind speed at height z using the power-law profile.

    Parameters
    ----------
    ws_ref : xr.DataArray
        Wind speed at the reference height.
    z : float
        Target height in meters.
    z_ref : float, optional
        Reference height in meters.
    alpha : float or xr.DataArray, optional
        Power-law exponent. Can be a constant or a time series.

    Returns
    -------
    xr.DataArray
        Wind speed time series at the target height.
    """
    if z <= 0 or z_ref <= 0:
        raise ValueError("Heights must be positive.")

    ws_z = ws_ref * (z / z_ref) ** alpha
    ws_z.attrs["long_name"] = f"Wind speed at {z} m"
    ws_z.attrs["units"] = "m/s"
    return ws_z


def extrapolate_to_height(
    point_ds: xr.Dataset,
    target_height: float,
    alpha: float | None = None,
    low_height: float = 10.0,
    high_height: float = 100.0,
) -> xr.Dataset:
    """
    Extrapolate wind speed to a target height using the power-law profile.

    Parameters
    ----------
    point_ds : xr.Dataset
        Dataset at one interpolated location containing ws10 and ws100.
    target_height : float
        Target height in meters.
    alpha : float or None, optional
        If provided, use a constant alpha.
        If None, estimate alpha from ws10 and ws100.
    low_height : float, optional
        Lower measurement height in meters.
    high_height : float, optional
        Higher measurement height in meters.

    Returns
    -------
    xr.Dataset
        Copy of the input dataset with:
        - alpha
        - ws_z
        - wd_z (copied from wd100 if available)
    """
    if target_height <= 0:
        raise ValueError("target_height must be positive.")

    required = ("ws10", "ws100")
    missing = [var for var in required if var not in point_ds.data_vars]
    if missing:
        raise KeyError(
            "Dataset is missing required variables for profile extrapolation: "
            + ", ".join(missing)
        )

    out = point_ds.copy()

    if alpha is None:
        alpha_da = estimate_power_law_exponent(
            out["ws10"],
            out["ws100"],
            z_low=low_height,
            z_high=high_height,
        )
        out["alpha"] = alpha_da
        out["alpha"].attrs["description"] = (
            "Estimated from ws10 and ws100 using the power-law profile"
        )
    else:
        out["alpha"] = xr.full_like(out["ws100"], fill_value=float(alpha))
        out["alpha"].attrs["description"] = "User-defined constant power-law exponent"
        out["alpha"].attrs["units"] = "-"

    out["ws_z"] = power_law_wind_speed(
        out["ws100"],
        z=target_height,
        z_ref=high_height,
        alpha=out["alpha"],
    )
    out["ws_z"].attrs["target_height_m"] = target_height
    out["ws_z"].attrs["reference_height_m"] = high_height
    out["ws_z"].attrs["method"] = "power_law_profile"

    if "wd100" in out.data_vars:
        out["wd_z"] = out["wd100"].copy()
        out["wd_z"].attrs["long_name"] = f"Wind direction at {target_height} m"
        out["wd_z"].attrs["units"] = "degrees"
        out["wd_z"].attrs["note"] = (
            "Approximated using wind direction at 100 m as stated in the project methodology"
        )
        out["wd_z"].attrs["target_height_m"] = target_height

    out.attrs["target_height_m"] = target_height
    out.attrs["profile_method"] = "power_law"

    return out