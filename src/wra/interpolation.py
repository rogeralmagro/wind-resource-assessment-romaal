from __future__ import annotations

import xarray as xr

from .wind import wind_direction_from_uv, wind_speed_from_uv


REQUIRED_UV_VARIABLES = ("u10", "v10", "u100", "v100")


def subset_dataset_years(
    ds: xr.Dataset,
    start_year: int = 1997,
    end_year: int = 2008,
) -> xr.Dataset:
    """
    Subset a dataset using start and end years (inclusive).
    """
    if start_year > end_year:
        raise ValueError("start_year must be smaller than or equal to end_year.")

    start_time = f"{start_year}-01-01"
    end_time = f"{end_year}-12-31T23:00:00"

    return ds.sel(time=slice(start_time, end_time))


def validate_location_inside_grid(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
) -> None:
    """
    Check that the target location is inside the ERA5 box.
    """
    lat_min = float(ds["latitude"].min())
    lat_max = float(ds["latitude"].max())
    lon_min = float(ds["longitude"].min())
    lon_max = float(ds["longitude"].max())

    if not (lat_min <= latitude <= lat_max):
        raise ValueError(
            f"Latitude {latitude} is outside the grid bounds [{lat_min}, {lat_max}]."
        )

    if not (lon_min <= longitude <= lon_max):
        raise ValueError(
            f"Longitude {longitude} is outside the grid bounds [{lon_min}, {lon_max}]."
        )


def interpolate_uv_to_location(
    ds: xr.Dataset,
    latitude: float,
    longitude: float,
    start_year: int = 1997,
    end_year: int = 2008,
) -> xr.Dataset:
    """
    Interpolate u/v wind components to a target location and compute
    wind speed/direction at 10 m and 100 m.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 dataset containing u10, v10, u100, v100.
    latitude : float
        Target latitude.
    longitude : float
        Target longitude.
    start_year : int, optional
        First year to include.
    end_year : int, optional
        Last year to include.

    Returns
    -------
    xr.Dataset
        Dataset at the target location with:
        u10, v10, u100, v100, ws10, wd10, ws100, wd100
    """
    missing_variables = [var for var in REQUIRED_UV_VARIABLES if var not in ds.data_vars]
    if missing_variables:
        raise KeyError(
            "Dataset is missing required variables: " + ", ".join(missing_variables)
        )

    ds_subset = subset_dataset_years(ds, start_year=start_year, end_year=end_year)
    validate_location_inside_grid(ds_subset, latitude=latitude, longitude=longitude)

    point_ds = ds_subset[list(REQUIRED_UV_VARIABLES)].interp(
        latitude=latitude,
        longitude=longitude,
        method="linear",
    )

    point_ds = point_ds.copy()

    point_ds["ws10"] = wind_speed_from_uv(point_ds["u10"], point_ds["v10"])
    point_ds["wd10"] = wind_direction_from_uv(point_ds["u10"], point_ds["v10"])
    point_ds["ws100"] = wind_speed_from_uv(point_ds["u100"], point_ds["v100"])
    point_ds["wd100"] = wind_direction_from_uv(point_ds["u100"], point_ds["v100"])

    point_ds["ws10"].attrs.update({"units": "m/s", "long_name": "10 m wind speed"})
    point_ds["wd10"].attrs.update(
        {"units": "degrees", "long_name": "10 m wind direction"}
    )
    point_ds["ws100"].attrs.update({"units": "m/s", "long_name": "100 m wind speed"})
    point_ds["wd100"].attrs.update(
        {"units": "degrees", "long_name": "100 m wind direction"}
    )

    point_ds.attrs["target_latitude"] = latitude
    point_ds.attrs["target_longitude"] = longitude
    point_ds.attrs["start_year"] = start_year
    point_ds.attrs["end_year"] = end_year

    return point_ds