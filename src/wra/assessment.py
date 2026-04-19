from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .energy import (
    compute_aep_from_power_series,
    compute_power_output_series,
    get_turbine_hub_height,
)
from .interpolation import interpolate_uv_to_location
from .io import load_all_power_curves, load_wind_dataset_from_directory
from .profiles import extrapolate_to_height
from .rose import plot_wind_rose
from .weibull import fit_weibull_distribution, plot_wind_speed_distribution
from .wind import add_wind_variables_to_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WindResourceAssessment:
    """
    Main class for the wind resource assessment package.

    This class manages project paths and acts as the main entry point
    to load wind data, turbine power curves, interpolation, profiles,
    Weibull analysis, wind rose analysis and AEP calculations.
    """

    HORNS_REV_1 = {
        "name": "Horns Rev 1",
        "latitude": 55.5297222222,
        "longitude": 7.9061111111,
    }

    def __init__(
        self,
        inputs_dir: str | Path | None = None,
        outputs_dir: str | Path | None = None,
    ) -> None:
        self.inputs_dir = (
            Path(inputs_dir) if inputs_dir is not None else PROJECT_ROOT / "inputs"
        )
        self.outputs_dir = (
            Path(outputs_dir) if outputs_dir is not None else PROJECT_ROOT / "outputs"
        )

        self.dataset = None
        self.power_curves: dict[str, Any] = {}

        self._ensure_outputs_dir()

    def _ensure_outputs_dir(self) -> None:
        """Create the outputs directory if it does not exist."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def list_wind_files(self) -> list[Path]:
        """Return all NetCDF wind data files found in the inputs directory."""
        return sorted(self.inputs_dir.glob("*.nc"))

    def list_turbine_files(self) -> list[Path]:
        """Return all CSV turbine power-curve files found in the inputs directory."""
        return sorted(self.inputs_dir.glob("*.csv"))

    def load_dataset(self):
        """Load and store the ERA5 wind dataset."""
        self.dataset = load_wind_dataset_from_directory(self.inputs_dir)
        return self.dataset

    def add_wind_variables(self):
        """Add wind speed and wind direction variables to the loaded dataset."""
        if self.dataset is None:
            raise ValueError("Load the dataset first using load_dataset().")

        self.dataset = add_wind_variables_to_dataset(self.dataset)
        return self.dataset

    def interpolate_to_location(
        self,
        latitude: float,
        longitude: float,
        start_year: int = 1997,
        end_year: int = 2008,
    ):
        """
        Interpolate u/v wind components to a target location and compute
        ws/wd at 10 m and 100 m.
        """
        if self.dataset is None:
            raise ValueError("Load the dataset first using load_dataset().")

        return interpolate_uv_to_location(
            self.dataset,
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year,
        )

    def extrapolate_to_height(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
    ):
        """
        Interpolate wind data to a location and extrapolate wind speed
        to a target height using the power-law profile.
        """
        point_ds = self.interpolate_to_location(
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year,
        )

        return extrapolate_to_height(
            point_ds=point_ds,
            target_height=target_height,
            alpha=alpha,
        )

    def get_wind_speed_series(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
    ):
        """
        Return the wind speed time series at a specified location and height.
        """
        point_ds = self.interpolate_to_location(
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year,
        )

        if np.isclose(target_height, 10.0):
            return point_ds["ws10"]

        if np.isclose(target_height, 100.0):
            return point_ds["ws100"]

        profile_ds = extrapolate_to_height(
            point_ds=point_ds,
            target_height=target_height,
            alpha=alpha,
        )
        return profile_ds["ws_z"]

    def get_wind_direction_series(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
    ):
        """
        Return the wind direction time series at a specified location and height.
        """
        point_ds = self.interpolate_to_location(
            latitude=latitude,
            longitude=longitude,
            start_year=start_year,
            end_year=end_year,
        )

        if np.isclose(target_height, 10.0):
            return point_ds["wd10"]

        if np.isclose(target_height, 100.0):
            return point_ds["wd100"]

        profile_ds = extrapolate_to_height(
            point_ds=point_ds,
            target_height=target_height,
            alpha=alpha,
        )
        return profile_ds["wd_z"]

    def fit_weibull_at_height(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
    ) -> dict[str, float]:
        """
        Fit a Weibull distribution to the wind speed series at a given location and height.
        """
        wind_speed = self.get_wind_speed_series(
            latitude=latitude,
            longitude=longitude,
            target_height=target_height,
            start_year=start_year,
            end_year=end_year,
            alpha=alpha,
        )

        result = fit_weibull_distribution(wind_speed)
        result["target_height"] = float(target_height)
        result["latitude"] = float(latitude)
        result["longitude"] = float(longitude)
        result["start_year"] = float(start_year)
        result["end_year"] = float(end_year)
        return result

    def plot_weibull_distribution_at_height(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
        bins: int = 30,
        output_path: str | Path | None = None,
    ):
        """
        Plot histogram vs fitted Weibull distribution at a given location and height.
        """
        wind_speed = self.get_wind_speed_series(
            latitude=latitude,
            longitude=longitude,
            target_height=target_height,
            start_year=start_year,
            end_year=end_year,
            alpha=alpha,
        )

        params = fit_weibull_distribution(wind_speed)

        title = (
            f"Wind speed distribution at {target_height:.1f} m\n"
            f"lat={latitude:.4f}, lon={longitude:.4f}"
        )

        return plot_wind_speed_distribution(
            wind_speed=wind_speed,
            k=params["k"],
            A=params["A"],
            bins=bins,
            title=title,
            output_path=output_path,
        )

    def plot_wind_rose_at_height(
        self,
        latitude: float,
        longitude: float,
        target_height: float,
        start_year: int = 1997,
        end_year: int = 2008,
        alpha: float | None = None,
        n_sectors: int = 12,
        output_path: str | Path | None = None,
    ):
        """
        Plot wind rose frequencies at a given location and height.
        """
        wind_direction = self.get_wind_direction_series(
            latitude=latitude,
            longitude=longitude,
            target_height=target_height,
            start_year=start_year,
            end_year=end_year,
            alpha=alpha,
        )

        title = (
            f"Wind rose at {target_height:.1f} m\n"
            f"lat={latitude:.4f}, lon={longitude:.4f}"
        )

        return plot_wind_rose(
            wind_direction=wind_direction,
            n_sectors=n_sectors,
            title=title,
            output_path=output_path,
        )

    def load_power_curves(self) -> dict[str, Any]:
        """Load and store all turbine power curves."""
        self.power_curves = load_all_power_curves(self.inputs_dir)
        return self.power_curves

    def get_power_curve(self, turbine_name: str):
        """
        Return the power curve for a turbine.
        """
        if not self.power_curves:
            self.load_power_curves()

        turbine_key = turbine_name.lower()
        if turbine_key not in self.power_curves:
            raise KeyError(
                f"Turbine '{turbine_name}' not found. "
                f"Available: {list(self.power_curves.keys())}"
            )

        return self.power_curves[turbine_key]

    def get_power_output_series(
        self,
        turbine_name: str,
        latitude: float,
        longitude: float,
        year: int,
        alpha: float | None = None,
    ):
        """
        Compute the turbine power output time series [kW] for a specific year.
        """
        hub_height = get_turbine_hub_height(turbine_name)
        wind_speed = self.get_wind_speed_series(
            latitude=latitude,
            longitude=longitude,
            target_height=hub_height,
            start_year=year,
            end_year=year,
            alpha=alpha,
        )

        power_curve = self.get_power_curve(turbine_name)

        return compute_power_output_series(
            wind_speed=wind_speed,
            power_curve=power_curve,
            turbine_name=turbine_name,
        )

    def compute_aep(
        self,
        turbine_name: str,
        latitude: float,
        longitude: float,
        year: int,
        alpha: float | None = None,
    ) -> dict[str, float]:
        """
        Compute AEP for a specific turbine, location and year.
        """
        power_series = self.get_power_output_series(
            turbine_name=turbine_name,
            latitude=latitude,
            longitude=longitude,
            year=year,
            alpha=alpha,
        )

        result = compute_aep_from_power_series(power_series)
        result["turbine_name"] = turbine_name
        result["hub_height_m"] = get_turbine_hub_height(turbine_name)
        result["latitude"] = float(latitude)
        result["longitude"] = float(longitude)
        result["year"] = float(year)

        return result

    def get_default_site(self) -> dict[str, float | str]:
        """Return the default project site (Horns Rev 1)."""
        return self.HORNS_REV_1.copy()

    def summary(self) -> dict[str, Any]:
        """Return a summary of the current project setup."""
        return {
            "inputs_dir": str(self.inputs_dir),
            "outputs_dir": str(self.outputs_dir),
            "n_wind_files": len(self.list_wind_files()),
            "n_turbine_files": len(self.list_turbine_files()),
            "dataset_loaded": self.dataset is not None,
            "loaded_power_curves": list(self.power_curves.keys()),
            "default_site": self.get_default_site(),
        }

    def __repr__(self) -> str:
        return (
            "WindResourceAssessment("
            f"inputs_dir='{self.inputs_dir}', "
            f"outputs_dir='{self.outputs_dir}')"
        )