from pathlib import Path

import matplotlib.pyplot as plt

from wra import WindResourceAssessment


wra = WindResourceAssessment()

print("Project summary:")
print(wra.summary())

print("\nLoading ERA5 dataset...")
ds = wra.load_dataset()
print(ds)

print("\nAdding wind speed and wind direction variables on the original grid...")
ds = wra.add_wind_variables()
print(ds[["ws10", "wd10", "ws100", "wd100"]])

print("\nLoading turbine power curves...")
curves = wra.load_power_curves()
print(curves.keys())

site = wra.get_default_site()

print("\nInterpolating wind time series at Horns Rev 1...")
point_ds = wra.interpolate_to_location(
    latitude=site["latitude"],
    longitude=site["longitude"],
    start_year=1997,
    end_year=2008,
)
print(point_ds[["u10", "v10", "u100", "v100", "ws10", "wd10", "ws100", "wd100"]])

print("\nFirst 5 time steps at Horns Rev 1:")
print(point_ds[["ws10", "wd10", "ws100", "wd100"]].isel(time=slice(0, 5)))

print("\nExtrapolating wind speed to 90 m using the power-law profile...")
profile_90m = wra.extrapolate_to_height(
    latitude=site["latitude"],
    longitude=site["longitude"],
    target_height=90.0,
    start_year=1997,
    end_year=2008,
)
print(profile_90m[["alpha", "ws_z", "wd_z"]])

print("\nFirst 5 time steps at 90 m:")
print(profile_90m[["alpha", "ws_z", "wd_z"]].isel(time=slice(0, 5)))

print("\nFitting Weibull distribution at 90 m...")
weibull_result = wra.fit_weibull_at_height(
    latitude=site["latitude"],
    longitude=site["longitude"],
    target_height=90.0,
    start_year=1997,
    end_year=2008,
)
print(weibull_result)

print("\nPlotting histogram vs fitted Weibull distribution at 90 m...")
weibull_plot_path = wra.outputs_dir / "weibull_distribution_90m.png"

fig, ax = wra.plot_weibull_distribution_at_height(
    latitude=site["latitude"],
    longitude=site["longitude"],
    target_height=90.0,
    start_year=1997,
    end_year=2008,
    output_path=weibull_plot_path,
)
print(f"Saved Weibull plot to: {weibull_plot_path}")
plt.close(fig)

print("\nPlotting wind rose at 90 m...")
wind_rose_path = wra.outputs_dir / "wind_rose_90m.png"

fig, ax, rose_data = wra.plot_wind_rose_at_height(
    latitude=site["latitude"],
    longitude=site["longitude"],
    target_height=90.0,
    start_year=1997,
    end_year=2008,
    n_sectors=12,
    output_path=wind_rose_path,
)
print(f"Saved wind rose plot to: {wind_rose_path}")
print("Directional frequencies (%):")
print(rose_data["frequencies_percent"])
plt.close(fig)

print("\nComputing AEP for NREL 5 MW at Horns Rev 1 in 2005...")
aep_5mw_2005 = wra.compute_aep(
    turbine_name="nrel_5mw",
    latitude=site["latitude"],
    longitude=site["longitude"],
    year=2005,
)
print(aep_5mw_2005)

print("\nComputing AEP for NREL 15 MW at Horns Rev 1 in 2005...")
aep_15mw_2005 = wra.compute_aep(
    turbine_name="nrel_15mw",
    latitude=site["latitude"],
    longitude=site["longitude"],
    year=2005,
)
print(aep_15mw_2005)