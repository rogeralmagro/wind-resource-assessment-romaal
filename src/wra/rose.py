from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def extract_valid_direction_values(
    wind_direction: xr.DataArray | np.ndarray | list[float],
) -> np.ndarray:
    """
    Extract valid wind direction values as a 1D NumPy array in [0, 360).
    """
    if isinstance(wind_direction, xr.DataArray):
        values = wind_direction.values
    else:
        values = np.asarray(wind_direction)

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    values = np.mod(values, 360.0)

    if values.size == 0:
        raise ValueError("No valid wind direction values were found.")

    return values


def compute_directional_frequencies(
    wind_direction: xr.DataArray | np.ndarray | list[float],
    n_sectors: int = 12,
) -> dict[str, np.ndarray | float]:
    """
    Compute directional frequencies for a wind rose.

    Parameters
    ----------
    wind_direction : array-like
        Wind direction values in degrees.
    n_sectors : int, optional
        Number of directional sectors.

    Returns
    -------
    dict
        Dictionary containing:
        - sector_edges_deg
        - sector_centers_deg
        - frequencies_percent
        - counts
        - sector_width_deg
    """
    if n_sectors <= 0:
        raise ValueError("n_sectors must be a positive integer.")

    direction = extract_valid_direction_values(wind_direction)

    sector_edges = np.linspace(0.0, 360.0, n_sectors + 1)
    counts, _ = np.histogram(direction, bins=sector_edges)

    frequencies = counts / counts.sum() * 100.0
    sector_centers = 0.5 * (sector_edges[:-1] + sector_edges[1:])
    sector_width = 360.0 / n_sectors

    return {
        "sector_edges_deg": sector_edges,
        "sector_centers_deg": sector_centers,
        "frequencies_percent": frequencies,
        "counts": counts,
        "sector_width_deg": sector_width,
    }


def plot_wind_rose(
    wind_direction: xr.DataArray | np.ndarray | list[float],
    n_sectors: int = 12,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any, dict[str, np.ndarray | float]]:
    """
    Plot a directional wind rose.

    Returns
    -------
    tuple
        (fig, ax, rose_data)
    """
    rose_data = compute_directional_frequencies(
        wind_direction=wind_direction,
        n_sectors=n_sectors,
    )

    theta = np.deg2rad(rose_data["sector_centers_deg"])
    radius = rose_data["frequencies_percent"]
    width = np.deg2rad(rose_data["sector_width_deg"] * 0.9)

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    ax.bar(theta, radius, width=width, bottom=0.0, align="center", alpha=0.8)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_title(title or "Wind rose", va="bottom")
    ax.set_ylabel("Frequency [%]")
    ax.grid(True, alpha=0.3)

    cardinal_angles = np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315])
    cardinal_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ax.set_xticks(cardinal_angles)
    ax.set_xticklabels(cardinal_labels)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax, rose_data