from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.stats import weibull_min


def extract_valid_wind_speed_values(wind_speed: xr.DataArray | np.ndarray | list[float]) -> np.ndarray:
    """
    Extract valid non-negative wind speed values as a 1D NumPy array.
    """
    if isinstance(wind_speed, xr.DataArray):
        values = wind_speed.values
    else:
        values = np.asarray(wind_speed)

    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    values = values[values >= 0.0]

    if values.size == 0:
        raise ValueError("No valid wind speed values were found.")

    return values


def fit_weibull_distribution(
    wind_speed: xr.DataArray | np.ndarray | list[float],
) -> dict[str, float]:
    """
    Fit a Weibull distribution to wind speed data using scipy.stats.weibull_min.

    The location parameter is fixed to zero, which is standard for wind-speed fitting.

    Returns
    -------
    dict
        Dictionary with:
        - k: Weibull shape parameter
        - A: Weibull scale parameter
        - loc: location parameter (fixed to 0)
        - mean_wind_speed
        - std_wind_speed
        - n_samples
    """
    values = extract_valid_wind_speed_values(wind_speed)

    k, loc, A = weibull_min.fit(values, floc=0.0)

    return {
        "k": float(k),
        "A": float(A),
        "loc": float(loc),
        "mean_wind_speed": float(np.mean(values)),
        "std_wind_speed": float(np.std(values, ddof=0)),
        "n_samples": float(values.size),
    }


def weibull_pdf(u: np.ndarray, k: float, A: float) -> np.ndarray:
    """
    Compute the Weibull probability density function.

    f(u) = (k / A) * (u / A)^(k - 1) * exp(-(u / A)^k)
    """
    u = np.asarray(u, dtype=float)
    pdf = np.zeros_like(u, dtype=float)

    valid = u >= 0.0
    u_valid = u[valid]

    pdf[valid] = (k / A) * (u_valid / A) ** (k - 1.0) * np.exp(-(u_valid / A) ** k)
    return pdf


def plot_wind_speed_distribution(
    wind_speed: xr.DataArray | np.ndarray | list[float],
    k: float,
    A: float,
    bins: int = 30,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> tuple[Any, Any]:
    """
    Plot histogram of wind speed data together with the fitted Weibull PDF.

    Returns
    -------
    tuple
        (fig, ax)
    """
    values = extract_valid_wind_speed_values(wind_speed)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(values, bins=bins, density=True, alpha=0.7, label="Wind speed histogram")

    x_max = max(values.max() * 1.05, A * 3.0)
    x = np.linspace(0.0, x_max, 400)
    y = weibull_pdf(x, k=k, A=A)

    ax.plot(x, y, linewidth=2, label=f"Weibull fit (k={k:.3f}, A={A:.3f})")

    ax.set_xlabel("Wind speed [m/s]")
    ax.set_ylabel("Probability density [-]")
    ax.set_title(title or "Wind speed distribution and fitted Weibull")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig, ax