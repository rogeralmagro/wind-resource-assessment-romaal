# Wind Resource Assessment

Python package for wind resource assessment using ERA5 reanalysis data.

## Project objective
This package is developed for the final project of the course 46120 Scientific Programming for Wind Energy at DTU Wind. The goal is to process ERA5 wind data and estimate key wind resource metrics, including wind speed distributions, wind roses, and annual energy production (AEP) for selected turbines.

## Main functionalities
- Load and parse multiple NetCDF4 ERA5 files
- Compute wind speed and wind direction from u and v components
- Interpolate wind time series at a user-defined location inside the bounding box
- Extrapolate wind speed to other heights using the power law profile
- Fit Weibull distributions
- Plot histogram and fitted Weibull distribution
- Plot wind rose
- Compute AEP using turbine power curves

## Repository structure
- `inputs/`: provided input data
- `outputs/`: generated results
- `src/wra/`: package source code
- `tests/`: unit tests
- `examples/main.py`: example script for evaluation

## Installation
Create and activate a virtual environment, then run:

```bash
pip install -e .[dev]
cat > examples/main.py <<'EOF'
"""
Example script for the Wind Resource Assessment package.

This script will later demonstrate:
1. Loading multiple NetCDF files
2. Computing wind speed and direction
3. Interpolating to a target location
4. Extrapolating wind speed to another height
5. Fitting a Weibull distribution
6. Plotting wind distribution and wind rose
7. Computing AEP for a selected turbine
"""

def main():
    """Run the example workflow."""
    print("Wind Resource Assessment example script")
    print("Implementation in progress.")


if __name__ == "__main__":
    main()
