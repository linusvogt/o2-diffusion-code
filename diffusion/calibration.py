"""Spatially-smoothed variance-inflation factor for a DDPM ensemble.

Extracted from ml_framework/diffusion_model/calibration_pipeline.py (only
``fit_alpha_map`` and its dependencies; used by the extrapolation figure's
calibrated OMZ ensemble).

Assumes an xarray Dataset `ds` with:
    o2_true  (model, lat, lon)          — ground-truth oxygen
    o2_gen   (model, sample, lat, lon)  — DDPM-generated oxygen

Land grid points are NaN in o2_true.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

# Spatial smoothing sigma (in grid points) for the alpha map.
# ~5 degrees at 1-degree resolution.  Increase for heavier smoothing.
SMOOTH_SIGMA = 5.0


def ensemble_stats(ds: xr.Dataset) -> xr.Dataset:
    """Ensemble mean, std, and error (true minus mean)."""
    ens_mean = ds["o2_gen"].mean(dim="sample")
    ens_std = ds["o2_gen"].std(dim="sample")
    error = ds["o2_true"] - ens_mean
    return xr.Dataset(
        {"ens_mean": ens_mean, "ens_std": ens_std, "error": error}
    )


def _nan_aware_smooth(field_2d: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian-smooth a 2D array that contains NaNs.
    Uses the normalised-convolution trick: smooth both a zero-filled
    copy and a weight mask, then take their ratio.
    """
    filled = np.where(np.isnan(field_2d), 0.0, field_2d)
    weights = np.where(np.isnan(field_2d), 0.0, 1.0)
    smooth_num = gaussian_filter(filled, sigma=sigma)
    smooth_den = gaussian_filter(weights, sigma=sigma)
    smooth_den = np.where(smooth_den > 1e-8, smooth_den, np.nan)
    return smooth_num / smooth_den


def fit_alpha_map(
    ds: xr.Dataset,
    mask: xr.DataArray,
    sigma: float = SMOOTH_SIGMA,
    model_indices=None,
) -> xr.DataArray:
    """
    Fit the spatially-smoothed variance-inflation factor alpha(lat, lon).

    alpha_raw(i,j) = RMSE_across_models(error) / mean_across_models(ens_std)

    Then Gaussian-smooth and clip to [0.5, 10].

    Parameters
    ----------
    ds : full dataset
    mask : boolean ocean mask (lat, lon)
    sigma : Gaussian smoothing width in grid points
    model_indices : subset of models to fit on (default: all)
    """
    ds_fit = ds.isel(model=model_indices) if model_indices is not None else ds
    stats = ensemble_stats(ds_fit)

    # Actual error scale across models at each grid point
    error_rmse = np.sqrt((stats["error"] ** 2).mean(dim="model"))
    # Predicted spread (mean of per-model ensemble std)
    mean_ens_std = stats["ens_std"].mean(dim="model")

    alpha_raw = (error_rmse / mean_ens_std).values
    alpha_raw = np.where(np.isfinite(alpha_raw), alpha_raw, np.nan)

    # Spatial smoothing
    alpha_smooth = _nan_aware_smooth(alpha_raw, sigma)
    alpha_smooth = np.clip(alpha_smooth, 0.5, 10.0)
    alpha_smooth = np.where(mask.values, alpha_smooth, np.nan)

    return xr.DataArray(
        alpha_smooth,
        dims=("lat", "lon"),
        coords={"lat": ds.lat, "lon": ds.lon},
        name="alpha",
    )
