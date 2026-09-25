"""Shared evaluation library: locate a run, load model + data, sample, score, plot.

Extracted from general/eval/eval_core.py (``resolve_predictors`` inlined from
general/launch_sweep.py).

  * run location  -- ``run_save_dir`` builds a checkpoint path from a base config
    + axis overrides via ``diffusion.naming.save_dir_from_config`` (the same
    naming the training code writes);
  * model loading -- ``load_run`` rebuilds the UNet/DDPM at the *expanded*
    predictor-channel count (density runs need this) and asserts the derived
    channels match the checkpoint's saved normalization keys;
  * data sources  -- CMIP per-model datasets (possibly for a *different*
    experiment than the run trained on -- that is what the out-of-scenario axis
    needs), built through ``diffusion.data_loading`` so channels/preprocessing
    match training exactly;
  * metric        -- ``skill_map`` = the canonical years-window, model-mean
    relative-error map (average fields over years then models, *then* divide);
  * plotting      -- ``rel_error_panel`` / ``error_panel`` / ... on ultraplot
    ``robin`` axes via ``contourf_plot``.

Import-safe on CPU; GPU is only needed once you actually sample.
"""
from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tqdm.autonotebook import tqdm

import paths
from diffusion import cmip_io as util_ml
from diffusion import inference as util
from diffusion import data_loading as dl
from diffusion.naming import save_dir_from_config


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CKPT_ROOT = paths.CHECKPOINTS
FIG_DIR = paths.EVAL_DIR

DEPTH_INT = 2000.0            # depth-integration thickness -> mol/m^3
O2 = r'$\mathrm{O}_2$'
CBAR_LABEL = 'mol/' + r'$\mathrm{m}^3$'

# Bathymetry mask: fields are depth-*integrated* over 0-2000 m, so cells whose
# seafloor is shallower than 2000 m have a truncated (small) integral. That
# inflates the relative-error map (small denominator) and the absolute-error map
# (shelf-break ring), and lets a shallow-cell outlier set the color scale. We
# drop those cells from every map plot AND from the scalar skill metric so both
# describe only the well-posed deep ocean.
BATHY_MASK_FILE = paths.BATHY_MASK
BATHY_MIN_DEPTH = 2000.0      # keep cells at least this deep; None disables
_BATHY_CACHE = {}


def bathy_mask(min_depth=BATHY_MIN_DEPTH):
    """(lat, lon) 1/NaN mask keeping only cells at least ``min_depth`` m deep.

    Loaded once per threshold from ``BATHY_MASK_FILE`` and cached. ``method=
    'nearest'`` so an int/float depth label can't KeyError (2000 -> the 2000 m
    coordinate). Returns None when ``min_depth`` is None (masking disabled).
    """
    if min_depth is None:
        return None
    if min_depth not in _BATHY_CACHE:
        ds = xr.open_dataset(BATHY_MASK_FILE)
        _BATHY_CACHE[min_depth] = (
            ds['mask'].sel(depth=min_depth, method='nearest')
            .reset_coords(drop=True).load())
    return _BATHY_CACHE[min_depth]


def mask_shallow(da, min_depth=BATHY_MIN_DEPTH):
    """Set cells shallower than ``min_depth`` m to NaN via the bathymetry mask.

    A no-op when ``da`` is None or ``min_depth`` is None. Idempotent (already-NaN
    cells stay NaN), so it is safe to apply at every plot boundary. The mask
    shares the r360x180 grid (lat -89.5..89.5, lon 0..359), so the multiply
    aligns by coordinate with the eval fields.
    """
    if da is None:
        return da
    m = bathy_mask(min_depth)
    if m is None:
        return da
    return da * m

MONTH_NAMES = {0: 'Jan', 1: 'Feb', 2: 'Mar', 3: 'Apr', 4: 'May', 5: 'Jun',
               6: 'Jul', 7: 'Aug', 8: 'Sep', 9: 'Oct', 10: 'Nov', 11: 'Dec'}

# In-sample held-out test window. split_train_test_insample uses (0-based year
# index) years_train = 0..80 & 101..150, years_test = 81..100, so the genuine
# held-out window is 81..100 -- year 80 is a *training* year and is excluded.
# This is the canonical CMIP evaluation window used across all axes.
DEFAULT_YEAR_WINDOW = (81, 100)

# Minimal config carrying exactly the keys save_dir_from_config reads. Analyses
# override individual axes; the *authoritative* per-run config (architecture,
# etc.) is always re-read from the checkpoint's config.json by load_run.
DEFAULT_CONFIG = {
    'target': 'o2',
    'predictors': ['thetao', 'so'],
    'experiments': ['1pctCO2'],
    'resolution': 'annual',        # 'annual' | 'monthly'
    'field_type': 'depthint',      # 'depthint' | 'density'
    'density_var': 'sigma_1',
    'density_levels': 'all',
    'train_test_split': 'insample',  # 'insample' | 'model'
    'models_test': None,             # [name] when split == 'model'
    'anom': False,
    'penalize_non_negative': False,
    'integral_loss': False,
    'w_integral': 0.0,
    # Ensemble-composition cap. None selects the published (uncapped) runs;
    # an int selects the `_mem<N>` runs -- overrides flow straight into
    # save_dir_from_config.
    'max_members_per_model': None,
}


# resolution -> concrete MLD variable name for the 'mld' token
RESOLVE_MLD = {'annual': 'mld_max', 'monthly': 'mld'}


def resolve_predictors(predictors, resolution):
    """Resolve the 'mld' token to the resolution-appropriate variable name."""
    return [RESOLVE_MLD[resolution] if p == 'mld' else p for p in predictors]


# ---------------------------------------------------------------------------
# run location (decoupled from data)
# ---------------------------------------------------------------------------
def make_config(base=None, **overrides):
    """A naming-config dict = DEFAULT_CONFIG (or ``base``) with overrides applied."""
    cfg = copy.deepcopy(base if base is not None else DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def run_save_dir(base=None, **overrides):
    """Checkpoint dir for a config, resolving the ``mld`` token per resolution.

    Returns a ``Path`` under ``CKPT_ROOT``. The path is byte-identical to what
    the training launcher writes, so ``.is_dir()`` / glob of ``ckpt_epoch*.pt``
    is a reliable "does this run exist yet" check.
    """
    cfg = make_config(base, **overrides)
    cfg['predictors'] = resolve_predictors(cfg['predictors'], cfg['resolution'])
    return Path(save_dir_from_config(cfg))


def run_exists(save_dir, epoch=None):
    """True if the run has a usable checkpoint.

    With ``epoch=None`` (default) any ``ckpt_epoch*.pt`` counts. With an explicit
    ``epoch`` it checks for *that* checkpoint specifically -- so a caller that
    will ``load_run(..., epoch=E)`` can skip a run that exists but has not yet
    reached epoch E, instead of erroring when ``load_model`` fails to open the
    missing file. Epoch is zero-padded to width 3 to match ``load_model``.
    """
    save_dir = Path(save_dir)
    if epoch is None:
        return bool(sorted(save_dir.glob('ckpt_epoch*.pt')))
    return (save_dir / f'ckpt_epoch{str(epoch).zfill(3)}.pt').is_file()


# ---------------------------------------------------------------------------
# model loading + channel guard
# ---------------------------------------------------------------------------
@dataclass
class RunHandle:
    """A loaded run: model + normalization + the config it was trained with."""
    ddpm: object
    ema: object
    scaling: dict
    predictor_channels: list
    config: dict
    save_dir: Path
    device: torch.device


def _first_test_model(config):
    models = get_models(config, require_target=True)
    assert models, f'no test models with predictors+target for {config}'
    return models[0]


def load_run(save_dir, device, epoch=None, _compile=True):
    """Load the run at ``save_dir`` into a :class:`RunHandle`.

    ``_compile=False`` loads the (compiled-saved) weights into an eager model by
    stripping the ``_orig_mod.`` prefix -- much faster when loading many runs in
    one process (skips a per-run torch.compile), at a small per-sample speed cost.

    ``load_model`` sizes the UNet from ``len(config['predictors'])+1``, but the
    density runs expand each tracer into one channel per sigma layer, so we swap
    in the *expanded* channel names (re-derived from the data pipeline -- they
    are not stored in config.json) before building the model, then assert those
    channels match the checkpoint's saved normalization keys (order matters).
    """
    save_dir = Path(save_dir)
    config = util.load_config(str(save_dir))

    # expanded predictor channel names, from the exact training data pipeline
    ds0 = build_per_model_ds(config, _first_test_model(config))
    predictor_channels = [v for v in ds0.data_vars if v != config['target']]

    # Checkpoints are saved from a torch.compile'd model (state_dict keys prefixed
    # "_orig_mod."); load_model compiles too by default, or strips the prefix when
    # _compile=False (fast multi-run loading).
    model_config = {**config, 'predictors': predictor_channels}
    ddpm, scaling, ema = util.load_model(
        config=model_config, device=device, epoch=epoch, _compile=_compile,
        log=True)

    expected = predictor_channels + [config['target']]
    got = list(scaling['means'].keys())
    assert got == expected, (
        f'channel mismatch for {save_dir.name}:\n  ckpt={got}\n  derived={expected}')
    logging.info(f'  loaded {save_dir.name}: {len(predictor_channels)} '
                 f'predictor channels, scaling order verified')
    return RunHandle(ddpm=ddpm, ema=ema, scaling=scaling,
                     predictor_channels=predictor_channels, config=config,
                     save_dir=save_dir, device=device)


# ---------------------------------------------------------------------------
# CMIP data source (independent of the loaded run; may use a different exp)
# ---------------------------------------------------------------------------
def get_models(config, data_experiment=None, require_target=True, max_models=None):
    """Test models for a run/data config.

    ``data_experiment`` overrides ``config['experiments'][0]`` -- used by the
    out-of-scenario axis (load a 1pctCO2 run, evaluate on abrupt-4xCO2 data).
    ``require_target=False`` (extrapolation) keeps models that have the
    predictors even if the ground-truth target is missing.
    """
    exp = data_experiment or config['experiments'][0]
    m_pred = dl._get_predictor_models(
        exp, config['predictors'], config['field_type'],
        config['resolution'], config.get('density_var', 'sigma_1'), dl.DEPTH)
    if require_target:
        m_tgt = dl._get_target_models(
            exp, config['target'], config['resolution'], dl.DEPTH)
        models = m_pred & m_tgt
    else:
        models = m_pred
    models = sorted(models, key=lambda m: m.model)
    models = util_ml.filter_first_model(models)   # one member per model
    if max_models is not None:
        models = models[:max_models]
    return models


def build_per_model_ds(config, model, data_experiment=None):
    """Per-model dataset (target + expanded predictor channels), lazy.

    Uses the same construction as training so channel names / preprocessing
    match the checkpoint. Predictors here are the *base* names from the run
    config; density expansion happens inside ``dl._build_per_model_ds``.
    """
    exp = data_experiment or config['experiments'][0]
    return dl._build_per_model_ds(
        exp=exp, model=model,
        base_predictors=config['predictors'], target=config['target'],
        field_type=config['field_type'], resolution=config['resolution'],
        density_var=config.get('density_var', 'sigma_1'),
        levels=config.get('density_levels', 'all'),
        depth=dl.DEPTH, anom=config['anom'])


def build_predictors_only_ds(config, model, data_experiment=None):
    """Per-model dataset with the expanded predictor channels but NO target.

    For the extrapolation axis: models that have T/S but no ground-truth O2.
    ``dl._build_per_model_ds`` always loads the target (and would raise for
    these models), so we replicate its assembly here, minus the target.
    """
    exp = data_experiment or config['experiments'][0]
    resolution = config['resolution']
    sample_dim = 'year' if resolution == 'annual' else 'time'
    channels = dl._predictor_channels(
        exp, model, config['predictors'], config['field_type'], resolution,
        config.get('density_var', 'sigma_1'),
        config.get('density_levels', 'all'), dl.DEPTH)
    names = list(channels.keys())
    arrs = list(xr.align(*channels.values(), join='inner'))
    year_vals = arrs[0]['year'].values if 'year' in arrs[0].coords else None
    arrs = [a.reset_coords(drop=True) for a in arrs]
    ds = xr.Dataset({n: a for n, a in zip(names, arrs)})
    if resolution != 'annual' and year_vals is not None:
        ds = ds.assign_coords(year=(sample_dim, year_vals))
    if config['anom']:
        ds = dl._subtract_anom(ds, sample_dim, resolution)
    ds = dl._restrict_years(ds, exp, sample_dim, resolution)
    return ds.assign_coords(model=model, exp=exp)


#: Timesteps materialized per read by :func:`materialize_window`. The source
#: files are chunked in blocks of 116 along time, so a per-slice read
#: decompresses a whole chunk (x21 variables) to keep one step -- and the next
#: step re-reads the same chunk. Anything of this order amortizes that; 120 caps
#: the peak at ~0.7 GB for the widest config (monthly density, 5.7 MB/step).
WINDOW_BLOCK = 120


def materialize_window(ds_model, resolution, years, months=None,
                       block=WINDOW_BLOCK):
    """Eagerly read the timesteps the sampling loop will visit, in few reads.

    ``select_case`` reads ONE timestep per call, which on these files costs a
    full 116-step chunk decompression across every variable, leaving the GPU
    idle most of the time. Reading in blocks amortizes the chunk, and slicing
    out of the result is free.

    Returns a dataset with the same coords/vars, numpy-backed, holding only the
    visited steps -- so ``select_case`` still selects from it by (year[, month])
    and its trailing ``.load()`` becomes a no-op. **Numerically identical**: same
    values, same order, no effect on the sampler or its noise stream.
    """
    idx = _window_indices(ds_model, resolution, years, months)
    if not idx:
        return ds_model
    dim = 'year' if resolution == 'annual' else 'time'
    parts = [ds_model.isel({dim: idx[i:i + block]}).load()
             for i in range(0, len(idx), block)]
    return parts[0] if len(parts) == 1 else xr.concat(parts, dim)


def _window_indices(ds_model, resolution, years, months=None):
    """Positional indices along the sample dim for the (year[, month]) cases
    the loop will visit, ascending and deduplicated."""
    if resolution == 'annual':
        avail = ds_model['year'].values
        return sorted({int(i) for y in years
                       for i in np.where(avail == y)[0]})
    months = range(12) if months is None else months
    yr, tv = ds_model['year'].values, ds_model['time'].values
    return sorted({int(i) for y in years for m in months
                   for i in np.where((yr == y) & (tv % 12 == m))[0]})


def select_case(ds_model, resolution, year, month=None):
    """Select and load the (year[, month]) 2D slice from a per-model dataset.

    Returns an (lat, lon) Dataset, or None if that time is not present.

    Cheap when ``ds_model`` came from :func:`materialize_window` (already in
    memory, so ``.load()`` is a no-op); on a lazy dataset each call pays a full
    chunk read -- see that function.
    """
    if resolution == 'annual':
        if year not in ds_model['year'].values:
            return None
        sel = ds_model.sel(year=year)
    else:
        time_vals = ds_model['time'].values
        yr = ds_model['year'].values
        match = np.where((yr == year) & (time_vals % 12 == month))[0]
        if len(match) == 0:
            return None
        sel = ds_model.isel(time=int(match[0]))
    return sel.load()


# ---------------------------------------------------------------------------
# sampling  (mask on the o2 target like TracerDataset, not predictor[0])
# ---------------------------------------------------------------------------
@dataclass
class SampleCfg:
    n_samples: int = 10
    sampler: str = 'ddim'
    eta: float = 0.1
    steps: int = 100
    apply_ema: bool = True


@torch.no_grad()
def predict_ensemble(ds_sel, run: RunHandle, sample_cfg: SampleCfg,
                     mask_var=None, ocean=None):
    """Generate an O2 ensemble for one (lat, lon) conditioning slice.

    Returns a DataArray (sample, lat, lon) in physical units; land is NaN.
    ``mask_var`` defaults to the o2 target (matches TracerDataset at train time);
    for observational predictors where the target is not co-located it can be a
    predictor channel instead.

    ``ocean`` overrides both with an explicit (H, W) boolean array -- for
    comparisons that must put every model on the *same* domain (see
    ``extrapolation.py``, where a model-mean over two different model sets is
    only meaningful on one common mask). Note this is also closer to training
    than the per-slice default: ``TracerDataset`` derives ONE ocean mask from the
    first sample's target and reuses it for every sample.
    """
    means, stds = run.scaling['means'], run.scaling['stds']
    if ocean is None:
        mask_var = mask_var or run.config['target']
        ocean = np.isfinite(ds_sel[mask_var].values)          # (H, W) bool
    else:
        ocean = np.asarray(ocean, dtype=bool)

    # land-fill convention must match how this checkpoint was trained (see
    # util.normalize_channel / land_fill_mode) -- otherwise land cells are
    # off-distribution and the generator collapses to flat fields.
    land_fill = run.scaling.get('land_fill', 'zero')
    cond_arrs = []
    for ch in run.predictor_channels:
        cond_arrs.append(util.normalize_channel(
            ds_sel[ch], means[ch], stds[ch], land_fill))
    cond = torch.from_numpy(np.stack(cond_arrs, 0))[None].to(run.device)
    mask = torch.from_numpy(ocean.astype(np.float32))[None, None].to(run.device)

    target = run.config['target']
    # A 'meanstd'-era checkpoint trained the TARGET with land = -mean/std, but the
    # sampler clamps land to 0 each step -> a train/sample mismatch that biases the
    # ocean. Re-inject the trained land value at each reverse step for those runs;
    # 'zero'-era checkpoints trained with land=0 and keep the clamp-to-0 (None).
    target_land = (-means[target] / stds[target]) if land_fill == 'meanstd' else None
    samples = []
    run.ddpm.model.eval()
    for _ in range(sample_cfg.n_samples):
        if sample_cfg.apply_ema:
            run.ema.apply_shadow(run.ddpm.model)
            x = run.ddpm.sample(cond, mask, sampler=sample_cfg.sampler,
                                eta=sample_cfg.eta, steps=sample_cfg.steps,
                                target_land=target_land)
            run.ema.restore(run.ddpm.model)
        else:
            x = run.ddpm.sample(cond, mask, sampler=sample_cfg.sampler,
                                eta=sample_cfg.eta, steps=sample_cfg.steps,
                                target_land=target_land)
        arr = np.squeeze(x.detach().cpu().numpy()) * stds[target] + means[target]
        arr = np.where(ocean, arr, np.nan)
        samples.append(arr)

    return xr.DataArray(
        np.stack(samples, 0), dims=('sample', 'lat', 'lon'),
        coords={'lat': ds_sel.lat, 'lon': ds_sel.lon})


# ---------------------------------------------------------------------------
# per-model time-mean field, and the model-mean skill map
# ---------------------------------------------------------------------------
def _years_in_window(year_window, stride=1, years=None):
    """Year list for a (lo, hi) window at ``stride``, or the explicit ``years``
    if given (an int, or an iterable of ints) which overrides window+stride --
    e.g. ``years=90`` restricts to the single held-out year 90."""
    if years is not None:
        return [int(years)] if np.isscalar(years) else [int(y) for y in years]
    lo, hi = year_window
    return list(range(lo, hi + 1, stride))


def model_timemean(run, ds_model, year_window, sample_cfg, months=None,
                   years_stride=1, want_truth=True, mask_var=None, years=None):
    """Time-mean generated (and truth) field for one model over a year window.

    Generates an ensemble for every (year[, month]) in the window, takes the
    ensemble mean per time, then averages the per-time fields. Returns
    ``dict(gen, truth_or_None, n_times)`` in raw (depth-integrated) units, or
    None if no times were present. ``mask_var`` is threaded to
    ``predict_ensemble`` -- extrapolation (no target) must mask on a predictor
    channel instead of the missing o2 target. ``years`` (int or list) restricts
    to specific held-out year(s), overriding ``year_window``/``years_stride``.
    """
    resolution = run.config['resolution']
    months = months if resolution == 'monthly' else [None]
    years_visited = _years_in_window(year_window, years_stride, years)
    # ONE blocked read up front instead of a chunk decompression per timestep
    # (see materialize_window). Same values, same order.
    ds_model = materialize_window(
        ds_model, resolution, years_visited,
        months=None if resolution == 'annual' else months)
    gen_times, truth_times = [], []
    for year in years_visited:
        for month in months:
            ds_sel = select_case(ds_model, resolution, year, month)
            if ds_sel is None:
                continue
            gen_times.append(
                predict_ensemble(ds_sel, run, sample_cfg,
                                 mask_var=mask_var).mean('sample'))
            if want_truth:
                truth_times.append(ds_sel[run.config['target']])
    if not gen_times:
        return None
    gen = xr.concat(gen_times, 'time').mean('time')
    truth = (xr.concat(truth_times, 'time').mean('time')
             if want_truth and truth_times else None)
    return dict(gen=gen, truth=truth, n_times=len(gen_times))


def skill_map(run, models, year_window=DEFAULT_YEAR_WINDOW, sample_cfg=None,
              months=None, years_stride=1, data_experiment=None,
              want_truth=True, years=None, per_model=False):
    """Model-mean truth / generated / error / relative-error maps.

    Canonical metric: average the fields over the year window and over models
    *first*, then form the relative error (stable vs per-year ratios where
    truth -> 0). All fields in mol/m^3 (depth-integrated / DEPTH_INT).

    Returns ``dict(truth, gen, err, rel, n_models, model_names)`` -- where
    ``model_names`` are the models that actually contributed (models with no
    times in the window are dropped, so it is not always ``models``). With
    ``want_truth=False``
    (extrapolation) truth/err/rel are None and only ``gen`` is filled.
    ``years`` (int or list) restricts to specific held-out year(s), overriding
    ``year_window``/``years_stride`` -- e.g. ``years=90`` for year 90 only.
    ``per_model=True`` additionally returns the raw per-model ``gens``/``truths``
    lists, for metrics that must aggregate per model (see :func:`model_mean_abs`).
    """
    sample_cfg = sample_cfg or SampleCfg()
    months = months if months is not None else list(range(12))
    gens, truths, used = [], [], []
    for model in models:
        ds_model = build_per_model_ds(run.config, model, data_experiment)
        res = model_timemean(run, ds_model, year_window, sample_cfg,
                             months=months, years_stride=years_stride,
                             want_truth=want_truth, years=years)
        if res is None:
            logging.info(f'    [warn] no times for {model.model}')
            continue
        used.append(model.model)
        gens.append(res['gen'])
        if want_truth and res['truth'] is not None:
            truths.append(res['truth'])

    if not gens:
        return None

    gen = xr.concat(gens, 'model').mean('model') / DEPTH_INT
    if want_truth and truths:
        truth = xr.concat(truths, 'model').mean('model') / DEPTH_INT
        err = gen - truth
        rel = 100.0 * err / truth
    else:
        truth = err = rel = None
    out = dict(truth=truth, gen=gen, err=err, rel=rel, n_models=len(gens),
               model_names=used)
    if per_model:
        # raw (depth-integrated) per-model fields, for aggregations that must not
        # average the fields first -- see model_mean_abs.
        out['gens'], out['truths'] = gens, truths
    return out


def model_mean_abs(gens, truths):
    """Model-mean of the ABSOLUTE per-model error -- no sign cancellation.

    The canonical :func:`skill_map` metric averages the fields over models and
    *then* forms the error, so per-model biases of opposite sign cancel: the T+S
    annual depthint LOMO fleet pools to 1.9% while its per-model MAPEs run
    3.5-80.5% (mixed signs). This computes each model's error FIRST and averages
    the magnitudes:

        err = <|gen_m - truth_m|>_m            mol/m^3, >= 0
        rel = <|100*(gen_m - truth_m)/truth_m|>_m   %,  >= 0  (mean per-model MAPE)

    so both maps are non-negative and nothing cancels. ``truth``/``gen`` in the
    result stay the model-mean fields (for reference); note ``err`` is therefore
    NOT ``gen - truth`` under this metric.

    ``gens``/``truths`` are per-model raw (depth-integrated) fields, as returned
    by ``skill_map(..., per_model=True)``. Caveat: the per-model ratio uses each
    model's OWN truth as denominator, so cells where a single model's truth
    approaches 0 (OMZ cores) are noisier than under the pooled metric -- that
    stability is what the field-averaging bought.
    """
    if not gens or not truths:
        return None
    g = xr.concat(gens, 'model') / DEPTH_INT
    t = xr.concat(truths, 'model') / DEPTH_INT
    err = np.abs(g - t).mean('model')
    rel = np.abs(100.0 * (g - t) / t).mean('model')
    return dict(truth=t.mean('model'), gen=g.mean('model'), err=err, rel=rel,
                n_models=len(gens))


def mape(rel, min_depth=BATHY_MIN_DEPTH):
    """Mean absolute percentage error (%) = area-weighted mean of |rel|.

    ``rel`` is the signed relative-error map (already in %, i.e.
    ``100*(gen-truth)/truth``), so this is the area-weighted MAPE of the
    model-mean fields. Like :func:`skill_scalar` it masks shallow (<``min_depth``
    m) cells first, because a truncated 0-2000 m integral gives a small
    denominator that would dominate the mean.
    """
    if rel is None:
        return np.nan
    rel = mask_shallow(rel, min_depth)
    return float(util.global_mean(np.abs(rel)).values)


def skill_scalar(rel, min_depth=BATHY_MIN_DEPTH):
    """Single-number skill = area-weighted RMS of the relative-error map (%).

    Shallow (<``min_depth`` m) cells are masked out first so the metric matches
    the deep-ocean maps and is not dominated by truncated-integral shelves;
    ``util.global_mean`` (an area-weighted mean) skips the NaN'd cells.
    """
    if rel is None:
        return np.nan
    rel = mask_shallow(rel, min_depth)
    return float(np.sqrt(util.global_mean(rel ** 2).values))


# ---------------------------------------------------------------------------
# raw ensemble generation  (the building block for custom evaluations)
# ---------------------------------------------------------------------------
def _nan_slab(lat, lon, n_samples=None):
    """An all-NaN (sample?, lat, lon) placeholder for a missing time step."""
    if n_samples is None:
        return xr.DataArray(np.full((lat.size, lon.size), np.nan),
                            dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    return xr.DataArray(np.full((n_samples, lat.size, lon.size), np.nan),
                        dims=('sample', 'lat', 'lon'),
                        coords={'lat': lat, 'lon': lon})


def generate_samples(base=None, *, device, sample_cfg=None, models=None,
                     year_window=DEFAULT_YEAR_WINDOW, years_stride=1, years=None,
                     months=None, epoch=None, want_truth=True, max_models=None,
                     progress=True, **config_overrides):
    """Load data + checkpoint(s) and generate an O2 ensemble as a labelled Dataset.

    This is the one-config building block for hand-rolled evaluations: it does
    three steps -- (1) load the input fields for a config's
    variables/experiment/resolution/field_type, (2) load the matching diffusion
    checkpoint(s), (3) sample -- and returns

        Dataset
          gen    (sample, model, time, lat, lon)   generated O2 ensemble
          truth  (model, time, lat, lon)           ground-truth O2 (if want_truth)
        dim coords : sample 0..n-1, model (names), time 0..T-1, lat, lon
        time coords: year (and month, monthly only) along `time`
        scalar coords (concat labels): target, predictors ("thetao+so"),
          experiment, resolution, field_type, split, sampler, epoch

    **The output is dimensionally invariant.** However many predictors go in and
    whatever the field_type, the generated field is always single-channel O2
    `(sample, lat, lon)`; `gen`/`truth` are the *only* data variables. That is
    exactly what lets you stack runs that differ in predictors / resolution /
    field_type: you concatenate the invariant *output*, with those config axes as
    coordinates -- never as input-derived dimensions. Use
    `xr.concat(..., dim='predictors')` to add such an axis.

    Config is `make_config(base, **config_overrides)` -- e.g.
    `generate_samples(predictors=['thetao','so','dissic'], field_type='density',
    device=dev)`. The run is located with `run_save_dir` (so the `mld` token and
    checkpoint naming match training); the authoritative per-run config
    (architecture, resolved channel names, normalization) is re-read from the
    checkpoint, and data is built from *that*, so channels/preprocessing match.

    Checkpoint selection:
      * `train_test_split='insample'` (or a scenario run): one checkpoint, every
        test model evaluated on it. `models` may name a subset; else all models
        with data are used (capped by `max_models`).
      * `train_test_split='model'`: *per-model* checkpoints -- each held-out model
        is evaluated on its own `split-model-<M>` run (pass `models=[names]`).

    Units are `predict_ensemble`'s native target units with land as NaN; the
    depth-integrated `/DEPTH_INT -> mol/m^3` rescale is a caller step (as in
    `skill_map`). Missing (model, time) steps are filled with NaN so every model
    shares one time axis and the result concatenates cleanly. Returns None if no
    model produced any field. Cost ~ n_samples x models x times x sampler steps.

    ``years`` (an int, or a list of ints) restricts generation to specific
    held-out year(s), overriding ``year_window``/``years_stride`` -- e.g.
    ``years=90`` generates only year 90 instead of the full 81-100 window.

    ``progress`` (default True) shows a tqdm bar over models plus a nested bar
    over each model's time steps, and logs a one-line summary + per-model data
    load; set it False for quiet batch runs.
    """
    sample_cfg = sample_cfg or SampleCfg()
    cfg = make_config(base, **config_overrides)
    split = cfg['train_test_split']
    times = _years_in_window(year_window, years_stride, years)
    n_months = len(months if months is not None else range(12))

    # ---- resolve which model names to evaluate, and (for insample) the run ----
    shared_run = None
    if split == 'model':
        if not models:
            raise ValueError(
                "train_test_split='model' needs an explicit models=[names] "
                "(each held-out model uses its own checkpoint)")
        model_names = list(models)
    else:
        save_dir = run_save_dir(cfg)
        if not run_exists(save_dir, epoch):
            raise FileNotFoundError(
                f'no checkpoint at epoch {epoch} in {save_dir}')
        shared_run = load_run(save_dir, device, epoch)
        avail = [m.model for m in get_models(
            shared_run.config, require_target=want_truth, max_models=max_models)]
        model_names = [n for n in (models or avail) if n in avail]

    logging.info(
        f'[generate] {"+".join(cfg["predictors"])} {cfg["experiments"][0]} '
        f'{cfg["resolution"]}/{cfg["field_type"]} split={split}: '
        f'{len(model_names)} model(s) x {len(times)} yr'
        f'{f" x {n_months} mo" if cfg["resolution"] == "monthly" else ""} x '
        f'{sample_cfg.n_samples} samples @ epoch {epoch}')

    # ---- per model: locate/load the run, build data, sample every time step ----
    per_model, ref_config = [], None
    model_bar = tqdm(model_names, desc='models', unit='model',
                     disable=not progress or len(model_names) <= 1)
    for name in model_bar:
        model_bar.set_description(f'model {name}')
        if split == 'model':
            sdir = run_save_dir(cfg, train_test_split='model', models_test=[name])
            if not run_exists(sdir, epoch):
                logging.info(f'[skip] {name}: no run at {sdir.name}')
                continue
            run = load_run(sdir, device, epoch)
        else:
            run = shared_run
        ref_config = ref_config or run.config

        mobjs = [m for m in get_models(run.config, require_target=want_truth)
                 if m.model == name]
        if not mobjs:
            logging.info(f'[skip] {name}: not present in data')
            continue

        model_bar.set_description(f'model {name}: loading data')
        if want_truth:
            ds_model = build_per_model_ds(run.config, mobjs[0])
            mask_var = None
        else:
            ds_model = build_predictors_only_ds(run.config, mobjs[0])
            mask_var = run.predictor_channels[0]
        lat, lon = ds_model['lat'], ds_model['lon']
        resolution = run.config['resolution']
        mlist = ((months if months is not None else list(range(12)))
                 if resolution == 'monthly' else [None])
        cases = [(y, m) for y in times for m in mlist]
        logging.info(f'  {name}: {len(run.predictor_channels)} predictor '
                     f'channels, sampling {len(cases)} time step(s)')

        gen_t, truth_t, yr_c, mo_c = [], [], [], []
        model_bar.set_description(f'model {name}: sampling')
        for year, month in tqdm(cases, desc=f'{name} steps', unit='step',
                                leave=False, disable=not progress):
            ds_sel = select_case(ds_model, resolution, year, month)
            if ds_sel is None:
                gen_t.append(_nan_slab(lat, lon, sample_cfg.n_samples))
                if want_truth:
                    truth_t.append(_nan_slab(lat, lon))
            else:
                g = predict_ensemble(ds_sel, run, sample_cfg, mask_var=mask_var)
                gen_t.append(g.reset_coords(drop=True))
                if want_truth:
                    truth_t.append(
                        ds_sel[run.config['target']].reset_coords(drop=True))
            yr_c.append(year)
            mo_c.append(-1 if month is None else month)

        data_vars = {'gen': xr.concat(gen_t, 'time')}
        if want_truth:
            data_vars['truth'] = xr.concat(truth_t, 'time')
        ds_m = xr.Dataset(data_vars).assign_coords(
            time=np.arange(len(yr_c)), year=('time', yr_c))
        if resolution == 'monthly':
            ds_m = ds_m.assign_coords(month=('time', mo_c))
        per_model.append(ds_m.assign_coords(model=name))

    if not per_model:
        return None

    out = xr.concat(per_model, 'model')
    out = out.assign_coords(sample=np.arange(sample_cfg.n_samples))
    out = out.transpose('sample', 'model', 'time', 'lat', 'lon',
                        missing_dims='ignore')
    out = out.assign_coords(
        target=ref_config['target'],
        predictors='+'.join(ref_config['predictors']),
        experiment=ref_config['experiments'][0],
        resolution=ref_config['resolution'],
        field_type=ref_config['field_type'],
        split=split,
        sampler=sample_cfg.sampler,
        epoch=(-1 if epoch is None else epoch))
    out['gen'].attrs['units'] = 'target native units (÷ DEPTH_INT for mol/m³)'
    return out

# ---------------------------------------------------------------------------
# plotting  (ultraplot robin axes via ocean_utils_min.contourf_plot)
# ---------------------------------------------------------------------------
def _pplt():
    import ultraplot as pplt
    return pplt


def _contourf():
    from vendored.ocean_utils_min import contourf_plot
    return contourf_plot


def shared_limits(rel_maps, o2_maps=None, err_maps=None, rel_limit=None):
    """Color limits shared across a set of maps so panels are comparable.

    Returns (vmax_o2, errmax, relmax). Any of the inputs may be empty/None.
    """
    def q(arrays, p, default=1.0):
        arrays = [a for a in arrays if a is not None]
        if not arrays:
            return default
        return max(float(np.nanquantile(
            np.concatenate([np.abs(a).ravel() for a in arrays]), p)), 1e-6)

    vmax = q([m.values for m in (o2_maps or [])], 0.98) if o2_maps else None
    errmax = q([m.values for m in (err_maps or [])], 0.98) if err_maps else None
    relmax = (float(rel_limit) if rel_limit is not None
              else q([m.values for m in rel_maps if m is not None], 0.95))
    return vmax, errmax, relmax


# Defined ABOVE ``_diverging_kw`` because that function takes
# ``log_decades=LOG_DECADES`` as a DEFAULT, which is evaluated at def time.
#: DEFAULT span of a LOGARITHMIC diverging scale, in decades below the limit,
#: and the mantissa ladder its boundaries step through -- see
#: :func:`log_levels`. ``(1, 2, 5)`` is the engineering ladder: three steps per
#: decade whose log10 spacings (0.301, 0.398, 0.301) are near-uniform, and
#: whose every boundary prints as a round number, which is what lets the bar
#: label its own boundaries (the ``nonlinear_levels`` principle).
LOG_DECADES = 2
LOG_LADDER = (1, 2, 5)


def log_levels(limit, decades=LOG_DECADES, ladder=LOG_LADDER):
    """Symmetric ``contourf`` level boundaries, uniform in ``log10|x|``.

    The LOGARITHMIC sibling of :func:`nonlinear_levels`, for a signed field
    whose interesting structure spans orders of magnitude. ``decades=2,
    limit=0.05`` gives the positive wing

        0.0005  0.001  0.002  0.005  0.01  0.02  0.05

    and the full symmetric ladder mirrors it, so the bins are

        [-0.05, -0.02] ... [-0.001, -0.0005]  [-0.0005, +0.0005]  [0.0005, ...

    -- i.e. ``2*len(ladder)*decades + 1`` bins, the CENTRAL one spanning zero
    linearly (``+/-limit/10**decades``) the way a symlog scale's linear region
    does. An odd bin count is deliberate: the middle bin is the middle of a
    diverging colormap, so "indistinguishable from zero" is drawn white rather
    than split across the two wings.

    **The bar this draws is SEGMENTED, not a smooth SymLogNorm ramp, and
    that is the point** -- the same argument :func:`nonlinear_levels` makes.
    Uneven ``levels`` make ultraplot build a ``DiscreteNorm`` and give
    every level interval an equal share of the bar, LABELLED WITH ITS OWN
    VALUE, so a reader reads a number off the bar without knowing the
    transform. A true ``SymLogNorm`` hides exactly that.

    ``limit`` must be a ladder value times a power of ten (``0.05``, ``100``,
    ``2e-3``); an autoscaled percentile limit will not be, and raises here
    rather than producing a ladder whose labels are 6.5 and 20.6: a figure on
    this scale pins a ROUND limit.

    Use :func:`log_ticks` for the colorbar ticks -- all
    ``2*len(ladder)*decades + 1`` boundaries rarely fit a short bar.
    """
    lim = float(limit)
    if not np.isfinite(lim) or lim <= 0:
        raise ValueError(f'log_levels needs a positive limit, got {limit!r}')
    dec = int(decades)
    if dec < 1:
        raise ValueError(f'need at least one decade, got {decades!r}')
    rungs = sorted(float(m) for m in ladder)
    if not rungs or rungs[0] < 1 or rungs[-1] >= 10:
        raise ValueError(f'ladder mantissas must lie in [1, 10), got {ladder!r}')
    # Decompose the limit into mantissa x 10**exponent and find its rung. round()
    # rather than == because 0.05 is not exactly 5e-2 in binary.
    exponent = math.floor(math.log10(lim))
    mantissa = lim / 10.0 ** exponent
    hit = [i for i, m in enumerate(rungs) if abs(m - mantissa) < 1e-9]
    if not hit:
        raise ValueError(
            f'log_levels needs a limit on the {tuple(ladder)} ladder -- '
            f'{limit!r} is {mantissa:.4g}e{exponent}, so its boundaries would '
            'print as unround numbers. Pin a round limit in the spec.')
    top = hit[0]
    pos = []
    for k in range(len(rungs) * dec + 1):
        i = top - k
        pos.append(rungs[i % len(rungs)] * 10.0 ** (exponent + math.floor(i / len(rungs))))
    pos = pos[::-1]
    return [-v for v in pos[::-1]] + pos


def log_ticks(levels, every=len(LOG_LADDER)):
    """Colorbar ticks for a :func:`log_levels` ladder: every ``every``-th
    boundary of each wing, counted OUT from the centre, and mirrored.

    ``every=len(ladder)`` (the default) labels one boundary per DECADE, which
    on the ``(1, 2, 5)`` ladder is the mantissa-1 rung: ``+/-0.0005, 0.005,
    0.05``. Six labels, evenly spaced along the bar (a decade is always three
    equal segments). ``every=1`` labels every boundary.
    """
    pos = [v for v in levels if v > 0]
    keep = pos[::int(every)]
    return [-v for v in keep[::-1]] + keep


def _diverging_kw(limit, log_scale=False, log_decades=LOG_DECADES,
                  cmap='Div'):
    """``(contourf_kw, cbar_kw)`` for ONE symmetric diverging map on ``limit``.

    The single place the linear and the LOGARITHMIC (:func:`log_levels`) forms
    of the same map are spelled out, so every panel drawn either way is drawn
    identically. Under ``log_scale`` the kwargs carry ``levels`` and NOT
    ``vmin``/``vmax``: the uneven ladder is what makes ultraplot build the
    ``DiscreteNorm`` that segments and self-labels the bar, and a vmin/vmax
    alongside it would be a second, contradicting statement of the same range
    (the ``rel_error_maps`` note).

    ``cbar_kw`` is None on the linear form -- ultraplot's automatic ticks are
    right for an even scale -- and pins one tick per decade on the log one,
    where the 13 boundaries of a two-decade ladder would collide.
    """
    lim = float(limit)
    if not log_scale:
        return dict(vmin=-lim, vmax=lim, extend='both', cmap=cmap), None
    levels = log_levels(lim, log_decades)
    return (dict(levels=levels, extend='both', cmap=cmap),
            dict(ticks=log_ticks(levels)))


def diverging_map(ax, da, limit, cbar=True, cbar_label='%', cmap='Div',
                  title=None, log_scale=False, log_decades=LOG_DECADES):
    """ONE deep-ocean-masked diverging map on a symmetric +/-``limit``.

    The smallest reusable unit of the panel helpers here, for a caller that
    wants a SINGLE map out of a figure that normally draws several.

    Deliberately identical in every argument to what the multi-panel helpers
    build for such a panel -- both go through :func:`_diverging_kw`
    (``mask_shallow`` first, then ``vmin=-limit, vmax=limit, extend='both',
    cmap='Div'``) -- so a map drawn through here and the same map inside its
    full figure are the same picture. ``log_scale``/``log_decades`` are that
    helper's logarithmic form.
    """
    contourf_plot = _contourf()
    contourf_kw, cbar_kw = _diverging_kw(limit, log_scale, log_decades, cmap)
    im = contourf_plot(
        ax=ax, da=mask_shallow(da), contourf_kw=contourf_kw,
        cbar=cbar, cbar_label=cbar_label, cbar_kw=cbar_kw)
    if title is not None:
        ax.format(title=title)
    return im


def sequential_map(ax, da, limit, cbar=True, cbar_label='%', cmap='Reds',
                   title=None, extend='max'):
    """ONE deep-ocean-masked MAGNITUDE map on a sequential 0..``limit`` scale.

    The non-diverging sibling of :func:`diverging_map`, and the single-panel
    unit :func:`error_maps_pair` builds each of its two panels from, so a
    caller wanting one such map (``lomo_signed_error.triptych_maps``) does not
    fork the kwargs.

    ``da`` is taken as a magnitude (``abs`` after :func:`mask_shallow`), so a
    signed field plots as its absolute value rather than silently using half a
    sequential colormap.

    ``extend`` is 'max' for an error map, whose scale is a percentile and so is
    exceeded by construction. Pass 'neither' for a field the scale BOUNDS --
    a ratio in [0, 1] -- where the arrow would advertise cells that cannot
    exist.
    """
    im = _contourf()(
        ax=ax, da=np.abs(mask_shallow(da)),
        contourf_kw=dict(vmin=0, vmax=float(limit), extend=extend, cmap=cmap),
        cbar=cbar, cbar_label=cbar_label)
    if title is not None:
        ax.format(title=title)
    return im


#: DEFAULT number of colour bins per WING of a nonlinear rel-error scale, and
#: the exponent of the power law placing their boundaries -- see
#: :func:`nonlinear_levels`. Both are overridable per figure
#: (``rel_error_maps(gamma=, bins=)``), and a published figure pins them
#: rather than leaning on these.
#:
#: **n is a colour-CONTRAST knob, not just a resolution one.** A bin gets
#: ``1/n`` of each wing of the colormap, so the same value sits ``bin/n`` along
#: it: fewer bins give the same value a stronger colour.
#:
#: **n also decides whether the bar can carry integer tick labels**, because
#: boundary *k* is ``(limit/n**2) * k**2`` under ``gamma=0.5``: pick ``limit``
#: so that ``limit/n**2`` is a quarter and every EVEN boundary is an integer
#: (``n=8, limit=16`` -> 0, 1, 4, 9, 16; ``n=10, limit=25`` -> the same plus
#: 25).
NONLINEAR_N = 10
NONLINEAR_GAMMA = 0.5


def nonlinear_levels(limit, gamma=NONLINEAR_GAMMA, n=NONLINEAR_N):
    """Symmetric ``contourf`` level boundaries, uniform in ``|x|**gamma``.

    The alternative to widening a *linear* diverging limit until the tail fits,
    which is unaffordable whenever the field is strongly peaked at zero: the
    predictor-set maps have a global pooled median well under 1 % and an OMZ
    p99 above 10 %, so no single linear limit both resolves the bulk and
    reaches the tail.

    ``gamma`` is matplotlib's ``PowerNorm`` exponent: colour position goes as
    ``(|x|/limit)**gamma``, so boundary *k* of *n* sits at
    ``limit * (k/n)**(1/gamma)`` and ``gamma=1`` reproduces even spacing.
    ``gamma=0.5`` with ``limit=16`` and ``n=8`` gives the positive wing
    ``0, 0.25, 1, 2.25, 4, 6.25, 9, 12.25, 16`` -- whose every-other boundary is
    an integer, which is what lets the colorbar label bin EDGES with whole
    numbers. That is a property of
    ``limit/n**2`` = 0.25, not of the shape: the same γ at ``limit=20, n=10``
    gives ``0.2k**2``, integral only at *k* = 0, 5, 10. ``limit=25, n=10`` is
    the same ladder with two more bins on top.

    **The bar this draws is SEGMENTED, not a smooth nonlinear ramp, and that
    is the point.** Uneven ``levels`` make ultraplot build a ``DiscreteNorm``
    and draw the colorbar on a *function* scale: every level interval gets
    exactly ``1/(2n)`` of the bar and is LABELLED WITH ITS OWN VALUE.
    That is what answers the objection to a nonlinear scale on a
    signed quantity -- a reader does not have to know the transform to read a
    number off it, because the boundaries are printed. A smooth ``SymLogNorm``
    ramp hides exactly that.

    Requires ``0 < gamma <= 1``: above 1 the boundaries crowd at the LIMIT
    rather than at zero, which is the opposite of the problem this exists for.
    """
    lim = float(limit)
    if not np.isfinite(lim) or lim <= 0:
        raise ValueError(f'nonlinear_levels needs a positive limit, got {limit!r}')
    g = float(gamma)
    if not 0 < g <= 1:
        raise ValueError(
            f'gamma must be in (0, 1] -- {gamma!r} would crowd the levels at '
            'the limit instead of at zero')
    n = int(n)
    if n < 2:
        raise ValueError(f'need at least 2 bins per wing, got {n!r}')
    pos = [lim * (k / n) ** (1.0 / g) for k in range(n + 1)]
    return [-v for v in pos[::-1]] + pos[1:]


def rel_error_maps(axs, cases, rel_limit, cbar='each', cbar_label='%',
                   gamma=None, bins=None, cbar_ticks=None):
    """Draw one relative-error map per case INTO ``axs``; return the mappables.

    The drawing half of :func:`rel_error_panel`, split out so that a caller who
    owns a *larger* figure can reuse it: a helper that creates and saves its
    own figure cannot be merged into another one.

    ``rel_limit`` is REQUIRED here, unlike in :func:`rel_error_panel` where
    ``None`` means "autoscale off these cases". A stacked figure whose blocks
    each autoscaled would put several different colour scales under one shared
    colorbar -- silently, since every block would look plausible on its own.
    Requiring the limit is what makes that unrepresentable rather than merely
    unlikely.

    ``cbar``: ``'each'`` one colorbar per panel, ``'last'`` one on the final
    case, ``'none'`` none at all (the caller draws a single figure-level bar
    across the whole stack).

    ``cases`` are mutated in place (``mask_shallow`` on ``rel``), matching what
    :func:`rel_error_panel` has always done; the mask is idempotent, so a caller
    that has already applied it -- as ``rel_error_panel`` must, to keep shallow
    outliers out of its autoscale -- pays nothing.

    ``gamma`` -- exponent of a NONLINEAR (power-law) diverging scale, or None
    for the linear one (the default).
    Given, the map is drawn on ``nonlinear_levels(rel_limit, gamma)`` instead of
    ``vmin/vmax``, and ``rel_limit`` stops being the point where the colour
    saturates in the usual sense: it is the top of a segmented bar whose bins
    grow with |rel|, so the same figure can resolve a p50 of half a per cent and
    still reach a p99 of 13 % without flattening either. This is the answer to a
    field peaked at zero with a long tail.

    ``bins`` -- colour bins per WING under ``gamma``, or None for
    ``NONLINEAR_N``. It sets CONTRAST as much as resolution: a bin takes
    ``1/bins`` of each wing, so fewer bins give the same value a stronger
    colour. See the note above ``NONLINEAR_N``.

    ``cbar_ticks`` -- explicit tick values for the colorbar, or None for
    ultraplot's automatic choice. Needed with ``gamma``: a 10-bin-per-wing
    ladder puts 21 labelled boundaries on one bar, and ultraplot thins tick
    LABELS by dropping them, which on a short bar can leave a nonuniform scale
    with almost nothing to read it by. Pass a SUBSET OF THE LEVELS -- a tick
    between two boundaries lands at the segmented scale's interpolation of it,
    which is legible but names no colour.
    """
    if cbar not in ('each', 'last', 'none'):
        raise ValueError(f"cbar must be 'each', 'last' or 'none', got {cbar!r}")
    if rel_limit is None:
        raise ValueError(
            'rel_error_maps needs an explicit rel_limit -- autoscaling per '
            'block would put different colour scales under one colorbar')
    contourf_plot = _contourf()
    relmax = float(rel_limit)
    if gamma is None:
        rel_kw = dict(vmin=-relmax, vmax=relmax, extend='both', cmap='Div')
    else:
        # levels INSTEAD of vmin/vmax, never both: the uneven ladder is what
        # makes ultraplot build the DiscreteNorm, and a vmin/vmax alongside it
        # would be a second, contradicting statement of the same range.
        rel_kw = dict(extend='both', cmap='Div',
                      levels=nonlinear_levels(
                          relmax, gamma,
                          NONLINEAR_N if bins is None else bins))
    cbar_kw = None if cbar_ticks is None else dict(ticks=list(cbar_ticks))
    ims = []
    for i, (ax, c) in enumerate(zip(axs, cases)):
        c['rel'] = mask_shallow(c['rel'])
        ims.append(contourf_plot(
            ax=ax, da=c['rel'], contourf_kw=rel_kw,
            cbar=(cbar == 'each' or (cbar == 'last' and i == len(cases) - 1)),
            cbar_label=cbar_label, cbar_kw=cbar_kw))
        title = c['label']
        if c.get('scalar') is not None and np.isfinite(c['scalar']):
            # One decimal, not zero: these RMS values are a few per cent, and
            # '%.0f' would round away the differences the figure exists to show.
            title += f"  (RMS {c['scalar']:.1f}%)"
        ax.format(title=title)
    return ims


def rel_error_panel(cases, suptitle, save_path, rel_limit=None, ncols=None,
                    hide_empty=False, layout=None, cbar='each',
                    gamma=None, bins=None, cbar_ticks=None):
    """Grid of relative-error maps, one per case.

    ``cases`` = list of dict(label=str, rel=DataArray[, scalar=float]).

    ``hide_empty`` blanks the grid slots no case fills (5 cases at ncols=3
    leaves one) instead of drawing a bare land map there, since an empty globe
    in a published figure reads as a failed panel.

    ``layout`` is an ultraplot subplot ARRAY (e.g. ``[[1, 2, 5], [3, 4, 5]]``),
    for the case where a plain nrows x ncols grid is the wrong shape: a repeated
    number spans its slots, so that example is a 2x2 block with a full-height
    fifth panel beside it. It replaces ncols/hide_empty entirely, and ``cases``
    are consumed in the array's NUMBERING order, not the order of the sweep --
    so the caller reorders ``cases`` to match (make_plots.plot_from_cache's
    ``order=``). A fixed-aspect projection in a taller slot is centred in it,
    which is what makes the spanning panel sit vertically centred.

    ``cbar='last'`` draws ONE colorbar, on the last case, instead of one per
    panel. Legitimate only because every panel is already on the same
    +/-``relmax`` scale (that is the whole point of ``shared_limits``), so the
    per-panel bars are copies of one axis. "Last case" is the last entry of
    ``cases``, which under a ``layout`` is the highest-numbered slot -- for
    ``[[1, 2, 5], [3, 4, 5]]`` that is the full-height panel on the right, so
    its own right-side colorbar spans both rows.

    ``gamma``/``bins``/``cbar_ticks`` are :func:`rel_error_maps`' nonlinear-scale
    arguments and mean the same here -- with one extra rule: ``gamma`` REQUIRES
    an explicit ``rel_limit``. Under a nonlinear scale the limit is the top of a
    ladder that the tail is meant to reach, not the point where the colour gives
    up, so the p95 ``shared_limits`` would autoscale to would spend the whole
    ladder on the bulk.
    """
    pplt = _pplt()
    if gamma is not None and rel_limit is None:
        raise ValueError(
            'gamma needs an explicit rel_limit -- an autoscaled p95 limit is '
            'the wrong quantity for a nonlinear scale, whose limit bounds the '
            'TAIL')
    # deep-ocean mask first, so shallow outliers set neither the map nor the scale
    for c in cases:
        c['rel'] = mask_shallow(c['rel'])
    rel_maps = [c['rel'] for c in cases]
    _, _, relmax = shared_limits(rel_maps, rel_limit=rel_limit)
    proj_kw = dict(proj='robin', proj_kw=dict(lon_0=202), refwidth='60mm',
                   share=0)
    if layout is not None:
        fig, axs = pplt.subplots(layout, **proj_kw)
        if len(axs) != len(cases):
            raise ValueError(
                f'layout has {len(axs)} panels but {len(cases)} cases -- a '
                f'mismatch would silently drop or blank one')
    else:
        ncols = ncols or min(len(cases), 3)
        nrows = int(np.ceil(len(cases) / ncols))
        fig, axs = pplt.subplots(nrows=nrows, ncols=ncols, **proj_kw)
    rel_error_maps(axs, cases, relmax, cbar=cbar, gamma=gamma, bins=bins,
                   cbar_ticks=cbar_ticks)
    for ax in axs[len(cases):]:
        if hide_empty:
            ax.set_visible(False)
        else:
            ax.format(land=True)
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


def obs_truth_gen_error_panel(d, suptitle, save_path, o2_max=None,
                              err_limit=None, truth_label=None,
                              rel_limit=None, log_scale=False,
                              log_decades=LOG_DECADES):
    """[truth | generated | signed error] in a row, for ONE case.

    Pass ``d['rel']`` (signed %, 100*err/truth) as well and it becomes the
    **2x2** [truth | generated | error | relative error]. The relative panel is the same quantity as the error panel with the
    denominator divided out, so it is drawn diverging on a symmetric scale like
    the error panel -- NOT like :func:`error_panel`'s ``rel``, which is a
    model-mean of per-model *magnitudes* and is sequential from 0.

    Fixed limits and the 'Observed' label are what make the obs_validation
    figures comparable across products and cells.

    ``d`` = dict(truth, gen, err[, rel]) in mol/m^3 (rel in %), ``err``
    **signed** (gen - truth) -- unlike :func:`error_panel`, whose magnitudes
    exist to stop opposite-signed *per-model* biases cancelling. There is no
    model axis here, so the sign is real information and is kept.

    Truth and generated necessarily share one colour scale (``o2_max``, or the
    98th percentile of both when None) -- a per-panel scale would make the visual
    comparison meaningless. ``err_limit`` / ``rel_limit`` fix the symmetric
    error scales.

    ``log_scale`` puts the two ERROR panels (and only those -- the observed and
    generated panels stay linear, being a positive field on a shared sequential
    bar) on the LOGARITHMIC diverging ladder of :func:`log_levels`, spanning
    ``log_decades`` decades below each panel's own limit, with the colorbar
    ticked once per decade by :func:`log_ticks`. Default **off**. Both limits
    must then be round ladder values; ``log_levels`` raises otherwise, rather
    than inherit an autoscaled percentile (whose ladder would print as 6.5 and
    20.6).
    """
    pplt, contourf_plot = _pplt(), _contourf()
    # deep-ocean mask first, so shallow outliers set neither maps nor scales
    d = {k: mask_shallow(v) for k, v in d.items()}
    with_rel = 'rel' in d
    if o2_max is None:
        o2_max = max(float(np.nanquantile(np.concatenate(
            [d['truth'].values.ravel(), d['gen'].values.ravel()]), 0.98)), 1e-6)
    if err_limit is None:
        err_limit = max(float(np.nanquantile(np.abs(d['err'].values), 0.98)),
                        1e-6)
    if with_rel and rel_limit is None:
        rel_limit = max(float(np.nanquantile(np.abs(d['rel'].values), 0.98)),
                        1e-6)

    shape = dict(nrows=2, ncols=2, refwidth='62mm') if with_rel else \
        dict(nrows=1, ncols=3, refwidth='58mm')
    fig, axs = pplt.subplots(proj='robin', proj_kw=dict(lon_0=202), share=0,
                             **shape)
    o2_kw = dict(vmin=0, vmax=float(o2_max), extend='max', cmap='algae')
    err_kw, err_cbar_kw = _diverging_kw(err_limit, log_scale, log_decades)
    contourf_plot(ax=axs[0], da=d['truth'], contourf_kw=o2_kw,
                  cbar=True, cbar_label=CBAR_LABEL)
    contourf_plot(ax=axs[1], da=d['gen'], contourf_kw=o2_kw,
                  cbar=True, cbar_label=CBAR_LABEL)
    contourf_plot(ax=axs[2], da=d['err'], contourf_kw=err_kw,
                  cbar=True, cbar_label=CBAR_LABEL, cbar_kw=err_cbar_kw)
    axs[0].format(title=truth_label or f'Observed {O2}')
    axs[1].format(title=f'Generated {O2}')
    axs[2].format(title='Error (gen $-$ obs)')
    if with_rel:
        rel_kw, rel_cbar_kw = _diverging_kw(rel_limit, log_scale, log_decades)
        contourf_plot(ax=axs[3], da=d['rel'], contourf_kw=rel_kw,
                      cbar=True, cbar_label='%', cbar_kw=rel_cbar_kw)
        axs[3].format(title='Relative error')
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


#: Default five-map grid: a)-c) two columns each on top, d)-e) two columns each
#: below and inset. Every panel the same size. A figure may pass its own
#: ``layout``, layout being a per-figure presentation choice.
EXTRAP_LAYOUT = [[1, 1, 2, 2, 3, 3],
                 [0, 4, 4, 5, 5, 0]]


def extrapolation_panel(d, suptitle, save_path, o2_max=None, rel_limit=None,
                        labels=None, layout=None, hratios=None):
    """The five-map extrapolation figure (see ``extrapolation.py``).

    Top row  : true O2 | generated O2, models WITH truth | generated O2, ALL
               models with the predictors -- one shared sequential scale, so the
               three are directly comparable.
    Bottom   : 100*(gen_tso - truth)/truth   (reconstruction error)
               100*(gen_ts - gen_tso)/gen_tso (the correction the extra models
               make) -- one shared diverging scale, so "is the correction bigger
               than the reconstruction error?" is readable off the figure.

    ``d`` = dict(truth, gen_tso, gen_ts, rel_recon, rel_corr) in mol/m^3 / %.
    ``labels`` = dict of the same keys -> panel title (the model counts differ
    per cell and belong in the titles).

    ``layout`` is an ultraplot subplot array, default :data:`EXTRAP_LAYOUT`.
    **Slot numbers are panel identity**, not position: 1-5 are consumed in the
    order (truth, gen_tso, gen_ts, rel_recon, rel_corr) and the panel titles
    refer to the resulting letters ("[b) $-$ a)]"), so a layout that renumbers
    the slots relabels the figure and breaks those titles. Reshaping which
    slots a number *spans* is safe; reordering the numbers is not.

    ``hratios`` weights the ROW heights, and on a fixed-aspect projection it is
    the only thing that changes a panel's SIZE. Widening a slot alone does
    **nothing**: the map keeps its aspect, the row height binds, and the extra
    width becomes margin.
    Spanning more columns is what makes the *row* usable; ``hratios`` is what
    lets the panel grow into it. Always verify with ``ax.get_position()`` after
    a draw rather than from the array.
    """
    pplt, contourf_plot = _pplt(), _contourf()
    # deep-ocean mask first, so shallow outliers set neither maps nor scales
    d = {k: mask_shallow(v) for k, v in d.items()}
    labels = labels or {}
    if o2_max is None:
        o2_max = max(float(np.nanquantile(np.concatenate(
            [d[k].values.ravel() for k in ('truth', 'gen_tso', 'gen_ts')]),
            0.98)), 1e-6)
    if rel_limit is None:
        rel_limit = max(float(np.nanquantile(np.abs(np.concatenate(
            [d[k].values.ravel() for k in ('rel_recon', 'rel_corr')])), 0.95)),
            1e-6)

    axgrid = EXTRAP_LAYOUT if layout is None else layout
    sub_kw = dict(proj='robin', proj_kw=dict(lon_0=202), refwidth='56mm',
                  share=0)
    if hratios is not None:
        sub_kw['hratios'] = tuple(hratios)
    fig, axs = pplt.subplots(axgrid, **sub_kw)
    if len(axs) != 5:
        raise ValueError(
            f'extrapolation_panel layout has {len(axs)} panels, needs exactly '
            f'5 (truth, gen_tso, gen_ts, rel_recon, rel_corr) -- a mismatch '
            f'would pair maps with the wrong axes silently')
    o2_kw = dict(vmin=0, vmax=float(o2_max), extend='max', cmap='algae')
    rel_kw = dict(vmin=-float(rel_limit), vmax=float(rel_limit), extend='both',
                  cmap='Div')
    # a-c share one scale and d-e share another, so ONE colourbar per row (on
    # its last panel) -- five identical bars would only cost map area.
    panels = [('truth', o2_kw, None), ('gen_tso', o2_kw, None),
              ('gen_ts', o2_kw, CBAR_LABEL), ('rel_recon', rel_kw, None),
              ('rel_corr', rel_kw, '%')]
    default_titles = {
        'truth': f'True {O2}', 'gen_tso': f'Generated {O2}',
        'gen_ts': f'Generated {O2}',
        'rel_recon': 'Relative reconstruction error\n[b) $-$ a)]',
        'rel_corr': f'Relative {O2} correction\nfrom additional models '
                    '[c) $-$ b)]'}
    for ax, (key, kw, cbar_label) in zip(axs, panels):
        if cbar_label is None:
            contourf_plot(ax=ax, da=d[key], contourf_kw=kw, cbar=False)
        else:
            contourf_plot(ax=ax, da=d[key], contourf_kw=kw, cbar=True,
                          cbar_label=cbar_label)
        ax.format(title=labels.get(key, default_titles[key]))
    # abc inside the axes: the default (left of the title block) collides with
    # the longer two-line panel titles this figure needs.
    axs.format(abc='a)', abcloc='ul')
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


def error_maps_pair(axs, d, err_limit, rel_limit, mape_value=None,
                    err_title=None, rel_title=None, cbar=True):
    """Draw :func:`error_panel`'s two maps INTO ``axs``; return the mappables.

    The drawing half of :func:`error_panel` -- see :func:`rel_error_maps` for why
    the split exists and why both limits are REQUIRED here (a stacked figure
    whose blocks autoscale independently is the failure mode this rules out).

    ``cbar=False`` suppresses both bars, for a caller drawing one bar per column
    across a stack of blocks. Note the two panels are DIFFERENT quantities
    (mol/m^3 and %), so a stacked figure needs one bar per column and never a
    single figure-wide one.
    """
    if err_limit is None or rel_limit is None:
        raise ValueError(
            'error_maps_pair needs explicit err_limit and rel_limit -- '
            'autoscaling per block would put different colour scales under one '
            'colorbar')
    # deep-ocean mask FIRST, so shallow outliers set neither map nor scale --
    # sequential_map does that (and the abs) for each panel.
    im_err = sequential_map(axs[0], d['err'], err_limit, cbar=cbar,
                            cbar_label=CBAR_LABEL)
    im_rel = sequential_map(axs[1], d['rel'], rel_limit, cbar=cbar,
                            cbar_label='%')
    axs[0].format(title=err_title if err_title is not None
                  else f'Absolute error $|$gen $-$ true$|$ {O2}')
    if rel_title is None:
        rel_title = 'Absolute percentage error'
        if mape_value is not None and np.isfinite(mape_value):
            rel_title += f'  (MAPE {mape_value:.1f}%)'
    axs[1].format(title=rel_title)
    return [im_err, im_rel]


def error_panel(d, suptitle, save_path, err_limit=None, rel_limit=None,
                mape_value=None, err_title=None, rel_title=None):
    """Two maps for ONE run: absolute error in O2 units (mol/m^3) and in %.

    ``d`` = dict(err, rel) -- both **non-negative** magnitude maps, the
    model-mean of the per-model absolute error (see
    :func:`model_mean_abs`): ``<|gen_m - truth_m|>`` on the left and
    ``<|100*(gen_m - truth_m)/truth_m|>`` (mean per-model MAPE, before any
    spatial averaging) on the right. Both scales are therefore sequential and
    start at 0. Any sign that survives in ``d`` is taken as magnitude anyway, so
    a signed input plots as its absolute value rather than silently mis-scaling.

    ``err_limit`` / ``rel_limit`` fix the upper colour limits (used for a common
    scale across figures); when None each is the 98th/95th percentile of the
    field. ``mape_value`` (%) -- the area-weighted mean of the right panel -- is
    annotated on its title.

    ``err_title`` / ``rel_title`` override the two panel titles; the defaults
    name the *quantity*. An explicit ``rel_title`` is taken literally and
    suppresses the ``(MAPE x%)`` suffix.
    """
    pplt = _pplt()
    # deep-ocean mask FIRST, so shallow outliers set neither map nor scale
    err = np.abs(mask_shallow(d['err']))
    rel = np.abs(mask_shallow(d['rel']))
    errmax = (float(err_limit) if err_limit is not None
              else max(float(np.nanquantile(err.values, 0.98)), 1e-6))
    relmax = (float(rel_limit) if rel_limit is not None
              else max(float(np.nanquantile(rel.values, 0.95)), 1e-6))

    fig, axs = pplt.subplots(nrows=1, ncols=2, proj='robin',
                             proj_kw=dict(lon_0=202), refwidth='68mm', share=0)
    error_maps_pair(axs, d, errmax, relmax, mape_value=mape_value,
                    err_title=err_title, rel_title=rel_title)
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


def skill_bar(labels, scalars, title, save_path, ylabel='rel. error RMS (%)'):
    """Bar chart of the scalar skill for a set of options."""
    pplt = _pplt()
    fig, ax = pplt.subplots(refwidth='90mm', refheight='55mm')
    x = np.arange(len(labels))
    ax.bar(x, scalars, width=0.7)
    ax.format(xlocator=x, xformatter=list(labels), ylabel=ylabel, title=title,
              xrotation=30)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=200)
    logging.info(f'  saved {save_path}')
    return fig
