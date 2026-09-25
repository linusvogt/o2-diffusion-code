"""
Unified training-data loader for the conditional diffusion model.

Supports the four combinations of
    resolution : 'annual' | 'monthly'
    field_type : 'depthint' | 'density'

The TARGET is always depth-integrated o2 (a single 2D channel) at the chosen
resolution -- the UNet/DDPM are unchanged. Only the PREDICTORS differ:

  * field_type='depthint': predictors are depth-integrated 2D fields (one
    channel per variable), exactly like the original setup.
  * field_type='density': predictors live on N sigma density layers; each
    layer is expanded into its own channel (``<var>_dlev<i>``), mirroring the
    existing fixed-depth ``thetao_10`` "single-level-as-channel" convention so
    that the existing ``TracerDataset`` can be reused unchanged.

``field_type`` only affects tracers that actually have both a depth-integrated
and a density-layer version (thetao, so, dissic, o2). Scalar/surface
predictors that have no density-layer analogue -- ``tauuo``, ``tauvo``,
``mld_max``, ``mld_mean`` (see ``SCALAR_VARS``) -- are always loaded via the
depth-integrated-style pipeline regardless of ``field_type``, so e.g.
``predictors=['thetao', 'so', 'mld_max', 'tauuo', 'tauvo']`` with
``field_type='density'`` expands thetao/so into density-layer channels while
loading mld_max/tauuo/tauvo as single depth-integrated channels.

Months are treated as independent samples (no month conditioning); the
train/test split is by year, so all months of a given year fall on the same
side of the split. A 0-based ``year`` coordinate (1pctCO2 -> 0..149) is kept on
the sample dimension so the existing split helpers
(``split_train_test_insample`` / ``split_train_test``) work unchanged.

Data locations (see paths.py)
-----------------------------
  depth-integrated, annual  : paths.REGRID_YEARLY/...
  depth-integrated, monthly : paths.REGRID_MONTHLY/...
  density layers, monthly   : paths.DENSITY_SURFACES/<density_var>/<var>/<exp>/...

The annual density predictors are obtained by annual-averaging the monthly
density files.

Returns from :func:`load_training_data`:
    ds_stacked          : xr.Dataset, dims (sample, lat, lon), variables =
                          target + expanded predictor channels
    predictor_channels  : list[str], the expanded predictor channel names
    target              : str

Extracted from general/data_loading.py.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import xarray as xr

import paths
from vendored.ocean_utils_min import ModelCMIP6, centers_from_modelname_dict

# cmip_io holds the needed parts of both original helper modules
# (ml_framework/util.py and cnn_2d/data_2d_new.py); the two aliases keep the
# call sites below unchanged.
from diffusion import cmip_io as ml_util
from diffusion import cmip_io as data_2d


DEPTH = slice(0, 2000)
DENSITY_ROOT = paths.DENSITY_SURFACES

# Unified MLD tree (annual max/mean and monthly all live here). Layout:
#   <root>/<exp>/{max,mean}/mld_<exp>_<center>_<model>_<member>_year{max,mean}.nc
#   <root>/<exp>/mon/mld_<exp>_<center>_<model>_<member>_mon.nc
MLD_ROOT = paths.MLD_DIR
# tauuo/tauvo (both annual and monthly) live in the unified regridded tree at
#   <root>/<exp>/Omon/<var>/<annual|monthly>/<var>_<exp>_<model>_<member>_<res>.nc
# The annual case is served by ml_util.tau_regrid_yearly_outpath (same root);
# TAU_ROOT here is used for the monthly tau helpers below. Filenames omit the
# modelling-center field, unlike the generic
# <var>_<exp>_<center>_<model>_<member>_... convention.
TAU_ROOT = paths.CMIP6_REGRIDDED

# Scalar/surface predictors that have no depth-integrated vs. density-layer
# distinction (there's only one version of each). These are always loaded via
# the depth-integrated-style pipeline, regardless of ``field_type`` -- only
# true tracers (thetao, so, dissic, o2) actually differ between the two
# field_type cases.
TAU_VARS = {'tauuo', 'tauvo'}
# MLD predictors: annual runs use mld_max / mld_mean (year statistics), monthly
# runs use the raw monthly field 'mld'. All come from MLD_ROOT.
MLD_VARS = {'mld', 'mld_max', 'mld_mean'}
SCALAR_VARS = TAU_VARS | MLD_VARS


# ----------------------------------------------------------------------------
# file paths / model discovery
# ----------------------------------------------------------------------------
def _nan_mask_path(config, ds_stacked, repr_vars):
    """Path of the cached all-NaN sample mask for this pre-drop dataset.

    Keyed by everything that determines the mask: the variables actually
    scanned (``repr_vars`` -- derived from target + predictor channels, so
    T+S and T+S+tau check different variables and must NOT share a mask) and
    the pre-drop sample identity (member coords plus year/time/exp). New source
    files landing therefore produce a different key rather than a stale hit,
    the same rule the raw cache's per-member fingerprints follow.

    Lives under the shared cache dir so it is keyed by data identity like the
    .npy files.
    """
    from diffusion.naming import (
        shared_cache_dir)
    h = hashlib.sha256()
    for v in sorted(repr_vars):
        h.update(f'{v}\n'.encode())
    h.update(np.asarray([len(ds_stacked.sample)], dtype=np.int64).tobytes())
    for coord in ('model', 'year', 'time', 'exp'):
        if coord in ds_stacked.coords:
            try:
                vals = np.asarray(ds_stacked[coord].values).astype('S64')
                h.update(np.ascontiguousarray(vals).tobytes())
            except Exception:
                pass
    return (Path(shared_cache_dir(config)) / '_nanmask' /
            f'{h.hexdigest()[:16]}.json')


def _load_nan_mask(path, n_expected):
    """Cached boolean mask, or None if absent/unusable.

    The length check is the important one: a mask of the wrong length would
    silently mis-select samples, corrupting n_samples and every downstream
    fingerprint. Any problem returns None so the caller just rescans.
    """
    try:
        with open(path) as fh:
            blob = json.load(fh)
        idx = blob['valid_idx']
        n = int(blob['n_samples'])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if n != n_expected:
        logging.warning(f'Cached all-NaN mask {path.name} is for n={n} but the '
                        f'dataset has {n_expected} samples; rescanning.')
        return None
    valid = np.zeros(n_expected, dtype=bool)
    a = np.asarray(idx, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() >= n_expected):
        return None
    valid[a] = True
    return valid


def _save_nan_mask(path, valid):
    """Write the mask (as surviving indices); per-PID tmp + atomic rename.

    Never fatal -- the mask is already in hand, this only spares later runs.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
        with open(tmp, 'w') as fh:
            json.dump({'n_samples': int(valid.size),
                       'valid_idx': np.flatnonzero(valid).tolist()}, fh)
        os.replace(tmp, path)
        logging.info(f'  cached all-NaN mask -> {path.name}')
    except Exception as e:
        logging.warning(f'  could not cache all-NaN mask ({e})')


def density_outpath(var, exp, model, density_var='sigma_1'):
    """Path of a density-layer field (tracer or 'depth') for one model.

    Matches the layout written by density_surfaces.py:
        <root>/<density_var>/<var>/<exp>/<var>_<exp>_<center>_<model>_<member>_<density_var>.nc
    (the 'depth' field uses the same pattern with var='depth').
    """
    p = DENSITY_ROOT / density_var / var / exp
    p /= f'{var}_{exp}_{model.fname_repr()}_{density_var}.nc'
    return p


def _models_in_dir(p):
    """ModelCMIP6 list parsed from CMIP filenames in directory ``p``.

    Filenames are ``<var>_<exp>_<center>_<model>_<member>_...nc`` so the
    (center, model, member) triple is parts [2:5], as elsewhere in the
    codebase.
    """
    p = Path(p)
    if not p.exists():
        return []
    return [ModelCMIP6(*str(f.stem).split('_')[2:5])
            for f in p.glob('*.nc') if 'TMP' not in f.stem]


# ----------------------------------------------------------------------------
# MLD (MLD_ROOT tree) and monthly tau -- dedicated path/load helpers (the
# shared ml_util loaders assume a single monthly depth-dir).
# ----------------------------------------------------------------------------
def _mld_which(var):
    """Map an MLD predictor name to its year-statistic subdir.

    'mld_max' -> 'max', 'mld_mean' -> 'mean', 'mld' -> 'max' (monthly ignores it).
    """
    return var.split('_', 1)[1] if '_' in var else 'max'


def _mld_outpath(exp, model, resolution, which='max'):
    if resolution == 'annual':
        p = MLD_ROOT / exp / which
        p /= f'mld_{exp}_{model.fname_repr()}_year{which}.nc'
    else:
        p = MLD_ROOT / exp / 'mon'
        p /= f'mld_{exp}_{model.fname_repr()}_mon.nc'
    return p


def _mld_models(exp, resolution, which='max'):
    sub = 'mon' if resolution == 'monthly' else which
    return set(_models_in_dir(MLD_ROOT / exp / sub))


def _load_mld(exp, model, resolution, which='max'):
    """MLD predictor channel. Annual -> year index (0-based for 1pctCO2/abrupt);
    monthly -> integer month index, mirroring the tracer loaders."""
    p = _mld_outpath(exp, model, resolution, which)
    if resolution == 'annual':
        da = xr.open_dataset(p, use_cftime=True)['mld'].squeeze()
        da['time'] = np.array([tt.year for tt in da.time.values])
        da = da.rename(time='year')
        da = _assign_year_dim(da, exp)
    else:
        da = xr.open_dataset(
            p, use_cftime=True, chunks={'time': 120})['mld'].squeeze()
        da = _assign_monthly_time(da, exp)
    return da


def _tau_monthly_dir(var, exp):
    """Directory of monthly tau files for (var, exp) in the unified tree."""
    return TAU_ROOT / exp / 'Omon' / var / 'monthly'


def _tau_monthly_models(var, exp):
    """Models with monthly tau available.

    Tau filenames omit the modelling-center field, so (model, member) is parts
    [2:4]; the center is reconstructed (matching the annual tau handling in
    ``_scalar_var_models``) so these ModelCMIP6 keys compare equal to the
    target/other-predictor model sets.
    """
    p = _tau_monthly_dir(var, exp)
    if not p.exists():
        return set()
    models = [ModelCMIP6(None, *str(f.stem).split('_')[2:4])
              for f in p.glob('*.nc') if 'TMP' not in f.stem]
    return {ModelCMIP6(centers_from_modelname_dict()[m.model][0],
                       m.model, m.member) for m in models}


def _load_tau_monthly(exp, model, var):
    p = _tau_monthly_dir(var, exp) / \
        f'{var}_{exp}_{model.model_member_repr()}_monthly.nc'
    da = xr.open_dataset(
        p, use_cftime=True, chunks={'time': 120})[var].squeeze()
    da = _assign_monthly_time(da, exp)
    return da


def _scalar_var_models(var, exp, resolution):
    """Models with a given scalar predictor (tau*/mld*) available.

    MLD comes from MLD_ROOT (annual max/mean and monthly). Monthly tau is
    globbed from the monthly tree. Annual tau keeps its dedicated parsing
    (filename has no center field, unlike the generic
    (center, model, member) = stem.split('_')[2:5] convention).
    """
    if var in MLD_VARS:
        return _mld_models(exp, resolution, which=_mld_which(var))

    # tauuo / tauvo
    if resolution == 'monthly':
        return _tau_monthly_models(var, exp)
    # annual tau -- filename: <var>_<exp>_<source_id>_<member_id>_annual.nc (no center)
    p = ml_util.tau_regrid_yearly_outpath(
        experiment_id=exp, institution_id=None, source_id=None,
        member_id=None, variable_id=var, depth=None).parent
    models = [ModelCMIP6(None, *str(f.stem).split('_')[2:4])
              for f in p.glob('*.nc') if 'TMP' not in f.stem] \
        if p.exists() else []
    models = [ModelCMIP6(centers_from_modelname_dict()[m.model][0],
                         m.model, m.member) for m in models]
    return set(models)


import csv as _csv

DENSITY_QC_CSV = paths.DENSITY_QC_CSV
_DENSITY_QC_CACHE = None


def load_density_qc():
    """Corrupt (all-NaN) density members from a QC scan of the density files,
    keyed by ``(density_var, var, exp)`` -> set of ModelCMIP6. Empty when the CSV
    is absent.

    These members exist on disk but their sigma-surface regridding produced an
    all-NaN file; a member with an all-NaN density channel generates garbage at
    eval, so it is excluded from model discovery below -- matching what training
    already does by dropping all-NaN samples.
    """
    global _DENSITY_QC_CACHE
    if _DENSITY_QC_CACHE is None:
        qc = {}
        if DENSITY_QC_CSV.is_file():
            with open(DENSITY_QC_CSV) as fh:
                for r in _csv.DictReader(fh):
                    if r.get('status') == 'corrupt':
                        qc.setdefault(
                            (r['density_var'], r['var'], r['exp']), set()
                        ).add(ModelCMIP6(r['center'], r['model'], r['member']))
        _DENSITY_QC_CACHE = qc
    return _DENSITY_QC_CACHE


def _get_predictor_models(exp, base_predictors, field_type, resolution,
                          density_var, depth):
    """Models that have every predictor variable available."""
    sets = []
    for var in base_predictors:
        if var in SCALAR_VARS:
            sets.append(_scalar_var_models(var, exp, resolution))
        elif field_type == 'density':
            p = DENSITY_ROOT / density_var / var / exp
            models = set(_models_in_dir(p))
            # drop members whose sigma-surface file is all-NaN (corrupt regrid),
            # so eval (which else includes them by file existence) matches training
            models -= load_density_qc().get((density_var, var, exp), set())
            sets.append(models)
        elif resolution == 'annual':
            p = ml_util.regrid_yearly_outpath(
                exp, None, None, None, var, depth).parent
            sets.append(set(_models_in_dir(p)))
        else:
            p = ml_util.regrid_monthly_outpath(
                exp, None, None, None, var, depth).parent
            sets.append(set(_models_in_dir(p)))
    return set.intersection(*sets) if sets else set()


def _get_target_models(exp, target, resolution, depth):
    """Models that have the (depth-integrated) target available."""
    if resolution == 'annual':
        p = ml_util.regrid_yearly_outpath(
            exp, None, None, None, target, depth).parent
    else:
        p = ml_util.regrid_monthly_outpath(
            exp, None, None, None, target, depth).parent
    return set(_models_in_dir(p))


# ----------------------------------------------------------------------------
# per-array time handling
# ----------------------------------------------------------------------------
def _assign_year_dim(da, exp):
    """Reindex an annual array's ``year`` dim to the codebase convention.

    Mirrors data_2d_new._load_exp: 1pctCO2/abrupt-4xCO2 -> 0-based index,
    historical/ssp -> actual calendar years.
    """
    da = da.copy()
    if exp in ['1pctCO2', 'abrupt-4xCO2']:
        da['year'] = np.arange(len(da.year))
    elif exp == 'historical':
        da = da.sel(year=slice(None, 2014))
        da['year'] = np.arange(1850, 2014 + 1)
    elif 'ssp' in exp:
        da = da.sel(year=slice(None, 2100))
        da['year'] = np.arange(2015, 2100 + 1)
    return da


def _assign_monthly_time(da, exp):
    """Replace a monthly cftime ``time`` axis with an integer month index and
    attach a ``year`` coordinate (0-based for 1pctCO2/abrupt, else calendar).

    Using an integer ``time`` index makes predictor and target arrays from the
    same model run align cleanly regardless of cftime calendar quirks.
    """
    yrs = np.array([t.year for t in da['time'].values])
    if exp in ['1pctCO2', 'abrupt-4xCO2']:
        year_coord = yrs - yrs[0]
    else:
        year_coord = yrs
    n = da.sizes['time']
    da = da.assign_coords(time=np.arange(n))
    da = da.assign_coords(year=('time', year_coord))
    return da


# ----------------------------------------------------------------------------
# per-(exp, model) loading
# ----------------------------------------------------------------------------
def _load_target(exp, model, target, resolution, depth):
    if resolution == 'annual':
        da = ml_util.load_regrid_yearly(
            exp, model.center, model.model, model.member, target, depth)
        da = _assign_year_dim(da, exp)
    else:
        da = ml_util.load_regrid_monthly(
            exp, model.center, model.model, model.member, target, depth)
        da = _assign_monthly_time(da, exp)
    return da.rename(target)


def _load_density_var(exp, model, var, density_var):
    """Monthly density-layer field, dims (time, lat, lon, lev), chunked."""
    p = density_outpath(var, exp, model, density_var)
    da = xr.open_dataset(p, use_cftime=True, chunks={'time': 120})[var].squeeze()
    return da


def _predictor_channels(exp, model, base_predictors, field_type, resolution,
                        density_var, levels, depth):
    """dict {channel_name: DataArray} of predictor channels for one model.

    For density predictors each level becomes a channel ``<var>_dlev<i>``.
    """
    channels = {}
    for var in base_predictors:
        if var in MLD_VARS:
            channels[var] = _load_mld(exp, model, resolution, _mld_which(var))
        elif var in TAU_VARS and resolution == 'monthly':
            channels[var] = _load_tau_monthly(exp, model, var)
        elif var in SCALAR_VARS or field_type == 'depthint':
            if resolution == 'annual':
                da = ml_util.load_regrid_yearly(
                    exp, model.center, model.model, model.member, var, depth)
                da = _assign_year_dim(da, exp)
            else:
                da = ml_util.load_regrid_monthly(
                    exp, model.center, model.model, model.member, var, depth)
                da = _assign_monthly_time(da, exp)
            channels[var] = da
        elif field_type == 'density':
            da = _load_density_var(exp, model, var, density_var)
            if resolution == 'annual':
                da = da.groupby('time.year').mean('time')
                da = _assign_year_dim(da, exp)
            else:
                da = _assign_monthly_time(da, exp)
            n_lev = da.sizes['lev']
            lev_idx = range(n_lev) if levels in (None, 'all') else list(levels)
            for i in lev_idx:
                channels[f'{var}_dlev{i}'] = da.isel(lev=i)
        else:
            raise ValueError(f'{field_type=}')
    return channels


def _build_per_model_ds(exp, model, base_predictors, target, field_type,
                        resolution, density_var, levels, depth, anom):
    """Assemble one (exp, model) dataset: target + expanded predictor channels.

    Predictors and target are aligned on the sample axis (year for annual,
    integer month index for monthly). A single clean ``year`` coordinate (from
    the target) is kept for the train/test split.
    """
    sample_dim = 'year' if resolution == 'annual' else 'time'

    da_target = _load_target(exp, model, target, resolution, depth)
    channels = _predictor_channels(
        exp, model, base_predictors, field_type, resolution,
        density_var, levels, depth)

    names = [target] + list(channels.keys())
    arrs = list(xr.align(da_target, *channels.values(), join='inner'))
    year_vals = (arrs[0]['year'].values
                 if 'year' in arrs[0].coords else None)

    # drop per-array non-dim coords (year, sigma_1, lev, ...) to avoid merge
    # conflicts, then attach one clean year coord below
    arrs = [a.reset_coords(drop=True) for a in arrs]
    ds = xr.Dataset({n: a for n, a in zip(names, arrs)})

    if resolution != 'annual' and year_vals is not None:
        ds = ds.assign_coords(year=(sample_dim, year_vals))

    if anom:
        ds = _subtract_anom(ds, sample_dim, resolution)
    ds = _restrict_years(ds, exp, sample_dim, resolution)

    ds = ds.assign_coords(model=model, exp=exp)
    return ds


def _subtract_anom(ds, sample_dim, resolution):
    """Anomaly wrt the first 20 years (mirrors data_2d_new.construct_ds)."""
    if resolution == 'annual':
        base = ds.isel({sample_dim: slice(None, 20)}).mean(sample_dim)
    else:
        y0 = int(ds.year.min())
        base = ds.where(ds.year < y0 + 20, drop=True).mean(sample_dim)
    return ds - base


def _restrict_years(ds, exp, sample_dim, resolution):
    """Keep the same year span as the original pipeline (150 yr for 1pctCO2)."""
    if exp in ['1pctCO2', 'abrupt-4xCO2']:
        if resolution == 'annual':
            ds = ds.isel({sample_dim: slice(None, 150)})
        else:
            ds = ds.where(ds.year < 150, drop=True)
    elif 'hist-ssp' in exp and resolution == 'annual':
        ds = ds.isel({sample_dim: slice(None, 251)})
    return ds


def _stack(ds, resolution):
    """Stack (model, exp, <time>) into a single ``sample`` dimension.

    Mirrors data_2d_new.stack but uses the monthly ``time`` index when the
    resolution is monthly. Singleton coords (e.g. a single experiment) are not
    stacked, matching the original behaviour.
    """
    time_dim = 'year' if resolution == 'annual' else 'time'
    ds = ds.squeeze()
    coords = tuple(c for c in ['model', 'exp', time_dim]
                   if c in ds.coords and ds[c].size != 1)
    return ds.stack(sample=coords)


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def load_training_data(config, max_models=None):
    """Load and stack training data for the diffusion model.

    Parameters
    ----------
    config : dict
        Must contain: ``resolution`` ('annual'|'monthly'), ``field_type``
        ('depthint'|'density'), ``experiments`` (list), ``predictors`` (base
        variable names), ``target``, ``anom``. Optional: ``density_var``
        (default 'sigma_1'), ``density_levels`` ('all' or list of ints).
    max_models : int, optional
        If given, keep only the first ``max_models`` models per experiment
        (useful for quick smoke tests).

    Returns
    -------
    ds_stacked : xr.Dataset  (dims: sample, lat, lon)
    predictor_channels : list[str]
    target : str
    """
    resolution = config.get('resolution', 'annual')
    field_type = config.get('field_type', 'depthint')
    exps = config['experiments']
    base_predictors = list(config['predictors'])
    target = config['target']
    density_var = config.get('density_var', 'sigma_1')
    levels = config.get('density_levels', 'all')
    anom = config['anom']
    depth = DEPTH

    # Annual depth-integrated setup: delegate to the original loader
    # (cmip_io.wrap_load_data). Skip the shortcut when an MLD predictor is
    # requested: wrap_load_data reads MLD from a different location, whereas
    # all MLD is sourced from MLD_ROOT here, so those combos go through the
    # per-model path below.
    mld_requested = any(v in MLD_VARS for v in base_predictors)
    if resolution == 'annual' and field_type == 'depthint' and not mld_requested:
        variables = base_predictors + [target]
        ds_stacked, *_ = data_2d.wrap_load_data(
            variables=variables, experiments=exps, depth=depth, anom=anom)
        return ds_stacked, list(base_predictors), target

    data_exps = []
    predictor_channels = None
    for exp in exps:
        m_pred = _get_predictor_models(
            exp, base_predictors, field_type, resolution, density_var, depth)
        m_tgt = _get_target_models(exp, target, resolution, depth)
        models = sorted(m_pred & m_tgt,
                        key=lambda m: (m.model, ml_util.member_key(m.member)))
        if max_models is not None:
            models = models[:max_models]
        logging.info(f'{exp}: {len(models)} models with predictors + target')

        data_models = []
        for i, model in enumerate(models):
            t0 = time.monotonic()
            try:
                ds = _build_per_model_ds(
                    exp, model, base_predictors, target, field_type,
                    resolution, density_var, levels, depth, anom)
            except Exception as e:
                logging.info(f'  skip {model.model} {model.member}: {e}')
                continue
            if predictor_channels is None:
                predictor_channels = [v for v in ds.data_vars if v != target]
            data_models.append(ds)
            logging.info(f'  [{i+1}/{len(models)}] {model.model} {model.member} '
                         f'({time.monotonic()-t0:.1f}s)')

        if data_models:
            ds_exp = xr.concat(data_models, 'model').assign_coords(exp=exp)
            data_exps.append(ds_exp)

    if not data_exps:
        raise RuntimeError(
            f'No models found for {resolution=} {field_type=} {exps=}. '
            f'(Monthly modes require the files under '
            f'{paths.REGRID_MONTHLY}/.)')

    ds = xr.concat(data_exps, 'exp') if len(data_exps) > 1 else data_exps[0]
    ds_stacked = _stack(ds, resolution)

    # Drop samples where any source variable is entirely NaN (missing timesteps).
    # Strategy: pick one representative channel per source file (e.g. one of the
    # 10 thetao_dlev* channels suffices because they all share the same dask
    # task graph and are deduplicated by the scheduler).  Batch all notnull()
    # masks into a single dask.compute() so each underlying file is read once.
    all_vars = [target] + list(predictor_channels)
    seen_prefixes: set = set()
    repr_vars: list = []
    for v in all_vars:
        prefix = v.split('_dlev')[0] if '_dlev' in v else v
        if prefix not in seen_prefixes:
            repr_vars.append(v)
            seen_prefixes.add(prefix)

    # This read is expensive (minutes for monthly/density) and its result is
    # invariant across launches and resumes, so cache the surviving-sample mask.
    n0 = len(ds_stacked.sample)
    mask_path = _nan_mask_path(config, ds_stacked, repr_vars)
    valid = _load_nan_mask(mask_path, n0)
    if valid is not None:
        logging.info(f'Reusing cached all-NaN mask ({mask_path.name}); '
                     f'skipped the {len(repr_vars)}-variable scan')
    else:
        logging.info(f'Dropping all-NaN samples (checking {repr_vars})...')
        t0 = time.monotonic()
        # has_data[i][sample] == True when that sample has >=1 non-NaN cell in var i
        has_data_list = [ds_stacked[v].notnull().any(['lat', 'lon'])
                         for v in repr_vars]
        try:
            import dask as _dask
            has_data_list = list(_dask.compute(*has_data_list))
        except ImportError:
            pass  # already concrete for non-dask datasets
        valid = np.ones(n0, dtype=bool)
        for mask in has_data_list:
            valid &= mask.values
        logging.info(f'  all-NaN scan took {time.monotonic()-t0:.1f}s')
        _save_nan_mask(mask_path, valid)
    ds_stacked = ds_stacked.isel(sample=valid)
    logging.info(f'Dropped {n0 - len(ds_stacked.sample)} all-NaN samples; '
                 f'{len(ds_stacked.sample)} samples, '
                 f'{len(predictor_channels)} predictor channels')

    return ds_stacked, predictor_channels, target


# ----------------------------------------------------------------------------
# ensemble composition (the ESGF member-count weighting)
# ----------------------------------------------------------------------------
def subsample_members(ds, max_per_model, sample_dim='sample'):
    """Keep at most ``max_per_model`` members of every model along ``sample_dim``.

    The training distribution is otherwise weighted by how many members a
    centre happened to publish to ESGF -- 1pctCO2 runs ~10:1 (CNRM-ESM2-1)
    against several single-member models -- a weighting with no scientific
    justification. Capping the members per model removes it.

    Which members survive is deterministic and independent of file order:
    models are taken in name order and their members sorted by
    ``ml_util.member_key`` (the (r, i, p, f) integer tuple), so r1i1p1f1 is
    kept first. That matters because the whole point is a controlled
    comparison against the uncapped run -- a shuffled or filesystem-ordered
    choice would make the sensitivity test unreproducible.

    ⚠️ This drops WHOLE MEMBERS, never samples within a member, which is what
    keeps the shared raw preproc cache valid: a kept member's time/exp coords
    are unchanged, so its per-(channel, member) fingerprint -- and therefore
    its cache filename -- is identical to the uncapped run's. The capped run
    reads a subset of the same files rather than re-decoding into new ones.

    Parameters
    ----------
    ds : xr.Dataset
        Stacked dataset with a ``model`` coordinate of ``ModelCMIP6`` on
        ``sample_dim``.
    max_per_model : int
        Members to keep per model name (``ModelCMIP6.model``, so two centres
        publishing the same model name share one budget -- they are the same
        model).

    Returns
    -------
    ds_sub : xr.Dataset
    kept : dict[str, list[str]]
        model name -> the member ids kept, for logging into the run's record.
    """
    if max_per_model is None:
        return ds, None
    if max_per_model < 1:
        raise ValueError(f'{max_per_model=} must be >= 1')

    models = ds['model'].values  # one ModelCMIP6 per sample

    by_name: dict = {}
    for m in set(models):
        by_name.setdefault(m.model, []).append(m)

    keep = set()
    kept: dict = {}
    for name in sorted(by_name):
        members = sorted(by_name[name], key=lambda m: ml_util.member_key(m.member))
        chosen = members[:max_per_model]
        keep.update(chosen)
        kept[name] = [m.member for m in chosen]

    sel = np.array([m in keep for m in models])
    ds_sub = ds.isel({sample_dim: sel})
    logging.info(
        f'subsample_members(max_per_model={max_per_model}): '
        f'{len(set(models))} -> {len(keep)} members over {len(by_name)} models, '
        f'{len(models)} -> {int(sel.sum())} samples')
    for name in sorted(kept):
        dropped = len(by_name[name]) - len(kept[name])
        if dropped:
            logging.info(f'  {name}: kept {kept[name]}, dropped {dropped}')
    return ds_sub, kept
