"""Fig. 9: extrapolation to CMIP6 models without oxygen (T+S, annual,
depth-integrated, 1pctCO2) -- a) true O2, b) generated O2 for the same models,
c) generated O2 for every model with the predictors, d) the relative
reconstruction error b)-a), e) the relative correction the extra models make,
c)-b).

Extracted from general/eval/extrapolation.py (compute and plot). This module
also holds the extrapolation cache and eastern-tropical-Pacific OMZ helpers
that Fig. 10 (figures/fig10_extrapolation_omz.py) draws from the same cache.

    python -m figures.fig09_extrapolation compute [--epoch 500]   # GPU
    python -m figures.fig09_extrapolation plot                    # no GPU
"""
from __future__ import annotations

from figures import render_common   # first: forces the headless Agg backend

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import paths
from diffusion import obs as util_obs
from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, get_models,
    build_per_model_ds, build_predictors_only_ds, select_case,
    predict_ensemble, SampleCfg, DEPTH_INT, FIG_DIR, DEFAULT_YEAR_WINDOW)


# ---------------------------------------------------------------------------
# pinned manuscript constants (general/manuscript/figures.py, main variant)
# ---------------------------------------------------------------------------
# Fixed colour limits: panels a-c share one O2 scale, d-e one relative scale.
EXTRAP_O2_MAX = 0.35      # mol/m^3, panels a-c
EXTRAP_REL_LIMIT = 20.0   # %, symmetric, panels d-e

# a)-c) on top, d)-e) below at three of six columns each. On a fixed-aspect
# projection a wider slot alone does not enlarge the panel (the row height
# binds); `hratios` is what resizes d)-e), and it saturates at ~1.53x, so
# (1, 1.5) is the largest value fully converted into panel area. Slot NUMBERS
# are panel identity (the titles refer to the letters), so only spans move.
EXTRAP_LAYOUT = [[1, 1, 2, 2, 3, 3],
                 [4, 4, 4, 5, 5, 5]]
EXTRAP_HRATIOS = (1, 1.5)

# The manuscript's main cell (Fig. 10 draws the same cache).
CELL = dict(predictors=['thetao', 'so'], resolution='annual',
            field_type='depthint', experiment='1pctCO2')


# ===========================================================================
# from general/eval/extrapolation.py
# ===========================================================================
# Generate O2 for the CMIP6 models that have the predictors but no O2. The
# unverifiable part is bracketed by a verifiable one: panel d) is a
# held-out-YEAR, in-sample-MODEL error (the run is the `insample` split), so it
# is a best case and NOT comparable with the leave-one-model-out numbers.
#
# One common ocean mask: all five maps share the intersection of the per-model
# O2 masks over the (usable) models that have O2. It is the sampler mask for
# EVERY model and is applied to the truth too; without it ``mean('model')`` is
# skipna and panel e) would be a coverage artifact.
#
# Data hygiene: model discovery is filename globbing, so some models carry an
# all-NaN regrid. Every model is checked slice by slice and a model with no
# usable time step is dropped and stamped in the cache (``models_skipped``).
#
# The OMZ series (Fig. 10) cannot be reconstructed from the five cached maps (a
# weighted mean of model-means is not the distribution over models), so it is
# computed inside the sampling pass and cached alongside. Each point is one
# (model, sample) pair with time averaged first.
OUT_DIR = FIG_DIR / 'extrapolation'
CACHE_DIR = OUT_DIR / 'cache'

BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# Epoch reached by every in-sample run of the sweep: pinning it keeps training
# length out of the comparison. '--epoch latest' opts out.
DEFAULT_EPOCH = 500

# A cell here spans 3-5x more models than the LOMO error maps do, so the year
# strides are coarser: the model-mean, not the time-mean, is what these maps
# are about.
DEFAULT_STRIDE = {'annual': 2, 'monthly': 10}

# A model needs this many cells of the common mask before it is worth
# generating for -- below it the file is a corrupt (near-)all-NaN regrid.
MIN_OCEAN_CELLS = 1000

_SHORT_PRED = {'thetao': 'T', 'so': 'S', 'tauuo': 'taux', 'tauvo': 'tauy',
               'mld': 'MLD'}
_PRED_TEX = {'thetao': 'T', 'so': 'S', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}


# ---------------------------------------------------------------------------
# eastern tropical Pacific OMZ
# ---------------------------------------------------------------------------
# The OMZ mask is observational and identical for every cell: RECCAP2 Pacific,
# |lat| <= 30, and observed 0-2000 m COLUMN-MEAN O2 below OMZ_O2_THRESHOLD. This
# is a column-mean criterion, not the conventional OMZ-core one, and it is NOT
# bathymetry-masked, so OMZ numbers are not comparable with the deep-ocean
# masked scores. Everything is reduced over the mask intersected with the
# cell's common ocean mask, the observed value included.
GRIDAREA_FILE = paths.GRIDAREA
REGION_MASK_FILE = paths.RECCAP2_MASK

OMZ_O2_THRESHOLD = 0.1   # mol/m^3, on the 0-2000 m COLUMN MEAN -- not OMZ-core
OMZ_MAX_LAT = 30.0       # deg, tropical band
OMZ_BASIN = 'pacific'    # RECCAP2 basin variable
#: Observational product behind the OMZ region mask, the offset in panels a/c
#: of the OMZ maps, and the reference line of the OMZ PDF. ONE knob on purpose,
#: so the region, the bias and the reference line come from one product.
#: Changing it changes the REGION, so the cached omz_* series (reduced inside
#: the sampling pass over that mask) do not carry over; `omz_ref` is stamped in
#: every cache.
OMZ_REF = 'woa'
_OMZ_LOADERS = {'gobai': 'load_gobai_levmean', 'woa': 'load_woa_levmean'}
_OMZ_LABELS = {'gobai': 'GOBAI-O2', 'woa': 'WOA23'}
OMZ_REF_LABEL = _OMZ_LABELS[OMZ_REF]


# Stamped in every cache: a cache written under a different OMZ definition must
# not be replotted as if it answered the current one. The reference product is
# IN the key, and '+cal' marks caches that carry the calibrated series.
OMZ_KEY = (f'{OMZ_BASIN}-lat{OMZ_MAX_LAT:g}-{OMZ_REF}-colmean-lt'
           f'{OMZ_O2_THRESHOLD:g}+cal')

# WOA's oxygen file, which carries the analysis' own statistics (o_sd, o_se,
# o_dd) beside o_an on the SAME grid -- so the uncertainty band on the observed
# line comes from the same product as the line. The obs loader reads only
# o_an, hence the separate path here.
WOA_O2_FILE = str(paths.WOA_O2_FILE)

_OMZ_CACHE = {}


def _cell_area():
    """(lat, lon) grid-cell area, loaded once. NaN -> 0 so it can weight."""
    if 'area' not in _OMZ_CACHE:
        _OMZ_CACHE['area'] = (xr.open_dataset(GRIDAREA_FILE).cell_area
                              .fillna(0.0).load())
    return _OMZ_CACHE['area']


def omz_obs(source=None):
    """Observed 0-2000 m column-mean O2 (mol/m^3) from ``source``, time-mean.

    Both loaders return the depth-*integral* (mol/m^2) like the CMIP fields, so
    this divides by ``DEPTH_INT`` exactly as the generated fields do. Loaded
    eagerly: GOBAI opens with ``chunks=dict(time=1)`` and a lazy array here
    would be recomputed once per model. WOA's O2 is a single climatological
    field with no time axis at all, hence the conditional reduction.
    """
    source = source or OMZ_REF
    key = f'obs_{source}'
    if key not in _OMZ_CACHE:
        da = getattr(util_obs, _OMZ_LOADERS[source])().o2
        for dim in ('year', 'time'):
            if dim in da.dims:
                da = da.mean(dim)
        _OMZ_CACHE[key] = da.load() / DEPTH_INT
    return _OMZ_CACHE[key]


def omz_mask():
    """Boolean (lat, lon) mask of the eastern tropical Pacific OMZ.

    Observational and identical for every cell, so the region never moves with
    the run being evaluated (column-mean criterion, not OMZ core).
    """
    key = f'mask_{OMZ_REF}'
    if key not in _OMZ_CACHE:
        obs = omz_obs()
        reccap = xr.open_dataset(REGION_MASK_FILE)
        region = (reccap[OMZ_BASIN] > 0) & (np.abs(obs.lat) <= OMZ_MAX_LAT)
        mask = obs.where(region) < OMZ_O2_THRESHOLD
        _OMZ_CACHE[key] = mask.reset_coords(drop=True).load()
    return _OMZ_CACHE[key]


def omz_mean(da):
    """Area-weighted mean over the OMZ, reducing lat/lon only.

    NaN cells (land, and anything outside the cell's common ocean mask) drop out
    of both the sum and the weight normalisation, so this is the mean over
    ``omz_mask & ocean``. No bathymetry mask.
    """
    return (da.where(omz_mask())
            .weighted(_cell_area()).mean(['lat', 'lon']))


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def pred_slug(preds):
    return '-'.join(_SHORT_PRED.get(p, p) for p in preds)


def pred_label(preds):
    return '+'.join(_PRED_TEX.get(p, p) for p in preds)


# Extrapolation models left OUT of the `extra` set in a sensitivity run; part
# of the slug so such a run can never overwrite the published cache. Empty for
# the published figures.
EXCLUDED_MODELS = ()


def _excl_tag():
    """Slug suffix naming the exclusion, or '' when there is none.

    Names rather than a hash, so the file is readable next to the unmodified
    cache; hashed only if the names would make an unwieldy filename.
    """
    if not EXCLUDED_MODELS:
        return ''
    joined = '+'.join(EXCLUDED_MODELS)
    if len(joined) > 48:
        import hashlib
        joined = (f'{len(EXCLUDED_MODELS)}models-'
                  f'{hashlib.sha1(joined.encode()).hexdigest()[:8]}')
    return f'_excl-{joined}'


def cell_slug(cell):
    """Filename stem for a cell -- stable, so cache/figure names don't drift."""
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    return (f"{pred_slug(cell['predictors'])}_{res}_{fld}_"
            f"{cell['experiment']}{_excl_tag()}")


def cell_title(cell):
    fld = {'depthint': 'depth-integrated', 'density': 'density layers'}[
        cell['field_type']]
    t = (f"{pred_label(cell['predictors'])} | {cell['resolution']} | {fld} | "
         f"{cell['experiment']}")
    return t


def cell_config(cell):
    """Naming config for a cell's run (always the in-sample split)."""
    return make_config(BASE_CONFIG, predictors=cell['predictors'],
                       experiments=[cell['experiment']],
                       resolution=cell['resolution'],
                       field_type=cell['field_type'])


# ---------------------------------------------------------------------------
# epoch resolution
# ---------------------------------------------------------------------------
def latest_epoch(save_dir):
    epochs = [int(m.group(1))
              for p in Path(save_dir).glob('ckpt_epoch*.pt')
              if (m := re.search(r'ckpt_epoch(\d+)\.pt$', p.name))]
    return max(epochs) if epochs else None


def resolve_epoch(save_dir, epoch):
    if epoch in (None, 'latest'):
        return latest_epoch(save_dir)
    return int(epoch) if run_exists(save_dir, int(epoch)) else None


# ---------------------------------------------------------------------------
# model discovery + data hygiene
# ---------------------------------------------------------------------------
def split_models(run):
    """(models_with_truth, models_predictors_only) for a loaded run.

    The two lists are queried separately rather than by filtering one, because
    ``get_models`` keeps one MEMBER per model and the first member with the
    predictors is not always the member that also has the target (MRI-ESM2-0:
    r1i1p1f1 has T/S, a later member has O2). Taking the target-side object for
    models that have truth is what makes ``build_per_model_ds`` find its file.
    """
    with_truth = get_models(run.config, require_target=True)
    have = {m.model for m in with_truth}
    extra = [m for m in get_models(run.config, require_target=False)
             if m.model not in have]
    return with_truth, extra


def _repr_channels(names):
    """One representative channel per SOURCE FILE, as training does.

    All ``<var>_dlev<i>`` channels of a tracer come from the same file, so
    checking one of them tells you whether that file is an all-NaN regrid.
    """
    seen, out = set(), []
    for n in names:
        prefix = n.split('_dlev')[0]
        if prefix not in seen:
            seen.add(prefix)
            out.append(n)
    return out


def usable_slice(ds_sel, channels):
    """True when every source variable of this time step has some finite data.

    Model discovery globs filenames, so a model can "have" a variable whose
    regrid is entirely NaN (IPSL-CM5A2-INCA / IPSL-CM6A-LR depth-integrated).
    Training drops such samples; generating from one would return an all-NaN
    field that then poisons the common mask and the model-mean.
    """
    return all(np.isfinite(ds_sel[ch].values).any()
               for ch in _repr_channels(channels))


def iter_times(resolution, years, months):
    return ([(y, None) for y in years] if resolution == 'annual'
            else [(y, m) for y in years for m in months])


# ---------------------------------------------------------------------------
# per-cell computation
# ---------------------------------------------------------------------------
def common_ocean_mask(run, models, years, months, ctx):
    """(mask DataArray, usable models) -- the intersection of the per-model O2
    masks, over the models whose data is not an all-NaN regrid.

    One cheap pass (one time slice per model) before any sampling: the mask has
    to be known BEFORE generating, since it is what is handed to the sampler.
    """
    resolution = run.config['resolution']
    target = run.config['target']
    masks, usable, dropped = [], [], []
    for m in models:
        try:
            ds_model = build_per_model_ds(run.config, m)
        except Exception as e:                     # missing/unreadable file
            dropped.append(f'{m.model} ({type(e).__name__})')
            continue
        found = None
        for year, month in iter_times(resolution, years, months):
            ds_sel = select_case(ds_model, resolution, year, month)
            if ds_sel is None or not usable_slice(ds_sel,
                                                  list(ds_sel.data_vars)):
                continue
            ok = np.isfinite(ds_sel[target].values)
            if ok.sum() < MIN_OCEAN_CELLS:
                continue
            found = xr.DataArray(ok, dims=('lat', 'lon'),
                                 coords={'lat': ds_sel.lat, 'lon': ds_sel.lon})
            break
        if found is None:
            dropped.append(f'{m.model} (no usable time step)')
            continue
        masks.append(found)
        usable.append(m)
        logging.info(f'  mask {m.model}: {int(found.sum())} ocean cells')

    if not masks:
        return None, [], dropped
    stack = xr.concat(masks, 'm')
    inter, union = stack.all('m'), stack.any('m')
    logging.info(f'  common mask: {int(inter.sum())} cells (intersection of '
                 f'{len(masks)} models; union {int(union.sum())})')
    if dropped:
        logging.info(f'  [drop] truth models: {", ".join(dropped)}')
    if int(inter.sum()) < MIN_OCEAN_CELLS:
        logging.info('  [warn] the common mask is nearly empty -- refusing to '
                     'generate on it')
        return None, [], dropped
    return inter, usable, dropped


def model_gen(run, ds_model, ocean, years, months, sample_cfg, target=None):
    """Time-mean generated (and optionally truth) field for one model.

    ``ocean`` is the cell's common mask, used for every model -- so the model
    axis is the only thing that differs between the pooled panels. Time steps
    whose source data is all-NaN are skipped; None when none survive.

    The OMZ-mean series is reduced HERE, from the full ensemble before the
    sample-mean: a weighted mean of the cached model-mean maps is not the
    distribution over (model, sample) the PDF needs, so it cannot be recovered
    afterwards. ``omz_gen`` is per-sample with time averaged first (matching the
    maps); ``omz_gen_t`` keeps every (sample, time) point.
    """
    resolution = run.config['resolution']
    channels = list(run.predictor_channels) + ([target] if target else [])
    gen_t, truth_t, omz_t, omz_truth_t, ens_t = [], [], [], [], []
    for year, month in iter_times(resolution, years, months):
        ds_sel = select_case(ds_model, resolution, year, month)
        if ds_sel is None or not usable_slice(ds_sel, channels):
            continue
        ens = predict_ensemble(ds_sel, run, sample_cfg, ocean=ocean.values)
        # predict_ensemble returns no 'sample' COORD, only the dim -- xr.concat
        # will not join unlabelled to labelled operands, so label it before it
        # meets anything else.
        ens = ens.assign_coords(sample=np.arange(ens.sizes['sample']))
        gen_t.append(ens.mean('sample'))
        ens_t.append(ens)
        omz_t.append(omz_mean(ens) / DEPTH_INT)          # (sample,)
        if target:
            truth_t.append(ds_sel[target].where(ocean))
            omz_truth_t.append(omz_mean(truth_t[-1]) / DEPTH_INT)   # scalar
    if not gen_t:
        return None
    gen = xr.concat(gen_t, 'time').mean('time')
    truth = xr.concat(truth_t, 'time').mean('time') if truth_t else None
    omz_gen_t = xr.concat(omz_t, 'time')                 # (time, sample)
    omz_truth_t = (xr.concat(omz_truth_t, 'time') if omz_truth_t else None)
    # ens_tavg (sample, lat, lon): the ensemble with TIME AVERAGED FIRST, which
    # is the population the PDF's points come from (omz_gen is its OMZ mean --
    # omz_mean is linear, so averaging time before or after is identical). Kept
    # because calibration is pointwise and CANNOT be applied to the reduced
    # scalars afterwards: <alpha*dev> != alpha*<dev> when alpha varies in space.
    ens_tavg = xr.concat(ens_t, 'time').mean('time')
    return dict(gen=gen, truth=truth, n_times=len(gen_t), ens_tavg=ens_tavg,
                omz_gen=omz_gen_t.mean('time'),          # (sample,)
                omz_gen_t=omz_gen_t.stack(case=('time', 'sample')),
                omz_truth=(float(omz_truth_t.mean('time'))
                           if omz_truth_t is not None else None),
                omz_truth_t=(omz_truth_t.values
                             if omz_truth_t is not None else None))


#: Gaussian smoothing width (grid points) for the alpha map.
OMZ_CALIB_SIGMA = 3.0


def omz_calibrated(res_tso, res_extra, truths, ocean, ctx):
    """OMZ-mean series from the VARIANCE-INFLATED ensemble.

    alpha is fitted by ``calibration.fit_alpha_map`` on the models that HAVE
    truth, then applied to every model including the extrapolated ones. That
    extrapolates the calibration itself, which is the same assumption the
    figure already makes about the generator; it is stamped as
    ``omz_calib_fit``.

    Two consequences worth knowing before reading the result:

    * inflation preserves the pointwise ensemble MEAN exactly, so the five maps
      and panels a-c of the OMZ figure are bit-identical either way -- only the
      WIDTH of the PDF curves moves, and ``omz_truth`` not at all;
    * it cannot be done downstream from the cached scalars, because alpha
      varies in space and ``<alpha*dev> != alpha*<dev>``. Hence ens_tavg.
    """
    from diffusion import calibration as calib

    gen = xr.concat([r['ens_tavg'] for r in res_tso + res_extra], 'model')
    fit = xr.Dataset(dict(
        o2_gen=xr.concat([r['ens_tavg'] for r in res_tso], 'model'),
        o2_true=xr.concat(truths, 'model')))
    alpha = calib.fit_alpha_map(fit, ocean,
                                sigma=ctx.get('calib_sigma', OMZ_CALIB_SIGMA))
    ens_mean = gen.mean('sample')
    cal = ens_mean + alpha * (gen - ens_mean)
    omz = omz_mean(cal) / DEPTH_INT                      # (model, sample)
    n_tso = len(res_tso)
    a = alpha.where(omz_mask() & ocean)
    logging.info(f'  OMZ calibrated: alpha over the OMZ '
                 f'{float(a.min()):.2f}-{float(a.max()):.2f} '
                 f'(mean {float(a.mean()):.2f}), fitted on {n_tso} model(s)')
    return dict(omz_tso_cal=omz.isel(model=slice(None, n_tso)).values.ravel(),
                omz_ts_cal=omz.values.ravel(),
                omz_alpha_omz_mean=float(a.mean()))


def compute_cell(cell, ctx):
    """The five maps for one cell, in mol/m^3 (relative panels in %)."""
    cfg = cell_config(cell)
    save_dir = run_save_dir(cfg)
    ep = resolve_epoch(save_dir, ctx['epoch'])
    if ep is None:
        logging.info(f'[skip] {cell_slug(cell)}: no checkpoint at epoch '
                     f'{ctx["epoch"]} in {save_dir.name}')
        return None
    run = load_run(save_dir, ctx['device'], ep, _compile=False)
    target = run.config['target']
    resolution = run.config['resolution']

    with_truth, extra = split_models(run)
    if ctx['max_models'] is not None:
        with_truth = with_truth[:ctx['max_models']]
        extra = extra[:ctx['max_models']]
    if not with_truth:
        logging.info(f'[skip] {cell_slug(cell)}: no model with ground-truth O2')
        return None
    logging.info(f'  {len(with_truth)} model(s) with O2, {len(extra)} with the '
                 f'predictors only')

    years = ctx['years'] if ctx['years'] is not None else list(range(
        ctx['year_window'][0], ctx['year_window'][1] + 1,
        ctx['years_stride'][resolution]))
    months = ctx['months']

    # ---- one common ocean mask, from the models that have (usable) truth ----
    ocean, with_truth, dropped = common_ocean_mask(
        run, with_truth, years, months, ctx)
    if ocean is None:
        logging.info(f'[skip] {cell_slug(cell)}: no usable common ocean mask')
        return None

    # ---- generate: models with truth first, then the extrapolation models ----
    gens_tso, truths, names_tso, res_tso = [], [], [], []
    for m in with_truth:
        res = model_gen(run, build_per_model_ds(run.config, m), ocean, years,
                        months, ctx['sample_cfg'], target=target)
        if res is None:
            dropped.append(f'{m.model} (no usable time step)')
            continue
        gens_tso.append(res['gen'])
        truths.append(res['truth'])
        names_tso.append(m.model)
        res_tso.append(res)
        logging.info(f'  [truth] {m.model}: {res["n_times"]} time(s)')

    gens_extra, names_extra, res_extra = [], [], []
    for m in extra:
        try:
            ds_model = build_predictors_only_ds(run.config, m)
            res = model_gen(run, ds_model, ocean, years, months,
                            ctx['sample_cfg'])
        except Exception as e:
            dropped.append(f'{m.model} ({type(e).__name__}: {e})')
            continue
        if res is None:
            dropped.append(f'{m.model} (no usable time step)')
            continue
        gens_extra.append(res['gen'])
        names_extra.append(m.model)
        res_extra.append(res)
        logging.info(f'  [extra] {m.model}: {res["n_times"]} time(s)')

    if not gens_tso:
        logging.info(f'[skip] {cell_slug(cell)}: no model produced both a '
                     f'generated and a truth field')
        return None
    if dropped:
        logging.info(f'  [drop] {len(dropped)}: {", ".join(dropped)}')

    truth = xr.concat(truths, 'model').mean('model') / DEPTH_INT
    gen_tso = xr.concat(gens_tso, 'model').mean('model') / DEPTH_INT
    gen_ts = (xr.concat(gens_tso + gens_extra, 'model').mean('model')
              / DEPTH_INT)

    # ---- OMZ populations: one flat point cloud per curve of the PDF figure ----
    res_all = res_tso + res_extra
    omz = dict(
        omz_truth=np.array([r['omz_truth'] for r in res_tso], dtype=float),
        omz_truth_all=np.concatenate([r['omz_truth_t'] for r in res_tso]),
        omz_tso=np.concatenate([r['omz_gen'].values for r in res_tso]),
        omz_tso_all=np.concatenate([r['omz_gen_t'].values for r in res_tso]),
        omz_ts=np.concatenate([r['omz_gen'].values for r in res_all]),
        omz_ts_all=np.concatenate([r['omz_gen_t'].values for r in res_all]),
        # Over the SAME domain as the curves (mask & this cell's ocean mask);
        # the mask-only value is kept beside it because it is the one number
        # that is identical across every cell.
        omz_obs=float(omz_mean(omz_obs().where(ocean))),
        omz_obs_mask_only=float(omz_mean(omz_obs())),
        n_omz=int((omz_mask() & ocean).sum()))
    omz.update(omz_calibrated(res_tso, res_extra, truths, ocean, ctx))
    n_omz = omz['n_omz']
    logging.info(f'  OMZ: {n_omz} of {int(omz_mask().sum())} mask cells; truth '
                 f'{omz["omz_truth"].mean():.4f}, gen(O2 models) '
                 f'{omz["omz_tso"].mean():.4f}, gen(all) '
                 f'{omz["omz_ts"].mean():.4f}, obs {omz["omz_obs"]:.4f} '
                 f'(mask only {omz["omz_obs_mask_only"]:.4f}) mol/m^3')

    return dict(truth=truth, gen_tso=gen_tso, gen_ts=gen_ts,
                rel_recon=100.0 * (gen_tso - truth) / truth,
                rel_corr=100.0 * (gen_ts - gen_tso) / gen_tso,
                n_tso=len(gens_tso), n_ts=len(gens_tso) + len(gens_extra),
                models_tso=names_tso, models_extra=names_extra,
                models_skipped=dropped, n_valid=int(ocean.sum()), epoch=ep,
                **omz)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
MAPS = ('truth', 'gen_tso', 'gen_ts', 'rel_recon', 'rel_corr')

# The OMZ point clouds, each on its own dim (the three populations have
# different lengths). '<name>' = per (model, sample), time averaged first;
# '<name>_all' = every (model, sample, time) point. The flat per-(model,
# sample) series are regular ``(n_models, n_samples)`` once reshaped, in the
# order of the stamped ``models_tso`` (+ ``models_extra`` for ``omz_ts``).
OMZ_SERIES = ('omz_truth', 'omz_tso', 'omz_ts',
              'omz_truth_all', 'omz_tso_all', 'omz_ts_all',
              # variance-inflated counterparts of omz_tso / omz_ts. No
              # omz_truth_cal: inflation acts on the generated ensemble only,
              # and no *_all_cal, since the calibrated series is built from the
              # time-averaged ensemble (see omz_calibrated).
              'omz_tso_cal', 'omz_ts_cal')


def cache_path(cell):
    return CACHE_DIR / f'{cell_slug(cell)}.nc'


def years_label(cell, ctx):
    """What was actually averaged in time -- never a window the run didn't sample."""
    if ctx['years'] is not None:
        return 'yrs ' + ','.join(str(y) for y in ctx['years'])
    lo, hi = ctx['year_window']
    stride = ctx['years_stride'][cell['resolution']]
    return f'yrs {lo}-{hi}' + (f' /{stride}' if stride != 1 else '')


def save_cache(cell, d, ctx):
    ds = xr.Dataset({k: d[k] for k in MAPS})
    for k in OMZ_SERIES:
        ds[k] = xr.DataArray(np.asarray(d[k], dtype=float), dims=(f'n_{k}',))
    ds.attrs.update(
        omz=OMZ_KEY, omz_obs=d['omz_obs'], omz_ref=OMZ_REF,
        omz_calib_fit=f'{d["n_tso"]} models with truth, applied to '
                      f'{d["n_ts"]}', omz_calib_sigma=OMZ_CALIB_SIGMA,
        omz_alpha_omz_mean=d['omz_alpha_omz_mean'],
        omz_obs_mask_only=d['omz_obs_mask_only'], n_omz_cells=d['n_omz'],
        cell=cell_slug(cell), title=cell_title(cell),
        predictors=' '.join(cell['predictors']), resolution=cell['resolution'],
        field_type=cell['field_type'], experiment=cell['experiment'],
        n_tso=d['n_tso'], n_ts=d['n_ts'], n_valid_cells=d['n_valid'],
        models_tso=' '.join(d['models_tso']),
        models_extra=' '.join(d['models_extra']),
        models_skipped='; '.join(d['models_skipped']),
        models_excluded=' '.join(EXCLUDED_MODELS),
        epoch=d['epoch'], epoch_arg=str(ctx['epoch']),
        n_samples=ctx['sample_cfg'].n_samples, steps=ctx['sample_cfg'].steps,
        sampler=ctx['sample_cfg'].sampler,
        year_window=' '.join(str(y) for y in ctx['year_window']),
        years_stride=ctx['years_stride'][cell['resolution']],
        years_label=years_label(cell, ctx),
        n_months=len(ctx['months']) if cell['resolution'] == 'monthly' else 1)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(cache_path(cell))
    logging.info(f'  cached {cache_path(cell)}')
    return ds


def load_cache(cell):
    path = cache_path(cell)
    return xr.open_dataset(path).load() if path.is_file() else None


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def plot_cell(cell, ds, o2_max=None, rel_limit=None, layout=None,
              hratios=None):
    n_tso = int(ds.attrs.get('n_tso', 0))
    n_ts = int(ds.attrs.get('n_ts', 0))
    preds = pred_label(cell['predictors'])
    labels = {
        'truth': f'True {ec.O2}\nfrom {n_tso} models with {preds}/{ec.O2}',
        'gen_tso': f'Generated {ec.O2}\nfrom the same {n_tso} models',
        'gen_ts': f'Generated {ec.O2}\nfrom all {n_ts} models with {preds}',
        'rel_recon': 'Relative reconstruction error\n[b) $-$ a)]',
        'rel_corr': f'Relative {ec.O2} correction\nfrom the {n_ts - n_tso} '
                    f'extra models [c) $-$ b)]',
    }
    bits = [ds.attrs.get('title', cell_title(cell)),
            str(ds.attrs.get('years_label', '')),
            f'{ds.attrs.get("n_valid_cells", "?")} cells',
            f'ep {ds.attrs.get("epoch", "?")}']
    suptitle = 'Extrapolation  |  ' + '  |  '.join(b for b in bits if b)
    return ec.extrapolation_panel(
        {k: ds[k] for k in MAPS}, suptitle,
        OUT_DIR / f'{cell_slug(cell)}.png',
        o2_max=o2_max, rel_limit=rel_limit, labels=labels, layout=layout,
        hratios=hratios)


# ===========================================================================
# entry points
# ===========================================================================
def compute(epoch=DEFAULT_EPOCH):
    """Sample the main cell and write its cache, which Figs. 9 and 10 share
    (5 samples, 100 DDIM steps, years 81-100 every 2nd year)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = dict(device=device,
               sample_cfg=SampleCfg(n_samples=5, sampler='ddim', steps=100),
               year_window=DEFAULT_YEAR_WINDOW, years=None,
               months=list(range(12)), years_stride=DEFAULT_STRIDE,
               max_models=None, epoch=epoch)
    d = compute_cell(CELL, ctx)
    if d is None:
        raise SystemExit(f'{cell_slug(CELL)}: nothing computed')
    return save_cache(CELL, d, ctx)


def plot():
    ds = load_cache(CELL)
    if ds is None:
        raise SystemExit(f'no cache at {cache_path(CELL)}; run compute first')
    fig = plot_cell(CELL, ds, o2_max=EXTRAP_O2_MAX, rel_limit=EXTRAP_REL_LIMIT,
                    layout=EXTRAP_LAYOUT, hratios=EXTRAP_HRATIOS)
    # abc off: ec.extrapolation_panel letters its own axes, and the panel
    # titles refer to those letters.
    render_common.finish(fig, abc=False)
    return render_common.save(fig, 'extrapolation')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    p.add_argument('--epoch', default=str(DEFAULT_EPOCH),
                   help="checkpoint epoch, or 'latest' (default 500)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        compute('latest' if args.epoch.lower() == 'latest' else int(args.epoch))
    else:
        plot()
