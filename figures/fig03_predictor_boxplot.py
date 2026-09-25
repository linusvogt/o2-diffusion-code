"""Fig. 3 -- per-model global-mean bias of generated O2 for each predictor set.

One x position per predictor set, one coloured dash per CMIP6 model:
``y = 100 * (<gen> - <truth>) / <truth>`` with ``<>`` the area-weighted global
mean of that model's time-mean field (annual depth-integrated in-sample runs,
years 81-100). This is a signed, spatially-averaged-first bias, so regional
over/under-prediction cancels by design. The y-axis breaks automatically when a
few models sit far off the rest (``find_break``).

Extracted from general/eval/predictor_boxplot.py (compute and plot), with
``pred_slug``/``pred_label``/``resolve_epoch`` from general/eval/error_maps.py.

    python -m figures.fig03_predictor_boxplot compute [--epoch 500]   # GPU
    python -m figures.fig03_predictor_boxplot plot                    # CPU
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from figures import render_common

import numpy as np
import torch
import xarray as xr

import paths
from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, get_models,
    build_per_model_ds, model_timemean, SampleCfg, DEPTH_INT, FIG_DIR,
    DEFAULT_YEAR_WINDOW)


# ---------------------------------------------------------------------------
# pinned manuscript constants (main variant)
# ---------------------------------------------------------------------------
# Main variant: annual depth-integrated cell, relative (%) bias, 5-member
# ensembles, epoch 500.
MAIN_CELL = dict(resolution='annual', field_type='depthint')
WHICH = 'rel'
N_SAMPLES = 5
STEPS = 100
SAMPLER = 'ddim'


# ---------------------------------------------------------------------------
# from error_maps.py
# ---------------------------------------------------------------------------

# Default checkpoint epoch: 500 is the largest epoch reached by every axis of
# the sweep (the monthly runs are the laggards), so pinning it keeps training
# length out of any figure-to-figure comparison. '--epoch latest' opts out.
DEFAULT_EPOCH = 500

# Monthly runs cost 12x an annual one per year, so they default to a coarser
# year stride (as in seasonal_cycle.py).
DEFAULT_STRIDE = {'annual': 1, 'monthly': 5}

_SHORT_PRED = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': 'taux',
               'tauvo': 'tauy', 'mld': 'MLD'}
_PRED_TEX = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}


def pred_slug(preds):
    return '-'.join(_SHORT_PRED.get(p, p) for p in preds)


def pred_label(preds):
    return '+'.join(_PRED_TEX.get(p, p) for p in preds)


def latest_epoch(save_dir):
    """Highest ``ckpt_epoch###.pt`` epoch in ``save_dir``, or None if empty."""
    epochs = [int(m.group(1))
              for p in Path(save_dir).glob('ckpt_epoch*.pt')
              if (m := re.search(r'ckpt_epoch(\d+)\.pt$', p.name))]
    return max(epochs) if epochs else None


def resolve_epoch(save_dir, epoch):
    """Concrete epoch for a run: ``epoch`` itself, or its newest checkpoint when
    ``epoch`` is 'latest'/None. Returns None when the run has no checkpoint at
    all (or not that one), so callers can skip-and-log."""
    if epoch in (None, 'latest'):
        return latest_epoch(save_dir)
    return int(epoch) if run_exists(save_dir, int(epoch)) else None


# ---------------------------------------------------------------------------
# predictor_boxplot
# ---------------------------------------------------------------------------
OUT_DIR = FIG_DIR / 'predictor_boxplot'
CACHE_DIR = OUT_DIR / 'cache'

BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# The sweep's 5 predictor sets, ordered as in the original figure: plain T+S,
# then the physical add-ons by size, then the biogeochemical one. Predictors use
# the ABSTRACT 'mld' token -- run_save_dir resolves it to mld_max (annual) /
# mld (monthly) while the labels stay parallel across resolutions.
PREDICTOR_SETS = [
    ['thetao', 'so'],
    ['thetao', 'so', 'mld'],
    ['thetao', 'so', 'tauuo', 'tauvo'],
    ['thetao', 'so', 'tauuo', 'tauvo', 'mld'],
    ['thetao', 'so', 'dissic'],
]

# Spatial domains to reduce over. 'global' is the whole ocean; the rest are the
# RECCAP2 basin masks used everywhere else in this project (util.basin_mean),
# regridded to the same r360x180 grid. ALL of them are computed and cached per
# run -- only sampling costs anything -- so --basins picks at plot time.
BASINS = ['global', 'atlantic', 'pacific', 'indian', 'arctic', 'southern']
GRIDAREA_FILE = str(paths.GRIDAREA)
REGION_MASK_FILE = str(paths.RECCAP2_MASK)

# Aggregation, stamped into every cache -- so a cache can never be mistaken for
# error_maps' 'model-mean-abs' magnitudes.
METRIC = 'global-mean-bias-per-model'


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def cell_slug(cell):
    """Filename stem for a cell -- stable, so cache/figure names don't drift."""
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    return f'{res}_{fld}'


def cell_title(cell):
    fld = {'depthint': 'depth-integrated', 'density': 'density layers'}[
        cell['field_type']]
    return f"{cell['resolution']} | {fld} | in-sample"


def cell_config(cell, predictors):
    return make_config(BASE_CONFIG, predictors=list(predictors),
                       resolution=cell['resolution'],
                       field_type=cell['field_type'])


# ---------------------------------------------------------------------------
# per-cell computation
# ---------------------------------------------------------------------------
def _common_model_names(cell, epoch, max_models=None):
    """Model names available to EVERY predictor set of a cell, at ``epoch``.

    Cheap: ``get_models`` only globs filenames, no netCDF is read. Intersecting
    on *names* (not on the ``ModelCMIP6`` objects) matters because each config's
    ``get_models`` picks its own member for a model -- we re-resolve the object
    per predictor set later. The intersection is per cell, so density's dropped
    sigma members cannot thin the depthint figure.
    """
    common = None
    for preds in PREDICTOR_SETS:
        cfg = cell_config(cell, preds)
        if resolve_epoch(run_save_dir(cfg), epoch) is None:
            continue          # no run: this set won't be drawn, don't constrain
        names = {m.model for m in get_models(cfg)}
        common = names if common is None else (common & names)
    names = sorted(common or [])
    return names[:max_models] if max_models is not None else names


_WEIGHT_CACHE = {}


def _basin_weight(basin):
    """Area weights (lat, lon) for ``basin``, zero outside it. Cached.

    Reproduces ``util.basin_mean``'s math exactly -- it does
    ``da.where(mask > 0).weighted(area.fillna(0)).mean()``, and zeroing the
    weight outside the basin instead of NaN-ing the field gives the identical
    ratio (a zero weight contributes to neither the numerator nor the
    denominator). Done here rather than by calling ``util.basin_mean`` because
    that reopens the area *and* region-mask files on every call: 6 basins x 4
    reductions x every (predictor set, model) is ~1700 file opens per cell.
    """
    if basin not in _WEIGHT_CACHE:
        area = xr.open_dataset(GRIDAREA_FILE).cell_area.fillna(0.0)
        if basin != 'global':
            mask = xr.open_dataset(REGION_MASK_FILE)[basin]
            area = area.where(mask > 0, 0.0)
        _WEIGHT_CACHE[basin] = area.reset_coords(drop=True).load()
    return _WEIGHT_CACHE[basin]


def _wmean(da, basin):
    """Area-weighted mean of ``da`` over ``basin``; NaN if the basin is empty."""
    w = _basin_weight(basin)
    if float(w.where(da.notnull()).sum()) == 0.0:
        return np.nan          # no valid cell of this model falls in the basin
    return float(da.weighted(w).mean(['lat', 'lon']).values)


def _reduce(gen, truth):
    """Per-basin gen/truth mean scalars (mol/m^3), unmasked and deep-only.

    ``gen``/``truth`` are per-model time-mean fields in RAW depth-integrated
    units (that is what ``model_timemean`` returns -- ``skill_map`` applies the
    /DEPTH_INT only to its model-mean output), so divide here.

    Both fields are first restricted to the cells valid in BOTH. They should
    already share a mask (``predict_ensemble`` masks on the o2 target), but the
    two are reduced to scalars *independently*, so any one-sided NaN would enter
    one mean and not the other and show up as a spurious bias proportional to
    the mismatch area.

    Every basin is reduced here, not just the requested one: sampling is the
    whole cost, so caching all of them lets ``--reuse --basins southern`` replot
    without a GPU (same reasoning as the deep/full pair).
    """
    gen = gen / DEPTH_INT
    truth = truth / DEPTH_INT
    valid = np.isfinite(gen) & np.isfinite(truth)
    gen, truth = gen.where(valid), truth.where(valid)
    deep = ec.mask_shallow(valid.astype(float)).notnull() & valid
    gen_d, truth_d = gen.where(deep), truth.where(deep)
    out = {}
    for b in BASINS:
        w = _basin_weight(b)
        out[b] = dict(
            gen_mean=_wmean(gen, b),
            truth_mean=_wmean(truth, b),
            gen_mean_deep=_wmean(gen_d, b),
            truth_mean_deep=_wmean(truth_d, b),
            n_valid_cells=int((valid & (w > 0)).sum()),
            n_valid_cells_deep=int((deep & (w > 0)).sum()))
    return out


def compute_cell(cell, ctx):
    """Per-(predictor set, model) global-mean gen/truth scalars for one cell.

    Returns a dict of (pred, model)-shaped arrays plus provenance, or None if no
    predictor set of this cell has a usable run.
    """
    want_models = (None if ctx['all_models']
                   else _common_model_names(cell, ctx['epoch'], ctx['max_models']))
    if want_models is not None and not want_models:
        logging.info(f'[skip] {cell_slug(cell)}: no model common to all '
                     f'predictor sets (use --all-models to draw anyway)')
        return None

    stride = ctx['years_stride'][cell['resolution']]
    rows, pred_labels, pred_slugs, epochs = {}, [], [], []
    used_models, land_fills = [], []
    for preds in PREDICTOR_SETS:
        cfg = cell_config(cell, preds)
        save_dir = run_save_dir(cfg)
        ep = resolve_epoch(save_dir, ctx['epoch'])
        if ep is None:
            logging.info(f'  [skip] {pred_slug(preds)}: no checkpoint at epoch '
                         f'{ctx["epoch"]} in {save_dir.name}')
            continue
        run = load_run(save_dir, ctx['device'], ep, _compile=False)
        models = get_models(run.config)
        if want_models is not None:
            models = [m for m in models if m.model in want_models]
        elif ctx['max_models'] is not None:
            models = models[:ctx['max_models']]
        if not models:
            logging.info(f'  [skip] {pred_slug(preds)}: no test models with data')
            continue

        # The land-fill convention actually used at sampling time (from the
        # checkpoint sidecar). Stamped so a figure can be traced back to it --
        # a mis-stamped run generates a collapsed field and a wild bias.
        land_fills.append(str(run.scaling.get('land_fill', 'zero')))

        col = {}
        for model in models:
            ds_model = build_per_model_ds(run.config, model)
            res = model_timemean(run, ds_model, ctx['year_window'],
                                 ctx['sample_cfg'], months=ctx['months'],
                                 years_stride=stride, years=ctx['years'])
            if res is None or res['truth'] is None:
                logging.info(f'    [warn] no times for {model.model}')
                continue
            r = _reduce(res['gen'], res['truth'])
            for b in BASINS:
                r[b]['n_times'] = res['n_times']
            col[model.model] = r
            g, t = r['global']['gen_mean'], r['global']['truth_mean']
            logging.info(f'    {model.model}: {res["n_times"]} times, '
                         f'global bias {100 * (g - t) / t:+.2f}%')
        if not col:
            logging.info(f'  [skip] {pred_slug(preds)}: no model produced times')
            continue
        rows[pred_slug(preds)] = col
        pred_slugs.append(pred_slug(preds))
        pred_labels.append(pred_label(preds))
        epochs.append(ep)
        used_models.append(set(col))

    if not rows:
        logging.info(f'[skip] {cell_slug(cell)}: no run usable')
        return None

    # Model axis: the union of what each column produced, sorted -- a model that
    # dropped out of one column stays NaN there rather than shifting the others.
    model_names = sorted(set().union(*used_models))
    fields = ('gen_mean', 'truth_mean', 'gen_mean_deep', 'truth_mean_deep',
              'n_valid_cells', 'n_valid_cells_deep', 'n_times')
    data = {f: np.full((len(BASINS), len(pred_slugs), len(model_names)), np.nan)
            for f in fields}
    for b, basin in enumerate(BASINS):
        for i, ps in enumerate(pred_slugs):
            for j, mn in enumerate(model_names):
                if mn in rows[ps]:
                    for f in fields:
                        data[f][b, i, j] = rows[ps][mn][basin][f]

    return dict(data=data, pred_slugs=pred_slugs, pred_labels=pred_labels,
                model_names=model_names, epochs=epochs,
                land_fills=sorted(set(land_fills)))


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(cell):
    return CACHE_DIR / f'{cell_slug(cell)}.nc'


def years_label(cell, ctx):
    """What was actually averaged in time -- explicit years, or window+stride
    (the window alone would mislabel a strided or --years run)."""
    if ctx['years'] is not None:
        return 'yrs ' + ','.join(str(y) for y in ctx['years'])
    lo, hi = ctx['year_window']
    stride = ctx['years_stride'][cell['resolution']]
    return f'yrs {lo}-{hi}' + (f' /{stride}' if stride != 1 else '')


def save_cache(cell, d, ctx):
    """Persist the (pred, model) scalars + provenance so --reuse replots free."""
    coords = {'basin': BASINS, 'pred': d['pred_slugs'],
              'model': d['model_names']}
    ds = xr.Dataset(
        {k: (('basin', 'pred', 'model'), v) for k, v in d['data'].items()},
        coords=coords)
    ds.attrs.update(
        cell=cell_slug(cell), title=cell_title(cell),
        # x-axis labels carry TeX ($\tau_x$); keep them out of a string DATA VAR
        # (netCDF string encoding round-trips to bytes on some engines) -- '|'
        # cannot occur in a label, so it is a safe separator.
        pred_labels=' | '.join(d['pred_labels']),
        resolution=cell['resolution'], field_type=cell['field_type'],
        split='insample',
        n_models=len(d['model_names']), models=' '.join(d['model_names']),
        epochs=' '.join(str(e) for e in d['epochs']),
        epoch_arg=str(ctx['epoch']),
        land_fill=' '.join(d['land_fills']),
        n_samples=ctx['sample_cfg'].n_samples, steps=ctx['sample_cfg'].steps,
        sampler=ctx['sample_cfg'].sampler,
        years_stride=ctx['years_stride'][cell['resolution']],
        year_window=' '.join(str(y) for y in ctx['year_window']),
        years_label=years_label(cell, ctx),
        n_months=len(ctx['months']) if cell['resolution'] == 'monthly' else 1,
        common_models=int(not ctx['all_models']),
        basins=' '.join(BASINS),
        metric=METRIC)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(cell)
    ds.to_netcdf(path)
    logging.info(f'  cached {path}')
    return ds


def load_cache(cell):
    path = cache_path(cell)
    return xr.open_dataset(path).load() if path.is_file() else None


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def bias(ds, which, deep=False, basin='global'):
    """(pred, model) bias DataArray: '%' for ``rel``, mol/m^3 for ``abs``."""
    if 'basin' in ds.dims:
        ds = ds.sel(basin=basin)
    elif basin != 'global':      # pre-basin cache: only the global reduction
        raise KeyError(f'cache has no basin dimension -- {basin!r} needs a '
                       f'recompute (drop --reuse)')
    suf = '_deep' if deep else ''
    gen, truth = ds['gen_mean' + suf], ds['truth_mean' + suf]
    if which == 'rel':
        return 100.0 * (gen - truth) / truth
    if which == 'abs':
        return gen - truth
    raise ValueError(which)


# --- broken y-axis -----------------------------------------------------------
# Thresholds for find_break. Calibrated on the 4 caches x {rel, abs} x 6 basins
# x {full, deep}: the 8 default (global) figures have gap/core-span ratios
# 1.44-15.5 and a 4-5 point far group, while the figures that must NOT break
# sit at 0.26-1.03 -- except monthly-depthint arctic, whose largest gap splits a
# BIMODAL fleet (12 of 50 points, far group itself 4x wider than the gap), which
# is what MAX_FAR_SPAN catches.
BREAK_MIN_GAP = 1.0        # gap >= this * span(core group)
BREAK_MAX_FAR_SPAN = 1.0   # gap >= this * span(far group): a wide far group is
                           # a second mode, not an outlier
BREAK_MAX_FAR_FRAC = 0.25  # far group must be a small minority
BREAK_MIN_POINTS = 8       # below this, "outlier" is not a meaningful call
BREAK_PAD = 0.10           # padding of each panel's data range, as a fraction
BREAK_FRAC_LIM = (0.15, 0.35)   # far panel's share of the axes height


def find_break(values, min_gap=BREAK_MIN_GAP, max_far_span=BREAK_MAX_FAR_SPAN,
               max_far_frac=BREAK_MAX_FAR_FRAC, min_points=BREAK_MIN_POINTS):
    """Detect an outlier group to cut out of the y-axis, or None.

    ``values`` is the POOLED set of plotted y values (every predictor set x
    model of one figure) -- the columns share one y-axis, so a per-column
    detection could not be drawn.

    The break interval must be **empty**, so this is a gap statistic, not an
    IQR/MAD fence (a fence flags a point as unusual without telling you the
    interval below it is free of data). Take the largest gap between consecutive
    sorted values, call the more populous side the ``core`` and the other the
    ``far`` group, and require all three guards:

    * ``gap >= min_gap * span(core)``  -- the cut is worth making;
    * ``gap >= max_far_span * span(far)`` -- the far group is a tight cluster,
      not a second mode (without this, a bimodal fleet gets drawn as if the
      larger half were a handful of outliers);
    * ``len(far) <= max_far_frac * n``  -- and it is a small minority.

    Only the single largest gap is considered: a break at a runner-up gap would
    cut somewhere the eye does not read as empty.
    """
    v = np.sort(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    n = len(v)
    if n < min_points:
        return None
    gaps = np.diff(v)
    i = int(np.argmax(gaps))          # split between v[i] and v[i + 1]
    gap = float(gaps[i])
    lo_grp, hi_grp = v[:i + 1], v[i + 1:]
    far_above = len(lo_grp) >= len(hi_grp)
    core, far = (lo_grp, hi_grp) if far_above else (hi_grp, lo_grp)
    core_span = float(core[-1] - core[0])
    far_span = float(far[-1] - far[0])
    if gap <= 0 or (core_span > 0 and gap < min_gap * core_span):
        return None
    if far_span > 0 and gap < max_far_span * far_span:
        return None
    if len(far) > max_far_frac * n:
        return None
    return dict(split=(float(v[i]), float(v[i + 1])), far_above=far_above,
                core=(float(core[0]), float(core[-1])),
                far=(float(far[0]), float(far[-1])),
                n_far=int(len(far)), n=n, gap=gap,
                core_ratio=(gap / core_span if core_span > 0 else np.inf))


def break_limits(br, pad=BREAK_PAD, frac_lim=BREAK_FRAC_LIM):
    """(core_ylim, far_ylim, far_height_fraction) for a ``find_break`` result.

    Each panel is padded by ``pad`` of its own data range -- but a far group of
    one point has zero range, so both panels fall back to a fraction of the core
    span. The heights are proportional to the padded spans so the two panels
    keep a comparable data-per-pixel scale, then clipped: an unclipped
    single-point far panel collapses to a sliver (and a wide one would eat the
    core panel the break exists to enlarge).

    The core panel is then stretched to include y=0. The metric is a SIGNED
    bias read against that line ("tight around 0" = no net bias), and several
    figures -- the monthly-density cells, whose whole core sits at -0.5..0% --
    have an all-negative core that would otherwise be drawn without it.
    """
    # 0 joins the core range BEFORE padding, so the line gets the same headroom
    # as the data instead of sitting on the spine.
    core_lo, core_hi = min(br['core'][0], 0.0), max(br['core'][1], 0.0)
    core_span = core_hi - core_lo
    far_span = br['far'][1] - br['far'][0]
    ref = max(core_span, far_span) or abs(br['split'][1]) or 1.0
    core_pad = pad * (core_span or ref)
    far_pad = pad * (far_span or ref)
    core_lim = (core_lo - core_pad, core_hi + core_pad)
    far_lim = (br['far'][0] - far_pad, br['far'][1] + far_pad)
    # ... and the core panel is then clamped to the middle of the gap, so it can
    # never grow into the empty interval the break exists to remove (does not
    # bind on today's data, but a far group close to 0 would).
    mid = br['split'][0] + br['gap'] / 2
    core_lim = ((core_lim[0], min(core_lim[1], mid)) if br['far_above']
                else (max(core_lim[0], mid), core_lim[1]))
    core_h, far_h = core_lim[1] - core_lim[0], far_lim[1] - far_lim[0]
    frac = float(np.clip(far_h / (core_h + far_h), *frac_lim))
    return core_lim, far_lim, frac


def _draw_break_marks(ax_top, ax_bot, size=9):
    """Slanted cut marks on the two facing spines.

    Called AFTER ``format()``: ultraplot restyles spines there, so hiding the
    facing ones earlier would be undone.
    """
    ax_top.spines['bottom'].set_visible(False)
    ax_bot.spines['top'].set_visible(False)
    ax_top.tick_params(axis='x', bottom=False, labelbottom=False, which='both')
    ax_bot.tick_params(axis='x', top=False, which='both')
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=size, linestyle='none',
              color='k', mec='k', mew=1, clip_on=False)
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **kw)


def far_models(br, da, models):
    """Names of the models that occupy the FAR panel in at least one column.

    Same membership test as :func:`break_report` (a value on the far side of
    the split), factored out because :func:`straddlers` needs the set, not the
    sentence.
    """
    v = da.values
    sel = v >= br['far'][0] if br['far_above'] else v <= br['far'][1]
    return {models[j] for j in np.where(np.isfinite(v) & sel)[1]}


def straddlers(br, da, models):
    """``[(pred_index, model, value)]`` -- far-panel models sitting in the core.

    A model that occupies the far panel in most columns can drop into the core
    panel in one of them, so the far panel reads dash, dash, dash, dash, BLANK
    -- and a reader tracking that row cannot tell "moved into the core" from
    "not evaluated". Here the blank *is* the result (adding DIC collapses
    CanESM5-CanOE's +5..+8% bias to ~0%). The figure does not mark it; this
    feeds :func:`straddle_report`, a log line naming the moves.

    Deliberately a general rule over far-panel occupancy rather than a named
    model: ``CMCC-ESM2`` straddles in ``ann_dens`` too, and ``ann_int``'s split
    sits one rounding step from CanESM5-CanOE's own value, so which models are
    far is not stable enough to hardcode.
    """
    far = far_models(br, da, models)
    v = da.values
    is_far = v >= br['far'][0] if br['far_above'] else v <= br['far'][1]
    out = []
    for pp in range(v.shape[0]):
        for mm, model in enumerate(models):
            y = v[pp, mm]
            if model in far and np.isfinite(y) and not is_far[pp, mm]:
                out.append((pp, model, float(y)))
    return out


def straddle_report(br, da, models, labels=None):
    """One line naming the far-panel models that sit in the core, or ''.

    The figure does not mark them, so this log line is the only place the
    move is reported.
    """
    items = straddlers(br, da, models)
    if not items:
        return ''

    def col(pp):
        return labels[pp] if labels and pp < len(labels) else f'col {pp + 1}'
    parts = [f'{model} @ {col(pp)} = {y:+.4g}' for pp, model, y in items]
    return f'straddles the break: {"; ".join(parts)}'


def break_report(br, da, models):
    """One-line description of an applied break, naming the far-side models."""
    lo, hi = br['split']
    side = 'above' if br['far_above'] else 'below'
    v = da.values
    sel = v >= br['far'][0] if br['far_above'] else v <= br['far'][1]
    names = sorted({models[j] for j in np.where(np.isfinite(v) & sel)[1]})
    return (f'break ({lo:.4g}, {hi:.4g}): {br["n_far"]}/{br["n"]} point(s) '
            f'{side}, gap {br["gap"]:.4g} = {br["core_ratio"]:.2f}x core span '
            f'-- {", ".join(names)}')


def _epoch_tag(ds):
    """'ep 500' or 'ep 500-975' -- columns can mix epochs under --epoch latest."""
    eps = sorted({int(e) for e in str(ds.attrs.get('epochs', '')).split() if e})
    if not eps:
        return ''
    return f'ep {eps[0]}' if len(eps) == 1 else f'ep {eps[0]}-{eps[-1]}'


def model_colors(models):
    """{model: colour}, assigned by POSITION in ``models``.

    Split out of :func:`plot_cell` because a COMBINED figure must colour a model
    the same in every row, and the colour is an index into the cell's own model
    list: the four cells have different fleets (10/12/8/10 models), so stacking
    them on a per-cell mapping would give one model a different colour in each
    row and the shared legend would be a lie. The combined caller builds one
    mapping over the union; ``plot_cell`` builds it over its own list, which
    reproduces the previous ``colors[mm % len(colors)]`` exactly.
    """
    import ultraplot as pplt
    colors = pplt.get_colors('tab20')
    return {m: colors[i % len(colors)] for i, m in enumerate(models)}


def block_ylabel(which, deep=False, basin='global'):
    """'Global mean error (%)' -- the y label for one block."""
    unit = '%' if which == 'rel' else 'mol/' + r'$\mathrm{m}^3$'
    domain = 'Global' if basin == 'global' else basin.title()
    if deep:
        domain += ' (>2000 m)'
    return f'{domain} mean error ({unit})'


def block_shape(ds, which='rel', deep=False, basin='global', ylim=None,
                allow_break=True):
    """``(n_axes, hratios, br, ylims)`` for this cell's block.

    Pure and cache-only -- ``find_break`` reads the values that would be
    plotted, nothing else -- so a caller can ask how many gridspec ROWS a
    variant needs *before* creating any figure. That is what a combined figure
    has to do: a block here is 1 or 2 rows depending on whether the break fires
    for that particular variant, so the master gridspec cannot be a fixed array
    the way the map figures' can, and its ``hratios`` have to be assembled from
    the per-variant answers.
    """
    da = bias(ds, which, deep=deep, basin=basin)
    fixed = tuple(ylim) if (ylim is not None and which == 'rel') else None
    br = None if (fixed or not allow_break) else find_break(da.values.ravel())
    if br is None:
        return 1, (1.0,), None, [fixed]
    core_lim, far_lim, frac = break_limits(br)
    hratios = (frac, 1 - frac) if br['far_above'] else (1 - frac, frac)
    lims = (far_lim, core_lim) if br['far_above'] else (core_lim, far_lim)
    return 2, tuple(hratios), br, list(lims)


def draw_block(axs, ds, which='rel', deep=False, basin='global', ylim=None,
               allow_break=True, colors=None, xticklabels=True, ylabel='',
               log_break=True):
    """Scatter one cell's dashes into ``axs``; return ``{'handles', 'br'}``.

    The drawing half of :func:`plot_cell`, so a combined figure can stack one
    cell per row without reimplementing the scatter (see
    ``eval_common.rel_error_maps`` for the pattern).

    It deliberately does NOT draw the break marks: those must be drawn after the
    last ``fig.format()``, which restyles spines, and the caller is the only one
    that knows when that is. ``br`` comes back so the caller can call
    :func:`_draw_break_marks` at the right moment.

    Every point is scattered on BOTH axes of a broken pair and the per-axes
    ``ylim`` decides where it shows -- as before, because the alternative (each
    point on "its" panel only) would split the ``handles`` dict that drives the
    legend and the model count.
    """
    from matplotlib.ticker import MaxNLocator

    n_axes, _, br, lims = block_shape(ds, which=which, deep=deep, basin=basin,
                                      ylim=ylim, allow_break=allow_break)
    if len(axs) != n_axes:
        raise ValueError(f'block needs {n_axes} axes, got {len(axs)}')
    da = bias(ds, which, deep=deep, basin=basin)
    labels = str(ds.attrs.get('pred_labels', '')).split(' | ')
    if len(labels) != ds.sizes['pred']:      # cache without the attr
        labels = [str(p) for p in ds['pred'].values]
    models = [str(m) for m in ds['model'].values]
    if colors is None:
        colors = model_colors(models)
    if br is not None and log_break:
        logging.info(f'  {which}: {break_report(br, da, models)}')
        msg = straddle_report(br, da, models, labels)
        if msg:
            logging.info(f'  {which}: {msg}')

    handles = {}
    for ax in axs:
        for pp, _ in enumerate(ds['pred'].values):
            for mm, model in enumerate(models):
                y = float(da.isel(pred=pp, model=mm))
                if not np.isfinite(y):
                    continue    # model absent from this predictor set's run
                handles[model] = ax.scatter(
                    x=pp + 1, y=y, color=colors[model], marker='_', label=model)

    # A model can sit on the cache's `model` coordinate and draw NOTHING: the
    # IPSL members' depth-integrated regrids are entirely NaN, so their bias is
    # non-finite in every column and they never enter `handles`. That is the
    # difference between the cached and the drawn fleet, so say it out loud
    # rather than let the next reader infer a fleet from the coordinate.
    absent = [m for m in models if m not in handles]
    if absent:
        logging.info(f'  {which}: {len(absent)}/{len(models)} cached model(s) '
                     f'draw nothing (all-NaN in every column): '
                     f'{", ".join(absent)}')

    far_ax = None if br is None else axs[0 if br['far_above'] else 1]
    for i, (ax, lim) in enumerate(zip(axs, lims)):
        bottom = i == len(axs) - 1
        ax.format(
            xticks=list(range(1, len(labels) + 1)),
            xticklabels=(labels if (bottom and xticklabels) else []),
            xrotation=45,
            # PINNED, not autoscaled. `_draw_break_marks` plots a two-point line
            # in transAxes coordinates, which perturbs the x autoscale --
            # differently per cell, and enough to put columns 1 and 5 ON the
            # spines with their dashes half cut off. Half a column of pad
            # on each side is what a categorical axis wants anyway, and setting
            # it here disables autoscalex so nothing drawn later can move it.
            xlim=(0.5, len(labels) + 0.5),
            ylabel=ylabel,
            # the far panel is a sliver; at the default tick density a wide
            # outlier range crowds it with overlapping labels. `steps` is
            # spelled out because the unconstrained locator picks increments
            # like 0.6 for these odd ranges.
            **(dict(ylocator=MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
               if ax is far_ax else {}),
        )
        if lim is not None:
            ax.format(ylim=tuple(lim))
        if lim is None or lim[0] <= 0 <= lim[1]:
            ax.axhline(ls='--', color='grey5')
    return dict(handles=handles, br=br)


def plot_cell(cell, ds, which, deep=False, basin='global', ylim=None,
              allow_break=True):
    """The boxplot figure: one dash per (predictor set, model).

    Drawn on a single axes, or -- when ``find_break`` finds an outlier group in
    the pooled values -- on two stacked axes with the empty interval cut out.
    Every point is scattered on BOTH panels and the per-panel ``ylim`` decides
    where it shows: the alternative (each point on "its" panel only) would split
    the ``handles`` dict that drives the legend and the model count.
    """
    import ultraplot as pplt

    # --ylim is in the units of the REL panel (%); silently applying it to the
    # abs panel would clamp mol/m^3 values of ~1e-3 to a %-scale window and
    # render it blank. An explicit window also overrides the break: the user
    # asked for these limits, and a cut inside them is not what they asked for.
    n_axes, hratios, br, _ = block_shape(ds, which=which, deep=deep,
                                         basin=basin, ylim=ylim,
                                         allow_break=allow_break)
    if br is not None:
        frac = hratios[0] if br['far_above'] else hratios[1]
        # refnum -> the CORE panel: refwidth/refaspect size the *reference*
        # subplot, so pointing them at the sliver would blow the figure up by
        # 1/frac. Dividing the aspect by (1 - frac) keeps the two panels
        # together the same height the single axes had.
        fig, axs = pplt.subplots(nrows=2, hratios=hratios, hspace='0.7em',
                                 refaspect=2 / (1 - frac), refwidth='80mm',
                                 refnum=(2 if br['far_above'] else 1))
    else:
        fig, ax = pplt.subplots(refaspect=2, refwidth='80mm')
        axs = [ax]

    ylabel = block_ylabel(which, deep=deep, basin=basin)
    got = draw_block(axs, ds, which=which, deep=deep, basin=basin, ylim=ylim,
                     allow_break=allow_break,
                     # one shared label for a broken pair (supylabel below), so
                     # the two panels don't each claim to be "the" error axis
                     ylabel=(ylabel if n_axes == 1 else ''))
    handles = got['handles']
    # the basin is already named in the ylabel; don't repeat it in the suptitle
    bits = [ds.attrs.get('title', cell_title(cell)),
            str(ds.attrs.get('years_label', '')),
            f'n={len(handles)} models', _epoch_tag(ds)]
    fig.format(suptitle='  |  '.join(b for b in bits if b))
    if len(axs) > 1:
        fig.supylabel(ylabel)
        _draw_break_marks(axs[0], axs[1])
    fig.legend(list(handles.values()), loc='r', ncols=1)

    # 'global' stays out of the filename so the default figure keeps the name it
    # had before basins existed.
    suffix = ('' if basin == 'global' else f'_{basin}') + ('_deep' if deep else '')
    path = OUT_DIR / f'{cell_slug(cell)}_{which}{suffix}.png'
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.save(path, dpi=300)
    logging.info(f'  wrote {path}')
    return fig


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def compute(epoch=DEFAULT_EPOCH):
    """Sample every predictor set of the main cell and write its cache
    (``predictor_boxplot.py --cell 0``)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = dict(device=device,
               sample_cfg=SampleCfg(n_samples=N_SAMPLES, sampler=SAMPLER,
                                    steps=STEPS),
               year_window=DEFAULT_YEAR_WINDOW, years=None,
               months=list(range(12)), years_stride=DEFAULT_STRIDE,
               max_models=None, all_models=False, epoch=epoch)
    d = compute_cell(MAIN_CELL, ctx)
    if d is None:
        raise SystemExit(f'{cell_slug(MAIN_CELL)}: no run usable')
    return save_cache(MAIN_CELL, d, ctx)


def plot():
    """Replot the manuscript figure from the cache (no GPU)."""
    ds = load_cache(MAIN_CELL)
    if ds is None:
        raise FileNotFoundError(f'no cache at {cache_path(MAIN_CELL)}; '
                                f'run compute')
    fig = plot_cell(MAIN_CELL, ds, which=WHICH)
    render_common.finish(fig, abc=False)
    return render_common.save(fig, 'predictor_boxplot')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    p.add_argument('--epoch', type=int, default=DEFAULT_EPOCH)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        compute(args.epoch)
    else:
        plot()
