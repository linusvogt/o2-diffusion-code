"""Fig. 5 -- skill across every training x evaluation scenario combination.

A 2x2 of absolute relative error maps, rows = training experiment (1pctCO2,
abrupt-4xCO2), columns = evaluation experiment, for the T+S annual
depth-integrated in-sample runs, years 81-100: the diagonal is in-scenario, the
off-diagonal applies a run to a scenario it never trained on. Each map is the
model-mean of per-model ``|100*(gen_m - truth_m)/truth_m|`` over the models
present in both experiments. A third row differences each column, transfer
minus native, on its own signed scale.

Extracted from general/eval/scenario_transfer.py (compute and plot).

    python -m figures.fig05_scenario_transfer compute [--epoch 500]   # GPU
    python -m figures.fig05_scenario_transfer plot                    # CPU
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

from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, get_models,
    build_per_model_ds, model_timemean, select_case, mape, SampleCfg,
    FIG_DIR, DEFAULT_YEAR_WINDOW)


# ---------------------------------------------------------------------------
# pinned manuscript constants (main variant)
# ---------------------------------------------------------------------------
# Colour limit, %, for the four quad panels: |rel| p99 over the deep ocean is
# ~13.6 (p98 10.4), so 15 covers the 99th percentile. The scale extends, so
# the isolated cells past it saturate visibly.
SCENARIO_REL_LIMIT = 15.0
# The difference row (transfer minus native, SIGNED) gets its own symmetric
# scale: the pooled p98 of |diff| (9.94) rounded up. The tail is long where a
# near-zero truth makes the percentage explode, hence a percentile, not a max.
SCENARIO_DIFF_LIMIT = 10.0
WHICH = 'rel'

# The main variant: T+S, annual, depth-integrated; 5-member ensembles, 100-step
# DDIM, years 81-100 at stride 1, models common to both experiments.
PREDICTORS = ['thetao', 'so']
RESOLUTION = 'annual'
FIELD_TYPE = 'depthint'
N_SAMPLES = 5
STEPS = 100
SAMPLER = 'ddim'
COMMON_MODELS = True


OUT_DIR = FIG_DIR / 'scenario_transfer'
CACHE_DIR = OUT_DIR / 'cache'

BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

OTHER_EXP = {'1pctCO2': 'abrupt-4xCO2', 'abrupt-4xCO2': '1pctCO2'}

# Every in-sample run of both experiments has a checkpoint at 500, so pinning
# it keeps training length out of every comparison and leaves no pair
# half-empty.
DEFAULT_EPOCH = 500

# Monthly runs cost 12x an annual one per year, so they default to a coarser
# year stride (as in error_maps.py / seasonal_cycle.py).
DEFAULT_STRIDE = {'annual': 1, 'monthly': 5}

# ONE fixed pair of colour limits for the whole sweep -- both directions and both
# data experiments. Per-direction limits would rescale the native cell against
# its own transfer cell and destroy the one comparison this figure exists for.
# The two panels encode the SAME threshold under error_maps' equivalence
# (deep-ocean depth-integrated O2 median ~0.2 mol/m^3, so x% ~ 0.002*x mol/m^3):
# 30% <-> 0.06. Both scales `extend`, so a cell past the limit saturates visibly
# rather than clipping silently. Only the fallback when no limit is passed; the
# figure passes SCENARIO_REL_LIMIT.
DEFAULT_ERR_LIMIT = 0.06     # mol/m^3, 0..
DEFAULT_REL_LIMIT = 30.0     # %, 0..

# A model needs this many finite truth cells in a time step before it is worth
# generating for -- below it the regrid is corrupt (the IPSL-style entirely-NaN
# depth-integrated files). Training's all-NaN drop never ran on the second
# experiment, so unlike error_maps this check has to be made here.
MIN_OCEAN_CELLS = 1000

# Stamped into every cache, so a cache written under a different aggregation is
# not mistaken for this one. Identical to error_maps' metric by design.
METRIC = 'model-mean-abs'

_SHORT_PRED = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': 'taux',
               'tauvo': 'tauy', 'mld': 'MLD'}
_PRED_TEX = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}
_SHORT_EXP = {'1pctCO2': '1pct', 'abrupt-4xCO2': '4x'}


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def pred_slug(preds):
    return '-'.join(_SHORT_PRED.get(p, p) for p in preds)


def pred_label(preds):
    return '+'.join(_PRED_TEX.get(p, p) for p in preds)


def is_transfer(cell):
    return cell['train_exp'] != cell['data_exp']


def cell_slug(cell):
    """Filename stem -- stable, so cache/figure names don't drift.

    ``<preds>_<ann|mon>_<int|dens>_<data exp>_from_<train exp>``: the DATA
    experiment leads, so the two cells of an intended pair sort adjacent.
    """
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    return (f"{pred_slug(cell['predictors'])}_{res}_{fld}_"
            f"{_SHORT_EXP[cell['data_exp']]}_from_"
            f"{_SHORT_EXP[cell['train_exp']]}")


def pair_slug(cell):
    """Stem shared by a transfer cell and its native baseline (same data exp)."""
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    return (f"{pred_slug(cell['predictors'])}_{res}_{fld}_"
            f"{_SHORT_EXP[cell['data_exp']]}")


def cell_title(cell):
    fld = {'depthint': 'depth-integrated', 'density': 'density layers'}[
        cell['field_type']]
    kind = ('OUT-OF-SCENARIO' if is_transfer(cell) else 'in-scenario')
    return (f"{pred_label(cell['predictors'])} | {cell['resolution']} | {fld} | "
            f"{cell['data_exp']} data from {cell['train_exp']} run ({kind})")



def cell_config(cell):
    """Naming config for the cell's RUN -- the training experiment, in-sample."""
    return make_config(BASE_CONFIG, predictors=cell['predictors'],
                       experiments=[cell['train_exp']],
                       resolution=cell['resolution'],
                       field_type=cell['field_type'])


# ---------------------------------------------------------------------------
# epoch resolution
# ---------------------------------------------------------------------------
def latest_epoch(save_dir):
    """Highest ``ckpt_epoch###.pt`` epoch in ``save_dir``, or None if empty."""
    epochs = [int(m.group(1))
              for p in Path(save_dir).glob('ckpt_epoch*.pt')
              if (m := re.search(r'ckpt_epoch(\d+)\.pt$', p.name))]
    return max(epochs) if epochs else None


def resolve_epoch(save_dir, epoch):
    """Concrete epoch for a run, or None when it has no such checkpoint."""
    if epoch in (None, 'latest'):
        return latest_epoch(save_dir)
    return int(epoch) if run_exists(save_dir, int(epoch)) else None


# ---------------------------------------------------------------------------
# per-cell computation
# ---------------------------------------------------------------------------
def cell_models(run_config, cell, ctx):
    """Models to evaluate for a cell, as objects built for its DATA experiment.

    With ``common_models`` (the default) the set is intersected BY NAME with the
    models available in the other experiment, so all four cells of a
    (predictors, resolution, field type) quad score the same models and a
    difference between them cannot be a model-composition artifact. The
    intersection is by name and the objects are then taken from the data
    experiment's own discovery, because ``get_models`` picks a member per
    experiment and the choice can differ.
    """
    models = get_models(run_config, data_experiment=cell['data_exp'])
    if ctx['common_models']:
        other = {m.model for m in get_models(
            run_config, data_experiment=OTHER_EXP[cell['data_exp']])}
        dropped = [m.model for m in models if m.model not in other]
        models = [m for m in models if m.model in other]
        if dropped:
            logging.info(f'  [common-models] not in '
                         f'{OTHER_EXP[cell["data_exp"]]}: {", ".join(dropped)}')
    if ctx['max_models'] is not None:
        models = models[:ctx['max_models']]
    return models


def screen_models(run, models, cell, ctx):
    """Drop models whose truth for THIS experiment is a corrupt all-NaN regrid.

    One cheap pass (one time slice per model, no sampling) before generating.
    ``build_per_model_ds`` is called on an experiment the run never trained on,
    so training's all-NaN drop has not filtered these -- and because this
    analysis *requires* the target, a NaN truth would poison ``model_mean_abs``
    rather than merely be absent. Returns ``(kept, dropped_labels)`` where
    ``kept`` are ``(model, ds_model)`` pairs, so the lazy dataset built here is
    reused for the sampling pass.
    """
    resolution = run.config['resolution']
    target = run.config['target']
    years = ec._years_in_window(ctx['year_window'],
                                ctx['years_stride'][resolution], ctx['years'])
    months = ctx['months'] if resolution == 'monthly' else [None]
    kept, dropped = [], []
    for m in models:
        try:
            ds_model = build_per_model_ds(run.config, m,
                                          data_experiment=cell['data_exp'])
        except Exception as e:                     # missing / unreadable file
            dropped.append(f'{m.model} ({type(e).__name__})')
            continue
        ok = None
        for year in years:
            for month in months:
                ds_sel = select_case(ds_model, resolution, year, month)
                if ds_sel is None:
                    continue
                n = int(np.isfinite(ds_sel[target].values).sum())
                if n >= MIN_OCEAN_CELLS:
                    ok = n
                    break
            if ok is not None:
                break
        if ok is None:
            dropped.append(f'{m.model} (no usable truth slice)')
            continue
        kept.append((m, ds_model))
    if dropped:
        logging.info(f'  [drop] {", ".join(dropped)}')
    return kept, dropped


def compute_cell(cell, ctx):
    """Model-mean absolute error maps (mol/m^3, %) for one cell."""
    cfg = cell_config(cell)
    save_dir = run_save_dir(cfg)
    ep = resolve_epoch(save_dir, ctx['epoch'])
    if ep is None:
        logging.info(f'[skip] {cell_slug(cell)}: no checkpoint at epoch '
                     f'{ctx["epoch"]} in {save_dir.name}')
        return None
    run = load_run(save_dir, ctx['device'], ep, _compile=False)
    land_fill = run.scaling.get('land_fill', 'zero')
    logging.info(f'  run {save_dir.name}: epoch {ep}, land_fill={land_fill}')

    models = cell_models(run.config, cell, ctx)
    if not models:
        logging.info(f'[skip] {cell_slug(cell)}: no models with data in '
                     f'{cell["data_exp"]}')
        return None
    kept, dropped = screen_models(run, models, cell, ctx)
    if not kept:
        logging.info(f'[skip] {cell_slug(cell)}: no model has usable truth')
        return None

    gens, truths, used = [], [], []
    for m, ds_model in kept:
        res = model_timemean(run, ds_model, ctx['year_window'],
                             ctx['sample_cfg'], months=ctx['months'],
                             years_stride=ctx['years_stride'][cell['resolution']],
                             years=ctx['years'])
        if res is None or res['truth'] is None:
            logging.info(f'  [skip] {m.model}: no times in the window')
            dropped.append(f'{m.model} (no times in window)')
            continue
        gens.append(res['gen'])
        truths.append(res['truth'])
        used.append(m.model)
        logging.info(f'  {m.model}: {res["n_times"]} times')

    if not gens:
        logging.info(f'[skip] {cell_slug(cell)}: nothing generated')
        return None
    d = ec.model_mean_abs(gens, truths)
    d['model_names'] = used
    d['epoch'] = ep
    d['land_fill'] = land_fill
    d['models_skipped'] = dropped
    return d


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(cell):
    return CACHE_DIR / f'{cell_slug(cell)}.nc'


def years_label(cell, ctx):
    """What was actually averaged in time -- explicit years, or window+stride.

    The window alone would mislabel a strided or --years run (the figure would
    claim 'yrs 81-100' after sampling only year 81).
    """
    if ctx['years'] is not None:
        return 'yrs ' + ','.join(str(y) for y in ctx['years'])
    lo, hi = ctx['year_window']
    stride = ctx['years_stride'][cell['resolution']]
    return f'yrs {lo}-{hi}' + (f' /{stride}' if stride != 1 else '')


def save_cache(cell, d, ctx):
    """Persist the four maps + provenance so the figure replots without a GPU."""
    ds = xr.Dataset({k: d[k] for k in ('truth', 'gen', 'err', 'rel')})
    ds.attrs.update(
        cell=cell_slug(cell), pair=pair_slug(cell), title=cell_title(cell),
        predictors=' '.join(cell['predictors']), resolution=cell['resolution'],
        field_type=cell['field_type'], data_exp=cell['data_exp'],
        train_exp=cell['train_exp'],
        direction='transfer' if is_transfer(cell) else 'native',
        land_fill=d.get('land_fill', ''),
        n_models=d['n_models'], models=' '.join(d.get('model_names', [])),
        models_skipped='; '.join(d.get('models_skipped', [])),
        epoch=d.get('epoch', -1), epoch_arg=str(ctx['epoch']),
        common_models=int(bool(ctx['common_models'])),
        n_samples=ctx['sample_cfg'].n_samples, steps=ctx['sample_cfg'].steps,
        sampler=ctx['sample_cfg'].sampler,
        years_stride=ctx['years_stride'][cell['resolution']],
        year_window=' '.join(str(y) for y in ctx['year_window']),
        years_label=years_label(cell, ctx),
        n_months=len(ctx['months']) if cell['resolution'] == 'monthly' else 1,
        metric=METRIC,
        mape_pct=float(mape(d['rel'])))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(cell)
    ds.to_netcdf(path)
    logging.info(f'  cached {path}')
    return ds


def load_cache(cell):
    path = cache_path(cell)
    return xr.open_dataset(path).load() if path.is_file() else None



# ---------------------------------------------------------------------------
# composite: the transfer-vs-native quad (one figure, four cells)
# ---------------------------------------------------------------------------
# 2x2 as a train x eval factorial: ROWS are the TRAINING experiment, COLUMNS the
# EVALUATION experiment. So the DIAGONAL is in-scenario (native) and the
# OFF-DIAGONAL is out-of-scenario (transfer).
#
# The intended comparison is therefore down a COLUMN: it holds the evaluation
# data, models, window and metric fixed and varies only the training experiment,
# i.e. a transfer cell is read against the native cell with the SAME data
# experiment. Reading along a ROW instead crosses data
# experiments -- it varies what the run is asked to reproduce, not what it was
# trained on -- and conflates the unseen scenario with "4x fields are simply
# harder"; with a percentage metric the two columns also have different
# denominators. Each panel's title names both experiments so the reader can
# orient without row/column labels.
QUAD_TRAIN_ROWS = ['1pctCO2', 'abrupt-4xCO2']    # training exp, top to bottom
QUAD_DATA_COLS = ['1pctCO2', 'abrupt-4xCO2']     # evaluation exp, left to right


def quad_cells(predictors, resolution, field_type):
    """The four cells of a train x eval quad, in 2x2 row-major order."""
    return [dict(predictors=list(predictors), resolution=resolution,
                 field_type=field_type, data_exp=data_exp, train_exp=train_exp)
            for train_exp in QUAD_TRAIN_ROWS
            for data_exp in QUAD_DATA_COLS]


def _panel_scalar(ds, which):
    """Area-weighted mean of the plotted (deep-ocean) map, in its own units.

    For 'rel' this is the cached ``mape_pct``; for 'err' the same reduction on
    the mol/m^3 map, which has no cached attr. Recomputed from the map either
    way so the number always describes the panel actually drawn.
    """
    da = ec.mask_shallow(np.abs(ds[which]))
    return float(ec.util.global_mean(da).values)


def quad_maps(axs, cells, datasets, which='err', limit=None, show_scalar=True,
              cbar=True):
    """Draw the quad's four maps INTO ``axs`` (row-major); return the mappables.

    The drawing half of :func:`plot_quad`, split out so a caller can stack one
    quad per (resolution, field type) variant in a single figure.

    ``limit`` is REQUIRED when the caller owns the figure: all four panels of a
    quad (and, stacked, all sixteen) must share ONE scale, which is the whole
    point of the figure, and an autoscaled block would break that invisibly.
    ``cbar=False`` suppresses the per-panel bars for a single figure-level one --
    legitimate here precisely because every panel is already on that one scale.
    """
    contourf_plot = _contourf_local()
    if len(cells) != 4 or len(datasets) != 4:
        raise ValueError(f'quad needs 4 cells and 4 datasets, '
                         f'got {len(cells)} and {len(datasets)}')
    if limit is None:
        raise ValueError(
            'quad_maps needs an explicit limit -- the single shared scale is '
            'what makes transfer and native panels comparable')
    maps = [np.abs(ec.mask_shallow(ds[which])) for ds in datasets]
    unit = ec.CBAR_LABEL if which == 'err' else '%'
    kw = dict(vmin=0, vmax=float(limit), extend='max', cmap='Reds')
    ims = []
    for ax, cell, ds, da in zip(axs, cells, datasets, maps):
        ims.append(contourf_plot(ax=ax, da=da, contourf_kw=kw, cbar=cbar,
                                 cbar_label=unit))
        title = (f'Trained on {cell["train_exp"]}\n'
                 f'Evaluated on {cell["data_exp"]}')
        if show_scalar:
            scalar = _panel_scalar(ds, which)
            title += (f'  ({scalar:.4f} {unit})' if which == 'err'
                      else f'  ({scalar:.1f}%)')
        ax.format(title=title)
    return ims


def column_diff(cells, datasets, which='rel'):
    """Per evaluation experiment: the TRANSFER panel's error minus the NATIVE
    panel's. One entry per column of the quad, in ``QUAD_DATA_COLS`` order.

    This is the comparison the quad exists for, done arithmetically instead of
    by eye -- four near-identical maps otherwise make the reader difference
    colours across panels. It is taken DOWN a column and only down a
    column: the evaluation data, models, window, metric and percentage
    denominator are then all fixed and the single remaining difference is the
    training experiment. Along a row none of that holds (see this module's
    ``QUAD_*`` comment), so there is no row difference to draw.

    Sign convention: **positive means transfer is worse**, since the cached maps
    are already model-means of per-model |error| (``model_mean_abs``, so
    non-negative) and the subtraction of two magnitudes is what "how much skill
    does the unseen scenario cost" means here. That also makes the result
    SIGNED, unlike every panel of the quad -- so it needs its own diverging
    scale and its own colorbar, not the quad's one-sided ``Reds``.

    The two cells' fleets are compared and a mismatch RAISES: differencing two
    maps whose model populations differ gives arrays of the right shape whose
    result is part composition artefact. (The T+S annual depth-integrated quad
    carries the same 12 models in all four cells, so it passes.)
    """
    by_key = {(c['data_exp'], c['train_exp']): (c, ds)
              for c, ds in zip(cells, datasets)}
    out = []
    for data_exp in QUAD_DATA_COLS:
        native = by_key.get((data_exp, data_exp))
        transfer = [v for k, v in by_key.items()
                    if k[0] == data_exp and k[1] != data_exp]
        if native is None or not transfer:
            continue
        (_, ds_n), (_, ds_t) = native, transfer[0]
        f_n = str(ds_n.attrs.get('models', '')).split()
        f_t = str(ds_t.attrs.get('models', '')).split()
        if sorted(f_n) != sorted(f_t):
            raise ValueError(
                f'{data_exp}: the transfer and native cells of this column are '
                f'over different model fleets ({len(f_t)} vs {len(f_n)}; '
                f'symmetric difference '
                f'{sorted(set(f_n) ^ set(f_t))}). Differencing them would mix '
                f'the scenario-transfer signal with a composition artefact -- '
                f're-run the pair with --common-models.')
        diff = ec.mask_shallow(ds_t[which]) - ec.mask_shallow(ds_n[which])
        out.append(dict(data_exp=data_exp, diff=diff, n_models=len(f_n)))
    return out


def quad_diff_maps(axs, diffs, which='rel', limit=None, cbar='last',
                   show_scalar=True):
    """Draw :func:`column_diff`'s panels INTO ``axs``; return the mappables.

    ``cbar`` -- True one bar per panel, ``'last'`` (the default) one bar on the
    final panel, False none. 'last' because both difference panels are on the
    one pinned symmetric scale, so a second bar is a copy of the first; the same
    reasoning as ``rel_error_maps``' ``cbar='last'``. The quad above keeps its
    own per-panel bars.
    """
    contourf_plot = _contourf_local()
    if limit is None:
        raise ValueError(
            'quad_diff_maps needs an explicit limit -- both difference panels '
            'must share one scale for the two columns to be readable against '
            'each other')
    unit = ec.CBAR_LABEL if which == 'err' else '%'
    kw = dict(vmin=-float(limit), vmax=float(limit), extend='both', cmap='Div')
    ims = []
    for i, (ax, d) in enumerate(zip(axs, diffs)):
        show_bar = (cbar is True
                    or (cbar == 'last' and i == len(diffs) - 1))
        ims.append(contourf_plot(ax=ax, da=d['diff'], contourf_kw=kw,
                                 cbar=show_bar,
                                 cbar_label=unit if show_bar else None))
        title = (f'Transfer $-$ native\nEvaluated on {d["data_exp"]}')
        if show_scalar:
            # The area-weighted mean of the SIGNED difference: "what the unseen
            # scenario costs on average", which is the sentence this panel is
            # for. Not the mean of |difference|, which would report a cost even
            # for a column that is uniformly better.
            v = float(ec.util.global_mean(d['diff']).values)
            title += (f'  ({v:+.4f} {unit})' if which == 'err'
                      else f'  ({v:+.2f}%)')
        ax.format(title=title)
    return ims


def plot_quad(cells, datasets, which='err', limit=None, save_path=None,
              suptitle=None, show_scalar=True, diff_limit=None):
    """One 2x2 figure from four cached cells: training experiment x evaluation
    experiment, so the diagonal is in-scenario and the off-diagonal transfer.

    ``which`` picks ONE map per panel: 'rel' (absolute percentage error) or
    'err' (absolute error, mol/m^3). Either is read the same way -- DOWN a
    column, where the evaluation data, models, window and metric are all fixed
    and only the training experiment differs. With 'rel' the two columns
    additionally have different denominators (different data experiments), so a
    percentage in the left column is not comparable with one in the right;
    'err' is in physical units and does not have that caveat. Neither licenses
    reading ALONG a row.

    The single scale across all four panels is the point of the figure and is
    NOT negotiable per column: rescaling each column against its own transfer
    cell would destroy the transfer-vs-native comparison, exactly as noted for
    ``DEFAULT_ERR_LIMIT``. Faint diagonal panels are the honest result.

    ``show_scalar`` appends each panel's area-weighted mean to its title.

    ``diff_limit`` adds a THIRD ROW: :func:`column_diff`'s transfer-minus-native
    map for each column, placed directly under the column it differences. Off by
    default. The row is on its own symmetric diverging scale with its own colorbar because it
    is a **signed** quantity where the quad's four panels are magnitudes -- one
    shared bar would be a category error.
    """
    pplt = _pplt_local()
    if limit is None:
        limit = DEFAULT_ERR_LIMIT if which == 'err' else DEFAULT_REL_LIMIT
    diffs = column_diff(cells, datasets, which=which) if diff_limit else []
    nrows = 2 + (1 if diffs else 0)
    fig, axs = pplt.subplots(nrows=nrows, ncols=2, proj='robin',
                             proj_kw=dict(lon_0=202), refwidth='68mm', share=0)
    quad_maps(axs[:4], cells, datasets, which=which, limit=limit,
              show_scalar=show_scalar)
    if diffs:
        quad_diff_maps(axs[4:4 + len(diffs)], diffs, which=which,
                       limit=diff_limit, show_scalar=show_scalar)
    if suptitle is not None:
        fig.format(suptitle=suptitle)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.save(save_path, dpi=250)
        logging.info(f'  saved {save_path}')
    return fig



def _pplt_local():
    import ultraplot as pplt
    return pplt


def _contourf_local():
    from vendored.ocean_utils_min import contourf_plot
    return contourf_plot



# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def main_cells():
    """The four cells of the main quad, row-major: 1pct_from_1pct,
    4x_from_1pct, 1pct_from_4x, 4x_from_4x."""
    return quad_cells(PREDICTORS, RESOLUTION, FIELD_TYPE)


def compute(epoch=DEFAULT_EPOCH):
    """Sample the four cells of the quad and write one cache per cell
    (``scenario_transfer.py --cell N`` for each)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = dict(device=device,
               sample_cfg=SampleCfg(n_samples=N_SAMPLES, sampler=SAMPLER,
                                    steps=STEPS),
               year_window=DEFAULT_YEAR_WINDOW, years=None,
               months=list(range(12)), years_stride=DEFAULT_STRIDE,
               max_models=None, epoch=epoch, common_models=COMMON_MODELS)
    out = []
    for cell in main_cells():
        logging.info(f'\n=== {cell_slug(cell)} :: {cell_title(cell)} ===')
        d = compute_cell(cell, ctx)
        if d is None:
            raise SystemExit(f'{cell_slug(cell)}: nothing generated')
        out.append(save_cache(cell, d, ctx))
    return out


def plot():
    """Replot the manuscript figure from the four caches (no GPU)."""
    cells = main_cells()
    datasets = [load_cache(c) for c in cells]
    missing = [str(cache_path(c)) for c, ds in zip(cells, datasets) if ds is None]
    if missing:
        raise FileNotFoundError(f'no cache at {missing}; run compute')
    fig = plot_quad(cells, datasets, which=WHICH, limit=SCENARIO_REL_LIMIT,
                    diff_limit=SCENARIO_DIFF_LIMIT)
    render_common.finish(fig, abc=True)
    return render_common.save(fig, 'scenario_transfer')


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
