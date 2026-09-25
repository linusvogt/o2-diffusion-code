"""Fig. 6: out-of-sample error for held-out CMIP6 models (T+S, annual,
depth-integrated, leave-one-model-out): a) the signed relative error, b) the
absolute relative error, c) the cross-model coherence of the error.

Extracted from general/eval/error_maps.py (compute) and
general/eval/lomo_signed_error.py (plot).

    python -m figures.fig06_lomo_error compute [--epoch 500]   # GPU
    python -m figures.fig06_lomo_error plot                    # no GPU
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
from diffusion import inference as util
from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, get_models,
    build_per_model_ds, model_timemean, mape, SampleCfg,
    FIG_DIR, DEFAULT_YEAR_WINDOW)


# ---------------------------------------------------------------------------
# pinned manuscript constants (general/manuscript/figures.py, main variant)
# ---------------------------------------------------------------------------
# Colour limits, shared by all four T+S variants. Pooled p98 of the four T+S
# OOS caches (deep-ocean masked), rounded up, so at most ~2% of cells saturate
# and the scales `extend` rather than clip.
LOMO_REL_LIMIT = 75.0
LOMO_SIGNED_REL_LIMIT = 10.0
# Not a measurement: the ratio is in [0, 1] by Jensen, and the panel is drawn
# on that whole range so the two ends mean what the title says they mean.
LOMO_COHERENCE_LIMIT = 1.0

# Panel titles carry the FORMULA: the three panels are three different poolings
# of one error. NB ``_AVG`` CONTINUES the math span the '$|' before it opened,
# so it has no leading '$'; ``_AVG_OPEN`` opens its own. Swapping them does not
# raise -- mathtext renders the backslashes literally.
_AVG = r'\rangle_{\mathrm{models}}$'
LOMO_REL_TITLE = ('Absolute relative LOMO error\n'
                  r'$\langle|$100$\times$(gen $-$ true)/true$|' + _AVG)
_AVG_OPEN = '$' + _AVG
LOMO_SIGNED_TITLE = ('Signed relative LOMO error\n'
                     r'100$\times\langle$gen $-$ true' + _AVG_OPEN
                     + r' / $\langle$true' + _AVG_OPEN)
LOMO_COH_TITLE = ('Cross-model coherence\n'
                  r'$|\langle$gen $-$ true' + _AVG_OPEN
                  + r'$|$ / $\langle|$gen $-$ true$|' + _AVG)
# Bare colorbar labels: the panel title carries the formula directly above
# each bar. The sign convention (positive = generated exceeds truth) and which
# end of 0..1 means the models cancel are stated in the caption.
LOMO_SIGNED_CBAR = '%'
LOMO_COH_CBAR = 'Coherence'

# The manuscript's main cell: T+S, annual, depth-integrated, out-of-sample.
CELL = dict(predictors=['thetao', 'so'], resolution='annual',
            field_type='depthint', split='oos')


# ===========================================================================
# compute side -- from general/eval/error_maps.py
# ===========================================================================
MODELS_FILE = paths.MODELS_TXT      # the 14 CMIP6 models of the LOMO fleet

CACHE_DIR = FIG_DIR / 'error_maps' / 'cache'

BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# Default checkpoint epoch: 500 is the largest epoch reached by every axis of
# the sweep, so pinning it keeps training length out of any figure-to-figure
# comparison. '--epoch latest' opts out.
DEFAULT_EPOCH = 500

# Monthly runs cost 12x an annual one per year, so they default to a coarser
# year stride.
DEFAULT_STRIDE = {'annual': 1, 'monthly': 5}

# Aggregation over models, stamped into every cache. 'model-mean-abs' =
# per-model |error| first, then average the magnitudes over models; averaging
# the fields first would let opposite-signed per-model biases cancel.
METRIC = 'model-mean-abs'

_SHORT_PRED = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': 'taux',
               'tauvo': 'tauy', 'mld': 'MLD'}
_PRED_TEX = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def pred_slug(preds):
    return '-'.join(_SHORT_PRED.get(p, p) for p in preds)


def pred_label(preds):
    return '+'.join(_PRED_TEX.get(p, p) for p in preds)


def cell_slug(cell):
    """Filename stem for a cell -- stable, so cache/figure names don't drift."""
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    return f"{pred_slug(cell['predictors'])}_{res}_{fld}_{cell['split']}"


def cell_title(cell):
    res = cell['resolution']
    fld = {'depthint': 'depth-integrated', 'density': 'density layers'}[
        cell['field_type']]
    split = {'insample': 'in-sample',
             'oos': 'out-of-sample (held-out models)'}[cell['split']]
    return f"{pred_label(cell['predictors'])} | {res} | {fld} | {split}"


def cell_config(cell):
    """Naming config for a cell (the in-sample variant; LOMO overrides split)."""
    return make_config(BASE_CONFIG, predictors=cell['predictors'],
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
    """Concrete epoch for a run: ``epoch`` itself, or its newest checkpoint when
    ``epoch`` is 'latest'/None. Returns None when the run has no checkpoint at
    all (or not that one), so callers can skip-and-log."""
    if epoch in (None, 'latest'):
        return latest_epoch(save_dir)
    return int(epoch) if run_exists(save_dir, int(epoch)) else None


# ---------------------------------------------------------------------------
# per-cell computation
# ---------------------------------------------------------------------------
def _read_models():
    return [ln.strip() for ln in MODELS_FILE.read_text().splitlines() if ln.strip()]


def compute_oos(cell, ctx):
    """Model-mean absolute error maps aggregated over held-out models.

    Each held-out model is generated by its OWN leave-one-model-out run; each
    model's own absolute error is formed FIRST and the magnitudes are then
    averaged over models (``ec.model_mean_abs``), so opposite-signed per-model
    biases cannot cancel. Still ONE figure per cell, not one per held-out model.
    """
    cfg = cell_config(cell)
    model_names = _read_models()
    if ctx['max_models'] is not None:
        model_names = model_names[:ctx['max_models']]

    gens, truths, used, epochs = [], [], [], []
    for name in model_names:
        save_dir = run_save_dir(cfg, train_test_split='model', models_test=[name])
        ep = resolve_epoch(save_dir, ctx['epoch'])
        if ep is None:
            logging.info(f'  [skip] {name}: no checkpoint at epoch {ctx["epoch"]}')
            continue
        # _compile=False: this loop loads one checkpoint per held-out model, and
        # a torch.compile per run would dominate the runtime.
        run = load_run(save_dir, ctx['device'], ep, _compile=False)
        mobjs = [m for m in get_models(run.config) if m.model == name]
        if not mobjs:
            logging.info(f'  [skip] {name}: model absent from this data config')
            continue
        ds_model = build_per_model_ds(run.config, mobjs[0])
        res = model_timemean(run, ds_model, ctx['year_window'], ctx['sample_cfg'],
                             months=ctx['months'],
                             years_stride=ctx['years_stride'][cell['resolution']],
                             years=ctx['years'])
        if res is None or res['truth'] is None:
            logging.info(f'  [skip] {name}: no times in the window')
            continue
        gens.append(res['gen'])
        truths.append(res['truth'])
        used.append(name)
        epochs.append(ep)
        logging.info(f'  {name}: epoch {ep}, {res["n_times"]} times')

    if not gens:
        logging.info(f'[skip] {cell_slug(cell)}: no held-out run usable')
        return None
    d = ec.model_mean_abs(gens, truths)
    d['model_names'], d['epochs'] = used, epochs
    return d


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(cell):
    return CACHE_DIR / f'{cell_slug(cell)}.nc'


def years_label(cell, ctx):
    """What was actually averaged in time -- explicit years, or window+stride."""
    if ctx['years'] is not None:
        return 'yrs ' + ','.join(str(y) for y in ctx['years'])
    lo, hi = ctx['year_window']
    stride = ctx['years_stride'][cell['resolution']]
    return f'yrs {lo}-{hi}' + (f' /{stride}' if stride != 1 else '')


def save_cache(cell, d, ctx):
    """Persist the four maps + provenance so the plot step needs no GPU."""
    ds = xr.Dataset({k: d[k] for k in ('truth', 'gen', 'err', 'rel')})
    ds.attrs.update(
        cell=cell_slug(cell), title=cell_title(cell),
        predictors=' '.join(cell['predictors']), resolution=cell['resolution'],
        field_type=cell['field_type'], split=cell['split'],
        n_models=d['n_models'], models=' '.join(d.get('model_names', [])),
        epochs=' '.join(str(e) for e in d.get('epochs', [])),
        epoch_arg=str(ctx['epoch']),
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


def _epoch_tag(ds):
    """'ep 500' or 'ep 225-500' -- OOS cells can mix epochs under --epoch latest."""
    eps = sorted({int(e) for e in str(ds.attrs.get('epochs', '')).split() if e})
    if not eps:
        return ''
    return f'ep {eps[0]}' if len(eps) == 1 else f'ep {eps[0]}-{eps[-1]}'


# ===========================================================================
# plot side -- from general/eval/lomo_signed_error.py
# ===========================================================================
# The signed LOMO error is derived from the error_maps cache with no
# re-sampling: that cache stores the model-mean ``truth`` and ``gen`` beside
# its two magnitude maps, and the mean is linear, so
#     gen - truth == <gen_m>_m - <truth_m>_m == <gen_m - truth_m>_m
# exactly. ``verify_cache`` checks the two conditions under which that holds.
#
# The relative signed panel is model-mean-then-DIVIDE (a ratio of two pooled
# means), not the per-model ratio: the per-model denominator is noisiest where
# a single model's truth approaches zero, i.e. in the OMZ cores.
OUT_DIR = FIG_DIR / 'lomo_signed_error'

# Upper limit of the |bias|/err coherence map. A ratio in [0, 1] by Jensen, so
# this is a fixed 0..1 sequential scale, not a measurement.
COHERENCE_LIMIT = 1.0


# ---------------------------------------------------------------------------
# derivation + verification
# ---------------------------------------------------------------------------
def verify_cache(ds, slug):
    """Check the two conditions under which ``gen - truth`` IS the signed
    model-mean error, and raise with the cell named if either fails.

    1. ``gen`` and ``truth`` must be finite on exactly the same cells. Both are
       ``.mean('model')`` results and that reduction skips NaN, so a cell where
       one model contributed to ``gen`` but not to ``truth`` would have the two
       means taken over different model subsets -- and their difference would no
       longer be the mean of the per-model differences.
    2. ``|gen - truth| <= err`` pointwise (Jensen), since ``err`` is
       ``<|gen_m - truth_m|>_m`` over the same models.

    A tiny tolerance is allowed on (2) for the float32 round-trip through
    netCDF; (1) is exact and gets none.
    """
    g, t, e = ds['gen'].values, ds['truth'].values, ds['err'].values
    bad_mask = int((np.isfinite(g) != np.isfinite(t)).sum())
    if bad_mask:
        raise ValueError(
            f'{slug}: gen and truth are finite on different cells ({bad_mask} '
            f'of {g.size}) -- gen-truth is then NOT the signed model-mean '
            f'error there, because the two model-means cover different models')
    ok = np.isfinite(g) & np.isfinite(t) & np.isfinite(e)
    excess = np.abs(g - t)[ok] - e[ok]
    tol = 1e-6 * max(float(np.nanmax(e)), 1e-12)
    if excess.size and float(excess.max()) > tol:
        raise ValueError(
            f'{slug}: |gen-truth| exceeds the cached |error| by '
            f'{float(excess.max()):.3g} mol/m^3 at {int((excess > tol).sum())} '
            f'cell(s) -- Jensen forbids it, so the cache is inconsistent')
    # Cells where every held-out model errs the SAME way sit exactly on the
    # bound (|<e_m>| == <|e_m>|), so the worst violation is float32 round-trip
    # noise and lands just under `tol`.
    return dict(n_cells=int(ok.sum()), tol=tol,
                max_violation=float(max(excess.max(), 0.0))
                if excess.size else np.nan)


def signed_maps(ds):
    """``dict(bias, bias_rel, err, rel, coherence)`` for one ``error_maps``
    OOS cache.

    ``bias`` is signed mol/m^3, ``bias_rel`` signed %, ``err`` and ``rel`` the
    cache's two MAGNITUDE maps (mol/m^3 and %), carried through so the caller
    can form ratios against the same footprint and so a figure drawing a signed
    and an unsigned panel side by side needs only this one dict --
    ``coherence`` = ``|bias|/err`` in [0, 1]. Nothing is masked here --
    ``ec.mask_shallow`` is applied at every plot and scalar boundary instead,
    and is idempotent.

    ``bias_rel`` and ``rel`` are NOT the signed and unsigned versions of one
    number: ``bias_rel`` divides two POOLED model means while ``rel`` is the
    model-mean of per-model ratios. ``coherence`` is therefore formed from the
    mol/m^3 pair (``bias``/``err``), where Jensen bounds it by 1; the ratio of
    the two % maps mixes the two poolings and can exceed 1.
    """
    bias = ds['gen'] - ds['truth']
    bias_rel = 100.0 * bias / ds['truth']
    # err is 0 only where gen == truth for every model, in which case bias is 0
    # too; 0/0 would be NaN, which would read as "no data" rather than "no
    # error". Land must stay NaN, so the substitution is keyed on `zero`
    # (finite AND non-positive) rather than on `err > 0` -- a NaN comparison is
    # False, so `.where(err > 0, 0.0)` would paint every land cell 0.
    err = ds['err']
    zero = np.isfinite(err) & (err <= 0)
    coherence = (np.abs(bias) / err.where(err > 0)).where(~zero, 0.0)
    return dict(bias=bias, bias_rel=bias_rel, err=err, rel=ds['rel'],
                coherence=coherence)


# ---------------------------------------------------------------------------
# scalars
# ---------------------------------------------------------------------------
def _deep_mean(da):
    """Area-weighted mean over the DEEP ocean -- ``mape``/``skill_scalar``'s
    footprint, so these numbers sit beside the cached ``mape_pct``."""
    return float(util.global_mean(ec.mask_shallow(da)).values)


def cell_scalars(d, ds):
    """Every number this cell contributes to the panel titles.

    The ``*_pct`` biases are **mean-first**: the ratio of two area-weighted
    means, not the mean of the pointwise ratio.
    """
    out = {}
    out['glob_bias'] = _deep_mean(d['bias'])
    out['glob_truth'] = _deep_mean(ds['truth'])
    out['glob_bias_pct'] = 100.0 * out['glob_bias'] / out['glob_truth']
    out['glob_abs'] = _deep_mean(np.abs(d['err']))
    # magnitudes at each cell BEFORE the spatial average -> across-MODEL
    # cancellation only, with spatial cancellation excluded by construction.
    out['coherence'] = _deep_mean(np.abs(d['bias'])) / out['glob_abs']
    out['mape_pct'] = float(ds.attrs.get('mape_pct', np.nan))
    return out


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
# Sign convention, stated in the (eval) suptitle.
SIGN_NOTE = 'positive = generated exceeds truth (O$_2$ overestimated)'

# Default panel titles (the manuscript passes its own, above). NB this
# fragment OPENS its own math span ('$\\rangle...$'), unlike ``_AVG`` above,
# which continues the span the '$|' before it opened.
_MODELS_AVG = '$\\rangle_{\\mathrm{models}}$'
_REL_TITLE = ('Signed relative LOMO error\n'
              '100$\\times\\langle$gen $-$ true' + _MODELS_AVG
              + ' / $\\langle$true' + _MODELS_AVG)

# The three panels carry THREE poolings, which is why each draws its formula:
#
#   a)  100*<gen - true>_m / <true>_m      pooled numerator over pooled denom
#   b)  <|100*(gen - true)/true|>_m        per-model ratio, then mean of |.|
#   c)  |<gen - true>_m| / <|gen - true|>_m    both in mol/m^3
#
# so c) is NOT a)/b): c) is formed from the mol/m^3 pair, where Jensen bounds it
# by 1 (`verify_cache` asserts exactly that), while |a|/b mixes the two
# poolings and is not bounded. A LOW coherence means the held-out models'
# errors genuinely oppose one another rather than that there is no error.

# Default colorbar labels (the manuscript passes bare ones, above).
SIGNED_CBAR_LABEL = '%   ($+$: gen $>$ true)'
COHERENCE_CBAR_LABEL = '$|$mean error$|$ /\nmean $|$error$|$'

# The continuation form of :data:`_MODELS_AVG` -- the same text WITHOUT its
# leading '$', for use after a '$|' that has already opened the math span.
_MODELS_AVG_CONT = _MODELS_AVG[1:]

_ABS_REL_TITLE = ('Absolute relative LOMO error\n'
                  '$\\langle|$100$\\times$(gen $-$ true)/true$|'
                  + _MODELS_AVG_CONT)
# Line 1 states which end of the scale means what, because "coherence" alone
# runs OPPOSITE to the word a reader supplies for themselves ("cancellation").
_COH_TITLE = ('Cross-model coherence  (0 = models cancel)\n'
              '$|\\langle$gen $-$ true' + _MODELS_AVG
              + '$|$ / $\\langle|$gen $-$ true$|' + _MODELS_AVG_CONT)


def _titled(title, value, fmt):
    """``title`` with its area-weighted scalar in parentheses on the FIRST line.

    The number is derived from the cache at draw time and never pinned, so a
    cell re-run against a different fleet cannot leave a stale number on the
    canvas. A missing or non-finite scalar drops the parenthetical rather than
    drawing 'nan'.
    """
    if value is None or not np.isfinite(value):
        return title
    head, sep, rest = title.partition('\n')
    return f'{head}  ({fmt.format(value)}){sep}{rest}'


def triptych_maps(axs, d, signed_rel_limit, rel_limit,
                  coherence_limit=COHERENCE_LIMIT, cbar=True,
                  signed_title=None, abs_title=None, coherence_title=None,
                  signed_cbar_label=SIGNED_CBAR_LABEL,
                  coherence_cbar_label=COHERENCE_CBAR_LABEL, scalars=None):
    """Draw the three panels INTO ``axs``; return the mappables.

    Both % limits are REQUIRED: blocks that each autoscaled would sit under one
    shared colorbar showing only one of them. ``coherence_limit`` defaults
    because it is not a measurement -- the ratio is in [0, 1] by Jensen and the
    panel is drawn on that whole range.

    ``scalars`` is :func:`cell_scalars`' dict; the three keys used are
    ``glob_bias_pct``, ``mape_pct`` and ``coherence``, each appended to its own
    panel title. Pass None for formula-only titles.

    The three panels are three different quantities on three scales -- two
    of them % and NOT comparable with each other (a) is signed and cancels
    across models, b) is a mean of magnitudes and cannot).
    """
    if signed_rel_limit is None or rel_limit is None:
        raise ValueError(
            'triptych_maps needs explicit signed_rel_limit and rel_limit -- '
            'autoscaling per block would put different colour scales under one '
            'colorbar.')
    sc = scalars or {}
    im_signed = ec.diverging_map(
        axs[0], d['bias_rel'], signed_rel_limit, cbar=cbar,
        cbar_label=signed_cbar_label,
        title=_titled(signed_title if signed_title is not None else _REL_TITLE,
                      sc.get('glob_bias_pct'), '{:+.1f}%'))
    im_abs = ec.sequential_map(
        axs[1], d['rel'], rel_limit, cbar=cbar, cbar_label='%',
        title=_titled(abs_title if abs_title is not None else _ABS_REL_TITLE,
                      sc.get('mape_pct'), '{:.1f}%'))
    im_coh = ec.sequential_map(
        axs[2], d['coherence'], coherence_limit, cbar=cbar, cmap='Fire',
        # bounded by Jensen, so no 'max' arrow for cells that cannot exist
        extend='neither', cbar_label=coherence_cbar_label,
        title=_titled(coherence_title if coherence_title is not None
                      else _COH_TITLE, sc.get('coherence'), '{:.2f}'))
    return [im_signed, im_abs, im_coh]


def triptych_panel(d, suptitle, save_path, signed_rel_limit, rel_limit,
                   coherence_limit=COHERENCE_LIMIT, scalars=None, **kwargs):
    """The three-panel LOMO figure for one cell -- the figure-owning half."""
    pplt = ec._pplt()
    fig, axs = pplt.subplots(nrows=1, ncols=3, proj='robin',
                             proj_kw=dict(lon_0=202), refwidth='58mm', share=0)
    triptych_maps(axs, d, signed_rel_limit, rel_limit,
                  coherence_limit=coherence_limit, scalars=scalars, **kwargs)
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


def suptitle_for(ds, slug):
    bits = [ds.attrs.get('title', slug),
            str(ds.attrs.get('years_label', '')),
            f'n={ds.attrs.get("n_models", "?")} models',
            _epoch_tag(ds),
            SIGN_NOTE]
    return '  |  '.join(b for b in bits if b)


def plot_cell_triptych(cell, ds, signed_rel_limit=None, rel_limit=None,
                       coherence_limit=COHERENCE_LIMIT, suptitle=None,
                       **kwargs):
    """The three-panel LOMO figure for one cell, from a loaded error_maps cache.

    The scalars on the three titles are deep-ocean means; no panel of this
    figure is over the OMZ region.
    """
    slug = cell_slug(cell)
    verify_cache(ds, slug)          # the identity the whole figure rests on
    d = signed_maps(ds)
    return triptych_panel(
        d, (suptitle if suptitle is not None else suptitle_for(ds, slug)),
        OUT_DIR / f'{slug}_lomo_triptych.png',
        signed_rel_limit, rel_limit, coherence_limit=coherence_limit,
        scalars=cell_scalars(d, ds), **kwargs)


# ===========================================================================
# entry points
# ===========================================================================
def compute(epoch=DEFAULT_EPOCH):
    """Sample every held-out model with its own LOMO run and write the cache
    (5 samples, 100 DDIM steps, years 81-100, stride 1)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = dict(device=device,
               sample_cfg=SampleCfg(n_samples=5, sampler='ddim', steps=100),
               year_window=DEFAULT_YEAR_WINDOW, years=None,
               months=list(range(12)), years_stride=DEFAULT_STRIDE,
               max_models=None, epoch=epoch)
    d = compute_oos(CELL, ctx)
    if d is None:
        raise SystemExit(f'{cell_slug(CELL)}: no held-out run usable')
    return save_cache(CELL, d, ctx)


def plot():
    ds = load_cache(CELL)
    if ds is None:
        raise SystemExit(f'no cache at {cache_path(CELL)}; run compute first')
    fig = plot_cell_triptych(CELL, ds,
                             signed_rel_limit=LOMO_SIGNED_REL_LIMIT,
                             rel_limit=LOMO_REL_LIMIT,
                             coherence_limit=LOMO_COHERENCE_LIMIT,
                             signed_title=LOMO_SIGNED_TITLE,
                             abs_title=LOMO_REL_TITLE,
                             coherence_title=LOMO_COH_TITLE,
                             signed_cbar_label=LOMO_SIGNED_CBAR,
                             coherence_cbar_label=LOMO_COH_CBAR)
    render_common.finish(fig, abc=True)
    return render_common.save(fig, 'lomo_error')


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
