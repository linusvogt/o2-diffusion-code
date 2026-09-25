"""Fig. 2 -- signed relative error of generated O2 for each predictor set.

In-sample runs, annual depth-integrated, years 81-100; every panel pools over
the models common to all five predictor sets. The map is the model-mean-then-
divide signed relative error ``100*(<gen>_m - <truth>_m)/<truth>_m``.

Extracted from general/eval/make_plots.py (``axis_sweep`` for the
``predictor_set`` sweep, compute and plot) with ``resolve_epoch`` from
general/eval/error_maps.py.

    python -m figures.fig02_predictor_set compute [--epoch 500]   # GPU
    python -m figures.fig02_predictor_set plot                    # CPU
"""
from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from figures import render_common

import numpy as np
import torch
import xarray as xr

from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, get_models,
    skill_map, skill_scalar, SampleCfg, FIG_DIR, DEFAULT_YEAR_WINDOW)


# ---------------------------------------------------------------------------
# pinned manuscript constants (main variant)
# ---------------------------------------------------------------------------
EPOCH = 500

# Nonlinear (square-root) diverging scale, ec.nonlinear_levels(16, 0.5, 8):
# boundaries are uniform in sqrt(|rel|), so the positive wing is
#     0   0.25   1   2.25   4   6.25   9   12.25   16
# with each interval 1/16 of the bar. The bulk of the deep ocean is a few per
# cent while the OMZs reach >10 %, which no linear limit resolves at once; the
# limit bounds the tail (~0.04 % of deep cells saturate). limit/n**2 = 0.25
# puts an integer at every even boundary, which is what the ticks label.
PREDICTOR_SET_REL_LIMIT = 16.0
PREDICTOR_SET_GAMMA = 0.5
PREDICTOR_SET_BINS = 8
# Ticks are bin EDGES only (the even boundaries = the integers), so every
# printed number names a colour boundary.
PREDICTOR_SET_CBAR_TICKS = (-16.0, -9.0, -4.0, -1.0, 0.0, 1.0, 4.0, 9.0, 16.0)

# A 2x2 block of the four predictor sets without DIC, and T+S+DIC beside them,
# spanning both rows (the fixed-aspect projection keeps it the same size and
# centres it vertically):
#
#     a) T+S            b) T+S+MLD
#                                        e) T+S+DIC
#     c) T+S+tau        d) T+S+tau+MLD
PREDICTOR_SET_LAYOUT = [[1, 2, 5],
                        [3, 4, 5]]
# Which case goes in which numbered slot, by option string (the labels carry
# TeX). reorder_cases raises if one is missing or unplaced.
PREDICTOR_SET_ORDER = ['thetao+so',
                       'thetao+so+mld',
                       'thetao+so+tauuo+tauvo',
                       'thetao+so+tauuo+tauvo+mld',
                       'thetao+so+dissic']

# Sampling of the manuscript cache: 10-member ensembles, 100-step DDIM, years
# 81-100 at stride 1, every panel over the models common to all predictor sets.
# The variant (resolution, field_type) the sweep sits on; it reduces to the
# plain 'predictor_set_common' cache name (variant_tag drops the defaults).
VARIANT = dict(resolution='annual', field_type='depthint')
N_SAMPLES = 10
STEPS = 100
SAMPLER = 'ddim'
YEARS_STRIDE = 1
COMMON_MODELS = True


# Cache root for the axis sweeps (see save_cache); figures go to ctx.fig_dir.
AXIS_CACHE_DIR = FIG_DIR / 'axis_sweep_cache'

# Aggregation behind every axis_sweep figure, stamped into the cache. NOT the
# same statistic as error_maps' 'model-mean-abs' (which takes |.| per model
# before averaging, so opposite-signed per-model biases cannot cancel) nor
# predictor_boxplot's 'global-mean-bias-per-model' (spatial mean first). Any
# figure built from these caches must say which one it is.
METRIC = 'model-mean-then-divide-signed-rel'

# Reference run all analyses start from; each overrides one axis.
BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# Predictor sets to compare (mld resolves per-resolution inside run_save_dir).
PREDICTOR_SETS = [
    ['thetao', 'so'],
    ['thetao', 'so', 'dissic'],
    ['thetao', 'so', 'tauuo', 'tauvo'],
    ['thetao', 'so', 'mld'],
    ['thetao', 'so', 'tauuo', 'tauvo', 'mld'],
]


# ---------------------------------------------------------------------------
# shared evaluation context (sampling settings + window + caps)
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    device: torch.device
    sample_cfg: SampleCfg
    year_window: tuple = DEFAULT_YEAR_WINDOW
    years: list = None          # explicit year(s), overriding window+stride
    months: list = field(default_factory=lambda: list(range(12)))
    years_stride: int = 1
    max_models: int = None
    epoch: int = None
    rel_limit: float = None
    fig_dir: Path = FIG_DIR
    # axis_sweep cache control
    reuse: bool = False
    overwrite: bool = False
    # pool every case of a sweep over the models common to all of them
    common_models: bool = False


# ---------------------------------------------------------------------------
# epoch resolution (from error_maps.py)
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
# generic axis sweep: compare skill across one config axis
# ---------------------------------------------------------------------------
# A *variant* is the (resolution, field_type) the whole sweep sits on -- the
# same comparison redrawn on another data axis, which is what the manuscript's
# appendix versions are. It is part of the cache/figure name; ``base_overrides``
# (the sweep's own fixed definition) is NOT.
_RES_TAG = {'annual': 'ann', 'monthly': 'mon'}
_FLD_TAG = {'depthint': 'int', 'density': 'dens'}


def variant_tag(variant):
    """'' for the default (BASE_CONFIG) variant, else e.g. '_mon_dens'.

    Canonical: a key whose value already IS the BASE_CONFIG default contributes
    nothing, so ``{}`` and ``{'resolution': 'annual', 'field_type':
    'depthint'}`` name the same cache instead of computing it twice. A sweep
    that *varies* one of these axes simply never pins it in ``variant``.
    """
    tags = {'resolution': _RES_TAG, 'field_type': _FLD_TAG}
    parts = [table[variant[key]] for key, table in tags.items()
             if key in variant and variant[key] != BASE_CONFIG[key]]
    return ('_' + '_'.join(parts)) if parts else ''


def sweep_slug(name, variant=None, common_models=False):
    """Cache/figure stem. ``common_models`` is IN the name, not just an attr:
    the two fleet conventions are different scientific content, so they must
    not overwrite each other's cache."""
    return f'{name}{variant_tag(variant)}{"_common" if common_models else ""}'


def cache_path(slug):
    return AXIS_CACHE_DIR / f'{slug}.nc'


def years_label(ctx):
    """What was actually averaged in time -- explicit years, or window+stride.

    The window alone would mislabel a strided or --years run (the figure would
    claim 'yrs 81-100' after sampling only year 81).
    """
    if ctx.years is not None:
        return 'yrs ' + ','.join(str(y) for y in ctx.years)
    lo, hi = ctx.year_window
    return f'yrs {lo}-{hi}' + (f' /{ctx.years_stride}'
                               if ctx.years_stride != 1 else '')


def save_cache(slug, cases, ctx, meta):
    """Persist the per-case relative-error maps + provenance.

    Sampling is the only GPU cost; the maps themselves are small, so caching
    them makes every restyle/rescale a CPU replot. Labels carry TeX ($\\tau_x$), which does not
    round-trip reliably through a netCDF string variable, so they live in an
    attr joined on '|' -- the same trick predictor_boxplot.save_cache uses.
    """
    ds = xr.Dataset(
        {'rel': xr.concat([c['rel'] for c in cases], dim='case'),
         'scalar': ('case', [float(c['scalar']) for c in cases])},
        coords={'case': np.arange(len(cases))})
    ds.attrs.update(
        slug=slug, name=meta['name'], suptitle=meta['suptitle'],
        vary=meta['vary'],
        resolution=meta['resolution'], field_type=meta['field_type'],
        base_predictors=' '.join(meta['base_predictors']),
        labels=' | '.join(c['label'] for c in cases),
        options=' | '.join(c['option'] for c in cases),
        # The model fleet is IN _PROVENANCE: a case drawn over a smaller fleet
        # must not silently replot later.
        case_models=' | '.join(' '.join(c['models']) for c in cases),
        case_n_models=' '.join(str(len(c['models'])) for c in cases),
        epochs=' '.join(str(c['epoch']) for c in cases),
        land_fill=' '.join(c['land_fill'] for c in cases),
        epoch_arg=str(ctx.epoch),
        n_samples=ctx.sample_cfg.n_samples, steps=ctx.sample_cfg.steps,
        sampler=ctx.sample_cfg.sampler,
        years_stride=ctx.years_stride,
        year_window=' '.join(str(y) for y in ctx.year_window),
        years_label=years_label(ctx),
        n_months=len(ctx.months) if meta['resolution'] == 'monthly' else 1,
        common_models=int(ctx.common_models),
        metric=METRIC)
    AXIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(slug)
    ds.to_netcdf(path)
    logging.info(f'  cached {path}')
    return ds


def load_cache(slug):
    path = cache_path(slug)
    return xr.open_dataset(path).load() if path.is_file() else None


# Attrs that define WHAT was computed: a cache written under different values is
# not the figure the current arguments ask for. `case_models` and `labels` are
# in here so a fleet that grew (or a case that appeared) invalidates the cache
# instead of being replotted at its old composition.
_PROVENANCE = ('epoch_arg', 'n_samples', 'steps', 'sampler', 'years_label',
               'metric', 'labels', 'case_models', 'common_models')


def cache_matches(ds, ctx, cases=None):
    """(ok, differences) -- does this cache describe the requested computation?

    ``cases`` (the fleet resolved right now, cheap: get_models only globs
    filenames) enables the fleet check; without it the two fleet-derived keys
    are skipped, which is what a pure ``--reuse`` replot wants.
    """
    want = {'epoch_arg': str(ctx.epoch),
            'n_samples': ctx.sample_cfg.n_samples,
            'steps': ctx.sample_cfg.steps,
            'sampler': ctx.sample_cfg.sampler,
            'years_label': years_label(ctx),
            'common_models': int(ctx.common_models),
            'metric': METRIC}
    if cases is not None:
        want['labels'] = ' | '.join(c['label'] for c in cases)
        want['case_models'] = ' | '.join(' '.join(c['models']) for c in cases)
    diffs = [f'{k}: cache={ds.attrs.get(k, "?")} requested={want[k]}'
             for k in _PROVENANCE
             if k in want and str(ds.attrs.get(k, '')) != str(want[k])]
    return (not diffs), diffs


def cases_from_cache(ds):
    """Rebuild the plot-ready case list from a cache (no GPU, no run loading).

    ``option`` (the '+'-joined option string, e.g. 'thetao+so+mld') comes along
    because ``label`` carries TeX and is a display string; the option is the
    stable identifier a caller can select or reorder by.
    """
    labels = str(ds.attrs['labels']).split(' | ')
    options = str(ds.attrs.get('options', '')).split(' | ')
    return [dict(label=lab, option=options[i] if i < len(options) else '',
                 rel=ds['rel'].isel(case=i),
                 scalar=float(ds['scalar'].isel(case=i)))
            for i, lab in enumerate(labels)]


def reorder_cases(cases, order):
    """``cases`` in the given order of option strings.

    Loud on a miss, deliberately: a mistyped option that silently dropped a
    panel would produce a figure that looks fine and is missing a predictor set,
    and with a fixed layout array the panel count must match exactly anyway.
    """
    by_option = {c['option']: c for c in cases}
    missing = [o for o in order if o not in by_option]
    if missing:
        raise KeyError(f'no case for option(s) {missing}; cache has '
                       f'{sorted(by_option)}')
    extra = [o for o in by_option if o not in order]
    if extra:
        raise KeyError(f'order omits {extra}; every case must be placed')
    return [by_option[o] for o in order]


def plot_from_cache(ds, save_path, rel_limit=None, ncols=None, suptitle=None,
                    hide_empty=False, order=None, layout=None, cbar='each',
                    gamma=None, bins=None, cbar_ticks=None):
    """Draw the panel figure from a cache. Returns the figure (not closed).

    ``order`` (option strings) and ``layout`` (an ultraplot subplot array) go
    together: the array's numbering decides where each case lands, so a
    non-default arrangement needs the cases in the matching order. Both default
    to None (plain reading-order grid).

    ``gamma``/``bins``/``cbar_ticks`` (nonlinear diverging scale) are forwarded
    to ``ec.rel_error_panel`` and likewise default to off.
    """
    cases = cases_from_cache(ds)
    if order:
        cases = reorder_cases(cases, order)
    return ec.rel_error_panel(
        cases, ds.attrs['suptitle'] if suptitle is None else suptitle,
        save_path, rel_limit=rel_limit, ncols=ncols, hide_empty=hide_empty,
        layout=layout, cbar=cbar, gamma=gamma, bins=bins,
        cbar_ticks=cbar_ticks)


def _resolve_cases(ctx, name, vary, options, label_fn, base):
    """Which cases exist right now + their fleets, WITHOUT loading any model.

    ``get_models`` only globs filenames, so this is cheap enough to run before
    deciding whether a cache is stale.

    With ``ctx.common_models`` the cases are reduced to the models available to
    EVERY case, so a panel-to-panel difference is the config and not which
    models happened to be trained -- the same fairness rule
    ``predictor_boxplot._common_model_names`` applies across its five columns.
    Intersecting on model NAMES, not on the objects, matters because each
    config's ``get_models`` picks its own member for a model.
    """
    cases = []
    for opt in options:
        save_dir = run_save_dir(base, **{vary: opt})
        epoch = resolve_epoch(save_dir, ctx.epoch)
        if epoch is None:
            logging.info(f'[skip] {name}={label_fn(opt)}: no run at {save_dir.name}')
            continue
        cfg = make_config(base, **{vary: opt})
        models = get_models(cfg, max_models=ctx.max_models)
        cases.append(dict(label=label_fn(opt), option=_opt_str(opt),
                          save_dir=save_dir, epoch=epoch,
                          models=sorted({m.model for m in models}),
                          model_objs=models))
    if ctx.common_models and len(cases) > 1:
        common = set.intersection(*(set(c['models']) for c in cases))
        if not common:
            logging.warning(f'  [warn] {name}: no model is common to all '
                            f'{len(cases)} cases -- keeping per-case fleets')
            return cases
        dropped = sorted(set.union(*(set(c['models']) for c in cases)) - common)
        for c in cases:
            c['model_objs'] = [m for m in c['model_objs'] if m.model in common]
            c['models'] = sorted(common)
        logging.info(f'  common-model intersection: {len(common)} models '
                     f'kept, {len(dropped)} dropped ({", ".join(dropped)})')
    return cases


def _opt_str(opt):
    return '+'.join(opt) if isinstance(opt, (list, tuple)) else str(opt)


def _warn_case_fleets(slug, cases):
    """Say so when the panels of one sweep are over different model fleets.

    Each case resolves its OWN models, so the panels are model-means over
    different fleets -- typically the DIC set is short, and the density cells
    drop the corrupt-sigma members. Note this is NOT what predictor_boxplot
    does: it intersects models across its five columns
    (``_common_model_names``), so the two figures are matched differently even
    where they compare the same predictor sets. Part of any panel-to-panel
    difference here is therefore a composition effect.
    """
    fleets = {c['label']: frozenset(c['models']) for c in cases}
    if len(set(fleets.values())) <= 1:
        return
    common = set.intersection(*(set(f) for f in fleets.values()))
    union = set.union(*(set(f) for f in fleets.values()))
    detail = ', '.join(f'{lab}: n={len(f)}' for lab, f in fleets.items())
    logging.warning(
        f'  [warn] {slug}: panels are over DIFFERENT model fleets '
        f'({len(common)} common of {len(union)}) -- {detail}. Part of any '
        f'panel-to-panel difference is composition, not skill.')


def axis_sweep(ctx, name, vary, options, label_fn, suptitle,
               base_overrides=None, variant=None):
    """Compare skill across one config axis, on one (resolution, field_type).

    Samples only when there is no usable cache; ``ctx.reuse`` forbids sampling
    outright and ``ctx.overwrite`` forces it.
    """
    base = make_config(BASE_CONFIG, **(base_overrides or {}), **(variant or {}))
    slug = sweep_slug(name, variant, ctx.common_models)
    ds = None if ctx.overwrite else load_cache(slug)

    if ds is not None:
        # The fleet check needs today's runs; skip it under --reuse, which
        # promises not to touch a GPU and may run where the data is absent.
        probe = None if ctx.reuse else _resolve_cases(
            ctx, name, vary, options, label_fn, base)
        ok, diffs = cache_matches(ds, ctx, cases=probe)
        if ok:
            logging.info(f'  reusing cache {cache_path(slug).name}')
        elif ctx.reuse:
            logging.info(f'  [warn] cache {cache_path(slug).name} was made with '
                         f'different settings ({"; ".join(diffs)}) -- plotting '
                         f'it anyway because --reuse is set')
        else:
            logging.info(f'  cache stale ({"; ".join(diffs)}) -- recomputing')
            ds = None

    if ds is None:
        if ctx.reuse:
            logging.info(f'[skip] {slug}: no cache and --reuse given')
            return None
        cases = _resolve_cases(ctx, name, vary, options, label_fn, base)
        drawn = []
        for c in cases:
            run = load_run(c['save_dir'], ctx.device, ctx.epoch)
            d = skill_map(run, c['model_objs'], ctx.year_window, ctx.sample_cfg,
                          months=ctx.months, years_stride=ctx.years_stride,
                          years=ctx.years)
            if d is None:
                continue
            drawn.append({**c, 'rel': d['rel'], 'scalar': skill_scalar(d['rel']),
                          'land_fill': run.scaling.get('land_fill', 'zero')})
        if not drawn:
            logging.info(f'[skip] {slug}: no runs available')
            return None
        _warn_case_fleets(slug, drawn)
        ds = save_cache(slug, drawn, ctx, dict(
            name=name, vary=vary, suptitle=suptitle,
            resolution=base['resolution'], field_type=base['field_type'],
            base_predictors=base['predictors']))

    fig = plot_from_cache(ds, ctx.fig_dir / f'{slug}.png',
                          rel_limit=ctx.rel_limit)
    _close(fig)
    return ds


def _close(fig):
    """Release a saved figure (import matplotlib lazily)."""
    import matplotlib.pyplot as plt
    plt.close(fig)


def _pred_label(pred_set):
    short = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}
    return '+'.join(short.get(p, p) for p in pred_set)


# Declarative definition of the axis sweep (only the predictor-set one here),
# so a caller can ask for it by name + variant without re-stating its options.
# `fixed` are the axes the sweep pins by definition; `vary` is the axis it
# compares along -- a variant must never pin that same axis.
AXIS_SWEEPS = {
    'predictor_set': dict(
        vary='predictors', options=PREDICTOR_SETS, label_fn=_pred_label,
        suptitle='Relative error vs predictor set (in-sample, yrs 81-100)',
        fixed=None),
}


def run_axis_sweep(ctx, name, variant=None):
    """Draw one axis sweep by registry name, on an optional variant."""
    spec = AXIS_SWEEPS[name]
    if variant and spec['vary'] in variant:
        raise ValueError(
            f"sweep '{name}' varies {spec['vary']}; a variant cannot pin it")
    return axis_sweep(ctx, name, spec['vary'], spec['options'],
                      spec['label_fn'], spec['suptitle'],
                      base_overrides=spec['fixed'], variant=variant)


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def compute(epoch=EPOCH):
    """Sample every predictor-set run and write the ``predictor_set_common``
    cache (``make_plots.py --plots predictor_set --sweep-variants ann_int
    --common-models``)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = Ctx(device=device,
              sample_cfg=SampleCfg(n_samples=N_SAMPLES, sampler=SAMPLER,
                                   steps=STEPS),
              years_stride=YEARS_STRIDE, epoch=epoch,
              common_models=COMMON_MODELS)
    return run_axis_sweep(ctx, 'predictor_set', VARIANT)


def plot():
    """Replot the manuscript figure from the cache (no GPU)."""
    slug = sweep_slug('predictor_set', VARIANT, COMMON_MODELS)
    ds = load_cache(slug)
    if ds is None:
        raise FileNotFoundError(f'no cache at {cache_path(slug)}; run compute')
    fig = plot_from_cache(ds, FIG_DIR / f'{ds.attrs["slug"]}.png',
                          rel_limit=PREDICTOR_SET_REL_LIMIT,
                          layout=PREDICTOR_SET_LAYOUT,
                          order=PREDICTOR_SET_ORDER, cbar='last',
                          gamma=PREDICTOR_SET_GAMMA, bins=PREDICTOR_SET_BINS,
                          cbar_ticks=PREDICTOR_SET_CBAR_TICKS)
    render_common.finish(fig, abc=True)
    return render_common.save(fig, 'predictor_set')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    p.add_argument('--epoch', type=int, default=EPOCH)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        compute(args.epoch)
    else:
        plot()
