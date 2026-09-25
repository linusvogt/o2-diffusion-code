"""Fig. 4 -- seasonal cycle of regional-mean O2, truth vs generated.

Month-of-year climatology of area-weighted, deep-ocean (>= 2000 m) regional
means of depth-integrated O2 over RECCAP2 regions, for the MONTHLY in-sample
1pctCO2 T+S runs: truth (black) and generated from depth-integrated (blue) and
density-layer (orange) predictors; shading is +/-1 sigma of the generated
ensemble, model-averaged. Absolute view (not the anomaly one), no per-panel
annotations.

Extracted from general/eval/seasonal_cycle.py (compute and plot).

    python -m figures.fig04_seasonal_cycle compute [--epoch 500]   # GPU
    python -m figures.fig04_seasonal_cycle plot                    # CPU
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from figures import render_common

import numpy as np
import torch
import xarray as xr

import paths
from figures import eval_common as ec
from figures.eval_common import (
    make_config, generate_samples, SampleCfg, DEPTH_INT, FIG_DIR,
    DEFAULT_YEAR_WINDOW, MONTH_NAMES, O2, bathy_mask, BATHY_MIN_DEPTH)


# ---------------------------------------------------------------------------
# pinned manuscript constants (main figure)
# ---------------------------------------------------------------------------
# The absolute view of the one cached series; 5-member ensembles, 100-step
# DDIM, every 5th year of 81-100 with all 12 months.
MAIN_CELL = dict(view='absolute')
N_SAMPLES = 5
STEPS = 100
SAMPLER = 'ddim'
YEARS_STRIDE = 5


AREA_FILE = paths.GRIDAREA
RECCAP_FILE = paths.RECCAP2_MASK

# Run this analysis is about: the monthly in-sample 1pctCO2 pair.
BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='monthly', train_test_split='insample')
FIELD_TYPES = ['depthint', 'density']

# Both field types are read at the SAME epoch (the largest both monthly
# in-sample runs have reached), so training length is not a confound in the
# depthint-vs-density comparison.
DEFAULT_EPOCH = 500

# ---------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------
# RECCAP2 basin variables carry integer sub-region codes (0 = outside basin);
# the code -> biome names, from each variable's `region_name` attribute, are:
#   atlantic : 1 NA SPSS, 2 NA STSS, 3 NA STPS, 4 AEQU, 5 SA STPS, 6 MED
#   pacific  : 1 NP SPSS, 2 NP STSS, 3 NP STPS, 4 PEQU-W, 5 PEQU-E, 6 SP STPS
#   indian   : 1 Arabian Sea, 2 Bay of Bengal, 3 Equatorial Indian, 4 S. Indian
#   southern : 1 SO STSS, 2 SO SPSS, 3 SO ICE
# `codes=None` means "the whole basin" (any code > 0).
# ``label`` is what the per-region panel titles use (``short`` is for a
# categorical x-axis in the eval module's amplitude figure).
REGIONS = {
    'global':   dict(label='Global',              short='Global',
                     var=None,       codes=None),
    'natl':     dict(label='N. Atlantic subpolar', short='N.Atl. SP',
                     var='atlantic', codes=(1, 2)),
    'natl_stps': dict(label='N. Atlantic subtrop.', short='N.Atl. ST',
                      var='atlantic', codes=(3,)),
    'satl':     dict(label='S. Atlantic subtrop.', short='S.Atl. ST',
                     var='atlantic', codes=(5,)),
    'npac':     dict(label='N. Pacific subpolar',  short='N.Pac. SP',
                     var='pacific',  codes=(1, 2)),
    'eqpac':    dict(label='Equatorial Pacific',   short='Eq. Pac.',
                     var='pacific',  codes=(4, 5)),
    'spac':     dict(label='S. Pacific subtrop.',  short='S.Pac. ST',
                     var='pacific',  codes=(6,)),
    'indian':   dict(label='Indian',               short='Indian',
                     var='indian',   codes=None),
    'southern': dict(label='Southern Ocean',       short='S. Ocean',
                     var='southern', codes=None),
    'arctic':   dict(label='Arctic',               short='Arctic',
                     var='arctic',   codes=None),
}
DEFAULT_REGIONS = ['global', 'natl', 'npac', 'eqpac', 'indian', 'southern']

_AREA_CACHE = {}


def _area():
    """Grid-cell area (lat, lon), loaded once."""
    if 'area' not in _AREA_CACHE:
        _AREA_CACHE['area'] = (xr.open_dataset(AREA_FILE).cell_area
                               .reset_coords(drop=True).load())
    return _AREA_CACHE['area']


def _reccap():
    """The RECCAP2 region-mask dataset, loaded once."""
    if 'reccap' not in _AREA_CACHE:
        _AREA_CACHE['reccap'] = xr.open_dataset(RECCAP_FILE).load()
    return _AREA_CACHE['reccap']


def region_selector(name):
    """Boolean (lat, lon) DataArray: True inside region ``name``.

    ``'global'`` is every cell (the ocean/land distinction comes from the data's
    own NaNs, which the weighted mean skips). Otherwise the basin variable is
    thresholded at > 0, or restricted to the listed biome codes.
    """
    spec = REGIONS[name]
    area = _area()
    if spec['var'] is None:
        return xr.ones_like(area, dtype=bool)
    m = _reccap()[spec['var']]
    if spec['codes'] is None:
        return m > 0
    return m.isin(list(spec['codes']))


def region_weights(name, min_depth=BATHY_MIN_DEPTH):
    """Area weights for region ``name``, zero outside it and on shallow cells.

    Returns ``(weights, area_fraction)`` where ``area_fraction`` is the share of
    the region's *ocean* area that survives the >= ``min_depth`` bathymetry cut
    -- the fields are depth-integrated 0-2000 m, so shallower cells have a
    truncated integral and are excluded here exactly as in
    ``eval_common.skill_scalar``.

    The fraction is reported against ocean area, not total area, so it means the
    same thing in every panel. The basin masks are ocean-only already, but the
    ``'global'`` selector is every cell, and dividing by the whole globe would
    have made its number a share of Earth's surface (0.57) rather than of the
    ocean (0.81) -- not comparable with the basins beside it.

    The *weights* need no such correction: land is NaN in the fields and
    xarray's weighted mean skips it, so the curves are unaffected either way.
    """
    area = _area().fillna(0.0)
    sel = region_selector(name)
    w_all = area.where(sel, 0.0)
    bm = bathy_mask(min_depth)
    w = w_all if bm is None else w_all.where(bm.notnull(), 0.0)
    # Denominator only -- leaving the weights untouched keeps the curves (and
    # every cached run) byte-identical; the seamask is applied here purely so
    # the reported fraction is per ocean area in every panel.
    ocean = float(area.where(sel & (_reccap()['seamask'] > 0), 0.0).sum())
    frac = float(w.sum()) / ocean if ocean > 0 else np.nan
    return w, frac


# ---------------------------------------------------------------------------
# reduction: (sample, model, time, lat, lon) fields -> regional time series
# ---------------------------------------------------------------------------
def reduce_to_regions(ds, regions, min_depth=BATHY_MIN_DEPTH):
    """Collapse a ``generate_samples`` output to area-weighted regional means.

    ``ds`` has ``gen (sample, model, time, lat, lon)`` and
    ``truth (model, time, lat, lon)`` in the target's native depth-integrated
    units. Returns a Dataset on dims ``(sample, model, time, region)`` in
    mol/m^3:

        gen     regional mean of each ensemble member
        truth   regional mean of the truth
        rmse    within-region spatial RMSE of the ensemble-mean field
        spread  within-region RMS of the ensemble standard deviation
        area_frac  (region) deep-ocean area fraction retained

    Land cells are NaN in the fields and xarray's weighted mean skips them, so
    the weights need only encode the region and the bathymetry cut.
    """
    gen = ds['gen'] / DEPTH_INT
    truth = ds['truth'] / DEPTH_INT
    gen_mean = gen.mean('sample')
    gen_std = gen.std('sample')
    sq_err = (gen_mean - truth) ** 2
    sq_spread = gen_std ** 2

    parts, fracs = [], []
    for name in regions:
        w, frac = region_weights(name, min_depth)
        part = xr.Dataset(dict(
            gen=gen.weighted(w).mean(('lat', 'lon')),
            truth=truth.weighted(w).mean(('lat', 'lon')),
            rmse=np.sqrt(sq_err.weighted(w).mean(('lat', 'lon'))),
            spread=np.sqrt(sq_spread.weighted(w).mean(('lat', 'lon'))),
        ))
        parts.append(part.assign_coords(region=name))
        fracs.append(frac)
        logging.info(f'    region {name:9s}: deep-ocean area fraction '
                     f'{frac:.2f}')

    out = xr.concat(parts, 'region')
    out['area_frac'] = xr.DataArray(fracs, dims='region',
                                    coords={'region': list(regions)})
    return out


def to_year_month(ts):
    """Reshape the ``time`` dim into ``(year, month)`` and drop partial years.

    Averaging month-by-month across years only removes the 1pctCO2 trend if
    every contributing year has all twelve months; a year missing a month would
    leave a month-dependent trend residual. So a (model, region, year) whose
    truth is not finite in all twelve months is dropped whole, for both truth
    and gen. Adds ``n_years`` (model, region) = complete years retained.
    """
    ts = ts.set_index(time=['year', 'month']).unstack('time')
    complete = ts['truth'].notnull().all('month')          # (model, region, year)
    n_years = complete.sum('year')
    kept = ts[['gen', 'truth', 'rmse', 'spread']].where(complete)
    kept['area_frac'] = ts['area_frac']
    kept['n_years'] = n_years
    return kept


def seasonal_cycle(ts):
    """Average the complete years away: ``(..., year, month) -> (..., month)``."""
    keep = ts[['area_frac', 'n_years']]
    cyc = ts[['gen', 'truth', 'rmse', 'spread']].mean('year')
    return cyc.merge(keep)


# Only named in the (unused here) annotated-title variant of plot_cycle.
AMPLITUDE_DEF = 'max-min'


def curve_mean_bias(cyc):
    """Annual-mean bias **of the drawn model-mean curves**, per (ft, region).

    The curves are ``.mean('model')`` of gen and of truth, so their gap is the
    model *mean* bias, while the eval module's annotated number is the model
    *median*; the fleet is skewed, so the two differ (see :func:`plot_cycle`).
    """
    gen = cyc['gen'].mean('sample')
    parts = []
    for region in [str(r) for r in cyc['region'].values]:
        t = _truth(cyc, region).mean('month')          # already model-mean
        g = gen.sel(region=region).mean('model').mean('month')
        parts.append((100 * (g - t) / t).assign_coords(region=region))
    return xr.concat(parts, 'region')


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------
def compute(device, sample_cfg, regions, field_types=FIELD_TYPES,
            year_window=DEFAULT_YEAR_WINDOW, years_stride=1, years=None,
            epoch=DEFAULT_EPOCH, max_models=None, min_depth=BATHY_MIN_DEPTH,
            base=None, cache_file=None):
    """Sample each field type, reduce to regional (year, month) series, stack.

    Returns a Dataset on ``(field_type, sample, model, region, year, month)``.
    Written to ``cache_file`` after *each* field type, so a job that dies in the
    second one still leaves the first on disk.
    """
    base = base or BASE_CONFIG
    attrs = _provenance_attrs(base, sample_cfg, epoch, year_window,
                              years_stride, years, min_depth, field_types)
    parts = []
    for ft in field_types:
        logging.info(f'[seasonal] generating {ft} ...')
        ds = generate_samples(
            base, device=device, sample_cfg=sample_cfg, field_type=ft,
            year_window=year_window, years_stride=years_stride, years=years,
            months=list(range(12)), epoch=epoch, max_models=max_models,
            want_truth=True)
        if ds is None:
            logging.info(f'[seasonal] no data for {ft}, skipping')
            continue
        logging.info(f'  reducing {ft} to {len(regions)} region(s)')
        ts = to_year_month(reduce_to_regions(ds, regions, min_depth))
        parts.append(ts.assign_coords(field_type=ft))
        del ds
        if cache_file is not None and parts:
            _write_cache(_stamp(xr.concat(parts, 'field_type', join='outer'),
                                attrs), cache_file)
    if not parts:
        return None
    return _stamp(xr.concat(parts, 'field_type', join='outer'), attrs)


def _provenance_attrs(base, sample_cfg, epoch, year_window, years_stride,
                      years, min_depth, field_types):
    """What this cache IS, stamped on it -- not just what it holds.

    A cache that does not say which epoch (ensemble size, years ...) it came
    from cannot be checked against the figure it is drawn into, so it is
    stamped here, at the one place that has the arguments in hand.
    """
    lo, hi = year_window
    label = ('yrs ' + ','.join(str(y) for y in years) if years else
             f'yrs {lo}-{hi}' + (f' /{years_stride}' if years_stride != 1
                                 else ''))
    return dict(
        epoch=int(epoch), n_samples=int(sample_cfg.n_samples),
        steps=int(sample_cfg.steps), sampler=str(sample_cfg.sampler),
        years_label=label, year_window=f'{lo} {hi}',
        # 0 encodes "no bathymetry cut" (netCDF attrs cannot hold None).
        min_depth=float(min_depth) if min_depth else 0.0,
        field_types=' '.join(field_types),
        predictors=' '.join(base['predictors']),
        experiment=' '.join(base['experiments']),
        split=str(base['train_test_split']),
        resolution=str(base['resolution']))


def _stamp(ts, attrs):
    """Attach the provenance attrs + the fleet actually sampled.

    ``models`` is space-joined, as every other eval cache here stamps it; the
    model *dimension* is kept as well and is the authority.
    """
    ts.attrs.update(attrs)
    ts.attrs['models'] = ' '.join(str(m) for m in np.atleast_1d(
        ts['model'].values))
    # deliberately NOT an n_models attr: the cache's fleet is not the fleet a
    # figure draws (common_coverage drops models without complete coverage
    # under every field type).
    return ts


def _write_cache(ts, cache_file):
    cache_file = Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    # string coords (model, region, field_type) survive netCDF fine; drop the
    # scalar label coords that differ per field type and would clash.
    ts.to_netcdf(cache_file)
    logging.info(f'  cached regional time series -> {cache_file}')


# Truth wears ink (it is the reference, not a category); the two field types get
# the Okabe-Ito colourblind-safe blue/vermillion pair, and each series also has
# its own marker so identity never rests on colour alone.
STYLE = {
    'truth':    dict(color='#333333', marker=None, ls='-', lw=2.2, label='truth'),
    'depthint': dict(color='#0072B2', marker='o', ls='-', lw=2.0,
                     label='generated (depth-int.)'),
    'density':  dict(color='#D55E00', marker='s', ls='-', lw=2.0,
                     label='generated (density)'),
}
# On the error/skill figures the curve is an error, not "the generated field",
# so it is labelled by field type alone.
SHORT = {'depthint': 'depth-int.', 'density': 'density'}
MONTH_TICKS = list(range(12))
# Single-letter month labels: twelve ticks on a ~65 mm panel leave no room for
# 'Jan'..'Dec' unrotated, and rotating them runs the labels into the title of
# the panel below.
MONTH_LABELS = [MONTH_NAMES[i][0] for i in MONTH_TICKS]


def _panels(n):
    """Region panel grid. Panels keep independent y-limits (``share=0``) because
    regional O2 levels differ; only the leftmost column gets a y-label (see
    ``_fmt``), which is what keeps the labels out of the neighbouring panel."""
    pplt = ec._pplt()
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    fig, axs = pplt.subplots(nrows=nrows, ncols=ncols, refwidth='65mm',
                             refheight='45mm', share=0, hspace=5.0, wspace=6.0)
    return pplt, fig, axs


def common_coverage(ts):
    """Restrict to the (model, year) coverage every field type shares.

    Two different confounds, both of which would otherwise land inside the
    depthint-vs-density difference the figures exist to show:

    *Models* -- density model discovery drops members whose sigma-surface file
    is all-NaN (``data_loading.load_density_qc``), so the density run can have a
    smaller model set than depthint. Averaging each field type over its own set
    puts the model set and the field type in the same comparison.

    *Years* -- a member can have a complete year under one field type and not
    the other. Averaging over different years leaves a different amount of the
    1pctCO2 trend in each, which reads as a field-type difference. So the
    per-(model, year) completeness masks are intersected, not just counted.

    Together these make ``truth`` genuinely identical across field types, so
    there is one truth curve rather than two near-copies (see :func:`_truth`,
    which still checks rather than assumes).
    """
    complete = ts['truth'].notnull().all('month')      # field_type,model,region,year
    shared = complete.all('field_type')                # model, region, year
    ok = shared.any('year').all('region')
    keep = [str(m) for m, v in zip(ts['model'].values, ok.values) if bool(v)]
    dropped = [str(m) for m in ts['model'].values if str(m) not in keep]
    if dropped:
        logging.info(f'[seasonal] restricting to {len(keep)} model(s) common '
                     f'to all field types; dropped {dropped}')
    if not keep:
        raise SystemExit('no model has a complete year in every field type')

    n_before = int(complete.sum().values)
    out = ts.sel(model=keep)
    shared = shared.sel(model=keep)
    kept = out[['gen', 'truth', 'rmse', 'spread']].where(shared)
    kept['area_frac'] = out['area_frac']
    kept['n_years'] = shared.sum('year')
    n_after = int((kept['truth'].notnull().all('month')).sum().values)
    if n_after < n_before:
        logging.info(f'[seasonal] dropped {n_before - n_after} '
                     '(field_type, model, region, year) cell(s) not complete '
                     'under every field type')
    return kept


TRUTH_RTOL = 1e-3      # relative disagreement worth warning about


def _truth(cyc, region, per_model=False):
    """Truth curve for a region, with the ``field_type`` axis collapsed.

    ``truth`` is the same CMIP data for every field type, but concatenation
    gives it a ``field_type`` dim; leaving it in would make every curve and
    every relative-error denominator two-dimensional.

    The entries *should* be identical -- but ``common_models`` only requires
    each model to have contributed some complete year, not the same *set* of
    years. If a member has a missing month in year 86 under one field type
    only, the two field types average over different years and the 1pctCO2
    trend leaves a different offset in each. Collapsing with a mean would hide
    exactly that. So the disagreement is measured and warned about rather than
    assumed away.
    """
    t = cyc['truth'].sel(region=region)
    if t.sizes.get('field_type', 1) > 1:
        spread = (t.max('field_type') - t.min('field_type'))
        rel = float(np.nanmax(np.abs(spread / t.mean('field_type')).values))
        if np.isfinite(rel) and rel > TRUTH_RTOL:
            logging.info(
                f'  [warn] {region}: truth differs by {rel * 100:.2f}% between '
                'field types -- they are not averaging over the same years, so '
                'the depthint/density difference carries a trend offset')
        t = t.mean('field_type')
    return t if per_model else t.mean('model')


def _finish(fig, axs, n, save_path, suptitle):
    for ax in axs[n:]:
        ax.set_visible(False)
    axs[0].legend(loc='best', ncols=1, frame=False, fontsize=7)
    fig.format(suptitle=suptitle)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    return fig


def _region_title(cyc, region, st=None):
    """Panel title: the region name, optionally + the two scalar biases.

    The deep-ocean area fraction each region retains is a guard on how much of
    the region the curve rests on, but it belongs in the log, not on the
    figure.

    With ``st`` (an :func:`amplitude_stats` Dataset) a second line carries the
    median-over-models annual-mean bias and seasonal-amplitude bias, one pair
    per field type in the order they are drawn.

    Both numbers go on **both** cycle figures, deliberately, even though each
    figure can only show one of them: on the absolute panel the mean bias is
    the content but is far too small to see (a 0.1% offset on curves whose
    y-range is 0.09-0.25 mol/m^3 leaves truth and generated exactly on top of
    each other), and on the anomaly panel the mean has been subtracted out
    while the amplitude is what the curves show. Carrying the pair on each
    keeps them comparable without holding two figures side by side.
    """
    label = REGIONS[str(region)]['label']
    if st is None:
        return label
    fts = [str(f) for f in st['field_type'].values]
    s = st.sel(region=region)
    mb = '/'.join(_signed(float(s['mean_bias'].sel(field_type=f).median()), 2)
                  for f in fts)
    ab = '/'.join(_signed(float(s['amp_bias'].sel(field_type=f).median()), 0)
                  for f in fts)
    return f'{label}\nmean {mb}%   ampl. {ab}%'


def _signed(v, nd):
    """``+1.2``-style label, without a signed zero.

    The annual-mean biases round to 0.0 at one decimal and the amplitude biases
    to 0 at zero decimals; ``-0.0%`` reads as a formatting bug rather than as
    "no bias", so an exact zero loses its sign.
    """
    s = f'{v:+.{nd}f}'
    return s[1:] if float(s) == 0 else s


# ---------------------------------------------------------------------------


def _fmt(ax, ylabel):
    """Month axis + y-label, the latter only on the leftmost column.

    Panels have independent y-limits, so every panel keeps its own tick labels;
    repeating the y-*label* on the inner columns is what pushed it into the
    neighbouring panel.
    """
    first_col = ax.get_subplotspec().colspan.start == 0
    ax.format(xlocator=MONTH_TICKS, xformatter=MONTH_LABELS, xlim=(-0.4, 11.4),
              ylabel=ylabel if first_col else '', gridalpha=0.25)


def plot_cycle(cyc, save_path, anomaly=False, st=None):
    """Seasonal cycle of regional-mean O2: truth vs both generated field types.

    ``anomaly=True`` plots the departure from each curve's own annual mean --
    the cycle's shape and amplitude, which the absolute panel hides under the
    (much larger) mean.

    ``st`` (:func:`amplitude_stats`) adds the two scalar biases to each panel
    title; it is passed on the anomaly figure only (see :func:`_region_title`).
    """
    regions = [str(r) for r in cyc['region'].values]
    pplt, fig, axs = _panels(len(regions))
    gen_ens = cyc['gen'].mean('sample')                 # ensemble mean

    for ax, region in zip(axs, regions):
        r = dict(region=region)
        truth = _truth(cyc, region)
        if anomaly:
            truth = truth - truth.mean('month')
        ax.plot(MONTH_TICKS, truth.values, **STYLE['truth'])

        for ft in [str(f) for f in cyc['field_type'].values]:
            g = gen_ens.sel(field_type=ft, **r).mean('model')
            # ensemble spread (across samples), model-averaged
            sd = cyc['gen'].sel(field_type=ft, **r).std('sample').mean('model')
            if anomaly:
                g = g - g.mean('month')
            if np.isnan(g.values).all():
                continue
            ax.fill_between(MONTH_TICKS, (g - sd).values, (g + sd).values,
                            color=STYLE[ft]['color'], alpha=0.18, lw=0)
            ax.plot(MONTH_TICKS, g.values, ms=4, **STYLE[ft])
        ax.format(title=_region_title(cyc, region, st))
        _fmt(ax, f'{O2} anomaly (mol/m$^3$)' if anomaly
             else f'{O2} (mol/m$^3$)')

    kind = 'departure from annual mean' if anomaly else 'absolute'
    note = ('' if st is None else
            '\ntitles: median-over-models annual-mean bias and '
            f'seasonal-amplitude ({AMPLITUDE_DEF}) bias, '
            + '/'.join(SHORT[str(f)] for f in st['field_type'].values))
    if st is not None and not anomaly:
        # On this figure the eye reads the gap between the curves, which is the
        # model MEAN bias, against an annotated MEDIAN. They differ, so say so
        # rather than let the reader conclude one is broken (curve_mean_bias).
        cb = curve_mean_bias(cyc).values
        lo, hi = float(np.nanmin(cb)), float(np.nanmax(cb))
        note += (f'\nthe curves are the model MEAN, so the visible gap is the '
                 f'mean bias ({lo:+.2f}..{hi:+.2f}%) — larger than the median '
                 'because a few models carry the fleet\'s positive bias')
    return _finish(fig, axs, len(regions), save_path,
                   f'Seasonal cycle of depth-integrated {O2} ({kind})\n'
                   'monthly in-sample 1pctCO2, model mean, shading = ensemble '
                   r'$\pm1\sigma$' + note)



# ---------------------------------------------------------------------------
# figure entry points
# ---------------------------------------------------------------------------
OUT_DIR = FIG_DIR
CACHE_DIR = FIG_DIR
CACHE_NAME = 'seasonal_cycle_data.nc'


def cell_slug(cell):
    return f"seasonal_cycle_{cell['view']}"


def cache_path(cell):
    """One cache for both views (absolute and anomaly are two views of the
    same reduced (year, month) series)."""
    return Path(CACHE_DIR) / CACHE_NAME


def load_cache(cell):
    path = cache_path(cell)
    return xr.open_dataset(path).load() if path.is_file() else None


def cell_cycle(ds, restrict_models=True):
    """Cached (year, month) series -> the 12-month climatology the figures draw.

    ``restrict_models`` (the default): pool each field type
    over only the models and years common to both, so a depthint-vs-density
    difference is not a difference in who is in the average.
    """
    if restrict_models and ds.sizes.get('field_type', 1) > 1:
        ds = common_coverage(ds)
    return seasonal_cycle(ds)



def plot_cell(cell, ds, restrict_models=True):
    """One manuscript figure from the cache. Saves into ``OUT_DIR``, returns it.

    Drawn without the eval figure's per-panel bias numbers (``st=None``).
    """
    cyc = cell_cycle(ds, restrict_models=restrict_models)
    return plot_cycle(cyc, Path(OUT_DIR) / f'{cell_slug(cell)}.png',
                      anomaly=cell['view'] == 'anomaly')


def run_compute(epoch=DEFAULT_EPOCH):
    """Sample both field types and write the regional time-series cache
    (``seasonal_cycle.py --n-samples 5 --years-stride 5``)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
    sample_cfg = SampleCfg(n_samples=N_SAMPLES, sampler=SAMPLER, steps=STEPS)
    ts = compute(device, sample_cfg, DEFAULT_REGIONS, field_types=FIELD_TYPES,
                 year_window=DEFAULT_YEAR_WINDOW, years_stride=YEARS_STRIDE,
                 epoch=epoch, min_depth=BATHY_MIN_DEPTH, base=BASE_CONFIG,
                 cache_file=cache_path(MAIN_CELL))
    if ts is None:
        raise SystemExit('no run produced data -- nothing to plot')
    return ts


def plot():
    """Replot the manuscript figure from the cache (no GPU)."""
    ds = load_cache(MAIN_CELL)
    if ds is None:
        raise FileNotFoundError(f'no cache at {cache_path(MAIN_CELL)}; '
                                f'run compute')
    fig = plot_cell(MAIN_CELL, ds)
    render_common.finish(fig, abc=True)
    return render_common.save(fig, 'seasonal_cycle_absolute')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    p.add_argument('--epoch', type=int, default=DEFAULT_EPOCH)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        run_compute(args.epoch)
    else:
        plot()
