"""Fig. 7: validation against observed oxygen -- O2 generated from OBSERVED
predictors (WOA23 temperature and salinity; T+S annual depth-integrated run
trained on 1pctCO2) scored against observed WOA23 O2: observed | generated |
signed error | signed relative error.

Extracted from general/eval/obs_validation.py (compute and plot).

    python -m figures.fig07_obs_validation compute [--epoch 500]   # GPU
    python -m figures.fig07_obs_validation plot                    # no GPU
"""
from __future__ import annotations

from figures import render_common   # first: forces the headless Agg backend

import argparse
import logging
import re
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from diffusion import inference as util
from diffusion import obs as util_obs
from diffusion.data_loading import SCALAR_VARS
from figures import eval_common as ec
from figures.eval_common import (
    make_config, run_save_dir, run_exists, load_run, predict_ensemble,
    SampleCfg, DEPTH_INT, FIG_DIR)


# ---------------------------------------------------------------------------
# pinned manuscript constants (general/manuscript/figures.py, main variant)
# ---------------------------------------------------------------------------
# Both error panels on a LOGARITHMIC diverging ladder spanning two decades
# below each limit. Each limit must be a (1, 2, 5) mantissa times a power of
# ten (``ec.log_levels`` raises otherwise); the values put the bulk of both
# panels mid-wing rather than in the saturated top bin.
OBS_O2_MAX = 0.35
OBS_ERR_LIMIT = 0.2
OBS_REL_LIMIT = 200.0
OBS_LOG_SCALE = True
OBS_LOG_DECADES = 2
OBS_TRUTH = 'woa'

# The manuscript's cell, selected by slug (cell indices are not stable across a
# PRODUCTS change; caches are keyed by slug).
MAIN_CELL = 'T-S_T-S-woa_ann_int_1pctCO2'


# ===========================================================================
# from general/eval/obs_validation.py
# ===========================================================================
# Generation is independent of which O2 product it is scored against, so a cell
# samples once and caches the generated field beside every observed-O2 product
# it can be scored against. Observed O2 is annual or climatological for every
# product. The conditioning mask is predictor coverage only. Sampling goes
# through ``ec.predict_ensemble`` (not ``util.predict``), which re-injects the
# trained land value at each reverse step. Time reduction: generate for every
# year of a fixed window (2004-2017, the all-product overlap) and average the
# generated fields -- the average-fields-then-form-the-error order of
# ``ec.skill_map``. WOA is decadal and its O2 a single climatology, so a WOA
# cell is effectively a climatology comparison.
OUT_DIR = FIG_DIR / 'obs_validation'
CACHE_DIR = OUT_DIR / 'cache'

BASE_CONFIG = make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# Observational overlap window: GOBAI starts 2004, Roach & Bindoff ends 2017.
# Fixed for every cell so all cells average the same years.
OBS_YEAR_WINDOW = (2004, 2017)

# Epoch reached by every annual/depthint run of the sweep: pinning it keeps
# training length out of the comparison. '--epoch latest' opts out.
DEFAULT_EPOCH = 500

# Axes. A (resolution, field_type) combination is enumerated only where PRODUCTS
# has a variant for it: (annual, depthint), (annual, density) and (monthly,
# depthint) -- there are no monthly observations on sigma layers.
RESOLUTIONS = ['annual', 'monthly']
FIELD_TYPES = ['depthint', 'density']
EXPERIMENTS = ['1pctCO2', 'abrupt-4xCO2']

# Predictor sets, using the ABSTRACT 'mld' token (resolved per resolution by
# run_save_dir, and by resolve_channels below, so the run and the obs channel
# agree).
PREDICTOR_SETS = [
    ['thetao', 'so'],
    ['thetao', 'so', 'tauuo', 'tauvo'],
    ['thetao', 'so', 'mld'],
    ['thetao', 'so', 'tauuo', 'tauvo', 'mld'],
    ['thetao', 'so', 'dissic'],
]

TARGET = 'o2'


# ---------------------------------------------------------------------------
# channels: base variable and which product variant supplies them
# ---------------------------------------------------------------------------
def base_var(channel):
    """'thetao_dlev3' -> 'thetao'; leaves plain channels untouched.

    Density runs expand each tracer into one channel per sigma layer, and every
    layer of a tracer necessarily comes from the same product -- so labels,
    groups and product lookups are all formed on the base variable.
    """
    return re.sub(r'_dlev\d+$', '', channel)


# Channels whose data does NOT depend on the run's field_type. Training loads
# tau and MLD from the same depth-integrated-style source for both field types
# (scalar predictors never expand into sigma layers), and the TARGET is always
# single-channel depth-integrated o2. Their observational counterparts
# therefore live under the ('<resolution>', 'depthint') variant even for a
# density cell.
FIELD_TYPE_FREE = set(SCALAR_VARS) | {TARGET}


def variant_key(channel, resolution, field_type):
    """The PRODUCTS variant key that supplies ``channel`` for such a run."""
    return (resolution,
            'depthint' if base_var(channel) in FIELD_TYPE_FREE else field_type)


# ---------------------------------------------------------------------------
# product registry
# ---------------------------------------------------------------------------
# Each product maps (resolution, field_type) -> (loader, variables it provides).
#
# `provides` names BASE variables ('thetao'), never expanded channels
# ('thetao_dlev3'): how many sigma layers a density product has is only
# knowable from the data. A density variant therefore declares the same
# {'thetao','so','o2'} and returns a Dataset whose data_vars are
# `<var>_dlev<i>`; `build_obs_slice` expands each base variable into whatever
# layers the product actually carries, and `compute_cell` checks the result
# against the run's real channel list.
PRODUCTS = {
    'gobai': dict(
        label='GOBAI-O2', years=(2004, 2024),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_gobai_levmean,
                       provides={'thetao', 'so', 'o2'})}),
    'woa': dict(
        label='WOA23', years=(1960, 2020), climatology=True,
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_woa_levmean,
                       provides={'thetao', 'so', 'o2'}),
                  # T/S on sigma layers; no 'o2' because the TARGET is always
                  # depth-integrated whatever the predictors are, so a density
                  # cell is scored against the depthint O2 products (see
                  # FIELD_TYPE_FREE).
                  ('annual', 'density'):
                  dict(load=util_obs.load_woa_density,
                       provides={'thetao', 'so'})}),
    # The only product with MONTHLY predictors. The monthly variant provides
    # plain `mld` (what a monthly run conditions on) where the annual one
    # provides the mld_max / mld_mean reductions over the seasonal cycle.
    'ecco': dict(
        label='ECCO', years=(1992, 2019),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_ecco,
                       provides={'thetao', 'so', 'mld_max', 'mld_mean'}),
                  ('monthly', 'depthint'):
                  dict(load=util_obs.load_ecco_monthly,
                       provides={'thetao', 'so', 'mld'})}),
    'wang': dict(
        label='Wang 2025', years=(1960, 2021),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_wang_2025, provides={'o2'})}),
    'era5': dict(
        label='ERA5', years=(1940, 2025),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_era5_tau,
                       provides={'tauuo', 'tauvo'}),
                  # the source file is monthly; the annual variant is the one
                  # that reduces it
                  ('monthly', 'depthint'):
                  dict(load=util_obs.load_era5_tau_monthly,
                       provides={'tauuo', 'tauvo'})}),
    # Depth-integrated `dissic` only.
    'mobo': dict(
        label='MOBO-DIC', years=(2004, 2019),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_mobo_dic, provides={'dissic'})}),
    # The only product supplying every tracer on BOTH field types. A
    # climatology: mapped from 1972-2013 measurements, DIC normalized to 2002.
    # The density variant omits 'o2' for the same reason WOA's does.
    'glodap': dict(
        label='GLODAPv2', years=(1972, 2013), climatology=True,
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_glodap_levmean,
                       provides={'thetao', 'so', 'o2', 'dissic'}),
                  ('annual', 'density'):
                  dict(load=util_obs.load_glodap_density,
                       provides={'thetao', 'so', 'dissic'})}),
    'roach': dict(
        label='Roach & Bindoff 2023', years=(1960, 2017),
        variants={('annual', 'depthint'):
                  dict(load=util_obs.load_roach_bindoff,
                       provides={'thetao', 'so', 'o2'})}),
}


def variant(name, key):
    """The ``(resolution, field_type)`` variant of a product, or None."""
    return PRODUCTS[name]['variants'].get(tuple(key))


def provides(name, key):
    v = variant(name, key)
    return set(v['provides']) if v else set()


def sources_for(channel, resolution, field_type):
    """Products that can supply ``channel`` for this (resolution, field_type)."""
    key = variant_key(channel, resolution, field_type)
    return sorted(n for n in PRODUCTS if base_var(channel) in provides(n, key))


def resolve_channels(predictors, resolution):
    """Predictor tokens -> the channel names the run actually uses.

    Only the abstract 'mld' token is resolution-dependent (mld_max annual /
    mld monthly), exactly as for the save-dir name -- so the obs channel and
    the trained channel are the same variable.
    """
    return list(ec.resolve_predictors(list(predictors), resolution))


# ---------------------------------------------------------------------------
# channel grouping + source assignments
# ---------------------------------------------------------------------------
_SHORT = {'thetao': 'T', 'so': 'S', 'tauuo': 'taux', 'tauvo': 'tauy',
          'mld': 'MLD', 'mld_max': 'MLD', 'mld_mean': 'MLDmean',
          'dissic': 'DIC', 'o2': 'O2'}
_TEX = {'thetao': 'T', 'so': 'S', 'tauuo': r'$\tau_x$', 'tauvo': r'$\tau_y$',
        'mld': 'MLD', 'mld_max': 'MLD', 'mld_mean': 'MLD$_{mean}$',
        'dissic': 'DIC', 'o2': r'$\mathrm{O}_2$'}


def _short(channels):
    """Short label for a set of channels: 'T-S', 'taux-tauy', 'MLD'."""
    seen = []
    for ch in channels:
        s = _SHORT.get(base_var(ch), base_var(ch))
        if s not in seen:
            seen.append(s)
    return '-'.join(seen)


def _tex(channels):
    seen = []
    for ch in channels:
        s = _TEX.get(base_var(ch), base_var(ch))
        if s not in seen:
            seen.append(s)
    return '+'.join(seen)


def channel_groups(channels, resolution, field_type):
    """Group channels by *which products can supply them*.

    Channels with an identical candidate-source set share one source choice --
    so T and S (both from GOBAI/WOA/ECCO/Roach) are always taken from the same
    product, and so are all sigma layers of one tracer, while tau (ERA5 only)
    and MLD (ECCO only) choose independently.

    Grouping is on (candidate products, variant key) rather than the candidate
    products alone: two channels of one cell can need *different* variants of
    the same product (a density cell takes T/S from WOA-on-sigma but its O2
    from WOA-depth-integrated), and merging those into one group would hand the
    wrong variant key to ``load_product``.

    Returns [(tuple(channels), tuple(candidate products), variant key)],
    ordered by the channel order given.
    """
    groups, order = {}, []
    for ch in channels:
        key = (tuple(sources_for(ch, resolution, field_type)),
               variant_key(ch, resolution, field_type))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ch)
    return [(tuple(groups[k]), k[0], k[1]) for k in order]


def source_assignments(channels, resolution, field_type):
    """Every {group -> source product} assignment for a channel list.

    Empty when some channel has no product at all, so such a predictor set
    drops out of the enumeration.

    Each entry is (channels, source product, variant key) -- the key travels
    with the group so every later lookup uses the variant the channel was
    enumerated from.
    """
    groups = channel_groups(channels, resolution, field_type)
    if any(not cands for _, cands, _ in groups):
        return []
    out = []
    for combo in iterproduct(*[cands for _, cands, _ in groups]):
        out.append(tuple((chs, src, key)
                         for (chs, _, key), src in zip(groups, combo)))
    return out


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def cell_slug(cell):
    """Filename stem for a gen cell -- stable, so caches/figures don't drift."""
    res = {'annual': 'ann', 'monthly': 'mon'}[cell['resolution']]
    fld = {'depthint': 'int', 'density': 'dens'}[cell['field_type']]
    src = '_'.join(f'{_short(chs)}-{s}' for chs, s, _ in cell['sources'])
    return (f"{_short(cell['channels'])}_{src}_{res}_{fld}"
            f"_{cell['experiment']}")


def cell_title(cell):
    src = ', '.join(f'{_tex(chs)} from {PRODUCTS[s]["label"]}'
                    for chs, s, _ in cell['sources'])
    fld = {'depthint': 'depth-integrated', 'density': 'density layers'}[
        cell['field_type']]
    return (f"{_tex(cell['channels'])} | {src} | {cell['resolution']} | "
            f"{fld} | trained on {cell['experiment']}")


def cell_config(cell):
    """Naming config for a cell's run (in-sample, matching predictor set)."""
    return make_config(BASE_CONFIG, predictors=list(cell['predictors']),
                       experiments=[cell['experiment']],
                       resolution=cell['resolution'],
                       field_type=cell['field_type'])


def truth_key(source, cell):
    """The variant of ``source`` supplying the observed O2 a cell is scored
    against, or None if it supplies none.

    Two relaxations, both of which exist because the *truth* is compared to a
    field that has already been reduced in time and depth:

    - **field type** -- the target is always single-channel depth-integrated O2
      whatever the predictors are, so a density cell is scored against a
      depth-integrated product (that is ``variant_key`` / ``FIELD_TYPE_FREE``);
    - **resolution** -- the generated field is averaged over every time step of
      the window before scoring, so a *monthly* cell's generation is a
      2004-2017 mean and an annual observed-O2 product is exactly the right
      comparison for it. A product is therefore taken at the cell's own
      resolution when it has one, and at annual resolution otherwise.

    The truth is then iterated and selected at ``truth_key(...)[0]``, not the
    cell's resolution -- an annual product has no month to select.
    """
    for res in dict.fromkeys([cell['resolution'], 'annual']):
        key = variant_key(TARGET, res, cell['field_type'])
        if TARGET in provides(source, key):
            return key
    return None


def truth_sources(cell):
    """Observational O2 products available to score this cell against."""
    return sorted(n for n in PRODUCTS if truth_key(n, cell))


def all_cells():
    """The full gen-cell list, in a FIXED order.

    Ordered field_type -> resolution -> predictors -> experiment -> sources.
    """
    cells = []
    for field_type, resolution, preds, exp in iterproduct(
            FIELD_TYPES, RESOLUTIONS, PREDICTOR_SETS, EXPERIMENTS):
        channels = resolve_channels(preds, resolution)
        assigns = source_assignments(channels, resolution, field_type)
        if not assigns:
            continue
        for sources in assigns:
            # `channels` holds BASE variables (mld resolved); density expansion
            # to <var>_dlev<i> happens in build_obs_slice, from the data.
            cells.append(dict(predictors=list(preds), channels=channels,
                              resolution=resolution, field_type=field_type,
                              experiment=exp, sources=sources))
    return cells


def find_cell(slug):
    """Resolve a cell slug to its cell dict (by slug, never by index)."""
    for cell in all_cells():
        if cell_slug(cell) == slug:
            return cell
    raise KeyError(f'obs_validation: no cell {slug!r}')


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
# observational data
# ---------------------------------------------------------------------------
_PRODUCT_CACHE = {}


def load_product(name, key, years):
    """Load a product variant, reduced to ``years``, once per process.

    ``key`` is the (resolution, field_type) variant key, which for a scalar
    predictor or the target is not the cell's own field_type -- see
    ``variant_key``.

    Reduced *before* ``.load()`` where the product has a year dimension: some
    loaders (Roach & Bindoff, WOA) build 100+-level arrays lazily, and only the
    depth-integrated (year, lat, lon) slice is small.
    """
    key = tuple(key)
    cache_key = (name, key, tuple(years))
    if cache_key in _PRODUCT_CACHE:
        return _PRODUCT_CACHE[cache_key]
    v = variant(name, key)
    if v is None:
        raise KeyError(f'{name} has no {key} variant')
    logging.info(f'  [load] {PRODUCTS[name]["label"]} ({"/".join(key)})')
    ds = v['load']()
    # `provides` names BASE variables, the loaded Dataset carries real channels
    # (`thetao_dlev0..9` for a density product) -- so filter by base name, not
    # by literal membership, or a density product selects nothing at all.
    attrs = ds.attrs
    ds = ds[[c for c in ds.data_vars if base_var(c) in v['provides']]]
    ds.attrs.update(attrs)
    if 'year' in ds.dims:
        # nearest so decadal products (WOA) resolve; duplicates are fine, the
        # requested years are re-stamped in _sel_time.
        avail = np.unique(ds['year'].values)
        want = np.unique([_nearest(avail, y) for y in years])
        ds = ds.sel(year=want)
    ds = _to_ref_grid(ds, name).load()
    _PRODUCT_CACHE[cache_key] = ds
    return ds


def _nearest(avail, year):
    return int(avail[np.argmin(np.abs(np.asarray(avail) - year))])


def _to_ref_grid(ds, name, tol=0.6):
    """Relabel a product onto the reference r360x180 grid.

    Products are all regridded to r360x180 already, but not all label the cells
    identically -- Wang 2025 sits on 0.5..359.5 where everything else uses
    0..359, and a plain ``xr.merge`` of the two would outer-join into a
    720-column, mostly-NaN field. Relabel, log it, and refuse offsets larger
    than ``tol`` (a real grid mismatch, not a labelling convention).
    """
    ref = ec.bathy_mask()
    for dim in ('lat', 'lon'):
        if ds.sizes.get(dim) != ref.sizes[dim]:
            raise ValueError(
                f'{name}: {dim} has {ds.sizes.get(dim)} points, reference grid '
                f'has {ref.sizes[dim]} -- not on r360x180')
    off = {dim: float(np.abs(ds[dim].values - ref[dim].values).max())
           for dim in ('lat', 'lon')}
    if max(off.values()) > 1e-3:
        if max(off.values()) > tol:
            raise ValueError(
                f'{name}: grid offset lat {off["lat"]:.3g} / lon '
                f'{off["lon"]:.3g} deg exceeds tolerance {tol} -- this is a '
                f'different grid, not a labelling convention')
        logging.info(f'  [align] {name}: coords offset by lat {off["lat"]:.3g} / '
                     f'lon {off["lon"]:.3g} deg -- relabelled to the reference grid')
        ds = ds.assign_coords(lat=ref['lat'].values, lon=ref['lon'].values)
    return ds


def _sel_time(ds, resolution, year, month=None):
    """One (year[, month]) 2D slice of a product, or None if unavailable.

    Products without a year dimension (the WOA O2 climatology) are constant in
    time and returned as-is -- that is the honest reading of a climatology.
    Returns (slice, actual_year).
    """
    if 'year' not in ds.dims:
        return ds, None
    actual = _nearest(np.unique(ds['year'].values), year)
    sel = ds.sel(year=actual)
    if 'year' in sel.dims:                    # duplicate labels -> take the first
        sel = sel.isel(year=0)
    if resolution == 'monthly':
        if 'month' not in sel.dims:
            raise NotImplementedError(
                'monthly observational selection needs a `month` dimension; '
                'add one in the product loader when monthly obs land')
        sel = sel.sel(month=month)
    return sel.reset_coords(drop=True), actual


def _expand(sel, var):
    """The data_vars of ``sel`` belonging to base variable ``var``.

    ``['thetao']`` for a depth-integrated product, ``['thetao_dlev0', ...]`` in
    sigma-layer order for a density one -- so a cell can carry base variable
    names (the only thing knowable without reading the data) while the built
    slice carries the run's real channels.
    """
    matches = [v for v in sel.data_vars if base_var(v) == var]
    return sorted(matches, key=lambda v: (len(v), v))


def build_obs_slice(cell, year, month, years):
    """Predictor channels for one time step, on a single common ocean mask.

    Returns (Dataset, ocean mask, n_valid, actual_years) or None. The mask
    depends only on the predictors, never on which O2 product the result is
    later scored against -- which is what lets one generated field serve every
    truth source.

    Coverage is computed per BASE variable as the **union** over its sigma
    layers, and the domain is the intersection of those per-variable coverages.
    A density tracer only exists where its layer outcrops, so intersecting the
    layers themselves would leave ~500 of ~37000 cells; the union is where the
    product measured that tracer at all. Layers that are NaN inside the domain
    stay NaN and are land-filled by ``util.normalize_channel``, which is exactly
    what the run saw at training time: ``TracerDataset`` derives its ONE ocean
    mask from the depth-integrated target and land-fills every predictor NaN.
    For a depth-integrated cell each base variable expands to a single channel,
    so this reduces to the all-channel intersection.
    """
    names, arrays, actual, per_var = [], {}, {}, []
    for chs, src, key in cell['sources']:
        ds_src = load_product(src, key, years)
        sel, act = _sel_time(ds_src, cell['resolution'], year, month)
        if sel is None:
            return None
        actual[src] = act
        for ch in chs:
            expanded = _expand(sel, ch)
            if not expanded:
                logging.info(f'  [skip] {src} has no channel for {ch}')
                return None
            per_var.append(expanded)
            for name in expanded:
                names.append(name)
                arrays[name] = sel[name]
    ds = xr.Dataset({n: arrays[n] for n in names})
    valid = np.ones(ds[names[0]].shape, dtype=bool)
    for expanded in per_var:
        covered = np.zeros(valid.shape, dtype=bool)
        for n in expanded:
            covered |= np.isfinite(ds[n].values)
        valid &= covered
    ds = ds.where(xr.DataArray(valid, dims=('lat', 'lon'),
                               coords={'lat': ds.lat, 'lon': ds.lon}))
    return ds, valid, int(valid.sum()), actual


def obs_truth(source, cell, years, months):
    """Time-mean observed O2 (mol/m^3) over the same window as the generation.

    Iterated at the *truth product's* resolution (see ``truth_key``), which for
    a monthly cell scored against an annual product means 14 yearly fields
    against the generation's 14 x 12 -- both are the 2004-2017 mean.
    """
    key = truth_key(source, cell)
    if key is None:
        return None
    ds_src = load_product(source, key, years)
    if TARGET not in ds_src.data_vars:
        return None
    fields = []
    for year, month in _iter_times(key[0], years, months):
        sel, _ = _sel_time(ds_src[[TARGET]], key[0], year, month)
        if sel is not None:
            fields.append(sel[TARGET])
    if not fields:
        return None
    return xr.concat(fields, 'time').mean('time') / DEPTH_INT


def _iter_times(resolution, years, months):
    if resolution == 'annual':
        return [(y, None) for y in years]
    return [(y, m) for y in years for m in months]


# ---------------------------------------------------------------------------
# per-cell computation
# ---------------------------------------------------------------------------
def compute_cell(cell, ctx):
    """Time-mean generated O2 + every available observed O2, in mol/m^3.

    Generates an ensemble per time step, takes the ensemble mean, then averages
    the per-time fields -- the average-fields-first order of
    ``ec.skill_map``, so a year where the observed O2 approaches 0 cannot
    blow up a ratio.
    """
    cfg = cell_config(cell)
    save_dir = run_save_dir(cfg)
    ep = resolve_epoch(save_dir, ctx['epoch'])
    if ep is None:
        logging.info(f'[skip] {cell_slug(cell)}: no checkpoint at epoch '
                     f'{ctx["epoch"]} in {save_dir.name}')
        return None
    run = load_run(save_dir, ctx['device'], ep, _compile=False)

    years = list(range(ctx['year_window'][0], ctx['year_window'][1] + 1,
                       ctx['years_stride']))
    times = _iter_times(cell['resolution'], years, ctx['months'])
    gen_t, n_valid, used_years, actual_all = [], [], [], {}
    for year, month in times:
        built = build_obs_slice(cell, year, month, years)
        if built is None:
            continue
        ds_sel, ocean, nv, actual = built
        # The run's channel list is the authority (a density run wants one
        # channel per sigma layer, and only the data knows how many there are),
        # so the guard is against the BUILT slice, not the cell's base variables.
        missing = [ch for ch in run.predictor_channels
                   if ch not in ds_sel.data_vars]
        if missing:
            logging.info(f'[skip] {cell_slug(cell)}: the run needs channels '
                         f'{missing} that the assigned products do not provide')
            return None
        # Hand the sampler the conditioning mask explicitly. Taking it from one
        # channel instead would be wrong for a density cell, where the first
        # channel is a single sigma layer covering a fraction of the domain.
        gen_t.append(predict_ensemble(ds_sel, run, ctx['sample_cfg'],
                                      ocean=ocean).mean('sample'))
        n_valid.append(nv)
        used_years.append(year)
        for src, act in actual.items():
            actual_all.setdefault(src, set()).add(act)
    if not gen_t:
        logging.info(f'[skip] {cell_slug(cell)}: no usable time step')
        return None

    for src, acts in sorted(actual_all.items()):
        acts = sorted(a for a in acts if a is not None)
        if not acts:
            logging.info(f'  {PRODUCTS[src]["label"]}: no time axis -- '
                         f'climatology, constant over the window')
        elif len(acts) < len(used_years):
            logging.info(f'  {PRODUCTS[src]["label"]}: {len(acts)} distinct '
                         f'field(s) for {len(used_years)} requested year(s) '
                         f'({acts[0]}..{acts[-1]}) -- effectively a climatology')

    # Which sigma levels the observations were conditioned on. Level index i is
    # the i-th volume-weighted density quantile OF THAT DATASET, so obs and
    # models sit on different sigma values by design; record them rather than
    # let the figure imply the run was conditioned on the sigma values it was
    # trained on.
    levels = {src: lv for chs, src, key in cell['sources']
              if (lv := load_product(src, key, years).attrs.get('density_levels'))}
    for src, lv in sorted(levels.items()):
        logging.info(f'  {PRODUCTS[src]["label"]} sigma levels: {lv}')

    gen = xr.concat(gen_t, 'time').mean('time') / DEPTH_INT
    out = dict(gen=gen, n_times=len(gen_t), n_valid=int(np.min(n_valid)),
               epoch=ep, years=used_years, truths={},
               density_levels='; '.join(f'{s}={lv}'
                                        for s, lv in sorted(levels.items())))
    for src in truth_sources(cell):
        truth = obs_truth(src, cell, years, ctx['months'])
        if truth is not None:
            out['truths'][src] = truth
            logging.info(f'  truth {PRODUCTS[src]["label"]}: ready')
    if not out['truths']:
        logging.info(f'[skip] {cell_slug(cell)}: no observed O2 product')
        return None
    return out


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------
def cache_path(cell):
    return CACHE_DIR / f'{cell_slug(cell)}.nc'


def years_label(ctx):
    """What was actually averaged in time -- never a window the run didn't sample."""
    lo, hi = ctx['year_window']
    stride = ctx['years_stride']
    return f'yrs {lo}-{hi}' + (f' /{stride}' if stride != 1 else '')


def save_cache(cell, d, ctx):
    ds = xr.Dataset({'gen': d['gen'],
                     **{f'truth_{s}': t for s, t in d['truths'].items()}})
    ds.attrs.update(
        cell=cell_slug(cell), title=cell_title(cell),
        predictors=' '.join(cell['predictors']),
        channels=' '.join(cell['channels']),
        sources=' '.join(f'{_short(chs)}={s}' for chs, s, _ in cell['sources']),
        truth_sources=' '.join(sorted(d['truths'])),
        # Resolution each truth product was taken at -- not necessarily the
        # cell's own (see truth_key).
        truth_resolution=' '.join(f'{s}={truth_key(s, cell)[0]}'
                                  for s in sorted(d['truths'])),
        # Which truth products were *asked for*, as opposed to which produced a
        # field.
        truth_expected=' '.join(truth_sources(cell)),
        resolution=cell['resolution'], field_type=cell['field_type'],
        density_levels=d.get('density_levels', ''),
        experiment=cell['experiment'], epoch=d['epoch'],
        epoch_arg=str(ctx['epoch']), n_times=d['n_times'],
        n_valid_cells=d['n_valid'],
        n_samples=ctx['sample_cfg'].n_samples, steps=ctx['sample_cfg'].steps,
        sampler=ctx['sample_cfg'].sampler,
        year_window=' '.join(str(y) for y in ctx['year_window']),
        years_stride=ctx['years_stride'], years_label=years_label(ctx))
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
def _panel_for(cell, ds, src, o2_max=None, err_limit=None, rel_limit=None,
               with_rel=False, tres=None, log_scale=False,
               log_decades=ec.LOG_DECADES):
    """The figure for ONE observed-O2 product, or None if the cell has no such
    truth."""
    got = error_fields(ds, src)
    if got is None:
        return None
    err, rel, st = got
    truth, gen = ds[f'truth_{src}'], ds['gen']
    stats = f'bias {st["bias"]:+.3f} {ec.CBAR_LABEL}, MAPE {st["mape"]:.1f}%'
    res = (tres or {}).get(src)
    obs_bit = f'obs {ec.O2}: {PRODUCTS[src]["label"]}'
    if res and res != ds.attrs.get('resolution', cell['resolution']):
        obs_bit += f' ({res})'
    bits = [ds.attrs.get('title', cell_title(cell)),
            obs_bit,
            str(ds.attrs.get('years_label', '')),
            f'ep {ds.attrs.get("epoch", "?")}', stats]
    d = dict(truth=truth, gen=gen, err=err)
    if with_rel:
        d['rel'] = rel
    return ec.obs_truth_gen_error_panel(
        d, '  |  '.join(b for b in bits if b),
        OUT_DIR / f'{cell_slug(cell)}__obs-{src}.png',
        o2_max=o2_max, err_limit=err_limit, rel_limit=rel_limit,
        log_scale=log_scale, log_decades=log_decades,
        truth_label=f'Observed {ec.O2} ({PRODUCTS[src]["label"]})')


def _truth_resolutions(ds):
    """{product: resolution it was taken at} -- a monthly cell is scored against
    annual observed O2 (see truth_key), and the subtitle has to say so."""
    return dict(kv.split('=', 1)
                for kv in str(ds.attrs.get('truth_resolution', '')).split()
                if '=' in kv)


def error_fields(ds, src):
    """``(err, rel, stats)`` against observed-O2 product ``src``, or None.

    ``err`` = gen - obs in mol/m^3 and ``rel`` = 100*err/obs in %, both
    **SIGNED**: there is no model axis here, so the sign is real information
    and is kept. ``stats`` is the area-weighted bias and MAPE of the deep-ocean
    map, the same two numbers ``_panel_for`` puts in its subtitle.
    """
    var = f'truth_{src}'
    if var not in ds.data_vars:
        return None
    truth = ds[var]
    err = ds['gen'] - truth
    rel = 100.0 * err / truth
    bias = float(util.global_mean(ec.mask_shallow(err)).values)
    return err, rel, dict(bias=bias, mape=float(ec.mape(rel)))


def plot_cell_single(cell, ds, obs_o2, o2_max=None, err_limit=None,
                     rel_limit=None, with_rel=True, log_scale=False,
                     log_decades=ec.LOG_DECADES):
    """ONE figure, for ONE named observed-O2 product.

    Which product supplied the truth is a *plot* argument, not part of the cell
    slug (the cache carries every product the cell can be scored against).
    """
    fig = _panel_for(cell, ds, obs_o2, o2_max=o2_max, err_limit=err_limit,
                     rel_limit=rel_limit, with_rel=with_rel,
                     log_scale=log_scale, log_decades=log_decades,
                     tres=_truth_resolutions(ds))
    if fig is None:
        have = sorted(str(ds.attrs.get('truth_sources', '')).split())
        raise KeyError(f'{cell_slug(cell)}: no observed O2 from {obs_o2!r}; '
                       f'this cell carries {have}')
    return fig


# ===========================================================================
# entry points
# ===========================================================================
def compute(epoch=DEFAULT_EPOCH):
    """Sample the main cell and write its cache (5 samples, 100 DDIM steps,
    every year 2004-2017)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = dict(device=device,
               sample_cfg=SampleCfg(n_samples=5, sampler='ddim', steps=100),
               year_window=OBS_YEAR_WINDOW, years_stride=1,
               months=list(range(12)), epoch=epoch)
    cell = find_cell(MAIN_CELL)
    d = compute_cell(cell, ctx)
    if d is None:
        raise SystemExit(f'{MAIN_CELL}: nothing computed')
    return save_cache(cell, d, ctx)


def plot():
    cell = find_cell(MAIN_CELL)
    ds = load_cache(cell)
    if ds is None:
        raise SystemExit(f'no cache at {cache_path(cell)}; run compute first')
    fig = plot_cell_single(cell, ds, obs_o2=OBS_TRUTH, with_rel=True,
                           o2_max=OBS_O2_MAX, err_limit=OBS_ERR_LIMIT,
                           rel_limit=OBS_REL_LIMIT, log_scale=OBS_LOG_SCALE,
                           log_decades=OBS_LOG_DECADES)
    render_common.finish(fig, abc=True)
    return render_common.save(fig, 'obs_validation')


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
