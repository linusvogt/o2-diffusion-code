"""Fig. 8 -- prediction-error maps of the diffusion model and four baselines.

A 2 x 5 grid, rows = in-sample / leave-one-model-out (LOMO), columns =
diffusion | U-Net (MSE) | QRF | linear (pointwise) | climatology, each panel the
UNSIGNED relative error in % (mean per-model MAPE), one nonlinear colour scale
and one colorbar per row.

Extracted from general/baselines/method_error_maps.py (``grid_figure`` and the
helpers it calls, restricted to the unsigned / % variant its ``--grid`` flag
draws) with ``verify_cache`` / ``signed_maps`` copied verbatim from
general/eval/lomo_signed_error.py. The original wrote
``<EVAL_DIR>/baselines/methods_grid_unsigned.pdf`` (+ ``.png``, dpi 300,
``bbox_inches='tight'``); here the figure is returned and saved through
``render_common.save`` as ``baseline_grid``.

    python -m figures.fig08_baseline_grid compute   # prints how the caches are made
    python -m figures.fig08_baseline_grid plot

Reads caches only -- no sampling, no GPU, no raw netCDF. Every panel comes
from ``baselines/evaluate.py``, one cache per (method, split), so all ten go
through ONE code path.

What each panel draws
---------------------
``unsigned`` = ``error_maps``' ``rel``, i.e. ``<|100(gen_m - truth_m)/truth_m|>_m``
-- the mean *per-model* MAPE, magnitudes formed BEFORE the average over models
so opposite-signed per-model biases cannot cancel.

What the caption must carry (none of it is drawn)
-------------------------------------------------
* **What LOMO is** (the second row is labelled with the acronym).
* **The two rows are over different fleets** -- 16 models (14 with data)
  in-sample against 14 LOMO -- and under different colour scales. Side by side,
  never differenced.
* **The in-sample row is not a controlled comparison in method alone**: its
  diffusion run is ``land_fill='zero'`` and its forest ``min_samples_leaf=10``,
  against ``meanstd`` / ``msl=5`` for the LOMO row. The LOMO row is the clean
  one.
* **Which statistic**: mean per-model MAPE (magnitudes formed BEFORE the
  average over models).
* **The colour scale is nonlinear** (boundaries at ``limit*(k/n)**2``), with
  the two rows' limits.
* **No ranking claim is supported by these panels.** The fleet means are
  dominated by a few opposite-signed models; the ordering comes from a paired
  per-model sign test / signed-rank. On the LOMO row neither CNN is
  distinguishable from the climatology column.
* **Epoch 500** and the years label, from the caches' own stamps.

``grid_provenance_report`` prints all of this on every run.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from figures import render_common

import numpy as np
import xarray as xr

from figures import eval_common as ec
from baselines import evaluate as ev
from baselines import pointwise_linear as plin
from baselines import qrf, unet_mse


#: Panel order, left to right: the model under test, then its baselines in
#: DESCENDING order of machinery -- U-Net (same backbone, no diffusion), QRF
#: (pointwise, nonlinear), pointwise-linear (pointwise, linear).
KINDS = (ev.DIFFUSION, unet_mse.KIND, qrf.KIND, plin.KIND)

#: Everything drawable, in the same order. The grid's columns: the four above
#: plus the climatology floor.
ALL_KINDS = (ev.DIFFUSION, unet_mse.KIND, qrf.KIND, plin.KIND, plin.KIND_CLIM)

LABEL = {ev.DIFFUSION: 'diffusion', unet_mse.KIND: 'U-Net (MSE)',
         qrf.KIND: 'QRF', plin.KIND: 'linear (pointwise)',
         plin.KIND_CLIM: 'climatology'}

#: The manuscript variant: unsigned relative error in %.
SIGN = 'unsigned'
UNITS = 'pct'


# ---------------------------------------------------------------------------
# copied verbatim from general/eval/lomo_signed_error.py
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
    # Reported so the log shows the check was tight, not merely passed: cells
    # where every held-out model errs the SAME way sit exactly on the bound
    # (|<e_m>| == <|e_m>|), so the worst violation is pure float32 round-trip
    # noise and lands just under `tol` rather than comfortably below it.
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

    ⚠️ ``bias_rel`` and ``rel`` are NOT the signed and unsigned versions of one
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
# sources
# ---------------------------------------------------------------------------
def cache_for(kind, split):
    """The ``evaluate.py`` cache for one (method, split).

    Every panel comes from ONE code path on purpose -- ``evaluate.py``
    runs the diffusion model through the same per-model loop as the baselines,
    for exactly this reason. ``error_maps``' own cache for the same cell is NOT
    used even though it holds the same fields; :func:`crosscheck` compares the
    two instead, which is the end-to-end check that the path is equivalent.
    """
    return ev.cache_path(kind, split=split)


def load_cache(kind, split):
    """``(ds, None)`` or ``(None, why)``. Never raises on a missing cache."""
    path = cache_for(kind, split)
    if not path.is_file():
        return None, f'not written ({path})'
    try:
        return xr.open_dataset(path).load(), None
    except Exception as exc:                       # truncated / half-written
        return None, f'unreadable ({type(exc).__name__}: {exc})'


def crosscheck(ds, kind, split):
    """Log this cache's MAPE against ``error_maps``' own for the same cell.

    Only the diffusion has a counterpart there, and its agreement is the
    end-to-end evidence that ``evaluate.py``'s per-model loop reproduces
    ``error_maps``' -- the same check ``evaluate.save_cache`` prints for the
    baselines against the diffusion. Warned, never raised: a legitimately
    absent ``error_maps`` cache is not a reason to refuse to draw.
    """
    if not ev.is_diffusion(kind):
        return None
    ref = ev.pin_cache_for(split)
    if not ref.is_file():
        logging.warning(f'  [warn] no {ref.name} to cross-check against')
        return None
    with xr.open_dataset(ref) as r:
        want = float(r.attrs['mape_pct'])
    got = float(ds.attrs['mape_pct'])
    delta = abs(got - want)
    verdict = 'OK' if delta < 0.01 else 'DIFFERS'
    log = logging.info if delta < 0.01 else logging.warning
    log(f'  crosscheck {verdict}: {kind} {split} MAPE {got:.4f}% vs '
        f'{ref.name} {want:.4f}% (|d| = {delta:.4f} pp)')
    return delta


# ---------------------------------------------------------------------------
# the two quantities
# ---------------------------------------------------------------------------
def method_map(ds, sign, units):
    """The single map one panel draws, unmasked (``mask_shallow`` is applied
    by the drawing helpers and is idempotent).

    ``sign='unsigned'`` -> the cache's own magnitude map (``rel`` / ``err``);
    ``sign='signed'``   -> :func:`signed_maps`' ``bias_rel`` / ``bias``, i.e.
    ``gen - truth``, which is the signed model-mean error exactly once
    :func:`verify_cache` has passed.
    """
    if sign == 'unsigned':
        return ds['rel'] if units == 'pct' else ds['err']
    d = signed_maps(ds)
    return d['bias_rel'] if units == 'pct' else d['bias']


def panel_scalar(da, sign, units):
    """The number over each panel -- the area-weighted mean of the map
    DRAWN, over the deep ocean (``ec.mask_shallow`` first, ``mape``'s and
    ``skill_scalar``'s own footprint).

    For ``unsigned``/``pct`` this is exactly ``ec.mape``, so the title
    reproduces the cache's ``mape_pct`` stamp.
    """
    masked = ec.mask_shallow(da)
    if sign == 'unsigned':
        masked = np.abs(masked)
    return float(ec.util.global_mean(masked).values)


def cache_epoch(ds):
    """The epoch a cache was sampled at, however it happens to stamp it.

    ⚠️ The ``unet-mse`` LOMO cache predates the ``epoch_arg='pinned'`` /
    ``epoch_pinned=500`` pair: it stamps ``epoch_arg='500'`` and no
    ``epoch_pinned`` at all, so this reads around the difference.
    ``epochs`` (the per-model list) is stamped by every version and is the
    fact; a mixed list has no single comparison point and says so rather than
    picking one.
    """
    a = ds.attrs
    if 'epoch_pinned' in a:
        return str(a['epoch_pinned'])
    eps = sorted({int(e) for e in str(a.get('epochs', '')).split()})
    if len(eps) == 1:
        return str(eps[0])
    return str(a.get('epoch_arg', '?')) if not eps else f'mixed {eps}'


#: memo for :func:`run_provenance` -- it opens up to 16 checkpoints per call.
_PROV_CACHE = {}


def run_provenance(kind, split, models):
    """Land-fill convention (and forest tuning) of the runs behind one panel.

    ⚠️ **A row of this figure is NOT a set of models differing only in
    method**, and on the in-sample split it is not close:

      * the in-sample DIFFUSION run is ``land_fill='zero'`` (it predates the
        convention change; the value comes from its sidecar, since it predates
        the checkpoint stamp) while the in-sample baselines were trained under
        ``'meanstd'``;
      * the in-sample FOREST is ``min_samples_leaf=10`` against the LOMO
        fleet's 5 -- both validation-selected, on different validation year
        windows, because the in-sample split already holds out 81-100 as its
        test years.

    Neither is a bug: each run is *loaded* with its own convention, which is
    what makes its prediction correct. But a reader comparing the panels is
    entitled to know the row is not a controlled experiment in method alone.
    The LOMO row carries no such caveat -- every run there is ``'meanstd'`` and
    every forest ``msl=5``.

    Every pinned model is checked, not just the first. Resolution is
    cheapest-first -- ``qrf_meta.json`` for the forest, the ``land_fill``
    sidecar for a run that has one, and only then a ``torch.load`` of the
    checkpoint. Any failure degrades to ``'?'``; this is an annotation and must
    never be the reason a figure does not draw.
    """
    key = (kind, split)
    if key in _PROV_CACHE:
        return _PROV_CACHE[key]
    fills, msls, n_inferred = set(), set(), 0
    for name in models:
        sd = ev.baseline_save_dir(kind, name, split)
        try:
            if kind == qrf.KIND:
                meta = qrf.load_meta(sd)
                fills.add(str(meta.get('land_fill', '?')))
                msls.add(str(meta.get('min_samples_leaf', '?')))
                continue
            if kind in plin.KINDS:
                # closed form: no checkpoint to read a stamp out of, so the
                # convention comes from the artifact's own sidecar. It is
                # RECORDED, never inferred, so these kinds can never
                # contribute an '(N inferred)' to the line.
                meta = plin.load_meta(sd)
                fills.add(str(meta.get('land_fill', '?')))
                continue
            fill, inferred = _land_fill_of(sd)
            fills.add(fill)
            n_inferred += bool(inferred)
        except Exception as exc:                    # annotation, never fatal
            logging.warning(f'  [warn] no provenance for {kind}/{name}: '
                            f'{type(exc).__name__}: {exc}')
            fills.add('?')

    def one(vals):
        vals = sorted(v for v in vals if v)
        return vals[0] if len(vals) == 1 else 'mixed ' + '/'.join(vals)
    out = one(fills)
    if n_inferred:
        out += f' ({n_inferred} inferred)'
    if msls:
        out += f', msl {one(msls)}'
    _PROV_CACHE[key] = out
    return out


def _land_fill_of(save_dir):
    """``(convention, inferred)`` for one run, mirroring
    ``inference.resolve_land_fill``'s resolution order exactly: the checkpoint
    stamp, then a ``land_fill`` sidecar, then the timestamp heuristic.

    ``inferred`` marks the third case, and it is reported rather than collapsed
    into the answer because it is a materially weaker claim -- the heuristic is
    a date comparison against the land-fill convention change, and one LOMO
    run (``split-model-ACCESS-ESM1-5``) has neither a stamp nor a sidecar.
    """
    import torch
    from diffusion import inference as util
    sd = Path(save_dir)
    ck = torch.load(sd / f'ckpt_epoch{_latest_epoch(sd):03d}.pt',
                    map_location='cpu', weights_only=False)
    if ck.get('land_fill') in ('zero', 'meanstd'):
        return ck['land_fill'], False
    sidecar = sd / 'land_fill'
    if sidecar.is_file():
        v = sidecar.read_text().strip()
        if v in ('zero', 'meanstd'):
            return v, False
    return util.land_fill_mode(ck.get('timestamp')), True


def _latest_epoch(save_dir):
    """Largest ``ckpt_epoch*.pt`` in a run dir.

    ⚠️ Only ever used to pick a FILE to read a stamp out of -- never to decide
    what to evaluate, which is pinned from the diffusion cache. Parsed by regex
    and compared numerically, not by a ``[-3:]`` slice that would read
    ``ckpt_epoch1000`` as epoch 0.
    """
    import re
    eps = [int(m.group(1)) for m in
           (re.search(r'ckpt_epoch(\d+)\.pt$', str(f))
            for f in Path(save_dir).glob('ckpt_epoch*.pt')) if m]
    if not eps:
        raise FileNotFoundError(f'no checkpoints in {save_dir}')
    return max(eps)


def provenance_line(split, models, kinds=KINDS):
    """Each panel's run provenance, in panel order.

    See :func:`run_provenance` for why this is reported. Takes the kinds
    actually DRAWN, not the module default, so a subset never annotates a
    panel that is not there.
    """
    return '   '.join(f'{LABEL[k]}: {run_provenance(k, split, models)}'
                      for k in kinds)


# ---------------------------------------------------------------------------
# the 2 x 5 manuscript grid
# ---------------------------------------------------------------------------
#: Rows of :func:`grid_figure`, TOP to bottom. In-sample first: it is the row
#: where the methods separate most clearly, and LOMO reads as the harder case
#: beneath it.
#:
#: ⚠️ **The two rows are over DIFFERENT FLEETS** (14 LOMO against 16 in-sample
#: of which 14 carry data) and under DIFFERENT COLOUR SCALES. They are drawn in
#: one figure so a reader can see the same five methods under both protocols;
#: they must never be differenced, and this figure draws nothing that says so
#: -- see the caption list in the module docstring.
GRID_ROWS = ('insample', 'oos')

#: Left-margin row labels. 'LOMO', not 'leave-one-model-out': rotated into the
#: left margin the spelled-out form is longer than one Robinson panel is tall
#: and runs into the next row's label. The caption expands the acronym.
ROW_LABEL = {'insample': 'in-sample', 'oos': 'LOMO'}


def grid_column_label(kind):
    """Column header for the grid: :data:`LABEL` with an initial capital."""
    s = LABEL[kind]
    return s[:1].upper() + s[1:]

#: Exponent of the grid's power-law colour scale -- ``eval_common``'s own
#: default, and the value the ``predictor_set`` manuscript figure uses. See
#: :func:`ec.nonlinear_levels`: colour position goes as
#: ``(|x|/limit)**gamma``, so boundary *k* of *n* sits at ``limit*(k/n)**2``.
GRID_GAMMA = 0.5

#: ``(split, sign) -> (limit, bins-per-wing)`` for ``units='pct'``.
#:
#: These are NOT the linear limits of the single-method ``error_maps``
#: figures: the five in-sample fleet MAPEs span 1.50-23.44 % against that
#: figure's 15 % ceiling. Measured over all five methods pooled (deep-ocean
#: masked):
#:
#:   split     sign        p90    p95    p98    p99     max    inherited
#:   oos       unsigned   44.24  71.71  114.2  147.2    6384   50
#:   oos       signed      6.61  10.21   18.3  23.62    1856   14
#:   insample  unsigned   21.51  39.27  74.39  103.9    1360   15
#:   insample  signed     5.762  7.603  12.72  18.12   410.3    7
#:
#: The distribution is strongly peaked at zero with a tail four orders of
#: magnitude long, so NO linear limit both resolves the bulk and reaches the
#: tail -- which is the exact situation ``nonlinear_levels`` exists for. Each
#: limit is set near the pooled p98-p99 and the bins are chosen so that
#: ``limit/n**2`` is a round number and every second boundary is an integer
#: (100/10**2 = 1 -> 1, 4, 9, ... 100; 25/10**2 = 0.25 -> 0.25, 1, 2.25, ...).
#: That is what lets the bar label its own boundaries, so a reader reads a
#: NUMBER off it without knowing the transform.
#:
#: ⚠️ **The pooled percentile is the wrong statistic for the in-sample
#: unsigned row.** The pool is dominated by QRF / linear / climatology, but
#: what decides whether the figure resolves the two panels it exists to compare
#: is where the DIFFUSION and U-NET bulks sit on the ladder. Per-method
#: deep-ocean quantiles, against a (64, 8) ladder 0, 1, 4, 9, 16, 25, 36, 49,
#: 64:
#:
#:   method       q25    q50    q75    q90    q95   below the FIRST boundary
#:   diffusion   0.989   1.27   1.85   4.18   7.01   25.8 %
#:   U-Net       0.546  0.803   1.30   2.38   3.80   63.4 %
#:   QRF          4.96   6.79   10.4   19.5   32.6    0.0 %
#:   linear       7.77   11.2   18.3   40.6   67.5    0.0 %
#:
#: i.e. the U-Net panel would be 63 % one colour. (25, 10) drops the first
#: boundary 1 -> 0.25 while still reaching 25: the same ``limit/n**2 = 0.25``
#: ladder as the two signed rows, the same integer every-second boundary (1,
#: 4, 9, 16, 25), and it puts diffusion's quartiles at bins 1/2/2 and its
#: p90/p95 at 4/5. The cost is that linear and climatology saturate above
#: their q75 -- the panels that lose resolution are the weak ones, not the two
#: the comparison turns on.
GRID_REL_SCALE = {
    ('oos', 'unsigned'): (100.0, 10),
    ('insample', 'unsigned'): (25.0, 10),
    ('oos', 'signed'): (25.0, 10),
    ('insample', 'signed'): (16.0, 8),
}

#: The same in mol/m^3, for ``units='abs'``. Same construction, from the abs
#: half of the same table:
#:
#:   split     sign         p90      p95      p98      p99     max  inherited
#:   oos       unsigned   0.0308   0.0339   0.0372   0.0403   2.003   0.10
#:   oos       signed     0.0098   0.0117   0.0137   0.0156   1.970   0.011
#:   insample  unsigned   0.0234   0.0272   0.0299   0.0325   0.167   0.03
#:   insample  signed     0.0093   0.0109   0.0125   0.0138   0.167   0.008
#:
#: The two SIGNED rows deliberately share 0.016: in mol/m^3 the two splits'
#: signed distributions really are close (p99 0.0156 against 0.0138). Every
#: abs row takes 16 bins where the pct rows take 8-10, because in mol/m^3 each
#: method's error is narrowly distributed about a level that differs ~14x
#: between the CNNs and the floor.
GRID_ABS_SCALE = {
    ('oos', 'unsigned'): (0.04, 16),       # 0.00015625 k**2; ticks 0.0025 j**2
    ('insample', 'unsigned'): (0.032, 16),  # 0.000125 k**2;  ticks 0.002 j**2
    ('oos', 'signed'): (0.016, 16),        # 6.25e-05 k**2;   ticks 0.001 j**2
    ('insample', 'signed'): (0.016, 16),
}

#: Most boundaries a colorbar may LABEL. :func:`_tick_step` picks the sparsest
#: spacing that respects it. A signed bar labels BOTH wings, so it needs twice
#: the spacing, or its labels overlap.
GRID_MAX_TICKS = 7


def _tick_step(bins, sign):
    """The sparsest spacing that DIVIDES ``bins`` and labels at most
    :data:`GRID_MAX_TICKS` boundaries.

    ⚠️ It must divide ``bins`` or the ladder's top boundary -- the LIMIT, the
    one number a reader most needs off the bar -- goes unlabelled. That is why
    this searches divisors rather than taking a fixed 2 or 4.
    """
    n, wings = int(bins), (2 if sign == 'signed' else 1)
    for step in range(1, n + 1):
        if n % step == 0 and wings * (n // step) + 1 <= GRID_MAX_TICKS:
            return step
    raise ValueError(f'no tick step divides {bins} bins within '
                     f'{GRID_MAX_TICKS} labels')


def grid_scale_for(units):
    return GRID_REL_SCALE if units == 'pct' else GRID_ABS_SCALE


def grid_levels(limit, bins, sign):
    """``contourf`` level boundaries for one row of the grid.

    ``ec.nonlinear_levels`` returns the SYMMETRIC ladder (both wings), which is
    what a signed diverging map wants. An unsigned map is 0-up, so it takes the
    positive wing only -- element ``bins`` of that list is the 0 boundary, and
    everything after it is the wing. Slicing the same list rather than
    re-deriving the wing keeps one construction and one place for the arithmetic
    to be wrong.
    """
    full = ec.nonlinear_levels(float(limit), GRID_GAMMA, int(bins))
    if sign == 'signed':
        return full
    # ``full[bins]`` is the 0 boundary, and it arrives as ``-0.0`` (it is the
    # negated wing's last element). Equal to 0.0 for every comparison
    # matplotlib makes, but it prints as '-0' on a tick label.
    return [0.0] + full[int(bins) + 1:]


def grid_ticks(limit, bins, sign, step=None):
    """Every ``step``-th boundary, i.e. what the colorbar labels.

    ⚠️ **Boundaries, not round numbers of our own choosing.** Uneven ``levels``
    make ultraplot draw the bar on a function scale where each interval gets an
    equal share, so a tick anywhere but on a boundary would sit at a position
    the transform did not put it. The step (:func:`_tick_step` unless given)
    lands on round values with the pinned ``(limit, bins)`` pairs, because
    ``limit/bins**2`` is a round number by construction; a ``bins`` not
    divisible by the step would miss the limit itself, which is why every
    pinned value is 8, 10 or 16.
    """
    n, lim = int(bins), float(limit)
    step = _tick_step(n, sign) if step is None else int(step)
    pos = [lim * (k / n) ** (1.0 / GRID_GAMMA) for k in range(0, n + 1, step)]
    if sign == 'unsigned':
        return pos
    return [-v for v in pos[::-1] if v > 0] + pos


def grid_scalar_label(scalar, sign, units):
    """The one number a grid panel carries -- the area-weighted scalar of
    :func:`panel_scalar`, with the method name stripped (the column header
    carries that) and the statistic name stripped (the figure draws one
    statistic). Kept on the page because it is data: the colour scale is
    shared across methods that differ by 15x.
    """
    # THREE SIGNIFICANT figures in mol/m^3, not three decimals: the in-sample
    # CNN means are 0.0024 and 0.0015, which '.3f' renders as 0.002 and 0.002.
    # ``#`` keeps the trailing zeros ``g`` would strip, so every panel shows
    # the same three significant figures.
    fmt = '.1f' if units == 'pct' else '#.3g'
    unit = '%' if units == 'pct' else ec.CBAR_LABEL
    sgn = '+' if (sign == 'signed' and scalar > 0) else ''
    return f'{sgn}{scalar:{fmt}} {unit}'


def grid_figure(sign, units='pct', kinds=ALL_KINDS, rows=GRID_ROWS,
                scalars=True, refwidth='36mm'):
    """The 2 x N manuscript grid: rows = split, columns = method.

    One colour scale and one colorbar PER ROW, both on the right, because the
    two splits' error fields differ by an order of magnitude in the relative
    units -- a single figure-wide bar would render the in-sample row blank.
    Within a row every panel shares one scale, which is the comparison the
    figure exists for.

    Stripped to the axes: no suptitle, no provenance footer, no formula.
    Column headers, row labels, bold panel letters, each panel's scalar, two
    bars. Everything removed is printed to the log by
    :func:`grid_provenance_report` on every run and is listed in the module
    docstring as what the caption must carry.

    Returns ``(fig, cases)``; the caller saves the figure. (The original took
    ``out_dir``/``dpi``/``formats`` and saved ``methods_grid_<sign>[_abs]``
    itself.)
    """
    kinds, rows = tuple(kinds), tuple(rows)
    scale = grid_scale_for(units)

    cases = {}
    for split in rows:
        for kind in kinds:
            ds, why = load_cache(kind, split)
            if ds is None:
                raise FileNotFoundError(
                    f'no {kind} cache for split={split}: {why}. Every panel '
                    f'of this figure comes from one code path, so a missing '
                    f'one is a run to launch, not a panel to drop -- produce '
                    f'it with:\n    python -m baselines.evaluate --baseline '
                    f'{kind} --split {split} --aggregate')
            # ⚠️ verbatim, on every cache, before anything is derived from it
            # -- `gen - truth` is only the signed model-mean error where these
            # hold.
            info = verify_cache(ds, f'{kind}/{split}')
            logging.info(f'  verify {kind:18s} {split:8s} ok '
                         f'({info["n_cells"]} cells, worst Jensen violation '
                         f'{info["max_violation"]:.3g} <= tol '
                         f'{info["tol"]:.3g})')
            crosscheck(ds, kind, split)
            da = method_map(ds, sign, units)
            cases[(split, kind)] = dict(
                kind=kind, split=split, ds=ds, da=da,
                scalar=panel_scalar(da, sign, units))

    pplt = ec._pplt()
    fig, axs = pplt.subplots(nrows=len(rows), ncols=len(kinds), proj='robin',
                             proj_kw=dict(lon_0=202), refwidth=refwidth,
                             share=0)
    contourf_plot = ec._contourf()
    cbar_label = '%' if units == 'pct' else ec.CBAR_LABEL

    for r, split in enumerate(rows):
        limit, bins = scale[(split, sign)]
        kw = dict(cmap=('Div' if sign == 'signed' else 'Reds'),
                  extend=('both' if sign == 'signed' else 'max'),
                  levels=grid_levels(limit, bins, sign))
        cbar_kw = dict(ticks=grid_ticks(limit, bins, sign))
        for c, kind in enumerate(kinds):
            # 2D indexing only when there IS a second row: a 1 x N grid's
            # SubplotGrid does not promise `axs[0, c]` means what it does on a
            # 2 x N one.
            ax = axs[r, c] if len(rows) > 1 else axs[c]
            case = cases[(split, kind)]
            da = ec.mask_shallow(case['da'])
            if sign == 'unsigned':
                # any surviving sign is taken as magnitude rather than
                # silently mis-scaling
                da = np.abs(da)
            # The bar rides the LAST panel of the row, which is the right-hand
            # edge of the figure -- one bar per row, never per panel.
            contourf_plot(ax=ax, da=da, contourf_kw=kw,
                          cbar=(c == len(kinds) - 1),
                          cbar_label=cbar_label, cbar_kw=cbar_kw)
            if scalars:
                ax.format(title=grid_scalar_label(case['scalar'], sign, units))

    # Column = method, row = split. Drawn as labels rather than titles so the
    # titles are free for the one number each panel carries.
    #
    # Letters ABOVE the axes (``abcloc='l'``), not inside: a Robinson globe's
    # upper-left corner is empty white space *outside the limb*, so an inside
    # letter floats in a void. Above-left, the letter sits beside the panel's
    # own scalar title.
    axs.format(abc='a)', abcloc='l')
    fig.format(toplabels=[grid_column_label(k) for k in kinds],
               leftlabels=[ROW_LABEL[s] for s in rows])
    _bold_letters(axs)

    grid_provenance_report(cases, rows, kinds, sign, units)

    for case in cases.values():
        case['ds'].close()
    return fig, cases


def _bold_letters(axs):
    """Bold the a)-j) labels -- belt and braces, not the mechanism.

    ultraplot already draws them bold (``rc['abc.weight']`` defaults to
    ``'bold'`` on ultraplot 1.70.0). This exists so the style survives a change
    to that default, and it is set on the Text object rather than through an
    ``abc_kw`` because that keyword's spelling has moved between proplot and
    ultraplot. The label is ``ax._title_dict['abc']``; a miss warns and the
    figure still draws.
    """
    n = 0
    for ax in axs:
        t = (getattr(ax, '_title_dict', None) or {}).get('abc')
        if t is None:
            t = next((getattr(ax, a, None) for a in
                      ('_abc_label', '_abc_title')
                      if getattr(ax, a, None) is not None), None)
        if t is None or not hasattr(t, 'set_fontweight'):
            continue
        try:
            t.set_fontweight('bold')
            n += 1
        except Exception:
            pass
    if not n:
        logging.warning('  [warn] could not reach the panel letters to bold '
                        "them -- ultraplot's abc Text has moved out of "
                        '_title_dict["abc"]; see _bold_letters. They are '
                        'probably still bold by rc default -- look at the PNG')


def grid_provenance_report(cases, rows, kinds, sign, units):
    """Print everything the stripped figure does NOT draw.

    When a figure's prose moves into the caption, the numbers behind it have
    to keep being printed somewhere they can be read off a run. Everything here
    is caption material; the module docstring lists what of it is mandatory.
    """
    stat = ('signed model-mean bias' if sign == 'signed'
            else 'mean per-model magnitude')
    logging.info(f'\n  --- caption debt ({stat}, '
                 f'{"%" if units == "pct" else "mol/m^3"}) ---')
    for split in rows:
        ds = cases[(split, kinds[0])]['ds']
        a = ds.attrs
        n_eff = int(a.get('n_models_effective', a['n_models']))
        fleet = (f'{a["n_models"]} models' if n_eff == int(a['n_models'])
                 else f'{a["n_models"]} models ({n_eff} with data)')
        # Unwrap in case a ROW_LABEL is ever line-broken for the left margin
        # -- a newline mid-line silently splits this log record in two.
        row = ROW_LABEL[split].replace('\n', '')
        logging.info(f'  {row}: {a["title"]}; epoch '
                     f'{cache_epoch(ds)}, {a["years_label"]}, {fleet}')
        models = str(a['models']).split()
        logging.info(f'    provenance: {provenance_line(split, models, kinds)}')
        logging.info('    scalars: ' + '  '.join(
            f'{LABEL[k]} {grid_scalar_label(cases[(split, k)]["scalar"], sign, units)}'
            for k in kinds))
        limit, bins = grid_scale_for(units)[(split, sign)]
        logging.info(f'    colour scale: nonlinear gamma={GRID_GAMMA}, '
                     f'limit {limit:g}, {bins} bins/wing, boundaries '
                     + ', '.join(f'{v:g}' for v in
                                 grid_levels(limit, bins, 'unsigned')))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
#: How the ten caches were produced. ``<M>`` runs over the 14 LOMO models the
#: diffusion cache ``T-S_ann_int_oos.nc`` names (``evaluate.py`` reads the
#: fleet from it); every command is run once with ``--split oos`` and once
#: with ``--split insample``. Training of the runs these load is described in
#: baselines/README.md.
COMPUTE_STEPS = (
    'python -m baselines.evaluate --baseline diffusion --split <split>',
    'python -m baselines.evaluate --baseline unet-mse --split <split>',
    'python -m baselines.evaluate --baseline qrf --split <split> --model <M> '
    '--draw per-cell --seed 0          # one job per model',
    'python -m baselines.evaluate --baseline pointwise-linear --split <split>',
    'python -m baselines.evaluate --baseline climatology --split <split>',
    'python -m baselines.evaluate --baseline <kind> --split <split> '
    '--aggregate   # each of the five kinds (qrf: + --draw per-cell --seed 0)',
)


def compute():
    """The caches come from ``baselines/evaluate.py``, not from this script.

    One cache per (method, split) -- ``<kind>_T-S_ann_int_<split>.nc`` under
    ``paths.BASELINE_ROOT`` -- each pinned (fleet, epoch 500, 5 samples, 100
    DDIM steps, years 81-100) to the diffusion ``error_maps`` cache of the same
    split. This prints the invocations.
    """
    print('Fig. 8 caches are written by baselines/evaluate.py:')
    for step in COMPUTE_STEPS:
        print(f'  {step}')


def plot():
    """Load the ten caches and draw the unsigned / % grid."""
    fig, _ = grid_figure(SIGN, UNITS, kinds=ALL_KINDS, rows=GRID_ROWS)
    # the grid letters its own panels (above-left), so no abc here
    render_common.finish(fig, abc=False)
    return render_common.save(fig, 'baseline_grid', bbox_inches='tight')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        compute()
    else:
        print(plot())
