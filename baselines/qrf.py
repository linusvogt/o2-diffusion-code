"""The quantile regression forest baseline: features, fit, load, draw.

Extracted from general/baselines/qrf.py.

The probabilistic baseline: a deterministic U-Net has no spread at all, so it
alone cannot test "the advantage is chiefly in the spread" -- that claim would
be true by construction. This is the model that makes the spread comparison a
comparison.

What it is
----------
One forest per leave-one-model-out split, fit on rows of

    (predictor channels at a cell, sin/cos lat, sin/cos lon)  ->  target at that cell

in the SAME normalized space the diffusion model works in, using the SAME
``TracerDataset`` statistics. It is a per-cell model with a location encoding:
the fair minimum against a model that conditions on the whole field. It cannot
see its neighbours, which is the point --
that is the structure the diffusion model is being credited with.

⚠️ Three things here are easy to get silently wrong.

1. **Normalized space, this run's own stats.** ``ec.predict_ensemble``
   normalizes ``cond`` with ``run.scaling`` and denormalizes the sampler's
   output with ``* stds[target] + means[target]``. A forest fit on physical
   units would be double-scaled by that path and still produce a plausible-
   looking map.
2. **Ocean rows only.** ``dataset[i]['cond']`` fills land at ``-mean/std``
   (``land_fill='meanstd'``), and land is most of the domain. Land rows would
   put a large spurious mode at exactly the land-fill value into every leaf --
   the QRF analogue of an unmasked MSE. :func:`build_rows` samples from ocean
   cells only.
3. **The draw scheme is a scientific choice, not an implementation detail.**
   See below.

The draw scheme
---------------
A forest gives a *per-cell* predictive distribution and no joint distribution
over the map, so turning it into an ensemble MEMBER needs a rule:

``'per-cell'``    an independent uniform per grid cell -- the **headline**.
``'comonotone'``  one uniform per member, applied at every cell -- a stamped
                  sensitivity.

Per-cell marginals are identical under both, so **CRPS, rank histograms and
per-cell spread-skill do not depend on this choice**. What differs is the
spatial coherence of a single MEMBER: a ``'per-cell'`` member is spatial noise
around the median field, a ``'comonotone'`` member is a coherent field.
``'per-cell'`` is the headline because it is the truthful representation of a
model with no spatial covariance; ``'comonotone'`` imposes perfect global rank
correlation, i.e. hand-installs a spatial model the forest never fit, so it is
reported as a sensitivity rather than as the result. The scheme is stamped in
every evaluation cache.

⚠️ **The ensemble MEAN is not where the schemes differ** -- measured, not
assumed. Both average ``n`` draws of the same per-cell marginal, so the mean's
residual about the median shrinks as ``1/sqrt(n)`` either way and its magnitude
came out equal (0.166 vs 0.164 of the q05-q95 spread at n=10). The
ensemble-mean error maps are therefore essentially draw-scheme independent;
only member-level spatial statistics are not.

Layout on disk
--------------
``baselines.naming.save_dir_from_config`` with ``baseline='qrf'``, so a forest lands in
``<BASELINE_ROOT>/baseline-qrf/<the same directory name as the diffusion run it is
compared against>`` and pairing them stays a lookup.
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from baselines import shims


KIND = 'qrf'

#: Quantile levels stored per cell. The draw inverts this piecewise-linearly,
#: so the grid is the resolution of the predictive distribution -- 101 levels
#: put the draw's discretization error well below the ensemble spread, at a
#: cost that is linear and small next to the forest traversal itself.
#: ⚠️ Levels 0 and 1 are the min/max of the retained leaf values, so a draw can
#: never leave the range the training data spanned. That is inherent to a
#: forest and is one of the things the comparison measures.
QUANTILE_GRID = np.round(np.linspace(0.0, 1.0, 101), 4)

DRAW_SCHEMES = ('per-cell', 'comonotone')
DEFAULT_DRAW = 'per-cell'

#: Fit defaults, sized by a probe fit.
#:
#: ⚠️ ``max_samples_leaf`` is the memory lever and BOTH extremes are wrong here:
#:
#: * ``None`` (keep every leaf value -- the textbook Meinshausen QRF) is not
#:   merely expensive, it is *pathological in this implementation*:
#:   ``_get_y_train_leaves`` allocates a DENSE
#:   ``(n_trees, max_node_count, 1, max_samples_leaf)`` int64 array, where
#:   ``max_samples_leaf`` becomes the largest leaf in ANY tree. One fat leaf
#:   therefore sizes the array for every node of every tree -- measured **37 GB
#:   for a 200k-row forest** (max leaf ~2300 against a 20-sample minimum).
#: * ``1``, the package default, keeps a single value per leaf, so the
#:   predictive distribution comes from across-tree variation alone -- a
#:   different estimator wearing the same name.
#:
#: An INTEGER cap shuffles each oversized leaf and truncates it
#: (``random.shuffle`` then ``[:max_samples_leaf]``), i.e. an i.i.d. subsample
#: of that leaf's distribution, and bounds memory at
#: ``n_trees x max_node_count x k x 8 B`` -- predictable and small. With
#: ``min_samples_leaf=20`` a cap of 40 retains whole leaves in the ordinary
#: case and only thins the rare fat one.
DEFAULT_FIT = dict(n_estimators=100, min_samples_leaf=20, max_samples_leaf=40,
                   n_cells_per_map=200, seed=0)

ARTIFACT = 'qrf.joblib'
META = 'qrf_meta.json'

_INSTALL_HINT = (
    "quantile-forest / scikit-learn are not importable. Install them with:\n"
    "  pip install quantile-forest scikit-learn")


def _rfqr():
    """The forest class, or an actionable error -- never a bare ImportError."""
    try:
        from quantile_forest import RandomForestQuantileRegressor
    except ImportError as exc:                       # pragma: no cover
        raise RuntimeError(f'{_INSTALL_HINT}\n\noriginal error: {exc}') from exc
    return RandomForestQuantileRegressor


def _versions():
    import sklearn
    import quantile_forest
    return dict(sklearn=sklearn.__version__,
                quantile_forest=quantile_forest.__version__,
                numpy=np.__version__)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def location_features(lat, lon):
    """(H, W, 4) sin/cos of latitude and longitude.

    The location encoding the forest uses. sin/cos of
    longitude because longitude is periodic and a raw value would put a seam at
    the date line; sin/cos of latitude because the pair is a smooth encoding of
    a bounded coordinate (sin monotone in latitude, cos symmetric about the
    equator) and costs nothing next to getting the seam right.
    """
    LAT, LON = np.meshgrid(np.deg2rad(np.asarray(lat, dtype=np.float64)),
                           np.deg2rad(np.asarray(lon, dtype=np.float64)),
                           indexing='ij')
    return np.stack([np.sin(LAT), np.cos(LAT),
                     np.sin(LON), np.cos(LON)], axis=-1).astype(np.float32)


LOC_NAMES = ['sin_lat', 'cos_lat', 'sin_lon', 'cos_lon']


def feature_names(predictor_channels):
    return list(predictor_channels) + LOC_NAMES


def build_rows(dataset, loc, n_cells_per_map, seed, log_every=1000,
               return_cells=False, sample_mask=None):
    """``(X, y)`` -- ``n_cells_per_map`` ocean cells drawn from EVERY map.

    ⚠️ Subsampling is required, not an optimization: 3150-4500 training maps x
    ~4x10^4 ocean cells is ~1.6x10^8 candidate rows, and a QRF retains its
    training values at the leaves. It subsamples **cells, not maps**, so every
    year of the forcing trajectory is still represented -- the maps carry the
    trend that the LOMO test is about, and dropping maps would drop signal
    while dropping cells only drops redundancy within a field.

    Rows are drawn from the ocean mask only (see the module docstring, trap 2).
    ``sample_mask`` restricts which MAPS contribute -- used to hold out a year
    range for hyperparameter selection without touching the held-out model.
    ``return_cells`` additionally returns the flat cell index of every row, so a
    caller can prove the ocean restriction STRUCTURALLY. That matters: the obvious
    test -- "no target equals the land-fill value" -- is unusable here, because
    ``TracerDataset``'s in-RAM path stores float16, whose ~2e-3 spacing at these
    magnitudes puts **1.4% of genuine ocean rows** inside any tolerance wide
    enough to catch a real land cell. Indices are exact; values are not.
    """
    rng = np.random.default_rng(seed)
    ocean_flat = np.flatnonzero(dataset.mask.ravel())
    n_ocean = ocean_flat.size
    if n_cells_per_map > n_ocean:
        raise ValueError(f'{n_cells_per_map} cells/map requested but only '
                         f'{n_ocean} ocean cells exist')
    loc_flat = loc.reshape(-1, loc.shape[-1])[ocean_flat]     # (n_ocean, 4)

    idx_maps = (np.arange(len(dataset)) if sample_mask is None
                else np.flatnonzero(np.asarray(sample_mask, dtype=bool)))
    n_maps = len(idx_maps)
    if n_maps == 0:
        raise ValueError('sample_mask selected no maps')
    n_ch = len(dataset.predictors)
    X = np.empty((n_maps * n_cells_per_map, n_ch + loc.shape[-1]), np.float32)
    y = np.empty(n_maps * n_cells_per_map, np.float32)
    used_cells = np.empty(n_maps * n_cells_per_map, np.int64)
    t0 = time.time()
    for i, isamp in enumerate(idx_maps):
        s = dataset[int(isamp)]
        cond = s['cond'].numpy().reshape(n_ch, -1)
        tgt = s['y'].numpy().ravel()
        pick = rng.choice(n_ocean, size=n_cells_per_map, replace=False)
        cell = ocean_flat[pick]
        sl = slice(i * n_cells_per_map, (i + 1) * n_cells_per_map)
        X[sl, :n_ch] = cond[:, cell].T
        X[sl, n_ch:] = loc_flat[pick]
        y[sl] = tgt[cell]
        used_cells[sl] = cell
        if log_every and (i + 1) % log_every == 0:
            logging.info(f'  rows: {i+1}/{n_maps} maps '
                         f'({time.time()-t0:.0f}s)')
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('non-finite value in the QRF training rows -- a land '
                         'cell or a corrupt member reached the fit')
    logging.info(f'built {X.shape[0]:,} rows x {X.shape[1]} features from '
                 f'{n_maps} maps in {time.time()-t0:.0f}s '
                 f'({X.nbytes/1e9:.2f} GB)')
    return (X, y, used_cells) if return_cells else (X, y)


# ---------------------------------------------------------------------------
# fit / persist
# ---------------------------------------------------------------------------
#: Levels used for SCORING. The stored grid includes 0 and 1, which are the
#: min/max of the retained leaf values; scoring there rewards nothing but
#: extreme-value luck, so selection uses the interior only.
def _score_levels(grid):
    g = np.asarray(grid, dtype=float)
    return g[(g > 0.0) & (g < 1.0)]


def pinball_loss(qpred, y, quantiles):
    """Mean pinball (quantile) loss -- a PROPER scoring rule.

    ``qpred`` is (n, n_q) predicted quantiles, ``y`` (n,) the truth,
    ``quantiles`` (n_q,) the levels. Averaged over cells and levels this is the
    CRPS up to a constant factor and the grid's discretization, which is why it
    is the selection criterion:

    * selecting on **sharpness** would pick the most over-confident forest;
    * selecting on **RMSE** would ignore the spread entirely -- the one axis
      the QRF exists to measure.

    A proper rule rewards calibration and sharpness jointly, so the winner is
    the forest whose *distribution* is best, not merely whose median is.
    """
    d = np.asarray(y, dtype=np.float64)[:, None] - np.asarray(qpred, np.float64)
    q = np.asarray(quantiles, dtype=np.float64)[None, :]
    return float(np.mean(np.maximum(q * d, (q - 1.0) * d)))


def select_min_samples_leaf(dataset, loc, years, grid, *, n_cells_per_map,
                            n_cells_val, val_years, n_estimators,
                            max_samples_leaf, seed, n_jobs):
    """Choose ``min_samples_leaf`` on held-in models' held-out YEARS.

    This exists for symmetry: the U-Net baseline reports a validation-selected
    epoch so it cannot be accused of being understated, and an untuned forest
    against a tuned network would tilt the comparison toward the diffusion
    model on exactly the axis of interest -- the spread.

    The split mirrors the U-Net's: validation is ``val_years`` of the **held-in**
    models, never the held-out one, so nothing about the LOMO protocol leaks.
    Those are also the years the evaluation itself averages over
    (``years_label = 'yrs 81-100'`` in the diffusion cache), so the criterion is
    measured under the condition the model is judged in.

    Returns ``(best_min_samples_leaf, curve)``; the caller refits on ALL years
    with the winner, which is what keeps the final artifact trained on exactly
    the data every other baseline run is.
    """
    RFQR = _rfqr()
    years = np.asarray(years)
    lo, hi = val_years
    val_mask = (years >= lo) & (years <= hi)
    if not val_mask.any() or val_mask.all():
        raise ValueError(
            f'validation years {val_years} select {int(val_mask.sum())} of '
            f'{val_mask.size} maps -- cannot both train and validate')
    logging.info(f'selection: {int((~val_mask).sum())} train maps / '
                 f'{int(val_mask.sum())} validation maps (years {lo}-{hi} of '
                 f'the held-in models only)')

    X_tr, y_tr = build_rows(dataset, loc, n_cells_per_map, seed,
                            log_every=0, sample_mask=~val_mask)
    X_va, y_va = build_rows(dataset, loc, n_cells_val, seed + 1,
                            log_every=0, sample_mask=val_mask)
    levels = _score_levels(QUANTILE_GRID)
    logging.info(f'selection: {X_tr.shape[0]:,} train rows, '
                 f'{X_va.shape[0]:,} validation rows, '
                 f'{levels.size} scoring levels')

    curve = []
    for msl in grid:
        t0 = time.time()
        f = RFQR(n_estimators=n_estimators, min_samples_leaf=int(msl),
                 max_samples_leaf=max_samples_leaf, n_jobs=n_jobs,
                 random_state=seed)
        f.fit(X_tr, y_tr)
        qp = np.asarray(f.predict(X_va, quantiles=[float(q) for q in levels]))
        loss = pinball_loss(qp, y_va, levels)
        spread = float(np.median(np.percentile(qp, 95, axis=1)
                                 - np.percentile(qp, 5, axis=1)))
        curve.append(dict(min_samples_leaf=int(msl), pinball=loss,
                          spread_q05_q95=spread,
                          seconds=float(time.time() - t0)))
        logging.info(f'  min_samples_leaf={msl:>4}: pinball {loss:.5f}  '
                     f'spread {spread:.3f}  ({time.time()-t0:.0f}s)')
        # ⚠️ collect BEFORE the next fit. These forests are big (measured
        # 85 GB peak RSS for a grid reaching down to min_samples_leaf=2, whose
        # leaf store alone is ~25 GB and is transiently doubled while built),
        # and leaving the previous one for the cyclic collector stacks two of
        # them. The job wall is memory, not time.
        del f, qp
        gc.collect()
    best = min(curve, key=lambda r: r['pinball'])['min_samples_leaf']
    logging.info(f'selection: min_samples_leaf={best} wins on validation '
                 f'pinball loss')
    return best, curve


def fit(config, dataset, lat, lon, n_jobs=-1, years=None,
        select_grid=None, val_years=(81, 100), n_cells_val=50,
        **fit_kw):
    """Fit one forest and write it, with its stats and provenance, to save_dir.

    The stats written here are ``dataset.means``/``stds``/``land_fill``, i.e.
    exactly what ``Trainer`` stamps into a U-Net checkpoint, so
    :func:`load_baseline_run` can populate ``RunHandle.scaling`` identically and
    the evaluation path cannot tell the two baselines apart.
    """
    import joblib
    RFQR = _rfqr()
    p = {**DEFAULT_FIT, **{k: v for k, v in fit_kw.items() if v is not None}}
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    loc = location_features(lat, lon)

    # hyperparameter selection, mirroring the U-Net's validation-selected
    # epoch (see select_min_samples_leaf for why the symmetry matters)
    selection = None
    if select_grid:
        if years is None:
            raise ValueError(
                'select_grid needs the per-map year coordinate to hold out '
                'validation years; pass years=ds_train["year"].values')
        best, curve = select_min_samples_leaf(
            dataset, loc, years, select_grid,
            n_cells_per_map=p['n_cells_per_map'], n_cells_val=n_cells_val,
            val_years=val_years, n_estimators=p['n_estimators'],
            max_samples_leaf=p['max_samples_leaf'], seed=p['seed'],
            n_jobs=n_jobs)
        selection = dict(grid=[int(g) for g in select_grid], curve=curve,
                         selected=int(best), val_years=list(val_years),
                         n_cells_val=int(n_cells_val),
                         criterion='mean pinball loss over the interior '
                                   'quantile grid (CRPS up to a constant), on '
                                   'held-in models\' held-out years')
        p['min_samples_leaf'] = int(best)

    # refit on ALL years with the winner, so the shipped artifact is trained on
    # exactly the data every other baseline run is -- selection borrowed years, it
    # does not keep them
    X, y = build_rows(dataset, loc, p['n_cells_per_map'], p['seed'])

    if p['max_samples_leaf'] is None:
        logging.warning(
            'max_samples_leaf=None allocates a dense array sized by the '
            'LARGEST leaf in any tree -- 37 GB for a 200k-row forest in a '
            'probe fit. Use an integer cap unless you have measured '
            'this configuration.')
    logging.info(f'fitting {KIND}: {p["n_estimators"]} trees, '
                 f'min_samples_leaf={p["min_samples_leaf"]}, '
                 f'max_samples_leaf={p["max_samples_leaf"]}, n_jobs={n_jobs}')
    t0 = time.time()
    forest = RFQR(n_estimators=p['n_estimators'],
                  min_samples_leaf=p['min_samples_leaf'],
                  max_samples_leaf=p['max_samples_leaf'],
                  n_jobs=n_jobs, random_state=p['seed'])
    forest.fit(X, y)
    t_fit = time.time() - t0
    logging.info(f'fitted in {t_fit/60:.1f} min')

    meta = dict(
        kind=KIND,
        predictor_channels=list(dataset.predictors),
        target=dataset.target,
        feature_names=feature_names(dataset.predictors),
        means=dict(dataset.means), stds=dict(dataset.stds),
        land_fill=dataset.land_fill,
        n_samples=int(dataset.n_samples),
        n_rows=int(X.shape[0]),
        quantile_grid=[float(q) for q in QUANTILE_GRID],
        lat=[float(v) for v in np.asarray(lat)],
        lon=[float(v) for v in np.asarray(lon)],
        fit_seconds=float(t_fit),
        selection=selection,
        versions=_versions(),
        **{k: (None if v is None else int(v) if isinstance(v, (int, np.integer))
               else v) for k, v in p.items()},
    )
    joblib.dump(forest, save_dir / ARTIFACT, compress=0)
    with open(save_dir / META, 'w') as fh:
        json.dump(meta, fh, indent=4)
    size_gb = (save_dir / ARTIFACT).stat().st_size / 1e9
    logging.info(f'wrote {save_dir/ARTIFACT} ({size_gb:.2f} GB) and {META}')
    return forest, meta


def train(config, dataset, device=None, lat=None, lon=None, n_jobs=-1,
          years=None, resume_from_existing=True, **fit_kw):
    """``BUILDERS``-shaped entry point. ``device`` is accepted and ignored.

    ``resume_from_existing`` means "don't refit an artifact that is already
    there" -- a forest has no epochs, so resuming is all-or-nothing.
    """
    save_dir = Path(config['save_dir'])
    # ⚠️ A selection run must never be skipped by the "already fitted" guard:
    # the artifact on disk was fitted at a DIFFERENT min_samples_leaf, so
    # honouring it would silently ship the untuned forest under a config that
    # claims a selected one.
    if fit_kw.get('select_grid'):
        resume_from_existing = False
    if resume_from_existing and (save_dir / ARTIFACT).is_file():
        logging.info(f'{save_dir/ARTIFACT} already exists -- not refitting '
                     f'(a forest has no epochs; delete it to refit)')
        return None, load_meta(save_dir)
    return fit(config, dataset, lat, lon, n_jobs=n_jobs, years=years,
               **fit_kw)


def load_meta(save_dir):
    with open(Path(save_dir) / META) as fh:
        return json.load(fh)


def artifact_exists(save_dir):
    return (Path(save_dir) / ARTIFACT).is_file()


# ---------------------------------------------------------------------------
# draw:  per-cell quantile function  ->  one ensemble member
# ---------------------------------------------------------------------------
def _invert(u, grid, qvals):
    """Piecewise-linear inverse CDF: value at level ``u[i]`` for each row.

    ``qvals`` is (n, n_q) ascending along axis 1 (a quantile function is
    non-decreasing by construction, which is what makes this well defined).
    """
    j = np.clip(np.searchsorted(grid, u, side='right'), 1, grid.size - 1)
    g0, g1 = grid[j - 1], grid[j]
    v0 = np.take_along_axis(qvals, (j - 1)[:, None], 1)[:, 0]
    v1 = np.take_along_axis(qvals, j[:, None], 1)[:, 0]
    w = np.where(g1 > g0, (u - g0) / np.where(g1 > g0, g1 - g0, 1.0), 0.0)
    return v0 + w * (v1 - v0)


def draw_member(qvals, rng, scheme=DEFAULT_DRAW, grid=QUANTILE_GRID):
    """One ensemble member from per-cell quantiles. See the module docstring.

    ``'per-cell'`` draws an independent uniform per cell; ``'comonotone'``
    draws ONE uniform and applies it everywhere. Identical per-cell marginals,
    different spatial coherence.
    """
    if scheme not in DRAW_SCHEMES:
        raise ValueError(f'draw scheme must be one of {DRAW_SCHEMES}, '
                         f'got {scheme!r}')
    n = qvals.shape[0]
    u = (np.full(n, float(rng.random())) if scheme == 'comonotone'
         else rng.random(n))
    return _invert(u, np.asarray(grid, dtype=float), qvals)


class _SlicePredictor:
    """``predict_fn(cond, mask)`` for :class:`shims.QuantileSampler`.

    ⚠️ Memoizes the forest evaluation per conditioning slice.
    ``predict_ensemble`` calls ``.sample()`` ``n_samples`` times on the SAME
    ``cond``; without the memo each ensemble would traverse the forest 10 times
    for one slice's worth of information -- 10x the cost for none of it. The key
    is a hash of the actual conditioning bytes plus the mask, so a new slice can
    never hit a stale entry (the alternative, object identity, is wrong the
    moment a caller reuses a buffer).
    """

    def __init__(self, forest, loc, quantiles, draw, rng, n_ch):
        self.forest, self.loc = forest, loc
        self.quantiles = [float(q) for q in quantiles]
        self.grid = np.asarray(quantiles, dtype=float)
        self.draw, self.rng, self.n_ch = draw, rng, n_ch
        self._key = None
        self._qvals = None
        self._cells = None
        self.n_forest_calls = 0

    def _key_of(self, cond_np, ocean):
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(cond_np).view(np.uint8))
        h.update(np.ascontiguousarray(ocean).view(np.uint8))
        return h.digest()

    def __call__(self, cond, mask=None):
        cond_np = cond.detach().cpu().numpy()[0]             # (C, H, W)
        n_ch, H, W = cond_np.shape
        assert n_ch == self.n_ch, f'{n_ch} channels, forest wants {self.n_ch}'
        if mask is None:
            ocean = np.ones((H, W), dtype=bool)
        else:
            m = mask.detach().cpu().numpy()
            ocean = np.asarray(m).reshape(-1, H, W)[0] > 0.5
        key = self._key_of(cond_np, ocean)
        if not ocean.any():
            # ⚠️ A slice with NO ocean cell is not an error here, and must not
            # be one: a model whose regrids are entirely NaN has an empty mask,
            # and the CNN baselines simply produce an all-NaN field for it and
            # carry on. `RandomForestQuantileRegressor.predict` instead raises
            # "Found array with 0 sample(s)" on the empty X, which would be the
            # only asymmetry between the forest and the networks in the
            # evaluation path. The in-sample fleet includes IPSL-CM5A2-INCA and
            # IPSL-CM6A-LR, whose regrids are all-NaN; `ec.get_models`
            # discovers models by globbing FILENAMES, so nothing opens them
            # until the sampler does.
            #
            # NaN, not the zeros the normal path would leave in `out`: a zero
            # is a real predicted O2 concentration once de-normalized, and
            # would read as a confident prediction of nothing. The downstream
            # arithmetic is unaffected either way (truth is NaN there too, so
            # `gen - truth` is NaN, and model_mean_abs' mean over `model`
            # skips it) -- but the PART on disk is then self-describing, which
            # is what `evaluate.aggregate`'s all-NaN count reads.
            logging.warning(
                '  [warn] conditioning slice has 0 ocean cells -- returning an '
                'all-NaN field instead of traversing the forest. Expect this '
                'only for a model whose regrids are entirely NaN.')
            nan = np.full((1, 1, H, W), np.nan, dtype=np.float32)
            return torch.from_numpy(nan).to(cond.device)
        if key != self._key:
            cells = np.flatnonzero(ocean.ravel())
            X = np.empty((cells.size, n_ch + self.loc.shape[-1]), np.float32)
            X[:, :n_ch] = cond_np.reshape(n_ch, -1)[:, cells].T
            X[:, n_ch:] = self.loc.reshape(-1, self.loc.shape[-1])[cells]
            qv = np.asarray(self.forest.predict(X, quantiles=self.quantiles),
                            dtype=np.float64)
            # a quantile function is non-decreasing; enforce it so the inverse
            # is well defined even where interpolation between leaves wobbles
            self._qvals = np.maximum.accumulate(qv, axis=1)
            self._cells, self._key = cells, key
            self.n_forest_calls += 1
        vals = draw_member(self._qvals, self.rng, self.draw, self.grid)
        out = np.zeros(H * W, dtype=np.float32)
        out[self._cells] = vals.astype(np.float32)
        return torch.from_numpy(out.reshape(1, 1, H, W)).to(cond.device)


def load_baseline_run(save_dir, device=None, draw=DEFAULT_DRAW, seed=0,
                      config=None):
    """A :class:`eval_common.RunHandle` around a fitted forest.

    Everything downstream -- normalization, land fill, the ocean mask, the
    ``materialize_window``/``select_case`` time handling, ``model_timemean``'s
    averaging order, ``ec.model_mean_abs`` -- is then the diffusion model's own
    code, unchanged, which is the whole design (``shims.py``, seam 2).
    """
    import joblib
    from figures import eval_common as ec

    save_dir = Path(save_dir)
    meta = load_meta(save_dir)
    forest = joblib.load(save_dir / ARTIFACT)
    if config is None:
        with open(save_dir / 'config.json') as fh:
            config = json.load(fh)
    device = torch.device('cpu') if device is None else device

    loc = location_features(meta['lat'], meta['lon'])
    rng = np.random.default_rng(seed)
    predict_fn = _SlicePredictor(
        forest, loc, meta['quantile_grid'], draw, rng,
        n_ch=len(meta['predictor_channels']))
    sampler = shims.QuantileSampler(predict_fn, device=device,
                                    img_size=(loc.shape[0], loc.shape[1]))
    scaling = dict(means=meta['means'], stds=meta['stds'],
                   land_fill=meta['land_fill'])
    logging.info(f'  loaded {KIND} {save_dir.name}: '
                 f'{len(meta["predictor_channels"])} predictor channels, '
                 f'{meta["n_rows"]:,} rows, draw={draw}, seed={seed}')
    return ec.RunHandle(ddpm=sampler, ema=shims._NoOpEMA(), scaling=scaling,
                        predictor_channels=list(meta['predictor_channels']),
                        config=config, save_dir=save_dir, device=device)
