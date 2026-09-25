"""The pointwise LINEAR rung and its climatology floor: fit, load, draw.

Extracted from general/baselines/pointwise_linear.py.

The rung *below* both CNNs, and the floor below that: they give a scale on
which to read the CNNs' leave-one-model-out error.

What they are
-------------
``qrf.py``'s docstring settles the classification that makes the gap visible:
the forest is ONE global model over pointwise rows -- ``(predictors at a cell,
sin/cos lat, sin/cos lon)`` -- *not* one forest per cell. So the four methods
span

    |           | pointwise            | full field         |
    | linear    | THIS FILE            | --                 |
    | nonlinear | qrf                  | unet-mse / diffusion |

and ``U-Net - QRF`` is what the receptive field buys while ``QRF - linear`` is
what nonlinearity buys.

``pointwise-linear``   per grid cell, OLS of the target on that cell's
                       predictor channels:
                       ``o2(cell) = a(cell) + sum_k b_k(cell) * pred_k(cell)``.
``climatology``        the same fit with ``b = 0``, i.e. the per-cell
                       training-pool mean -- the intercept-only NESTED
                       submodel, and what says whether the predictors matter at
                       all.

Both are literally read off ONE set of normal equations (see :func:`accumulate`),
which is why a single pass can write both artifacts: the intercept-only fit
uses the ``A[0, 0]`` / ``b[0]`` corner of the same accumulators. That is the
nested structure made concrete rather than asserted.

Why closed form, and why that matters for the comparison
--------------------------------------------------------
⚠️ **The closed form is the EXACT global optimum of the same masked MSE
``unet-mse`` is trained on.** ``shims.DeterministicRegressor.p_losses`` is
``(mse * mask).sum() / mask.sum()``, a mean over ocean cells only; for a model
whose parameters are independent across cells that objective is a sum of
per-cell squared residuals, so the per-cell minimiser *is* the joint minimiser.
No "perhaps it was under-trained" rebuttal is available. It also means this
rung has **no epochs and no selected hyperparameter**.

⚠️ **OLS with an intercept is fitted in NORMALIZED space, and that is exactly
equivalent** to fitting in physical units -- an affine reparametrisation of both
the features and the target maps one solution onto the other. So working in the
space seam 2 requires costs nothing here. **This stops being true the moment a
ridge penalty appears** (a penalty is not scale-invariant).

⚠️ Three things here are easy to get silently wrong
---------------------------------------------------
1. **The evaluation mask is NOT the training mask.**
   ``eval_common.predict_ensemble`` builds its ocean mask from the held-out
   model's own target while ``TracerDataset`` takes ONE mask from its first
   training sample, and per-model O2 masks differ. This is the only baseline
   whose parameters are indexed by cell, so it is the only one for which that
   matters: the CNNs are convolutional and the QRF is global. A cell that is ocean in the held-out model and land in training has
   a **singular** ``XtX`` (its normalized ``cond`` is the land-fill constant in
   every training sample) and a naive ``lstsq`` would return something and draw
   a plausible map. The same pass therefore also fits a **globally pooled**
   model over the QRF's own feature set, which is the fallback at those cells --
   and because a pooled model's location terms are constant *at a given cell*,
   it collapses into the very same per-cell ``(intercept, slopes)`` array. One
   artifact, defined everywhere on the grid, one forward pass.
2. **Contamination is worse than coverage, because it looks fine.** A cell that
   is ocean in some training members and land in others has already been
   NaN -> filled by ``TracerDataset``, so those samples would enter the
   regression as rows pinned at the land-fill value at a cell that appears fully
   covered. :func:`accumulate` therefore takes a per-sample validity mask and
   zeroes each invalid row's whole design vector, so it contributes exactly
   nothing to ``A``, ``b`` or the sample count.
   ⚠️ Validity comes from ``isfinite`` on the RAW target, never from detecting
   the fill value: the ``cache_dir=None`` in-RAM path stores float16, whose
   ~2e-3 spacing at these magnitudes puts **1.4% of genuine ocean rows** inside
   any tolerance wide enough to catch a real land cell (measured; see
   ``qrf.build_rows``' docstring).
3. **Normalized space, this run's own stats.** ``ec.predict_ensemble``
   normalizes ``cond`` with ``run.scaling`` and denormalizes the output with
   ``* stds[target] + means[target]``. A fit in physical units would be
   double-scaled by that path and still look plausible.

Layout on disk
--------------
``naming.save_dir_from_config`` with ``baseline='pointwise-linear'`` (or
``'climatology'``), so an artifact lands in
``<BASELINE_ROOT>/baseline-<kind>/<the same directory name as the diffusion run it is
compared against>`` and pairing them stays a lookup.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from baselines import naming, qrf, shims


#: The two kinds this module fits. ``climatology`` is the intercept-only nested
#: submodel of ``pointwise-linear`` and is read off the same accumulators.
KIND = 'pointwise-linear'
KIND_CLIM = 'climatology'
KINDS = (KIND, KIND_CLIM)

#: Artifact filename inside a run's save dir. One small netCDF -- ~1 MB against
#: the QRF's 11.8 GB -- so there is no reason to shard or compress it.
ARTIFACT = 'linear_coef.nc'
META = 'linear_meta.json'

#: Name of the intercept row in the coefficient array's ``param`` coordinate.
#: Leading underscore so it can never collide with a channel name (``<var>``
#: or ``<var>_dlev<i>``).
INTERCEPT = '_intercept'

#: A cell needs at least this many VALID training maps before its own
#: regression is preferred to the pooled fallback. Not a tuned hyperparameter:
#: the fleet has 3150-4500 training maps, so every genuinely wet cell clears
#: this by two orders of magnitude and the threshold only ever fires on the
#: mask-disagreement fringe between training and evaluation masks. Raising it
#: cannot change a well-covered cell's fit, because the fit is exact.
MIN_SAMPLES = 50

#: Reject a per-cell system whose scaled eigenvalue ratio is below this, even
#: where the sample count is ample -- e.g. a cell where a predictor is
#: constant across the training pool. Scale-free (``A`` is ``sum x x^T``, so its
#: scale is O(n) and the RATIO is not), so one number serves every config.
#: ⚠️ Deliberately a rank test and NOT a ridge jitter: a jitter would make every
#: cell "solvable" and quietly break the exact-optimum claim above.
MIN_EIG_RATIO = 1e-8

#: ``source`` codes stamped per cell in the artifact.
SRC_PERCELL, SRC_POOLED = 0, 1


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class PointwiseLinear(nn.Module):
    """``bias(1,H,W) + sum_k weight(k,H,W) * cond(k,H,W)``, per cell.

    Deliberately an ``nn.Module`` rather than a bare numpy predict: it drops
    into ``shims.DeterministicRegressor``, which means seam 2 -- the land
    handling, the ``target_land`` re-injection, the ``eval()`` call
    ``predict_ensemble`` makes unconditionally -- comes for free and is the
    diffusion model's own code path rather than a second implementation of it.

    ``climatology`` is this class with ``weight`` identically zero, which is
    what makes it the nested submodel in code and not just in the docstring.

    ``t`` is accepted and ignored: ``DeterministicRegressor`` passes an all-zero
    timestep because the backbone it usually wraps needs one. There is no time
    embedding here, so the argument is inert.
    """

    def __init__(self, weight, bias):
        super().__init__()
        w = torch.as_tensor(np.asarray(weight, dtype=np.float32))
        b = torch.as_tensor(np.asarray(bias, dtype=np.float32))
        if w.ndim != 3:
            raise ValueError(f'weight must be (C, H, W), got {tuple(w.shape)}')
        if b.shape != w.shape[1:]:
            raise ValueError(f'bias {tuple(b.shape)} does not match '
                             f'weight {tuple(w.shape)}')
        # buffers, not parameters: nothing here is trained by gradient descent,
        # and registering them as parameters would make an optimizer built over
        # .parameters() silently non-empty
        self.register_buffer('weight', w[None])            # (1, C, H, W)
        self.register_buffer('bias', b[None, None])        # (1, 1, H, W)

    def forward(self, cond, t=None):
        if cond.shape[1] != self.weight.shape[1]:
            raise ValueError(
                f'cond has {cond.shape[1]} channels but this fit has '
                f'{self.weight.shape[1]} -- a permuted or truncated cond would '
                f'still produce a plausible-looking map')
        return self.bias + (self.weight * cond).sum(1, keepdim=True)


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------
def target_validity(ds_train, target):
    """``(n_sample, H, W)`` bool: where each training map's RAW target is finite.

    The answer to trap 2. Computed from ``ds_train`` BEFORE ``TracerDataset``
    has filled NaN -> 0, member slice by member slice so the whole variable is
    never materialized at once, and returned as bool (292 MB for 4500 annual
    maps) rather than as the float array it is read from.

    ⚠️ Index-aligned with ``TracerDataset.__getitem__``: both build paths index
    ``ds.sample`` positionally (``_preproc[v][idx]`` and the
    ``_sample_member``/``_sample_local`` tables, which are built from
    ``_member_slices`` in order), so dataset item ``i`` is
    ``ds_train.isel(sample=i)``. ``TracerDataset`` also transposes to
    ``(sample, lat, lon)``, which the loader already returns.
    """
    da = ds_train[target].transpose('sample', 'lat', 'lon')
    n = da.sizes['sample']
    out = np.empty((n, da.sizes['lat'], da.sizes['lon']), dtype=bool)
    step = 200
    t0 = time.time()
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        out[lo:hi] = np.isfinite(da.isel(sample=slice(lo, hi)).values)
        if lo and lo % (step * 10) == 0:
            logging.info(f'  validity: {hi}/{n} maps ({time.time()-t0:.0f}s)')
    frac = float(out.mean())
    logging.info(f'validity: {n} maps, {frac:.1%} of cells finite on average '
                 f'({time.time()-t0:.0f}s)')
    return out


def accumulate(dataset, valid, loc, log_every=500):
    """The normal equations for BOTH fits, in one pass over the training maps.

    Returns a dict with

        A   (H*W, p, p)   per-cell ``sum_i x_i x_i^T``, ``x = [1, cond...]``
        b   (H*W, p)      per-cell ``sum_i x_i y_i``
        n   (H*W,)        per-cell count of valid maps
        Ag  (q, q)        POOLED ``sum z z^T`` over every valid ocean cell of
                          every map, ``z = [cond..., sin/cos lat, sin/cos lon, 1]``
        bg  (q,)          pooled ``sum z y``
        ng  scalar        pooled row count

    Everything is float64. ``A`` is 4.7 MB at ``p = 3``, so the accumulators
    are never the memory constraint -- the ``valid`` mask is.

    Two properties this relies on, both worth stating because they are what
    make one pass enough:

    * **Zeroing an invalid row's whole design vector** (including its intercept
      entry) makes it contribute exactly 0 to ``A``, ``b`` and ``n``. That is
      trap 2's fix, and it is exact rather than a tolerance.
    * **OLS on a SUBSET of features uses the corresponding SUBMATRIX** of the
      normal equations. So the intercept-only (climatology) fit is the
      ``A[:, 0, 0]`` / ``b[:, 0]`` corner of the per-cell system, and the
      location-only pooled fit is a submatrix of ``Ag``/``bg``. Nothing is
      accumulated twice.
    """
    n_maps = len(dataset)
    s0 = dataset[0]
    n_ch = int(s0['cond'].shape[0])
    H, W = s0['y'].shape[-2:]
    if valid.shape != (n_maps, H, W):
        raise ValueError(f'validity mask {valid.shape} does not match '
                         f'{n_maps} maps of {(H, W)} -- it would silently '
                         f'mis-select samples')
    p = n_ch + 1                       # + intercept
    q = n_ch + loc.shape[-1] + 1       # + location encoding + intercept

    A = np.zeros((p, p, H, W), dtype=np.float64)
    b = np.zeros((p, H, W), dtype=np.float64)
    n = np.zeros((H, W), dtype=np.int64)
    Ag = np.zeros((q, q), dtype=np.float64)
    bg = np.zeros(q, dtype=np.float64)
    ng = 0

    loc_t = np.ascontiguousarray(
        loc.transpose(2, 0, 1).astype(np.float64))      # (4, H, W)
    x = np.empty((p, H, W), dtype=np.float64)
    z = np.empty((q, H, W), dtype=np.float64)
    t0 = time.time()
    for i in range(n_maps):
        s = dataset[i]
        cond = s['cond'].numpy().astype(np.float64)     # (C, H, W) normalized
        y = s['y'].numpy()[0].astype(np.float64)        # (H, W)
        v = valid[i]
        w = v.astype(np.float64)

        # per-cell system: x = [1, cond...], every entry zeroed where invalid
        x[0] = w
        np.multiply(cond, w, out=x[1:])
        yw = y * w
        A += x[:, None] * x[None, :]
        b += x * yw
        n += v

        # pooled system: z = [cond..., loc..., 1], same zeroing
        z[:n_ch] = x[1:]
        np.multiply(loc_t, w, out=z[n_ch:n_ch + loc_t.shape[0]])
        z[-1] = w
        Z = z.reshape(q, -1)
        Ag += Z @ Z.T
        bg += Z @ yw.ravel()
        ng += int(v.sum())

        if log_every and (i + 1) % log_every == 0:
            logging.info(f'  accumulate: {i+1}/{n_maps} maps '
                         f'({time.time()-t0:.0f}s)')
    logging.info(f'accumulated {n_maps} maps x {H*W} cells in '
                 f'{time.time()-t0:.0f}s; {ng:,} pooled rows, per-cell counts '
                 f'{int(n.min())}-{int(n.max())}')
    return dict(A=A.transpose(2, 3, 0, 1).reshape(H * W, p, p).copy(),
                b=b.transpose(1, 2, 0).reshape(H * W, p).copy(),
                n=n.ravel().copy(), Ag=Ag, bg=bg, ng=ng,
                shape=(H, W), n_ch=n_ch, n_maps=n_maps)


def _solve_pooled(Ag, bg, cols):
    """OLS coefficients for a SUBSET of the pooled features.

    Uses the submatrix of the normal equations, which is exactly the OLS system
    for those features -- so the location-only (climatology) pooled fit and the
    full pooled fit come from one accumulator.
    """
    cols = list(cols)
    Asub = Ag[np.ix_(cols, cols)]
    try:
        return np.linalg.solve(Asub, bg[cols])
    except np.linalg.LinAlgError:
        # a pooled system this rank-deficient means the training pool is
        # degenerate, not that a fallback is needed; lstsq keeps the artifact
        # writable and the meta records what happened
        logging.warning('pooled system is singular; falling back to lstsq')
        return np.linalg.lstsq(Asub, bg[cols], rcond=None)[0]


def coverage(acc, min_samples=MIN_SAMPLES, min_eig_ratio=MIN_EIG_RATIO):
    """``(good (N,) bool, eig_ratio (N,))`` -- where a per-cell fit is well-posed.

    ⚠️ **Computed ONCE from the full system and shared by both kinds**, rather
    than per kind. The climatology fit is a 1x1 system and would be well-posed
    at strictly more cells, but then the two artifacts would live on different
    domains and ``pointwise-linear`` vs ``climatology`` -- the comparison that
    says whether the predictors carry information -- would acquire a domain
    confound on top of the effect it is meant to isolate. Sharing the criterion
    keeps ``climatology`` the exact ``b = 0`` submodel *on the same cells*,
    which is the whole claim. The difference is confined to the mask-
    disagreement fringe either way.

    ``eig_ratio`` is the smallest/largest eigenvalue of the DIAGONALLY SCALED
    ``XtX``, so it is scale-free -- ``A`` is ``sum x x^T`` and its magnitude is
    O(n), while the ratio is not -- and one threshold serves every config.
    """
    A, n = acc['A'], acc['n']
    d = np.einsum('cii->ci', A).copy()              # (N, p) diagonal
    scale = np.sqrt(np.where(d > 0, d, 1.0))
    As = A / (scale[:, :, None] * scale[:, None, :])
    ev = np.linalg.eigvalsh(As)                     # ascending, symmetric PSD
    with np.errstate(divide='ignore', invalid='ignore'):
        eig_ratio = np.where(ev[:, -1] > 0, ev[:, 0] / ev[:, -1], 0.0)
    good = ((n >= min_samples) & (eig_ratio > min_eig_ratio)
            & (d > 0).all(1))
    return good, eig_ratio


def solve(acc, loc, good, eig_ratio, kind=KIND):
    """``(coef (p, H, W), source (H, W), diag)`` for one kind.

    ``coef[0]`` is the intercept and ``coef[1:]`` the per-channel slopes --
    identically zero for ``climatology``, which is what makes it the nested
    submodel rather than a second estimator.

    Cells in ``good`` (see :func:`coverage`) get their own fit. Everywhere else
    the POOLED fit is collapsed into the same per-cell form: its location terms
    are constant at a given cell, so ``a + sum_j c_j loc_j(cell)`` is that
    cell's intercept and the pooled slopes are its slopes. The field is
    therefore defined on every cell of the grid, including cells the training
    mask never saw wet -- which is the whole point (trap 1).
    """
    if kind not in KINDS:
        raise ValueError(f'kind must be one of {KINDS}, got {kind!r}')
    H, W = acc['shape']
    n_ch = acc['n_ch']
    A, b, n = acc['A'], acc['b'], acc['n']
    N = H * W
    p = n_ch + 1
    intercept_only = (kind == KIND_CLIM)
    good = np.asarray(good, dtype=bool).copy()

    # -- the pooled fallback, collapsed to per-cell form -------------------
    # pooled feature order is [cond..., sin_lat, cos_lat, sin_lon, cos_lon, 1]
    n_loc = loc.shape[-1]
    i_loc = list(range(n_ch, n_ch + n_loc))
    i_int = acc['Ag'].shape[0] - 1
    if intercept_only:
        cg = _solve_pooled(acc['Ag'], acc['bg'], i_loc + [i_int])
        slopes_pooled = np.zeros(n_ch)
        loc_coef, a_pooled = cg[:n_loc], cg[-1]
    else:
        cg = _solve_pooled(acc['Ag'], acc['bg'],
                           list(range(n_ch)) + i_loc + [i_int])
        slopes_pooled = cg[:n_ch]
        loc_coef, a_pooled = cg[n_ch:n_ch + n_loc], cg[-1]
    loc_flat = loc.reshape(-1, n_loc).astype(np.float64)
    fallback = np.empty((N, p), dtype=np.float64)
    fallback[:, 0] = a_pooled + loc_flat @ loc_coef
    fallback[:, 1:] = slopes_pooled[None, :]

    # -- solve the good cells ---------------------------------------------
    coef = fallback.copy()
    idx = np.flatnonzero(good)
    if idx.size:
        if intercept_only:
            coef[idx, 0] = b[idx, 0] / n[idx]
            coef[idx, 1:] = 0.0
        else:
            # ⚠️ ``b[idx][..., None]`` is required, not cosmetic. numpy >= 2.0
            # reads a 2-D second argument as a single (m, n) matrix rather than
            # a stack of m-vectors, so ``solve(A[idx], b[idx])`` raises a core-
            # dimension mismatch (size K against p) the moment K != p.
            coef[idx] = np.linalg.solve(A[idx], b[idx][..., None])[..., 0]
    if intercept_only:
        coef[:, 1:] = 0.0                # exactly zero everywhere, by design
    bad = ~np.isfinite(coef).all(1)
    if bad.any():                        # cannot happen via solve(); belt
        logging.warning(f'{int(bad.sum())} cells solved to non-finite '
                        f'coefficients; using the pooled fit there')
        coef[bad] = fallback[bad]
        good[bad] = False

    source = np.where(good, SRC_PERCELL, SRC_POOLED).astype(np.int8)
    eig_ratio = np.asarray(eig_ratio)
    diag = dict(
        n_percell=int(good.sum()), n_pooled=int((~good).sum()),
        n_wet=int((n > 0).sum()), n_thin=int(((n > 0) & ~good).sum()),
        eig_ratio_p01=(float(np.percentile(eig_ratio[n > 0], 1))
                       if (n > 0).any() else float('nan')),
        pooled_coef=[float(c) for c in cg],
    )
    logging.info(
        f'solve[{kind}]: {diag["n_percell"]} cells fitted per-cell, '
        f'{diag["n_pooled"]} from the pooled fallback (of which '
        f'{diag["n_thin"]} are wet but thin/degenerate); '
        f'{diag["n_wet"]} cells wet in training')
    return (coef.reshape(H, W, p).transpose(2, 0, 1).copy(), source.reshape(H, W),
            diag)


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------
def _param_names(channels):
    return [INTERCEPT] + list(channels)


def save_artifact(save_dir, kind, coef, source, acc, diag, dataset, config,
                  lat, lon):
    """Write ``linear_coef.nc`` + ``linear_meta.json`` for one kind.

    The netCDF carries what the prediction needs (``coef``, and ``means`` /
    ``stds`` / ``land_fill`` as attributes, because ``ec.predict_ensemble``
    reads exactly those through ``run.scaling``) plus the two diagnostic
    fields, ``n`` and ``source``, that say where the per-cell fit was used.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    channels = list(config['predictor_channels'])
    target = config['target']
    ds = xr.Dataset(
        {'coef': (('param', 'lat', 'lon'), coef.astype(np.float64)),
         'n': (('lat', 'lon'), acc['n'].reshape(coef.shape[1:]).astype(np.int32)),
         'source': (('lat', 'lon'), source)},
        coords={'param': _param_names(channels), 'lat': lat, 'lon': lon})
    ds['source'].attrs.update(
        description=f'{SRC_PERCELL} = own per-cell OLS, '
                    f'{SRC_POOLED} = globally pooled fallback')
    ds.attrs.update(
        baseline=kind, target=target,
        predictor_channels=' '.join(channels),
        land_fill=dataset.land_fill, n_samples=int(acc['n_maps']),
        means=json.dumps({k: float(v) for k, v in dataset.means.items()}),
        stds=json.dumps({k: float(v) for k, v in dataset.stds.items()}),
        n_percell=diag['n_percell'], n_pooled=diag['n_pooled'],
        n_wet=diag['n_wet'], n_thin=diag['n_thin'],
        min_samples=diag['min_samples'], min_eig_ratio=diag['min_eig_ratio'],
        pooled_rows=int(acc['ng']),
        fit='closed-form OLS (exact optimum of the masked MSE)')
    path = save_dir / ARTIFACT
    ds.to_netcdf(path)
    with open(save_dir / META, 'w') as fh:
        json.dump(dict(baseline=kind, diag=diag,
                       predictor_channels=channels, target=target,
                       land_fill=dataset.land_fill,
                       n_samples=int(acc['n_maps'])), fh, indent=4)
    logging.info(f'  wrote {path} ({path.stat().st_size/1e6:.2f} MB)')
    return path


def artifact_exists(save_dir):
    return (Path(save_dir) / ARTIFACT).is_file()


def load_meta(save_dir):
    with open(Path(save_dir) / META) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the entrypoint train_baseline.run calls
# ---------------------------------------------------------------------------
def fit(config, dataset, ds_train, lat, lon, also=(), min_samples=MIN_SAMPLES,
        min_eig_ratio=MIN_EIG_RATIO, log_every=500):
    """Accumulate once, then write an artifact per requested kind.

    ``also`` names the *other* kinds to write from the SAME accumulators --
    the fits pass ``climatology`` alongside ``pointwise-linear``,
    so both come from byte-identical normal equations rather than from two
    passes that could differ in anything at all. Each artifact goes to its own
    kind's save dir, so provenance stays one-artifact-per-kind.
    """
    kind = config['baseline']
    kinds = [kind] + [k for k in also if k != kind]
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f'{k!r} is not fitted by this module; have {KINDS}')

    loc = qrf.location_features(lat, lon)          # the QRF's OWN encoding
    valid = target_validity(ds_train, config['target'])
    acc = accumulate(dataset, valid, loc, log_every=log_every)
    del valid
    good, eig_ratio = coverage(acc, min_samples=min_samples,
                               min_eig_ratio=min_eig_ratio)

    out = {}
    for k in kinds:
        coef, source, diag = solve(acc, loc, good, eig_ratio, kind=k)
        diag.update(min_samples=int(min_samples),
                    min_eig_ratio=float(min_eig_ratio))
        cfg_k = {**config, 'baseline': k}
        sd = naming.save_dir_from_config(cfg_k)
        save_artifact(sd, k, coef, source, acc, diag, dataset, cfg_k, lat, lon)
        with open(Path(sd) / 'config.json', 'w') as fh:
            json.dump({kk: v for kk, v in cfg_k.items()
                       if not isinstance(v, np.ndarray)}, fh, indent=4)
        out[k] = dict(save_dir=sd, diag=diag)
    return out


def train(config, dataset, device=None, lat=None, lon=None, ds_train=None,
          resume_from_existing=True, also=(), **kw):
    """``train_baseline.run``'s uniform entrypoint. No epochs, no Trainer.

    ``resume_from_existing`` short-circuits on an existing artifact, matching
    ``qrf.train``'s guard -- but note there is nothing partial to resume here:
    the fit is one pass and one solve, so it either exists or is redone.
    """
    if ds_train is None:
        raise ValueError('pointwise_linear.fit needs ds_train: the per-sample '
                         'validity mask must be read from the RAW target, '
                         'before TracerDataset fills NaN -> 0')
    kind = config['baseline']
    kinds = [kind] + [k for k in also if k != kind]
    if resume_from_existing and all(
            artifact_exists(naming.save_dir_from_config({**config,
                                                         'baseline': k}))
            for k in kinds):
        logging.info(f'artifacts already fitted for {kinds}')
        return {k: dict(save_dir=naming.save_dir_from_config(
            {**config, 'baseline': k}), diag=None) for k in kinds}
    return fit(config, dataset, ds_train, lat, lon, also=also, **kw)


# ---------------------------------------------------------------------------
# load for evaluation
# ---------------------------------------------------------------------------
def load_baseline_run(save_dir, device=None, config=None):
    """A :class:`eval_common.RunHandle` around a fitted pointwise linear model.

    The counterpart of ``unet_mse.load_run`` / ``qrf.load_baseline_run``.
    Everything downstream of this -- ``predict_ensemble``, ``model_timemean``,
    ``ec.model_mean_abs`` -- is the diffusion model's own code, unchanged.

    ⚠️ Channel order is asserted against the artifact's own ``param``
    coordinate, as ``ec.load_run`` and ``unet_mse.load_run`` do. A silently
    permuted ``cond`` would still run and still produce a plausible-looking
    map; here it would also multiply the right slope by the wrong field.

    ⚠️ ``ema`` is ``shims._NoOpEMA``. ``predict_ensemble`` calls
    ``apply_shadow``/``restore`` around every draw whenever
    ``sample_cfg.apply_ema`` is set -- which the diffusion caches were produced
    with -- and a closed-form fit has no shadow weights. The no-op keeps that
    call site identical rather than growing a branch in ``eval_common``.
    """
    from figures import eval_common as ec

    save_dir = Path(save_dir)
    if config is None:
        with open(save_dir / 'config.json') as fh:
            config = json.load(fh)
    path = save_dir / ARTIFACT
    if not path.is_file():
        raise FileNotFoundError(
            f'no {ARTIFACT} in {save_dir} -- fit it with '
            f'python -m baselines.train_baseline --baseline pointwise-linear '
            f'--also climatology')
    ds = xr.open_dataset(path)

    channels = list(config['predictor_channels'])
    got = [str(p) for p in ds['param'].values]
    expected = _param_names(channels)
    assert got == expected, (
        f'channel mismatch for {save_dir.name}:\n  artifact={got}\n'
        f'  config={expected}')

    coef = ds['coef'].transpose('param', 'lat', 'lon').values
    model = PointwiseLinear(weight=coef[1:], bias=coef[0])
    dev = device or torch.device('cpu')
    model = model.to(dev)
    model.eval()

    means = json.loads(ds.attrs['means'])
    stds = json.loads(ds.attrs['stds'])
    scaling = dict(means=means, stds=stds, land_fill=str(ds.attrs['land_fill']))
    kind = str(ds.attrs['baseline'])
    n_pooled = int(ds.attrs.get('n_pooled', -1))
    ds.close()

    regressor = shims.DeterministicRegressor(model, device=dev,
                                             img_size=coef.shape[1:])
    logging.info(f'  loaded {kind} {save_dir.name}: {len(channels)} predictor '
                 f'channels, land_fill={scaling["land_fill"]}, '
                 f'{n_pooled} pooled-fallback cells')
    return ec.RunHandle(ddpm=regressor, ema=shims._NoOpEMA(), scaling=scaling,
                        predictor_channels=channels, config=config,
                        save_dir=save_dir, device=dev)
