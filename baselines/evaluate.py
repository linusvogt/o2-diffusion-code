"""Run a baseline (or the diffusion model) through the LOMO / in-sample protocol.

Extracted from general/baselines/evaluate.py; the few cell-naming helpers it
needs from general/eval/error_maps.py are copied in below.

    python -m baselines.evaluate --baseline unet-mse --list          # no GPU
    python -m baselines.evaluate --baseline unet-mse --model CanESM5 # one part
    python -m baselines.evaluate --baseline unet-mse                 # all parts
    python -m baselines.evaluate --baseline unet-mse --aggregate     # the cache
    # the same, on the IN-SAMPLE split (one run per kind, 16-model fleet)
    python -m baselines.evaluate --baseline unet-mse --split insample --list

``predict_ensemble`` / ``model_timemean`` / ``model_mean_abs`` are the diffusion
model's own code, called unchanged -- that identity is the deliverable, because
a second implementation of the protocol would be a second place for it to
drift and the comparison would stop being a comparison. What is re-written here
is only ``error_maps.compute_oos``'s ~30-line held-out-model loop, because
``ec.load_run`` rebuilds a UNet **+ DDPM** and none of the baselines is one.

Everything that defines *what* is computed is READ FROM THE DIFFUSION CACHE at
runtime
-------------------------------------------------------------------------------
The fleet, the epoch, the ensemble size, the sampler, the steps and the year
window all come out of the diffusion ``error_maps`` cache's own attributes
(``T-S_ann_int_oos.nc`` / ``T-S_ann_int_insample.nc``), never from a default
and never from a list in this file:

* ``error_maps``' ``cache_matches`` does **not** compare the model fleet, so a
  hardcoded fleet could drift from the very figure this exists to be compared
  against and nothing would catch it.
* Three different epoch numbers are in play (``error_maps.DEFAULT_EPOCH`` = 500
  as a module constant, 1000 on disk for the diffusion LOMO runs, and 500 in
  the cache). Only the **cache's** stamp is the comparison point, because it is
  what the published figure was actually sampled at.

⚠️ **A missing baseline run is FATAL here, not a skip.** ``compute_oos`` logs
``[skip]`` and carries on, which is right for a sweep still filling in and
exactly wrong here: silently dropping a model would report a different fleet
under the same figure. Every fatal path in this module names the fix.

Two-stage on purpose (parts, then aggregate)
--------------------------------------------
A QRF forest traversal is ~2 minutes per conditioning slice, so 14 held-out
models x 20 years in one process is a ~9 h job. Each held-out model's time-mean
``gen``/``truth`` pair is therefore computed and cached independently (one job
per model), and ``--aggregate`` forms ``ec.model_mean_abs`` over the parts. The
U-Net is fast enough to do all 14 in one process, but it uses the same two
stages so there is one code path.

⚠️ ``--aggregate`` re-checks every part against the pins **and against each
other** before combining: it is not enough for the knobs to match, the *fleet*
has to be the one the diffusion cache names.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from figures import eval_common as ec
from baselines import naming, pointwise_linear, qrf, unet_mse


# ---------------------------------------------------------------------------
# copied verbatim from general/eval/error_maps.py (cell naming + the directory
# of its caches, which pin the comparison)
# ---------------------------------------------------------------------------
#: ``error_maps``' cache directory. Holds the diffusion caches
#: ``T-S_ann_int_oos.nc`` / ``T-S_ann_int_insample.nc`` that define the fleet,
#: epoch and sampling of the comparison.
ERROR_MAPS_CACHE_DIR = ec.FIG_DIR / 'error_maps' / 'cache'

BASE_CONFIG = ec.make_config(
    target='o2', predictors=['thetao', 'so'], experiments=['1pctCO2'],
    resolution='annual', field_type='depthint', train_test_split='insample')

# Aggregation over models, stamped into every cache (and checked by
# cache_matches, so a cache from the previous metric is recomputed rather than
# replotted). 'model-mean-abs' = per-model |error| first, then average the
# magnitudes over models; the old 'model-mean-then-error' averaged the fields
# first, which let opposite-signed per-model biases cancel.
METRIC = 'model-mean-abs'

_SHORT_PRED = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': 'taux',
               'tauvo': 'tauy', 'mld': 'MLD'}
_PRED_TEX = {'thetao': 'T', 'so': 'S', 'dissic': 'DIC', 'tauuo': r'$\tau_x$',
             'tauvo': r'$\tau_y$', 'mld': 'MLD'}


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
    return ec.make_config(BASE_CONFIG, predictors=cell['predictors'],
                          resolution=cell['resolution'],
                          field_type=cell['field_type'])


# ---------------------------------------------------------------------------
# the cell
# ---------------------------------------------------------------------------


#: The two splits -- "is the machinery better *under model transfer*, or
#: better full stop?", which are different sentences.
#: ⚠️ They are over DIFFERENT FLEETS: the LOMO cache stamps **14** models and
#: the in-sample one **16**. Their panels may sit side by side and must NEVER
#: be differenced; ``error_maps``' ``cache_matches`` does not compare the
#: fleet, so nothing downstream would catch it.
SPLITS = ('oos', 'insample')

#: The default everywhere in this module (the in-sample split takes a suffix
#: on its part paths; see ``part_path``).
DEFAULT_SPLIT = 'oos'


def cell_for(split=DEFAULT_SPLIT):
    """The evaluation cell, in ``error_maps``' own vocabulary.

    The baselines cover the headline configuration only -- T+S, annual,
    depth-integrated, 1pctCO2 -- on either split. Kept in that vocabulary so a
    cell and its diffusion cache are one lookup apart rather than a translation
    table.
    """
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}, got {split!r}')
    return dict(predictors=['thetao', 'so'], resolution='annual',
                field_type='depthint', split=split)


#: The LOMO cell. Module-level because ``select_epoch`` imports it by name.
CELL = cell_for()


def pin_cache_for(split=DEFAULT_SPLIT):
    """The diffusion cache that defines the comparison on one split.

    Derived from ``cell_slug`` rather than spelled out, so the two files
    (``T-S_ann_int_oos.nc`` / ``T-S_ann_int_insample.nc``) cannot drift from
    the cells they belong to.
    """
    return ERROR_MAPS_CACHE_DIR / f'{cell_slug(cell_for(split))}.nc'


PIN_CACHE = pin_cache_for()

#: The diffusion model itself, run through THIS script's per-model path.
#: Not a baseline -- it is the thing the baselines are compared against -- but
#: ``error_maps`` throws its per-model fields away (``model_mean_abs`` then
#: ``save_cache`` writes only the aggregate). The comparison is PAIRED -- same
#: held-out models, same truth fields -- so the diffusion's parts are kept.
#: Its aggregate must reproduce ``T-S_ann_int_oos.nc``'s own ``mape_pct``,
#: which is the end-to-end check that this path is equivalent to
#: ``error_maps``'.
DIFFUSION = 'diffusion'

def is_diffusion(kind):
    """True for the diffusion model."""
    return kind == DIFFUSION


#: Kinds whose prediction is a single field, so n draws are n identical copies.
#: ⚠️ Their ``n_samples`` stays at the pinned value in every stamp so the
#: provenance is comparable attr-for-attr with the diffusion cache;
#: ``n_samples_effective`` is what records that only ONE was distinct. Their
#: SPREAD is therefore undefined, not zero.
DETERMINISTIC_KINDS = (unet_mse.KIND,) + tuple(pointwise_linear.KINDS)


def is_deterministic(kind):
    return kind in DETERMINISTIC_KINDS



#: Attributes a part must agree with the pins on before it can be aggregated.
_PART_PROVENANCE = ('baseline', 'epoch', 'epoch_arg', 'n_samples', 'steps',
                    'sampler', 'years_label', 'draw')

#: ``--epoch selected`` reads this per run, written by ``select_epoch.py``.
SELECTED = 'selected'


# ---------------------------------------------------------------------------
# the pins
# ---------------------------------------------------------------------------
def parse_years_label(label):
    """``error_maps.years_label`` in reverse: 'yrs 81-100' -> ((81, 100), 1, None).

    Also accepts the two other shapes that function can emit -- 'yrs 81-100 /5'
    (a stride) and 'yrs 81,82' (an explicit list). ⚠️ Anything else RAISES
    rather than falling back to a default window: silently evaluating a
    different time-mean than the diffusion figure is precisely the failure this
    whole module is arranged to prevent.
    """
    txt = str(label).strip()
    if not txt.startswith('yrs '):
        raise ValueError(f'cannot parse years_label {label!r} -- expected the '
                         f'"yrs ..." form written by error_maps.years_label')
    body = txt[4:].strip()
    stride = 1
    if '/' in body:
        body, _, s = body.partition('/')
        body, stride = body.strip(), int(s)
    if ',' in body:
        return None, stride, [int(y) for y in body.split(',')]
    if '-' in body:
        lo, _, hi = body.partition('-')
        return (int(lo), int(hi)), stride, None
    raise ValueError(f'cannot parse years_label {label!r}')


def read_pins(path=None, split=DEFAULT_SPLIT):
    """Everything the comparison must match, read off the diffusion cache.

    ⚠️ ``epochs`` is a per-model list in the cache (a single entry on the
    in-sample split, which is one run). They are required to be IDENTICAL
    here: a fleet sampled at mixed epochs has no single comparison point, and
    quietly taking the max (or the first) would put the baseline against a
    checkpoint the figure never used.

    ⚠️ The cache's OWN ``split`` attribute is checked against ``split``. Without
    that, ``--pin-cache`` pointed at the wrong file would pin a 16-model
    in-sample fleet under LOMO save-dir lookups (or the reverse) and every
    later check would agree with itself.
    """
    path = Path(path) if path is not None else pin_cache_for(split)
    if not path.is_file():
        raise FileNotFoundError(
            f'no diffusion cache at {path} -- the evaluation takes its fleet, '
            f'epoch and sampling from that file, so there is nothing to '
            f'compare against. It is written by '
            f'`python -m figures.fig06_lomo_error compute --split {split}`')
    ds = xr.open_dataset(path)
    a = ds.attrs
    if str(a.get('split')) != split:
        raise ValueError(
            f'{path.name} stamps split={a.get("split")!r} but this run asked '
            f'for {split!r}. The split decides the save dirs, the fleet and '
            f'the part paths, so pinning across it would compare two different '
            f'experiments under one name.')
    models = str(a['models']).split()
    epochs = sorted({int(e) for e in str(a['epochs']).split()})
    if len(epochs) != 1:
        raise ValueError(
            f'{path.name} stamps mixed epochs {epochs} across its fleet, so it '
            f'has no single comparison point. Re-run that cell with an '
            f'explicit --epoch before comparing a baseline to it.')
    if len(models) != int(a['n_models']):
        raise ValueError(f'{path.name}: {len(models)} names but n_models='
                         f'{a["n_models"]}')
    if str(a.get('metric')) != METRIC:
        raise ValueError(
            f'{path.name} stamps metric={a.get("metric")!r}, not {METRIC!r} -- '
            f'that cache is the older model-mean-THEN-error statistic and is '
            f'not comparable with what this script computes. Re-run the cell.')
    window, stride, years = parse_years_label(a['years_label'])
    pins = dict(
        path=str(path), models=models, epoch=epochs[0],
        n_samples=int(a['n_samples']), steps=int(a['steps']),
        sampler=str(a['sampler']), years_label=str(a['years_label']),
        year_window=window, years_stride=stride, years=years,
        metric=str(a['metric']), diffusion_mape_pct=float(a['mape_pct']),
        split=split)
    ds.close()
    return pins


def sample_cfg_from(pins, n_samples=None):
    """``SampleCfg`` matching the diffusion run.

    ⚠️ ``eta`` is NOT stamped in the cache (it is not in ``error_maps``'
    ``_PROVENANCE`` either), so it comes from ``SampleCfg``'s default -- which
    is what ``error_maps.main`` used, since it constructs ``SampleCfg`` without
    passing one. Both baselines ignore it; it is set for the record only.
    """
    return ec.SampleCfg(n_samples=pins['n_samples'] if n_samples is None
                        else int(n_samples),
                        sampler=pins['sampler'], steps=pins['steps'],
                        apply_ema=True)


# ---------------------------------------------------------------------------
# locating and loading a baseline run
# ---------------------------------------------------------------------------
def all_kinds():
    return list(naming.KINDS) + [DIFFUSION]


def baseline_save_dir(kind, model_name, split=DEFAULT_SPLIT):
    """The run that produces ``model_name``'s prediction on ``split``.

    Built from ``ec.run_save_dir`` and then re-rooted, rather than re-spelled:
    ``naming.py``'s whole design is that a baseline run and the diffusion run
    it is compared against share a directory *name*, so pairing them is a
    lookup. Deriving the name from the same function the diffusion side uses is
    what keeps that true if the encoding ever changes.

    ⚠️ **On the in-sample split ``model_name`` does NOT select a run.** There
    is exactly ONE in-sample run per kind and every model in the fleet is
    predicted from it -- ``error_maps.compute_insample`` loads one checkpoint
    and loops the fleet, and this mirrors that. The argument is still accepted
    so callers can stay split-agnostic, and it is what names the *part*.
    """
    # cell_config's base is already train_test_split='insample', so the
    # in-sample dir is the un-overridden one and LOMO is the override -- the
    # same asymmetry error_maps.cell_config's own docstring records.
    over = (dict() if split == 'insample'
            else dict(train_test_split='model', models_test=[model_name]))
    diffusion_dir = ec.run_save_dir(cell_config(cell_for(split)), **over)
    if is_diffusion(kind):
        return diffusion_dir          # the real run, under the diffusion root
    return naming.BASELINE_ROOT / f'baseline-{kind}' / diffusion_dir.name


def epoch_for(kind, save_dir, pins, epoch_arg):
    """The epoch to evaluate one run at: the pinned one, or its selected one.

    There are TWO readouts from one training run --
    ``epoch_arg='pinned'`` reproduces the diffusion cache's own epoch (the
    symmetric comparison, both models at what the published figure used) and
    ``'selected'`` uses each run's validation winner from ``select_epoch.py``
    (the baseline's best, so it cannot be accused of being understated).

    ⚠️ The selected epoch may EXCEED the diffusion model's, because the
    baselines were trained to 1000 while the cache sampled 500. That asymmetry
    is deliberate and can only ever favour the baseline -- but it means a
    ``'selected'`` number is NOT the like-for-like comparison, and the cache
    stamps ``epoch_arg`` so the two can never be confused for each other.
    """
    if kind != unet_mse.KIND or epoch_arg != SELECTED:
        return pins['epoch']          # the diffusion is always pinned
    # ⚠️ imported HERE, not at module scope. select_epoch imports this module
    # (for the pins and the save-dir lookup), and one of its defaults is
    # evaluated at def-time (`pin_cache=evaluate.PIN_CACHE`), so a module-level
    # import back would be a genuine circular-import failure rather than the
    # benign kind -- whichever module happened to be imported first would win.
    from baselines import select_epoch
    path = Path(save_dir) / select_epoch.OUT_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f'no {select_epoch.OUT_NAME} in {save_dir} -- --epoch '
            f'selected needs select_epoch to have run for this model. Produce it '
            f'with: python -m baselines.select_epoch --model <NAME> '
            f'(or --insample)')
    return int(json.loads(path.read_text())['selected_epoch'])


def load_baseline(kind, save_dir, pins, device, draw=qrf.DEFAULT_DRAW, seed=0,
                  epoch_arg='pinned'):
    """A ``RunHandle`` for either baseline, at the requested epoch."""
    if kind == unet_mse.KIND:
        return unet_mse.load_run(
            save_dir, device, epoch=epoch_for(kind, save_dir, pins, epoch_arg),
            _compile=False)
    if kind == qrf.KIND:
        return qrf.load_baseline_run(save_dir, device=device, draw=draw,
                                     seed=seed)
    if kind in pointwise_linear.KINDS:
        # closed form: no epoch to pick, no EMA shadow, no draw -- the handle
        # carries shims._NoOpEMA so predict_ensemble's apply_shadow/restore
        # call site stays identical
        return pointwise_linear.load_baseline_run(save_dir, device=device)
    if is_diffusion(kind):
        # ec.load_run is exactly what error_maps.compute_oos uses, so the
        # diffusion goes through its own loader and only the per-model
        # bookkeeping is this script's
        return ec.load_run(save_dir, device, epoch=pins['epoch'],
                           _compile=False)
    raise ValueError(f'unknown kind {kind!r}; have {all_kinds()}')


def check_run(kind, model_name, pins, epoch_arg='pinned', split=DEFAULT_SPLIT):
    """Fatal-if-unusable check for one fleet model. Returns its save dir.

    ⚠️ Deliberately raises where ``error_maps.compute_oos`` logs ``[skip]``.
    That skip is right for a sweep still filling in; here it would report a
    figure over a smaller fleet than the diffusion cache names.
    """
    sd = baseline_save_dir(kind, model_name, split)
    if not sd.is_dir():
        m = 'insample' if split == 'insample' else model_name
        train = f'python -m baselines.train_baseline -m {m} --baseline'
        how = {unet_mse.KIND: f'{train} {unet_mse.KIND} --epochs 1000 '
                              f'--save-every 25',
               qrf.KIND: f'{train} {qrf.KIND} --select-min-samples-leaf ...',
               pointwise_linear.KIND: f'{train} {pointwise_linear.KIND} '
                                      f'--also climatology',
               pointwise_linear.KIND_CLIM: f'{train} {pointwise_linear.KIND} '
                                           f'--also climatology',
               DIFFUSION: f'python -m diffusion.train -m {m}'}[kind]
        held = (f'the in-sample run (needed for {model_name})'
                if split == 'insample' else f'held-out {model_name}')
        raise FileNotFoundError(
            f'no {kind} run for {held} at {sd}. The fleet is '
            f'pinned to {Path(pins["path"]).name}, so a missing run is a '
            f'mismatch to fix, not a model to drop -- launch it with {how}')
    if is_diffusion(kind):
        if not ec.run_exists(sd, pins['epoch']):
            raise FileNotFoundError(
                f'{sd.name} has no ckpt_epoch{pins["epoch"]} -- but '
                f'{Path(pins["path"]).name} says it was sampled at that epoch, '
                f'so the run and the cache disagree.')
    elif kind == unet_mse.KIND:
        have = unet_mse.checkpoint_epochs(sd)
        want = epoch_for(kind, sd, pins, epoch_arg)
        if want not in have:
            raise FileNotFoundError(
                f'{sd.name} has no ckpt_epoch{want} (has {have}). Under '
                f'--epoch pinned the comparison epoch is {pins["epoch"]}, '
                f'because that is what {Path(pins["path"]).name} stamps; '
                f'evaluating at another one would compare against a '
                f'checkpoint the figure never used.')
    elif kind == qrf.KIND and not qrf.artifact_exists(sd):
        raise FileNotFoundError(
            f'{sd} holds no {qrf.ARTIFACT} -- the fit did not finish, or '
            f'finished without writing: check the job OUTPUT.')
    elif (kind in pointwise_linear.KINDS
          and not pointwise_linear.artifact_exists(sd)):
        raise FileNotFoundError(
            f'{sd} holds no {pointwise_linear.ARTIFACT} -- the fit did not '
            f'finish, or finished without writing: check the job OUTPUT.')
    return sd


# ---------------------------------------------------------------------------
# stage 1 -- one held-out model
# ---------------------------------------------------------------------------
def kind_cache_dir(kind):
    """``naming.cache_dir`` for a baseline; a sibling tree for the diffusion.

    The diffusion's own checkpoints live under the diffusion root, which is
    globbed by directory name -- so its evaluation parts go under the BASELINE
    root instead, beside the baselines they exist to be compared with, rather
    than seeding a cache directory inside the checkpoint tree.
    """
    if is_diffusion(kind):
        return naming.BASELINE_ROOT / f'_{kind}' / '_eval_cache'
    return naming.cache_dir(kind)


def part_path(kind, model_name, epoch_arg='pinned', split=DEFAULT_SPLIT):
    """⚠️ Parts are per-READOUT and per-SPLIT too.

    A selected-epoch part is a different field than a pinned-epoch one and must
    not overwrite it -- and the in-sample fleet shares MODEL NAMES with the
    LOMO one, so without a split component the two would collide file-for-file
    in one directory. Only a non-default split takes a suffix.
    """
    sub = 'parts' if epoch_arg == 'pinned' else f'parts_{epoch_arg}'
    if split != DEFAULT_SPLIT:
        sub = f'{sub}_{split}'
    return kind_cache_dir(kind) / sub / f'{model_name}.nc'


def compute_part(kind, model_name, pins, device, draw=qrf.DEFAULT_DRAW, seed=0,
                 n_samples=None, epoch_arg='pinned', split=DEFAULT_SPLIT):
    """Time-mean generated and true fields for ONE fleet model.

    The body mirrors ``error_maps.compute_oos``'s per-model block exactly --
    ``get_models`` -> ``build_per_model_ds`` -> ``model_timemean`` -- with the
    two differences that define this module: the run is a baseline, and every
    ``[skip]``
    is a raise. On ``split='insample'`` it mirrors ``compute_insample`` instead,
    which is the SAME three calls (that is what ``skill_map(per_model=True)``
    does per model); only the save dir differs, being shared across the fleet.
    """
    sd = check_run(kind, model_name, pins, epoch_arg, split)
    run = load_baseline(kind, sd, pins, device, draw=draw, seed=seed,
                        epoch_arg=epoch_arg)
    ep = epoch_for(kind, sd, pins, epoch_arg)
    mobjs = [m for m in ec.get_models(run.config) if m.model == model_name]
    if not mobjs:
        raise RuntimeError(
            f'{model_name} is absent from this config\'s data, so the baseline '
            f'cannot be evaluated on it. The diffusion '
            f'cache names it, so the data moved under one of the two.')
    ds_model = ec.build_per_model_ds(run.config, mobjs[0])
    res = ec.model_timemean(run, ds_model, pins['year_window'],
                            sample_cfg_from(pins, n_samples),
                            years_stride=pins['years_stride'],
                            years=pins['years'])
    if res is None or res['truth'] is None:
        raise RuntimeError(
            f'{model_name}: no times in {pins["years_label"]} -- the diffusion '
            f'cache averaged that window, so an empty one here means the two '
            f'are not looking at the same data.')
    logging.info(f'  {model_name}: {res["n_times"]} times, '
                 f'epoch {ep if kind == unet_mse.KIND else "n/a"}')
    res['epoch'] = ep
    return res


#: Coordinates a part keeps. Everything else is dropped -- see ``save_part``.
_KEEP_COORDS = ('lat', 'lon')


def save_part(kind, model_name, res, pins, draw, seed, n_samples=None,
              epoch_arg='pinned', split=DEFAULT_SPLIT):
    """One held-out model's fields, with enough provenance to be re-checked.

    ⚠️ Non-spatial coordinates are DROPPED before writing. ``build_per_model_ds``
    carries a ``model`` coordinate holding the loader's own model object, and
    netCDF cannot serialize an arbitrary Python object -- the write dies with
    ``unable to infer dtype on variable 'model'`` after all the sampling is
    done, which for a QRF part is ~35 min of forest traversal thrown away.
    ``error_maps`` never meets this because ``model_mean_abs`` concatenates
    along a fresh ``model`` dim and immediately averages it out; a per-model
    part is the first thing here to write a field with its original coords.
    The model's NAME is in ``attrs`` (and in the filename), which is the part of
    that coordinate anything downstream actually reads.
    """
    ds = xr.Dataset({'gen': res['gen'], 'truth': res['truth']})
    ds = ds.drop_vars([c for c in ds.coords if c not in _KEEP_COORDS])
    cfg = sample_cfg_from(pins, n_samples)
    ds.attrs.update(
        baseline=kind, split=split, model=model_name, n_times=res['n_times'],
        epoch=int(res.get('epoch', pins['epoch'])), epoch_arg=epoch_arg,
        n_samples=cfg.n_samples, steps=cfg.steps,
        sampler=cfg.sampler, eta=cfg.eta, years_label=pins['years_label'],
        draw=(draw if kind == qrf.KIND else 'n/a'),
        seed=(seed if kind == qrf.KIND else -1),
        # a deterministic regressor's n draws are identical, so its ensemble
        # mean IS the single field; n_samples stays at the pinned 5 so the
        # provenance is comparable attr-for-attr with the diffusion cache
        n_samples_effective=(1 if is_deterministic(kind) else cfg.n_samples),
        pinned_from=Path(pins['path']).name)
    path = part_path(kind, model_name, epoch_arg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logging.info(f'  wrote {path}')
    return ds


def open_part(path):
    """``(ds, None)``, or ``(None, why)`` if the part is missing or unreadable.

    ⚠️ A part can exist and still not open. A job killed mid-``to_netcdf``
    leaves a truncated file. Letting that surface as a raw HDF
    error deep in ``aggregate`` would read as data corruption rather than as
    "recompute this one part", so it is turned into a reason string at the one
    place that opens a part.
    """
    path = Path(path)
    if not path.is_file():
        return None, 'not computed'
    try:
        return xr.open_dataset(path).load(), None
    except Exception as exc:                      # truncated / half-written
        return None, f'unreadable ({type(exc).__name__}: {exc})'


def part_matches(ds, kind, pins, draw, seed, n_samples=None,
                 epoch_arg='pinned', split=DEFAULT_SPLIT):
    """(ok, differences) -- was this part computed under the requested settings?

    ⚠️ ``epoch_arg`` is compared, not ``epoch``. Under ``--epoch selected`` the
    concrete epoch differs BY MODEL (each run has its own validation winner), so
    a per-model equality check would call every part stale. ``epoch`` is still
    stamped as a fact and lands in the cache's ``epochs`` list.
    """
    cfg = sample_cfg_from(pins, n_samples)
    want = dict(baseline=kind, epoch_arg=epoch_arg, n_samples=cfg.n_samples,
                steps=cfg.steps, sampler=cfg.sampler,
                years_label=pins['years_label'],
                draw=(draw if kind == qrf.KIND else 'n/a'))
    # ⚠️ Under 'pinned' the concrete epoch IS constant across models, so it can
    # and must be compared: `part_path` keys only on `epoch_arg`, so pointing
    # --pin-cache at a different diffusion cache (a different epoch, say)
    # otherwise lands on the SAME part files and every one of them matches --
    # silently reporting epoch-500 fields under an epoch-1000 comparison. Under
    # 'selected' the epoch varies by model by design and is not compared.
    if epoch_arg != SELECTED:
        want['epoch'] = pins['epoch']
    diffs = [f'{k}: part={ds.attrs.get(k, "?")} requested={want[k]}'
             for k in _PART_PROVENANCE
             if k in want and str(ds.attrs.get(k, '')) != str(want[k])]
    # ⚠️ `split` is compared HERE rather than through _PART_PROVENANCE: the
    # oldest LOMO parts stamp no `split` at all, so a generic missing-attr
    # comparison would report every one of them stale. Absent means 'oos' --
    # the only split that existed when they were written.
    got_split = str(ds.attrs.get('split', DEFAULT_SPLIT))
    if got_split != split:
        diffs.append(f'split: part={got_split} requested={split}')
    if kind == qrf.KIND and int(ds.attrs.get('seed', -1)) != int(seed):
        diffs.append(f'seed: part={ds.attrs.get("seed")} requested={seed}')
    return (not diffs), diffs


# ---------------------------------------------------------------------------
# stage 2 -- the fleet
# ---------------------------------------------------------------------------
def cache_path(kind, epoch_arg='pinned', split=DEFAULT_SPLIT):
    """One cache per (baseline, readout, split).

    ⚠️ The two readouts get DIFFERENT files. Writing both to one path would
    make the last run win silently, and the pinned-epoch number is the
    like-for-like comparison while the selected-epoch one is deliberately
    asymmetric -- exactly the pair that must not overwrite each other.

    The split needs no suffix of its own: ``cell_slug`` already ends in it, so
    the two files are ``<kind>_T-S_ann_int_oos.nc`` and
    ``<kind>_T-S_ann_int_insample.nc``.
    """
    suffix = '' if epoch_arg == 'pinned' else f'_{epoch_arg}'
    return (kind_cache_dir(kind) /
            f'{kind}_{cell_slug(cell_for(split))}{suffix}.nc')


def aggregate(kind, pins, draw=qrf.DEFAULT_DRAW, seed=0, n_samples=None,
              epoch_arg='pinned', split=DEFAULT_SPLIT):
    """``ec.model_mean_abs`` over the parts -> the evaluation cache for one kind.

    ⚠️ Every part is re-checked here, against the pins **and** against the
    pinned fleet. ``error_maps``' ``cache_matches`` compares the knobs but not
    the fleet; the fleet is the thing this comparison cannot get wrong, so it
    is checked first and fatally.
    """
    gens, truths, used, epochs, allnan = [], [], [], [], []
    for name in pins['models']:
        path = part_path(kind, name, epoch_arg, split)
        ds, why = open_part(path)
        if ds is None:
            raise FileNotFoundError(
                f'part for held-out {name} at {path}: {why}. The fleet is '
                f'pinned to {Path(pins["path"]).name} '
                f'({len(pins["models"])} models), so aggregating without it '
                f'would report a different fleet under the same figure. '
                f'Compute it with: python -m baselines.evaluate --baseline '
                f'{kind} --split {split} --model {name}')
        ok, diffs = part_matches(ds, kind, pins, draw, seed, n_samples,
                                 epoch_arg, split)
        if not ok:
            raise RuntimeError(
                f'{path.name} was computed under different settings than the '
                f'pins ask for:\n    ' + '\n    '.join(diffs) +
                f'\n  Recompute it (--model {name}) rather than aggregating a '
                f'mixture.')
        gens.append(ds['gen'])
        truths.append(ds['truth'])
        used.append(name)
        epochs.append(int(ds.attrs.get('epoch', pins['epoch'])))
        # ⚠️ An all-NaN part contributes NOTHING to model_mean_abs (its mean
        # over `model` skips NaN) yet is still counted in n_models. The
        # in-sample fleet's 16 include IPSL-CM5A2-INCA
        # and IPSL-CM6A-LR, whose 1pctCO2 annual depth-integrated regrids are
        # entirely NaN, and ec.get_models discovers models by GLOBBING
        # FILENAMES -- nothing opens them. So the in-sample panels are drawn
        # over an EFFECTIVE fleet of 14, not the 16 the cache stamps. The
        # diffusion cache has the same property, so the comparison is still
        # like-for-like; what would be wrong is to report the 16.
        if not bool(np.isfinite(ds['truth'].values).any()):
            allnan.append(name)
        ds.close()

    d = ec.model_mean_abs(gens, truths)
    if d is None:
        raise RuntimeError('model_mean_abs returned nothing from '
                           f'{len(gens)} parts')
    d['model_names'] = used
    d['epochs'] = epochs
    d['models_allnan'] = allnan
    if allnan:
        logging.warning(
            f'  [warn] {len(allnan)} of {len(used)} parts are entirely NaN and '
            f'contribute nothing: {" ".join(allnan)}. n_models_effective = '
            f'{len(used) - len(allnan)} is stamped beside n_models.')
    return d


def save_cache(kind, d, pins, draw, seed, n_samples=None,
               epoch_arg='pinned', split=DEFAULT_SPLIT):
    """The evaluation cache. Stamped so it can never be read as a diffusion one."""
    ds = xr.Dataset({k: d[k] for k in ('truth', 'gen', 'err', 'rel')})
    cfg = sample_cfg_from(pins, n_samples)
    cell = cell_for(split)
    ds.attrs.update(
        baseline=kind,
        cell=cell_slug(cell), title=cell_title(cell),
        predictors=' '.join(cell['predictors']), resolution=cell['resolution'],
        field_type=cell['field_type'], split=cell['split'],
        n_models=d['n_models'], models=' '.join(d['model_names']),
        # see the all-NaN warning in aggregate(): n_models is the pinned fleet,
        # n_models_effective the part of it that actually carries data.
        n_models_effective=d['n_models'] - len(d.get('models_allnan', ())),
        models_allnan=' '.join(d.get('models_allnan', ())),
        epochs=' '.join(str(e) for e in d['epochs']),
        epoch_arg=epoch_arg,
        epoch_pinned=pins['epoch'],
        n_samples=cfg.n_samples, steps=cfg.steps, sampler=cfg.sampler,
        eta=cfg.eta,
        n_samples_effective=(1 if is_deterministic(kind) else cfg.n_samples),
        draw=(draw if kind == qrf.KIND else 'n/a'),
        seed=(seed if kind == qrf.KIND else -1),
        years_stride=pins['years_stride'],
        year_window=' '.join(str(y) for y in (pins['year_window'] or ())),
        years_label=pins['years_label'],
        n_months=1, metric=METRIC,
        mape_pct=float(ec.mape(d['rel'])),
        skill_rms_pct=float(ec.skill_scalar(d['rel'])),
        pinned_from=Path(pins['path']).name,
        diffusion_mape_pct=pins['diffusion_mape_pct'])
    path = cache_path(kind, epoch_arg, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logging.info(f'  cached {path}')
    logging.info(f'  MAPE {ds.attrs["mape_pct"]:.2f}%  (the diffusion model on '
                 f'the same fleet, epoch and window: '
                 f'{pins["diffusion_mape_pct"]:.2f}%)')
    return ds


# ---------------------------------------------------------------------------
def report(kind, pins, draw, seed, n_samples=None, epoch_arg='pinned',
           split=DEFAULT_SPLIT):
    """What is pinned, what exists, what is stale. No GPU, no sampling."""
    print(f'pins, read from {pins["path"]}:')
    for k in ('split', 'epoch', 'n_samples', 'steps', 'sampler', 'years_label',
              'metric'):
        print(f'  {k:12s} {pins[k]}')
    print(f'  {"fleet":12s} {len(pins["models"])} models')
    print(f'  {"diffusion":12s} MAPE {pins["diffusion_mape_pct"]:.2f}%')
    print(f'\nbaseline {kind} (draw={draw if kind == qrf.KIND else "n/a"}, '
          f'epoch={epoch_arg}):')
    n_ready = n_part = 0
    for name in pins['models']:
        try:
            check_run(kind, name, pins, epoch_arg, split)
            run_state = 'run ok'
            n_ready += 1
        except FileNotFoundError as exc:
            run_state = f'MISSING ({str(exc).splitlines()[0][:60]}...)'
        ds, why = open_part(part_path(kind, name, epoch_arg, split))
        if ds is None:
            part_state = 'part -' if why == 'not computed' else f'part {why}'
        else:
            ok, diffs = part_matches(ds, kind, pins, draw, seed, n_samples,
                                     epoch_arg, split)
            part_state = 'part ok' if ok else f'part STALE ({diffs[0]})'
            n_part += 1 if ok else 0
            ds.close()
        print(f'  {name:22s} {run_state:12s} {part_state}')
    print(f'\n{n_ready}/{len(pins["models"])} runs usable, '
          f'{n_part}/{len(pins["models"])} parts current')
    cp = cache_path(kind, epoch_arg, split)
    print(f'cache: {cp} {"(exists)" if cp.is_file() else "(not written)"}')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', required=True, choices=all_kinds(),
                   help="which model to run through the protocol; "
                        f"'{DIFFUSION}' is the model the baselines are "
                        "compared against, kept per-model so the comparison "
                        "can be paired")
    p.add_argument('--model', default=None, metavar='NAME',
                   help='compute the part for this held-out model only; '
                        'default: every pinned model')
    p.add_argument('--aggregate', action='store_true',
                   help='combine the parts into the evaluation cache (no '
                        'sampling, no GPU)')
    p.add_argument('--list', action='store_true',
                   help='print the pins and what exists, then exit (no GPU)')
    p.add_argument('--draw', default=qrf.DEFAULT_DRAW,
                   choices=list(qrf.DRAW_SCHEMES),
                   help=f'QRF ensemble draw rule (default {qrf.DEFAULT_DRAW}); '
                        f'ignored by the deterministic baseline')
    p.add_argument('--seed', type=int, default=0,
                   help='QRF draw seed, stamped into every cache')
    p.add_argument('--n-samples', type=int, default=None,
                   help='override the pinned ensemble size -- for quick tests '
                        'only; the comparison uses what the cache stamps')
    p.add_argument('--epoch', default='pinned', choices=['pinned', SELECTED],
                   help="'pinned' (default) uses the diffusion cache's own "
                        "epoch -- the like-for-like comparison; 'selected' "
                        "uses each U-Net run's validation winner from "
                        "select_epoch.py, which may EXCEED the diffusion's "
                        "and so is deliberately asymmetric in the baseline's "
                        "favour. The two write separate caches.")
    p.add_argument('--split', default=DEFAULT_SPLIT, choices=list(SPLITS),
                   help="'oos' (default) is the leave-one-model-out protocol; "
                        "'insample' evaluates the single "
                        "in-sample run of each kind over the 16-model "
                        "in-sample fleet. ⚠️ The two fleets DIFFER (14 vs 16), "
                        "so their figures may sit side by side but must never "
                        "be differenced. They write separate parts and caches.")
    p.add_argument('--pin-cache', default=None,
                   help='diffusion cache defining the comparison (default: '
                        'the one for --split, i.e. '
                        f'{PIN_CACHE.name} / '
                        f'{pin_cache_for("insample").name})')
    p.add_argument('--overwrite', action='store_true',
                   help='recompute parts that already exist and match')
    p.add_argument('--cpu', action='store_true',
                   help='force CPU (the QRF path never needs a GPU)')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    kind = args.baseline
    pins = read_pins(args.pin_cache, args.split)

    if kind != unet_mse.KIND and args.epoch == SELECTED:
        raise SystemExit(
            f'--epoch selected is unet-mse only. A forest has no epochs (the '
            f'QRF is tuned on min_samples_leaf instead), the linear kinds are '
            f'CLOSED FORM and have neither an epoch nor a tuned knob, and the '
            f'diffusion model has no selection pass -- it is read at whatever '
            f'{Path(pins["path"]).name} stamps, which is the whole point of '
            f'pinning to it.')

    if args.list:
        report(kind, pins, args.draw, args.seed, args.n_samples, args.epoch,
               args.split)
        return

    if args.aggregate:
        d = aggregate(kind, pins, args.draw, args.seed, args.n_samples,
                      args.epoch, args.split)
        save_cache(kind, d, pins, args.draw, args.seed, args.n_samples,
                   args.epoch, args.split)
        return

    use_gpu = (kind == unet_mse.KIND or is_diffusion(kind)) and not args.cpu
    if use_gpu and not torch.cuda.is_available():
        raise SystemExit(
            f'{kind} needs a GPU (or --cpu, fine for a smoke test but slow '
            f'for the fleet -- and very slow for {DIFFUSION}, which samples a '
            f'100-step chain per member)')
    device = torch.device('cuda' if use_gpu else 'cpu')

    todo = [args.model] if args.model else list(pins['models'])
    for name in todo:
        if name not in pins['models']:
            raise SystemExit(
                f'{name} is not in the pinned fleet '
                f'({", ".join(pins["models"])}) -- evaluating a model the '
                f'diffusion cache does not name would not be the comparison.')
        if not args.overwrite:
            ds, why = open_part(part_path(kind, name, args.epoch, args.split))
            if ds is not None:
                ok, diffs = part_matches(ds, kind, pins, args.draw, args.seed,
                                         args.n_samples, args.epoch,
                                         args.split)
                ds.close()
                if ok:
                    logging.info(f'  {name}: part current, skipping '
                                 f'(--overwrite to recompute)')
                    continue
                logging.info(f'  {name}: part stale ({diffs[0]}), recomputing')
            elif why != 'not computed':
                logging.info(f'  {name}: part {why}, recomputing')
        res = compute_part(kind, name, pins, device, draw=args.draw,
                           seed=args.seed, n_samples=args.n_samples,
                           epoch_arg=args.epoch, split=args.split)
        save_part(kind, name, res, pins, args.draw, args.seed, args.n_samples,
                  epoch_arg=args.epoch, split=args.split)


if __name__ == '__main__':
    main()
