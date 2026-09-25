"""Train one baseline run. Mirrors ``diffusion/train.py``.

Extracted from general/baselines/train_baseline.py.

    python -m baselines.train_baseline --baseline unet-mse -m insample
    python -m baselines.train_baseline --baseline unet-mse -m CanESM5
    python -m baselines.train_baseline --baseline qrf     -m CanESM5   # CPU
    python -m baselines.train_baseline --baseline pointwise-linear \
        --also climatology -m CanESM5                                   # CPU

Everything on the data side is the diffusion pipeline unchanged --
``data_loading.load_training_data``, ``diffusion.train._split``,
``TracerDataset``, and the same shared raw-preproc cache -- because a baseline
trained on even slightly different data would not be a baseline. What differs is
the model (``baselines/shims.py``) and where checkpoints land
(``baselines/naming.py``, its own scratch root).

All baselines share this entrypoint and everything up to ``TracerDataset``;
``run`` dispatches on the kind because the forest and the closed-form fits are
CPU jobs with no epochs and must not hold a GPU allocation.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch

from vendored.ocean_utils_min import log_setup
from diffusion.model import TracerDataset
from diffusion import data_loading
from diffusion import train as train_general
from baselines import naming, pointwise_linear, qrf, unet_mse


#: Every baseline kind, by name. ``unet-mse`` trains on a GPU through the stock
#: ``Trainer``; ``qrf`` fits a forest on CPU and has no epochs, and the two linear
#: kinds are a closed-form solve on CPU -- so ``run`` dispatches on this rather
#: than assuming torch (a forest, or a 3x3 linear solve, on a GPU node would be
#: a wasted allocation).
BUILDERS = {unet_mse.KIND: unet_mse, qrf.KIND: qrf,
            **{k: pointwise_linear for k in pointwise_linear.KINDS}}
TORCH_KINDS = (unet_mse.KIND,)

#: Kinds whose fit needs ``ds_train`` itself, not just the ``TracerDataset``
#: wrapped around it. A per-cell regression must know which of a cell's
#: training samples were genuinely wet, and ``TracerDataset`` has already
#: filled NaN -> the land-fill constant by the time it is constructed -- see
#: ``pointwise_linear.target_validity``.
RAW_TRAIN_KINDS = tuple(pointwise_linear.KINDS)


def base_config(**overrides):
    """The diffusion config, plus a ``baseline`` key. Same defaults on purpose."""
    cfg = copy.deepcopy(train_general.config)
    cfg.update(overrides)
    return cfg


def run(config, resume_from_existing=True, compile_model=True, max_models=None,
        fit_kw=None):
    """Load the data exactly as the diffusion pipeline does, then fit.

    Everything up to and including ``TracerDataset`` is shared by both
    baselines and is the diffusion code unchanged -- same loader, same split,
    same shared raw-preproc cache, same normalization. Only the model and the
    save-dir root differ.
    """
    kind = config['baseline']
    if kind not in BUILDERS:
        raise ValueError(f'no builder for baseline {kind!r}; have '
                         f'{sorted(BUILDERS)}')
    is_torch = kind in TORCH_KINDS
    if is_torch:
        if not torch.cuda.is_available():
            raise ValueError('Need GPU')
        device = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device('cpu')

    config['save_dir'] = naming.save_dir_from_config(config)
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    ds_stacked, predictor_channels, _ = data_loading.load_training_data(
        config, max_models=max_models)
    ds_train, ds_test = train_general._split(ds_stacked, config)
    logging.info(f'{len(ds_train.sample)} train / {len(ds_test.sample)} test '
                 f'samples, {len(predictor_channels)} predictor channels')

    cache_dir = (naming.shared_cache_dir(config) if config.get('use_cache')
                 else None)
    dataset = TracerDataset(ds_train, predictors=list(predictor_channels),
                            target=config['target'], cache_dir=cache_dir)

    config['predictor_channels'] = list(predictor_channels)
    if is_torch:
        fitted = BUILDERS[kind].train(
            config, dataset, device, resume_from_existing=resume_from_existing,
            compile_model=compile_model)
    elif kind in RAW_TRAIN_KINDS:
        # the closed-form fits need the grid coordinates (their pooled
        # fallback uses the QRF's own sin/cos lat/lon encoding) AND ds_train,
        # for the per-sample validity mask
        fitted = BUILDERS[kind].train(
            config, dataset, device, lat=ds_train['lat'].values,
            lon=ds_train['lon'].values, ds_train=ds_train,
            resume_from_existing=resume_from_existing, **(fit_kw or {}))
    else:
        # the forest needs the grid coordinates: its features are the per-cell
        # predictors PLUS a location encoding
        fitted = BUILDERS[kind].train(
            config, dataset, device, lat=ds_train['lat'].values,
            lon=ds_train['lon'].values,
            years=ds_train['year'].values,     # for the validation-year split
            resume_from_existing=resume_from_existing, **(fit_kw or {}))
    with open(Path(config['save_dir']) / 'config.json', 'w') as fh:
        json.dump({k: v for k, v in config.items()
                   if not isinstance(v, (np.ndarray,))}, fh, indent=4)
    return fitted, dataset


def leave_one_out(config, modelname_test, **kw):
    config['train_test_split'] = 'model'
    config['models_test'] = [modelname_test]
    ds_stacked, _, _ = data_loading.load_training_data(config)
    names = sorted({m.model for m in set(ds_stacked.model.values)})
    if modelname_test not in names:
        raise ValueError(
            f'{modelname_test} is not in this config\'s {len(names)} loaded '
            f'models, so no leave-one-out split exists. The fleet is pinned '
            f'to the diffusion cache, so a missing model is a mismatch to fix, '
            f'not a run to skip: {names}')
    config['models_train'] = [m for m in names if m != modelname_test]
    return run(config, **kw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', default=unet_mse.KIND, choices=sorted(BUILDERS))
    p.add_argument('-p', '--predictors', default='thetao so')
    p.add_argument('-t', '--target', default='o2')
    p.add_argument('-m', '--model', default='insample',
                   help="'insample' or a CMIP6 model name to hold out")
    p.add_argument('-r', '--resolution', default='annual',
                   choices=['annual', 'monthly'])
    p.add_argument('-f', '--field_type', default='depthint',
                   choices=['depthint', 'density'])
    p.add_argument('-e', '--experiments', default='1pctCO2')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--save-every', type=int, default=None, dest='save_every',
                   help='checkpoint interval -- select_epoch picks from these, '
                        'so a coarse interval limits the selection grid')
    p.add_argument('--use-cache', action='store_true', dest='use_cache')
    p.add_argument('--no-compile', action='store_true')
    # --- linear kinds only (a closed-form solve has no epochs and no tuned knob)
    lg = p.add_argument_group('pointwise-linear / climatology')
    lg.add_argument('--also', nargs='+', default=None,
                    choices=list(pointwise_linear.KINDS),
                    help='additionally write these kinds FROM THE SAME '
                         'accumulators. climatology is the intercept-only '
                         'nested submodel of pointwise-linear, so passing '
                         '`--baseline pointwise-linear --also climatology` '
                         'gets both from byte-identical normal equations in '
                         'one pass instead of two passes that could differ')
    lg.add_argument('--min-samples', type=int, default=None,
                    dest='min_samples',
                    help='valid training maps a cell needs before its own '
                         'regression is preferred to the pooled fallback. NOT '
                         'a tuned hyperparameter -- every deep evaluation cell '
                         'has >= 100 (measured), so it only ever '
                         f'fires on the shallow fringe. '
                         f'default {pointwise_linear.MIN_SAMPLES}')
    # --- qrf only (a forest has no epochs; these size the fit instead) -----
    qg = p.add_argument_group('qrf')
    qg.add_argument('--n-cells', type=int, default=None, dest='n_cells_per_map',
                    help='ocean cells sampled per map. Subsampling is REQUIRED '
                         '(~1.6e8 candidate rows) and subsamples cells, not '
                         'maps, so every year of the forcing keeps a vote. '
                         f'default {qrf.DEFAULT_FIT["n_cells_per_map"]}')
    qg.add_argument('--n-estimators', type=int, default=None)
    qg.add_argument('--min-samples-leaf', type=int, default=None,
                    dest='min_samples_leaf')
    qg.add_argument('--max-samples-leaf', type=int, default=None,
                    dest='max_samples_leaf',
                    help='leaf values retained per leaf. ⚠️ NOT None -- the '
                         'dense leaf array is then sized by the largest leaf '
                         'in any tree (37 GB at 200k rows). See qrf.py. '
                         f'default {qrf.DEFAULT_FIT["max_samples_leaf"]}')
    qg.add_argument('--qrf-seed', type=int, default=None, dest='seed',
                    help='stamped into the artifact')
    qg.add_argument('--n-jobs', type=int, default=-1,
                    help='forest workers; match --cpus-per-task')
    qg.add_argument('--select-min-samples-leaf', type=int, nargs='+',
                    default=None, dest='select_grid',
                    help='choose min_samples_leaf from these by validation '
                         'pinball loss on held-in models\' held-out years, '
                         'then refit on all years with the winner. The QRF '
                         'analogue of the U-Net\'s validation-selected epoch. '
                         'The shipped fleet used `5 10 20` and picked 5 on 14 '
                         'of 15 splits (the grid\'s lower edge; msl=2 lost in '
                         'a single-split probe). e.g. '
                         '--select-min-samples-leaf 5 10 20')
    qg.add_argument('--val-years', type=int, nargs=2, default=(81, 100),
                    help='validation year range, held-in models only')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    cfg = base_config(
        baseline=args.baseline, predictors=args.predictors.split(),
        target=args.target, resolution=args.resolution,
        field_type=args.field_type, experiments=args.experiments.split(),
        use_cache=args.use_cache,
        train_test_split='insample' if args.model == 'insample' else 'model',
        models_train=None, models_test=None)
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.save_every is not None:
        cfg['save_every'] = args.save_every

    kw = dict(compile_model=not args.no_compile)
    if args.baseline == qrf.KIND:
        kw = dict(fit_kw=dict(
            n_cells_per_map=args.n_cells_per_map,
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_samples_leaf=args.max_samples_leaf,
            seed=args.seed, n_jobs=args.n_jobs,
            select_grid=args.select_grid, val_years=tuple(args.val_years)))
    elif args.baseline in pointwise_linear.KINDS:
        fk = dict(also=tuple(args.also or ()))
        if args.min_samples is not None:
            fk['min_samples'] = args.min_samples
        kw = dict(fit_kw=fk)
    if args.model == 'insample':
        cfg['save_dir'] = naming.save_dir_from_config(cfg)
        log_setup(f'baseline_{Path(cfg["save_dir"]).name}')
        run(cfg, **kw)
    else:
        log_setup(f'baseline_{args.baseline}_{args.model}')
        leave_one_out(cfg, args.model, **kw)


if __name__ == '__main__':
    main()
