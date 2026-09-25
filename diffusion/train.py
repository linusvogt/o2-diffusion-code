"""
Training entrypoint for the conditional diffusion model.

The data side is routed through diffusion/data_loading.py so the user can
choose:

    resolution : 'annual' | 'monthly'
    field_type : 'depthint' | 'density'

The target is always depth-integrated o2 (single output channel); for the
'density' field type the predictors are expanded into one channel per density
layer, so ``in_ch`` is derived from the loaded dataset rather than assumed.

Checkpoints go to paths.CHECKPOINTS with a directory name that encodes the
configuration (see diffusion/naming.py), e.g.

    target-o2_pred-so-thetao_1pctCO2_res-monthly_fields-density-sigma1_split-insample_full

Example:
    python -m diffusion.train -p "thetao so" -t o2 -m insample -w 0.0 \
        -r monthly -f density -e "1pctCO2 abrupt-4xCO2"

Extracted from general/train_general.py.
"""
import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import torch

from diffusion.model import (
    DDPM, SimpleUNetCond, TracerDataset, Trainer)
from diffusion import data_loading
from diffusion.naming import (
    save_dir_from_config as _save_dir_from_config,
    shared_cache_dir as _shared_cache_dir)
from diffusion import cmip_io as util
from vendored.ocean_utils_min import log_setup


#: ``Trainer.checkpoint_path`` names checkpoints ``ckpt_epoch{epoch:0>3}.pt``,
#: so the digit count is 3 below epoch 1000 and 4 at or above it.
_CKPT_EPOCH_RE = re.compile(r'ckpt_epoch(\d+)\.pt$')


def _checkpoint_epochs(save_dir):
    """Epochs with a checkpoint in ``save_dir``, ascending NUMERICALLY.

    Parsed by regex rather than slicing the last three digits, so that
    ``ckpt_epoch1000.pt`` is read as 1000 (not 0) and sorts after 975.
    """
    epochs = []
    for ckpt in Path(save_dir).glob('ckpt_epoch*.pt'):
        m = _CKPT_EPOCH_RE.search(ckpt.name)
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


def main(config, ds_stacked=None, predictor_channels=None,
         resume_from_existing=True, setup_log=True, max_models=None,
         prep_only=False):
    # construct save_dir name and add to config
    config['save_dir'] = _save_dir_from_config(config)

    if setup_log:
        log_setup(f'general_diffusion_{Path(config["save_dir"]).name}')

    # save config
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    with open(Path(config['save_dir']) / 'config.json', 'w') as file:
        json.dump(config, file, indent=4)

    # use GPU. Under prep_only the netCDF decode runs on a CPU-only
    # allocation, so the GPU is never touched: everything up to and including
    # the raw-cache warm-up below is pure CPU/IO work.
    device = None
    if not prep_only:
        if not torch.cuda.is_available():
            raise ValueError('Need GPU')
        device = torch.device("cuda")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load data (target + expanded predictor channels)
    if ds_stacked is None:
        ds_stacked, predictor_channels, _ = data_loading.load_training_data(
            config, max_models=max_models)
        logging.info(f'Loaded data ({len(ds_stacked.sample)} samples, '
                     f'{len(predictor_channels)} predictor channels)')
    assert predictor_channels is not None

    # Split into train & test set
    ds_train, ds_test = _split(ds_stacked, config)
    logging.info(
        f'Split train/test: {len(ds_train.sample)} training samples, '
        f'{len(ds_test.sample)} test samples')

    # Ensemble-composition cap. Applied to the TRAIN split only and after the
    # split: the learned distribution is otherwise weighted by ESGF member
    # availability, so the cap belongs on what the model is fit to. Capping
    # ds_stacked instead would also thin the held-out model, changing the
    # evaluation target.
    if config.get('max_members_per_model'):
        mask_before = np.isfinite(
            ds_train[config['target']].isel(sample=0).values)
        ds_train, kept = data_loading.subsample_members(
            ds_train, config['max_members_per_model'])
        config['members_kept'] = kept
        # TracerDataset derives THE ocean mask from sample 0 of whatever it is
        # given, and per-model O2 masks differ. If the cap drops the
        # member that used to sit at sample 0, the training mask moves and the
        # comparison is no longer of composition alone.
        mask_after = np.isfinite(
            ds_train[config['target']].isel(sample=0).values)
        if not np.array_equal(mask_before, mask_after):
            raise RuntimeError(
                'max_members_per_model changed the training ocean mask '
                f'({int(mask_before.sum())} -> {int(mask_after.sum())} ocean '
                'cells): TracerDataset takes its mask from sample 0, and the '
                'capped split starts on a different member. The composition '
                'sensitivity test would confound the cap with a mask change; '
                'fix by masking to the intersection before capping.')
        logging.info(
            f'Capped composition: {len(ds_train.sample)} training samples '
            f'over {sum(len(v) for v in kept.values())} members / '
            f'{len(kept)} models (mask unchanged, '
            f'{int(mask_after.sum())} ocean cells)')
        # Re-dump: which members survived is the experiment's definition, and
        # the top-of-main dump ran before the data was even loaded. Recorded in
        # the run's own directory so the composition is recoverable from the
        # checkpoint alone, without re-deriving it from the loader.
        with open(Path(config['save_dir']) / 'config.json', 'w') as file:
            json.dump(config, file, indent=4)

    # construct torch dataset (predictors = expanded channel names). When
    # config['use_cache'] is set, raw per-(channel, member) data is cached to a
    # SHARED directory keyed by the data identity (not the split), so resumes AND
    # every leave-one-model-out run over the same data reuse it — the expensive
    # netCDF decode is paid ~once per predictor set instead of once per held-out
    # model. Stats/normalization are computed per-run on top, so the held-out
    # model never leaks in. With use_cache False (default) the in-RAM path is
    # used: nothing is written to or read from the cache.
    cache_dir = _shared_cache_dir(config) if config.get('use_cache') else None
    if prep_only and cache_dir is None:
        # Without a cache dir TracerDataset builds everything in RAM and throws
        # it away on return -- a prep job that caches nothing is pure waste.
        raise ValueError(
            'prep_only requires use_cache=True: with use_cache False nothing '
            'is written to the shared cache, so the decode would be discarded '
            'and the GPU run would pay it again.')
    dataset = TracerDataset(
        ds_train, predictors=predictor_channels, target=config["target"],
        cache_dir=cache_dir)
    logging.info(f'Constructed training dataset (cache_dir={cache_dir})')

    if prep_only:
        # TracerDataset swallows any cache-build failure and falls back to the
        # in-RAM path, which would let this job exit 0 having cached nothing.
        # _onfly is set only when the raw cache actually backs the dataset, so
        # it is the honest success signal.
        if not getattr(dataset, '_onfly', False):
            raise RuntimeError(
                f'prep_only: the raw cache at {cache_dir} was NOT built -- '
                f'TracerDataset fell back to its in-RAM path (see the preceding '
                f'"raw shared cache error" warning; usually disk quota). '
                f'Failing so dependent GPU jobs stay blocked.')
        # Everything expensive (netCDF decode -> float32 .npy per
        # (channel, member)) is now on disk under cache_dir. The GPU run for
        # this same config reads it back instead of re-decoding. Return before
        # any model/CUDA init.
        logging.info(
            f'prep_only: shared raw cache warmed at {cache_dir}; exiting '
            f'before model init (no GPU used).')
        return

    # initialize UNet -- derive in_ch from the dataset's conditioning channels
    cond_ch = dataset[0]["cond"].shape[0]
    in_ch = cond_ch + 1
    assert cond_ch == len(predictor_channels), (cond_ch, len(predictor_channels))
    config['in_ch'] = in_ch
    config['predictor_channels'] = list(predictor_channels)
    logging.info(f'{in_ch=} ({cond_ch} predictor channels + 1 noisy target)')

    model = SimpleUNetCond(
        in_ch=in_ch,
        base_ch=config["base_ch"],
        ch_mults=config["ch_mults"],
        time_emb_dim=config["time_emb_dim"],
        num_res_blocks=config["num_res_blocks"],
    )
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)
    logging.info('Initialized UNet (and compiled)')

    # initialize diffusion model
    H, W = dataset.mask.shape
    ddpm = DDPM(
        model,
        device=device,
        img_size=(H, W),
        num_timesteps=config["num_timesteps"],
        penalize_non_negative=config['penalize_non_negative'],
        integral_loss=config['integral_loss'],
        w_integral=config['w_integral'],
        schedule=config['schedule'],
        prediction=config['prediction'],
    )
    logging.info('Initialized DDPM')

    # initialize trainer
    trainer = Trainer(
        ddpm, dataset,
        save_dir=config["save_dir"],
        save_every=config['save_every'],
        batch_size=config["batch_size"],
        lr=config["learning_rate"],
        epochs=config["epochs"],
        num_workers=config["num_workers"],
        persistent_workers=config["persistent_workers"],
        prefetch_factor=config["prefetch_factor"],
        grad_accum=config["grad_accum"],
        amp=config["amp"],
        device=device,
    )
    logging.info('Initialized trainer')

    # resume from last saved checkpoint (if resume_from_existing is True)
    existing_epochs = _checkpoint_epochs(config['save_dir'])
    if resume_from_existing and existing_epochs:
        last_epoch = existing_epochs[-1]
        if last_epoch >= config['epochs']:
            logging.info('All epochs already trained')
            logging.info('Done')
            return
        logging.info(f'Resuming training from {last_epoch=}')
        resume_from = trainer.checkpoint_path(last_epoch)
    elif resume_from_existing and not existing_epochs:
        logging.info('No checkpoints saved, starting from epoch 0')
        resume_from = None
    else:
        logging.info('Starting from epoch 0')
        resume_from = None

    # train model
    trainer.train(resume_from=resume_from)
    logging.info('Done')


def main_leave_one_out(config, modelname_test, max_models=None,
                       prep_only=False):
    config['train_test_split'] = 'model'
    config['models_test'] = [modelname_test]
    config['save_dir'] = _save_dir_from_config(config)

    log_setup(f'general_diffusion_{Path(config["save_dir"]).name}')

    # load data once to get list of models
    ds_stacked, predictor_channels, _ = data_loading.load_training_data(
        config, max_models=max_models)
    logging.info(f'Loaded data: {len(ds_stacked.sample)=}')

    modelnames_all = sorted({m.model for m in set(ds_stacked.model.values)})
    logging.info(f'{modelnames_all=}')
    if modelname_test not in modelnames_all:
        # The held-out model has no data for this experiment/resolution/field-
        # type/predictor set (it lacks a required variable), so this leave-one-
        # out run can never train: exit cleanly rather than crashing.
        name = Path(config['save_dir']).name
        detail = (f'{modelname_test} absent from {len(modelnames_all)} loaded '
                  f'models: {modelnames_all}')
        logging.warning(
            f'DATA UNAVAILABLE: {modelname_test} not in the loaded dataset for '
            f'this config; exiting without training. {detail}')
        if prep_only:
            # A prep job must not report success for a cache it never warmed.
            raise RuntimeError(
                f'prep_only: cannot warm the cache for {name} -- the held-out '
                f'model is unavailable, so no training split exists. {detail}')
        return
    modelnames_train = [m for m in modelnames_all if m != modelname_test]
    assert len(modelnames_train) >= 1
    config['models_train'] = modelnames_train
    config['save_dir'] = _save_dir_from_config(config)

    main(config, ds_stacked=ds_stacked, predictor_channels=predictor_channels,
         setup_log=False, prep_only=prep_only)


def _split(ds_stacked, config):
    """Train/test split, reusing the existing helpers (identical to train.py)."""
    if config['train_test_split'] == 'insample':
        exp = config['experiments']
        if len(exp) == 1:
            exp = exp[0]
        return util.split_train_test_insample(ds_stacked, exp=exp)
    elif config['train_test_split'] == 'model':
        assert config['models_train'] is not None
        assert config['models_test'] is not None
        assert isinstance(config['models_train'][0], str)
        assert isinstance(config['models_test'][0], str)
        models_all = sorted(list(set(ds_stacked.model.values)))
        models_train = [m for m in models_all
                        if m.model in config['models_train']]
        models_test = [m for m in models_all
                       if m.model in config['models_test']]
        return util.split_train_test(
            ds_stacked, split_by='model',
            training_set=models_train, test_set=models_test,
            assert_all_samples=False)
    else:
        raise NotImplementedError(f"{config['train_test_split']=}")


config = {
    # --- TRAINING ---
    "learning_rate": 1e-4,
    "batch_size": 16,
    "epochs": 500,
    "save_every": 25,
    "grad_accum": 1,
    "amp": True,
    "num_workers": 4,
    # Optional DataLoader knobs, deliberately off (they raise peak memory).
    "persistent_workers": False,
    "prefetch_factor": None,

    # --- MODEL ---
    "base_ch": 32,
    "ch_mults": (1, 2, 4),
    "time_emb_dim": 256,
    "num_res_blocks": 2,
    "num_timesteps": 1000,
    "schedule": 'cos',
    "prediction": 'v',

    # --- DATA ---
    # Write/read the shared raw-preproc cache under preproc_cache_shared/. When
    # False, the in-RAM path is used (no cache written or read) — see
    # `use_cache` handling in main(). Disabled by default to save disk space.
    "use_cache": False,
    "anom": False,
    "experiments": ['1pctCO2'],
    "resolution": 'annual',     # 'annual' | 'monthly'
    "field_type": 'depthint',   # 'depthint' | 'density'
    "density_var": 'sigma_1',
    "density_levels": 'all',    # 'all' or list of ints
    # Ensemble-composition cap. None = the ESGF-weighted composition every
    # published run used. An int caps members per model on the TRAIN
    # split only and adds a `_mem<N>` tag to the save dir, so a capped run
    # never collides with -- or resumes -- its uncapped counterpart.
    "max_members_per_model": None,

    # --- LOSS ---
    'penalize_non_negative': False,
    'integral_loss': False,
    'w_integral': None,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predictors', type=str, required=True,
                        help='space-separated base predictor names, e.g. "thetao so"')
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='"insample" or a modelname for leave-one-out')
    parser.add_argument('-w', '--w_integral', type=str, default='0.0')
    parser.add_argument('-r', '--resolution', type=str, default='annual',
                        choices=['annual', 'monthly'])
    parser.add_argument('-f', '--field_type', type=str, default='depthint',
                        choices=['depthint', 'density'])
    parser.add_argument('-e', '--experiments', type=str, default='1pctCO2',
                        help='space-separated experiment names, e.g. '
                             '"1pctCO2" or "1pctCO2 abrupt-4xCO2"')
    parser.add_argument('--density_var', type=str, default='sigma_1')
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the default epoch target for this run '
                             'only (e.g. to continue a run past 500); resumes '
                             'from the highest existing checkpoint')
    parser.add_argument('--max-members-per-model', dest='max_members_per_model',
                        type=int, default=None,
                        help='cap the number of ensemble members per model in '
                             'the TRAINING split (composition sensitivity). '
                             'Members are chosen deterministically by (r,i,p,f), '
                             'so r1i1p1f1 first. Adds a "_mem<N>" tag to the '
                             'save dir; the uncapped run is untouched.')
    parser.add_argument('--use-cache', dest='use_cache', action='store_true',
                        help='write/read the shared raw-preproc cache '
                             '(off by default to save disk space)')
    parser.add_argument('--prep-only', dest='prep_only', action='store_true',
                        help='warm the shared raw-preproc cache for this config '
                             'and exit before touching CUDA. Meant for a '
                             'CPU-only allocation: the netCDF decode is pure '
                             'CPU work that would otherwise hold an idle GPU. '
                             'Implies --use-cache.')
    args = parser.parse_args()

    config['predictors'] = args.predictors.split(' ')
    config['target'] = args.target
    config['w_integral'] = float(args.w_integral)
    config['resolution'] = args.resolution
    config['field_type'] = args.field_type
    config['experiments'] = args.experiments.split(' ')
    config['density_var'] = args.density_var
    config['max_members_per_model'] = args.max_members_per_model
    # --prep-only exists to WRITE the cache, so it implies --use-cache.
    config['use_cache'] = args.use_cache or args.prep_only
    if args.epochs is not None:
        config['epochs'] = args.epochs

    if args.model == 'insample':
        config['train_test_split'] = 'insample'
        main(config, prep_only=args.prep_only)
    else:
        main_leave_one_out(config, modelname_test=args.model,
                           prep_only=args.prep_only)
