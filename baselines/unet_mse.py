"""The deterministic U-Net baseline: build, train, load.

Extracted from general/baselines/unet_mse.py.

Same backbone as the diffusion model (``SimpleUNetCond``, same ``base_ch``,
``ch_mults``, ``time_emb_dim``, ``num_res_blocks``), trained with a masked MSE
against the target instead of a denoising objective. That is the whole
difference, which is what makes the comparison attributable to the diffusion
machinery rather than to capacity.

⚠️ ``in_ch = N`` here, against the diffusion model's ``1 + N``: the deterministic
analogue takes the predictors and nothing else. Feeding a permanently-zero
channel to equalise the parameter count was considered and rejected -- the delta
is one conv's first layer (~288 parameters against millions), and a channel that
is always zero is not a cleaner comparison than not having it. Recorded because
the number of input channels is the one architectural difference, and a reader
comparing ``config['in_ch']`` across the two run families will see it.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import torch

from diffusion.model import EMA, SimpleUNetCond, Trainer
from baselines import shims


KIND = 'unet-mse'


def build(config, dataset, device):
    """``(regressor, in_ch)`` for a dataset -- the shim wrapping the backbone.

    ``in_ch`` is derived from the dataset's conditioning channels, never
    assumed: the density field type expands each tracer into one channel per
    sigma layer, which is the same reason ``diffusion.train.main`` derives it.
    """
    cond_ch = dataset[0]['cond'].shape[0]
    model = SimpleUNetCond(
        in_ch=cond_ch,                       # NOT cond_ch + 1 -- see module doc
        base_ch=config['base_ch'],
        ch_mults=config['ch_mults'],
        time_emb_dim=config['time_emb_dim'],
        num_res_blocks=config['num_res_blocks'],
    ).to(device).to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    logging.info(f'built {KIND}: in_ch={cond_ch} (predictors only), '
                 f'{n_par/1e6:.2f}M parameters')
    return shims.DeterministicRegressor(
        model, device=device, img_size=dataset.mask.shape), cond_ch


def train(config, dataset, device, resume_from_existing=True, compile_model=True):
    """Train through the stock ``Trainer``. Nothing about the loop is new.

    ``compile_model`` mirrors ``diffusion.train``: on for a real run, off for a
    smoke test, where the compile dominates a two-epoch job.
    """
    regressor, cond_ch = build(config, dataset, device)
    if compile_model:
        regressor.model = torch.compile(regressor.model)
        logging.info('compiled')
    config['in_ch'] = cond_ch

    trainer = Trainer(
        regressor, dataset,
        save_dir=config['save_dir'],
        save_every=config['save_every'],
        batch_size=config['batch_size'],
        lr=config['learning_rate'],
        epochs=config['epochs'],
        num_workers=config['num_workers'],
        persistent_workers=config['persistent_workers'],
        prefetch_factor=config['prefetch_factor'],
        grad_accum=config['grad_accum'],
        amp=config['amp'],
        device=device,
    )

    resume_from = None
    if resume_from_existing:
        existing = checkpoint_epochs(config['save_dir'])
        if existing:
            last = existing[-1]
            if last >= config['epochs']:
                logging.info(f'all {config["epochs"]} epochs already trained')
                return trainer
            logging.info(f'resuming from epoch {last}')
            resume_from = trainer.checkpoint_path(last)
    trainer.train(resume_from=resume_from)
    return trainer


#: ``Trainer.checkpoint_path`` names checkpoints ``ckpt_epoch{epoch:0>3}.pt``,
#: so the digit count is 3 BELOW epoch 1000 and 4 at or above it.
_EPOCH_RE = re.compile(r'ckpt_epoch(\d+)\.pt$')


def checkpoint_epochs(save_dir):
    """Every epoch this run has on disk, ascending.

    The baseline reports both a validation-selected epoch and the fixed epoch
    matching the diffusion runs, from ONE training run -- so the selection pass
    needs the full list, not just the latest.

    Parsed with a REGEX, not ``stem[-3:]``: that slice reads
    ``ckpt_epoch1000.pt`` as epoch 0, and these runs train to exactly 1000.
    """
    out = []
    for c in Path(save_dir).glob('ckpt_epoch*.pt'):
        m = _EPOCH_RE.search(c.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_run(save_dir, device, epoch=None, config=None, _compile=False):
    """A :class:`eval_common.RunHandle` around a trained deterministic U-Net.

    The counterpart of ``qrf.load_baseline_run``, and the reason the held-out
    loop of ``error_maps.compute_oos`` is re-written in ``evaluate.py`` rather
    than reused: ``ec.load_run`` goes through ``inference.load_model``, which
    rebuilds a UNet **+ DDPM** and sizes the input as ``len(predictors) + 1``.
    Neither is right here. Everything downstream of this function --
    ``predict_ensemble``, ``model_timemean``, ``ec.model_mean_abs`` -- is the
    diffusion model's own code, unchanged.

    ⚠️ ``epoch`` is REQUIRED in practice for the evaluation: the comparison
    point is whatever ``T-S_ann_int_oos.nc`` stamps (500), not the 1000 these
    runs reach on disk. Defaulting to the highest epoch is a convenience for diagnostics,
    never the comparison.

    ⚠️ Channel order is asserted against the checkpoint's own normalization
    keys, as ``ec.load_run`` does. A silently permuted ``cond`` would still run
    and still produce a plausible-looking map.
    """
    from figures import eval_common as ec

    save_dir = Path(save_dir)
    if config is None:
        with open(save_dir / 'config.json') as fh:
            config = json.load(fh)

    epochs = checkpoint_epochs(save_dir)
    if not epochs:
        raise FileNotFoundError(f'no ckpt_epoch*.pt in {save_dir}')
    ep = epochs[-1] if epoch is None else int(epoch)
    ckpt_path = Path(save_dir) / f'ckpt_epoch{str(ep).zfill(3)}.pt'
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f'{ckpt_path.name} not in {save_dir} -- have {epochs}')
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)

    # written by train_baseline.run() before it dumps config.json, so the
    # expanded channel names never have to be re-derived from the data pipeline
    channels = list(config['predictor_channels'])
    target = config['target']
    expected = channels + [target]
    got = list(ck['means'].keys())
    assert got == expected, (
        f'channel mismatch for {save_dir.name}:\n  ckpt={got}\n'
        f'  config={expected}')

    model = SimpleUNetCond(
        in_ch=len(channels),              # NOT +1 -- see the module docstring
        base_ch=config['base_ch'],
        ch_mults=config['ch_mults'],
        time_emb_dim=config['time_emb_dim'],
        num_res_blocks=config['num_res_blocks'],
    ).to(device).to(memory_format=torch.channels_last)
    if _compile:
        model = torch.compile(model)

    # checkpoints are written from a torch.compile'd model, so their keys carry
    # an "_orig_mod." prefix; strip it when loading eager (inference.load_model's
    # own convention, and why a --no-compile probe's checkpoints are unusable
    # to a compiled run)
    def _strip(sd):
        if _compile:
            return sd
        pfx = '_orig_mod.'
        return {(k[len(pfx):] if k.startswith(pfx) else k): v
                for k, v in sd.items()}

    model.load_state_dict(_strip(ck['model_state']))
    msd = dict(model.named_parameters())
    ema = EMA(model, decay=ck['ema_decay'])
    ema.shadow = {k: v.to(device=msd[k].device, dtype=msd[k].dtype)
                  for k, v in _strip(ck['ema_state']).items()}
    model.eval()

    scaling = dict(means=ck['means'], stds=ck['stds'],
                   land_fill=ck['land_fill'])
    regressor = shims.DeterministicRegressor(
        model, device=device, img_size=(180, 360))
    logging.info(f'  loaded {KIND} {save_dir.name}: epoch {ep}, '
                 f'{len(channels)} predictor channels, '
                 f'land_fill={scaling["land_fill"]}')
    return ec.RunHandle(ddpm=regressor, ema=ema, scaling=scaling,
                        predictor_channels=channels, config=config,
                        save_dir=save_dir, device=device)
