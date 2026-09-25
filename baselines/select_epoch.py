"""Pick the U-Net baseline's checkpoint on a validation split.

Extracted from general/baselines/select_epoch.py.

    python -m baselines.select_epoch --list                    # no GPU
    python -m baselines.select_epoch --model CanESM5           # one LOMO run
    python -m baselines.select_epoch --insample                # the in-sample run
    python -m baselines.select_epoch --summary                 # fleet table (no GPU)

The deterministic baseline has **two** readouts from ONE training run: the
epoch selected here, and the fixed epoch matching the diffusion cache (500).
``evaluate.py --epoch pinned`` (the default, and what the baseline-grid figure
reads) produces the second; ``evaluate.py --epoch selected`` reads the
``epoch_selection.json`` written here and produces the first, as a separate
cache.

Why this exists at all: the QRF's ``min_samples_leaf`` is selected on
validation, so the network must be tuned too. Comparing a tuned forest against
an arbitrarily-stopped network would tilt the comparison by an amount nobody
can quantify afterwards.

The criterion, and the three ways it could be quietly wrong
-----------------------------------------------------------
Validation loss is the **masked, area-weighted MSE** of the predicted field
against the truth, in mol/m^3 -- i.e. the quantity the baseline was trained to
minimize, evaluated out-of-training-years. The QRF's analogue is pinball loss (a
proper rule for the whole predictive distribution); for a deterministic
regressor the mean-squared error IS the proper rule for the mean, which is all
this baseline claims to produce.

1. ⚠️ **Validation is held-in models' held-out YEARS, never the held-out
   model.** A LOMO run trains on all years of its 13 held-in models, so the
   validation window is **81-100** -- the years the evaluation itself averages
   over, i.e. the condition the model is judged in. The **in-sample** run
   already excludes 81-100 as its *test* years, so it uses the adjacent
   **61-80**, which its training set does contain. This mirrors
   ``qrf.select_min_samples_leaf`` exactly, and for the same reason: selecting
   on the held-out model would leak the LOMO protocol into the choice.
   ⚠️ Note what this means for a LOMO run: the validation years ARE the
   reported years (the model has simply never seen the held-out *model*). That
   is the honest reading of "held-in models' held-out years" for a split whose
   held-out axis is the model, and it is what the QRF did -- but it is not a
   clean held-out set in the time axis, so the selected epoch is optimistic
   about those years and this is stamped rather than hidden.
2. ⚠️ **Per-model FIRST, then across models.** ``ec.model_mean_abs`` -- the
   metric the evaluation reports -- forms each model's error before averaging
   over models. Pooling every validation map into one MSE would select on a
   pooled statistic and report on a per-model-magnitude one. The averaging
   order is stamped into the output.
3. ⚠️ **No year stride.** A strided validation set selects the epoch on a
   different time-mean than the one being reported.

The evaluation path is ``ec.predict_ensemble``, unchanged
---------------------------------------------------------
Weights are swapped into ONE ``RunHandle`` per epoch rather than reloading the
run, so normalization, land fill and the ocean mask are literally the same code
that ``evaluate.py`` and the diffusion figures use. Re-normalizing by hand here
would be a second implementation of the thing the selection is supposed to be
selecting for.

⚠️ ``n_samples=1``, deliberately, and ONLY here: the network is deterministic,
so n draws are identical and the ensemble mean is the single field. That is a
selection statistic, not a cache anyone compares attr-for-attr against the
diffusion cache -- ``evaluate.py`` runs the pinned 5 for exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from diffusion import inference as util
from figures import eval_common as ec
from baselines import evaluate, naming, unet_mse


#: Validation year windows, by split. Different BY SPLIT and it must be -- see
#: the module docstring, and ``qrf.select_min_samples_leaf``, which makes the
#: same distinction for the same reason.
VAL_YEARS = {'model': (81, 100), 'insample': (61, 80)}

#: Written into each run's save dir.
OUT_NAME = 'epoch_selection.json'

#: How the per-map losses are combined. Stamped so a later reader cannot
#: mistake this for a pooled statistic (see trap 2).
AVERAGING = 'per-model-mean-of-per-year-MSE, then mean over models'


def validation_slices(run, model_names, year_window):
    """Every (model, year) validation case, materialized once.

    Uses ``ec.materialize_window`` -- the blocked read that exists because a
    per-timestep decode starves the GPU -- and then
    ``ec.select_case``, exactly as ``model_timemean`` does. Returns
    ``[(model_name, ds_sel), ...]``.
    """
    out = []
    years = list(range(year_window[0], year_window[1] + 1))
    by_name = {m.model: m for m in ec.get_models(run.config)}
    for name in model_names:
        mobjs = [by_name[name]] if name in by_name else []
        if not mobjs:
            raise RuntimeError(
                f'{name} is in this run\'s models_train but absent from its '
                f'data config -- the data moved under the run.')
        ds_model = ec.build_per_model_ds(run.config, mobjs[0])
        ds_model = ec.materialize_window(ds_model, run.config['resolution'],
                                         years, months=None)
        n = 0
        for year in years:
            ds_sel = ec.select_case(ds_model, run.config['resolution'], year)
            if ds_sel is not None:
                out.append((name, ds_sel))
                n += 1
        logging.info(f'  {name}: {n} validation maps')
    if not out:
        raise RuntimeError(
            f'validation window {year_window} selected NO maps. For a LOMO run '
            f'that means the years are missing from the held-in models; for '
            f'the in-sample run it means {VAL_YEARS["insample"]} is outside '
            f'its training years. Either way, selecting on an empty set would '
            f'silently return the first epoch.')
    return out


def load_checkpoint_into(run, save_dir, epoch, device):
    """Swap one checkpoint's weights into an existing handle, in place.

    Cheaper than ``unet_mse.load_run`` per epoch (41 of them), and -- the point
    -- it keeps ``run.scaling`` fixed, so every epoch is scored through
    identical normalization.

    ⚠️ Asserts the checkpoint's ``means``/``stds`` match the handle's. They are
    written per checkpoint by ``Trainer``, and its resume-compatibility check
    enforces they cannot change *within* a run -- so this should never fire.
    It is here because if it ever did, the selection would be comparing epochs
    scored in different units and the argmin would be meaningless.
    """
    ck = torch.load(Path(save_dir) / f'ckpt_epoch{str(epoch).zfill(3)}.pt',
                    map_location=device, weights_only=False)
    for key in ('means', 'stds'):
        have, want = ck[key], run.scaling[key]
        if set(have) != set(want) or any(
                not np.isclose(float(have[k]), float(want[k])) for k in want):
            raise RuntimeError(
                f'ckpt_epoch{epoch} of {Path(save_dir).name} carries different '
                f'{key} than its siblings -- the epochs are not on one scale, '
                f'so the validation curve does not compare like with like.')
    pfx = '_orig_mod.'
    strip = lambda sd: {(k[len(pfx):] if k.startswith(pfx) else k): v
                        for k, v in sd.items()}
    run.ddpm.model.load_state_dict(strip(ck['model_state']))
    msd = dict(run.ddpm.model.named_parameters())
    run.ema.shadow = {k: v.to(device=msd[k].device, dtype=msd[k].dtype)
                      for k, v in strip(ck['ema_state']).items()}
    run.ddpm.model.eval()
    return run


def score_epoch(run, slices, sample_cfg):
    """Per-model mean validation MSE at the currently loaded weights.

    Returns ``(loss_deep, loss_full, per_model)`` in (mol/m^3)^2, where
    ``loss_deep`` masks shallow cells the way ``ec.mape`` / ``ec.skill_scalar``
    do -- that is the footprint the evaluation reports on, so it is the one selected on.
    ``loss_full`` is carried alongside so a later reader can see whether the
    choice depended on the mask.
    """
    per_model = {}
    for name, ds_sel in slices:
        gen = ec.predict_ensemble(ds_sel, run, sample_cfg).mean('sample')
        truth = ds_sel[run.config['target']]
        sq = ((gen - truth) / ec.DEPTH_INT) ** 2
        rec = per_model.setdefault(name, {'deep': [], 'full': []})
        rec['deep'].append(float(util.global_mean(ec.mask_shallow(sq)).values))
        rec['full'].append(float(util.global_mean(sq).values))
    # per-model first, then over models (trap 2 in the module docstring)
    means = {k: {w: float(np.mean(v[w])) for w in ('deep', 'full')}
             for k, v in per_model.items()}
    return (float(np.mean([m['deep'] for m in means.values()])),
            float(np.mean([m['full'] for m in means.values()])),
            means)


def drop_nan_models(slices, per_model, save_dir):
    """Drop validation models whose loss is not finite, loudly. Returns (slices, dropped).

    ⚠️ **All-NaN source files are real here and discovery cannot see them.**
    ``get_models`` globs FILENAMES, so a model whose depth-integrated fields are
    entirely NaN is discovered and then poisons every statistic it enters --
    IPSL-CM5A2-INCA and IPSL-CM6A-LR are exactly that (`data_loading` reports
    "Dropped samples from 2 models that were all NaN" at training time). A LOMO
    run never meets them because its validation set is ``config['models_train']``,
    which the training loader already filtered; the **in-sample** run has
    ``models_train=None``, so it falls back to raw discovery and gets them back.

    Without this, the in-sample selection scored ``nan`` at all 41 epochs, and
    because ``min()`` compares NaN as False it returned the FIRST epoch --
    reporting epoch 0, an untrained network, as the validation winner.
    Dropping is what the rest of this toolchain does with those files; doing it
    silently is not, so the names are logged and stamped into the output.
    """
    dropped = sorted(k for k, v in per_model.items()
                     if not np.isfinite(v['deep']) or not np.isfinite(v['full']))
    if not dropped:
        return slices, []
    logging.warning(
        f'  ⚠️ dropping {len(dropped)} validation model(s) with non-finite '
        f'loss: {", ".join(dropped)} -- their source fields are all-NaN '
        f'(discovery globs filenames, not contents). {save_dir.name}')
    kept = [(n, ds) for n, ds in slices if n not in set(dropped)]
    if not kept:
        raise RuntimeError(
            f'{save_dir.name}: EVERY validation model scored non-finite, so '
            f'there is nothing to select on. That is a data problem, not a '
            f'training one -- check the source files for {sorted(per_model)}.')
    return kept, dropped


def select(save_dir, device, epochs=None, max_models=None):
    """Score every checkpoint of one run; return the curve and the winner."""
    save_dir = Path(save_dir)
    have = unet_mse.checkpoint_epochs(save_dir)
    if not have:
        raise FileNotFoundError(f'no checkpoints in {save_dir}')
    epochs = have if epochs is None else [e for e in epochs if e in have]
    if not epochs:
        raise ValueError(
            f'--epochs selected none of the checkpoints in {save_dir.name} '
            f'(has {have})')

    # one handle, then weights are swapped per epoch
    run = unet_mse.load_run(save_dir, device, epoch=epochs[0], _compile=False)
    split = run.config['train_test_split']
    if split not in VAL_YEARS:
        raise ValueError(f'no validation window for split {split!r}')
    window = VAL_YEARS[split]
    train_models = run.config.get('models_train')
    if not train_models:
        # the in-sample run trains on every model; its held-out axis is years
        train_models = [m.model for m in ec.get_models(run.config)]
    if max_models is not None:
        train_models = train_models[:max_models]
    logging.info(f'{save_dir.name}: split={split}, validating on years '
                 f'{window[0]}-{window[1]} of {len(train_models)} held-in '
                 f'models, {len(epochs)} checkpoints')

    slices = validation_slices(run, train_models, window)
    cfg = ec.SampleCfg(n_samples=1, apply_ema=True)
    curve, dropped = [], []
    for i, ep in enumerate(epochs):
        load_checkpoint_into(run, save_dir, ep, device)
        deep, full, per_model = score_epoch(run, slices, cfg)
        if i == 0:
            # checked ONCE, on the first epoch: an all-NaN source file is a
            # property of the data, not of the weights, so it cannot appear
            # later, and re-deriving it per epoch would cost 41 scans
            slices, dropped = drop_nan_models(slices, per_model, save_dir)
            if dropped:
                deep, full, per_model = score_epoch(run, slices, cfg)
                train_models = [m for m in train_models if m not in dropped]
        curve.append(dict(epoch=int(ep), val_mse=deep, val_mse_full=full))
        logging.info(f'  epoch {ep:>4}: val MSE {deep:.6e} (deep) '
                     f'{full:.6e} (full domain)')
    best = min(curve, key=lambda r: r['val_mse'])
    # ⚠️ min() over NaN returns the FIRST element, because every comparison
    # with NaN is False -- which is how a run once reported epoch 0, an
    # untrained network, as its validation winner. Refuse instead.
    if not np.isfinite(best['val_mse']):
        raise RuntimeError(
            f'{save_dir.name}: the whole validation curve is non-finite, so '
            f'the "winner" would just be the first epoch scored. Nothing is '
            f'selectable here -- see drop_nan_models.')
    logging.info(f'  selected epoch {best["epoch"]} '
                 f'(val MSE {best["val_mse"]:.6e})')
    return dict(
        save_dir=str(save_dir), split=split, selected_epoch=best['epoch'],
        val_years=list(window), n_val_models=len(train_models),
        n_val_maps=len(slices), models=list(train_models),
        dropped_models=dropped,
        criterion='area-weighted masked MSE of the predicted field, mol/m^3, '
                  'shallow cells masked as in ec.mape',
        averaging=AVERAGING, n_samples=cfg.n_samples,
        n_samples_note='deterministic: n draws are identical, so 1 is the mean',
        curve=curve)


def write(result):
    path = Path(result['save_dir']) / OUT_NAME
    path.write_text(json.dumps(result, indent=2))
    logging.info(f'  wrote {path}')
    return path


def summary(kind=unet_mse.KIND, pin_cache=None):
    """Fleet table: selected epoch per run, against the pinned comparison epoch.

    ⚠️ ``pin_cache`` defaults to None and is resolved INSIDE, not to
    ``evaluate.PIN_CACHE`` in the signature: a default is evaluated at def-time,
    and ``evaluate`` imports this module back, so the attribute may not exist
    yet depending on which module was imported first.
    """
    pins = evaluate.read_pins(pin_cache or evaluate.PIN_CACHE)
    rows = []
    for name in pins['models'] + ['insample']:
        sd = (evaluate.baseline_save_dir(kind, name) if name != 'insample'
              else naming.BASELINE_ROOT / f'baseline-{kind}' /
              ec.run_save_dir(evaluate.cell_config(evaluate.CELL),
                              train_test_split='insample').name)
        path = Path(sd) / OUT_NAME
        if not path.is_file():
            rows.append((name, None, None, 'not selected yet'))
            continue
        r = json.loads(path.read_text())
        at_pin = [c for c in r['curve'] if c['epoch'] == pins['epoch']]
        rows.append((name, r['selected_epoch'],
                     r['curve'][[c['epoch'] for c in r['curve']].index(
                         r['selected_epoch'])]['val_mse'],
                     f'pinned {pins["epoch"]}: '
                     f'{at_pin[0]["val_mse"]:.6e}' if at_pin else
                     f'no epoch {pins["epoch"]} in the curve'))
    print(f'{"run":24s} {"selected":>8s} {"val MSE":>12s}  vs the pinned epoch')
    for name, ep, mse, note in rows:
        ep_s = '-' if ep is None else str(ep)
        mse_s = '-' if mse is None else f'{mse:.6e}'
        print(f'{name:24s} {ep_s:>8s} {mse_s:>12s}  {note}')
    sel = [ep for _, ep, _, _ in rows if ep is not None]
    if sel:
        print(f'\n{len(sel)}/{len(rows)} selected; epochs '
              f'{min(sel)}-{max(sel)}, pinned comparison epoch {pins["epoch"]}')
    return rows


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default=None, metavar='NAME',
                   help='the LOMO run holding out this model')
    p.add_argument('--insample', action='store_true',
                   help='the in-sample run instead of a LOMO one')
    p.add_argument('--summary', action='store_true',
                   help='print the fleet table and exit (no GPU)')
    p.add_argument('--list', action='store_true',
                   help='alias for --summary')
    p.add_argument('--epochs', type=int, nargs='+', default=None,
                   help='score only these checkpoints (default: all on disk)')
    p.add_argument('--max-models', type=int, default=None,
                   help='cap held-in models -- smoke tests only')
    p.add_argument('--cpu', action='store_true', help='force CPU')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.summary or args.list:
        summary()
        return

    if args.insample == bool(args.model):
        raise SystemExit('give exactly one of --model NAME or --insample')

    if args.insample:
        name = ec.run_save_dir(evaluate.cell_config(evaluate.CELL),
                               train_test_split='insample').name
        save_dir = naming.BASELINE_ROOT / f'baseline-{unet_mse.KIND}' / name
    else:
        save_dir = evaluate.baseline_save_dir(unet_mse.KIND, args.model)
    if not Path(save_dir).is_dir():
        raise SystemExit(f'no run at {save_dir}')

    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit('needs a GPU (or --cpu, which is slow for 41 epochs)')
    device = torch.device('cpu' if args.cpu else 'cuda')
    write(select(save_dir, device, epochs=args.epochs,
                 max_models=args.max_models))


if __name__ == '__main__':
    main()
