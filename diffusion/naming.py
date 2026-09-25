"""Checkpoint- and cache-directory naming for the diffusion runs.

Dependency-free (stdlib plus ``paths``) so that any tooling can compute the
exact directory names a training job writes to.

Extracted from general/save_dir_naming.py.
"""
from pathlib import Path

import paths


def save_dir_from_config(config):
    """Checkpoint directory name, extending the original scheme with
    resolution and field-type tags.

    e.g. target-o2_pred-so-thetao_1pctCO2_res-monthly_fields-density-sigma1_split-insample_full
    """
    target, predictors = config['target'], config['predictors']
    part_target = f'target-{target}'
    part_pred = '-'.join(['pred'] + sorted(predictors))
    part_exp = '-'.join(config['experiments'])
    part_res = f'res-{config["resolution"]}'

    if config['field_type'] == 'density':
        dv = config.get('density_var', 'sigma_1').replace('_', '')
        part_fields = f'fields-density-{dv}'
        levels = config.get('density_levels', 'all')
        if levels not in (None, 'all'):
            part_fields += f'-lev{"-".join(str(i) for i in levels)}'
    else:
        part_fields = 'fields-depthint'

    part_split = f'split-{config["train_test_split"]}'
    if config['train_test_split'] == 'model':
        if len(config['models_test']) == 1:
            part_split += f'-{config["models_test"][0]}'
        else:
            raise NotImplementedError(f'{len(config["models_test"])=}')

    part_anom = '_full' if not config['anom'] else ''
    part_nonneg = '_nonneg' if config['penalize_non_negative'] else ''
    part_integral = (f'_integral-{config["w_integral"]:.2e}'
                     if config['integral_loss'] else '')

    # Ensemble-composition tag. Appended last and emitted only when the key is
    # set and truthy, so uncapped runs keep their name and a capped run lands in
    # its own directory rather than resuming the uncapped one.
    max_members = config.get('max_members_per_model')
    part_members = f'_mem{max_members}' if max_members else ''

    p = Path(paths.CHECKPOINTS)
    p /= (f'{part_target}_{part_pred}_{part_exp}_{part_res}_{part_fields}_'
          f'{part_split}{part_anom}{part_nonneg}{part_integral}{part_members}')
    return str(p)


def shared_cache_dir(config):
    """Directory of the shared RAW preproc cache for a run.

    Keyed by the DATA identity only — target, base predictors, experiment,
    resolution, field-type/density-var, and anomaly flag — and deliberately
    NOT by the train/test split or loss terms. This is what lets the insample
    run and every leave-one-model-out run over the same data share one raw
    cache (TracerDataset computes its own split-specific stats on top). See
    TracerDataset for how the per-(channel, member) files are used.

    Deliberately NOT keyed by the predictor list either, so the five predictor
    sets of the sweep share one directory instead of each re-storing the
    channels they have in common (thetao_dlev*/so_dlev*/o2 are the bulk of
    every set).

    This is safe because a file is identified by <channel>/<member>_<fp>.npy,
    where fp hashes the member identity AND its exact sample time/exp coords:
    * same content -> same name. Channel data for a member at given times does
      not depend on what OTHER channels the run loaded. Density levels come
      from a per-model file on disk (data_loading.density_outpath) and
      isel(lev=i) picks index i from it, so <var>_dlev<i> is not run-dependent.
    * different content -> different name. Adding a predictor adds it to
      load_training_data's all-NaN drop check, which can drop more samples;
      that changes the member's coords, hence fp, hence the filename -- a clean
      miss, never a stale hit.

    Deliberately NOT keyed by ``max_members_per_model`` either, for the same
    reason: capping members drops WHOLE members (data_loading.subsample_members)
    and never samples within one, so a kept member's coords -- hence its
    fingerprint, hence its filename -- are what the uncapped run wrote. The
    capped run reads a subset of the same files. Keying by the cap would make
    every composition-sensitivity run re-decode a cache it already has.

    e.g. target-o2_1pctCO2_res-monthly_fields-density-sigma1_full
    """
    part_target = f'target-{config["target"]}'
    part_exp = '-'.join(config['experiments'])
    part_res = f'res-{config["resolution"]}'

    if config['field_type'] == 'density':
        dv = config.get('density_var', 'sigma_1').replace('_', '')
        part_fields = f'fields-density-{dv}'
        levels = config.get('density_levels', 'all')
        if levels not in (None, 'all'):
            part_fields += f'-lev{"-".join(str(i) for i in levels)}'
    else:
        part_fields = 'fields-depthint'

    part_anom = '_full' if not config['anom'] else '_anom'

    p = Path(paths.PREPROC_CACHE)
    p /= f'{part_target}_{part_exp}_{part_res}_{part_fields}{part_anom}'
    return str(p)
