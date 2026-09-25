"""Save-dir and cache naming for the baselines.

Extracted from general/baselines/naming.py.

Dependency-free (stdlib + ``paths`` only), like ``diffusion/naming.py`` and for
the same reason: a launcher must be able to compute the exact directory a
training job will write to, without importing torch.

Baselines live under their own root, ``paths.BASELINE_ROOT/baseline-<kind>/``,
not under the diffusion checkpoint root, so that anything scanning that root by
directory name never counts a baseline as a diffusion run.

The directory *name* is deliberately NOT re-spelled here: it is whatever
``diffusion.naming.save_dir_from_config`` produces, re-rooted. A baseline run
and the diffusion run it is compared against therefore share a directory name,
which is what makes pairing them a lookup.
"""
from pathlib import Path

import paths
from diffusion import naming as dnaming


#: Root for every baseline checkpoint. NOT the diffusion checkpoints root --
#: see the module docstring.
BASELINE_ROOT = paths.BASELINE_ROOT

#: The baselines. 'unet-mse' is the deterministic U-Net, 'qrf' the quantile
#: regression forest, and 'pointwise-linear' / 'climatology' the per-cell
#: linear rung below both CNNs and the no-predictor floor below that.
#: Kebab-case so the tag reads cleanly in a directory name.
#:
#: The last two are fitted in CLOSED FORM -- no ``Trainer``, no epochs, no
#: selected hyperparameter -- so nothing that keys off an epoch applies to
#: them. ``climatology`` is the intercept-only NESTED submodel of
#: ``pointwise-linear`` and is read off the same normal equations; see
#: ``pointwise_linear.py``.
KINDS = ('unet-mse', 'qrf', 'pointwise-linear', 'climatology')


def _kind(config):
    kind = config.get('baseline')
    if kind not in KINDS:
        raise ValueError(
            f'config["baseline"] must be one of {KINDS}, got {kind!r} -- a '
            f'baseline config is the diffusion config dict plus this key')
    return kind


def save_dir_from_config(config):
    """Checkpoint directory for a baseline run.

    The same encoding as the diffusion run this will be compared against
    (target, predictors, experiment, resolution, field type, split), under the
    baseline root and tagged by kind. Two runs that differ only in being a
    baseline or the diffusion model therefore share a directory *name*, which
    is what makes pairing them a lookup rather than a translation table.
    """
    name = Path(dnaming.save_dir_from_config(config)).name
    return str(BASELINE_ROOT / f'baseline-{_kind(config)}' / name)


def cache_dir(kind):
    """Where the evaluation caches for one baseline go."""
    if kind not in KINDS:
        raise ValueError(f'unknown baseline {kind!r}; have {KINDS}')
    return BASELINE_ROOT / f'baseline-{kind}' / '_eval_cache'


def shared_cache_dir(config):
    """The RAW preproc cache -- deliberately the diffusion one, unchanged.

    It is keyed by data identity only (predictors, target, experiment,
    resolution, field type), never by the model or the split, so a baseline
    reading it gets byte-identical inputs to the diffusion run and pays no
    second decode. That identity is the whole point: a baseline trained on even
    slightly different data would not be a baseline.
    """
    return dnaming.shared_cache_dir(config)
