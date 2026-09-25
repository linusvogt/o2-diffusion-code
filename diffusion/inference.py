"""Model loading and area-weighted means for inference/evaluation.

Extracted from ml_framework/diffusion_model/util.py.
"""
from diffusion.model import SimpleUNetCond, DDPM, EMA

import json
import torch
import re
from pathlib import Path
import numpy as np
import xarray as xr

import paths


def global_mean(da):
    """Mean in lat-lon dimensions weighted by grid cell area."""
    area = xr.open_dataset(str(paths.GRIDAREA)).cell_area
    out = da.weighted(area.fillna(0.)).mean(['lat', 'lon'])
    return out

def basin_mean(da, basin):
    """Mean in lat-lon dimensions weighted by grid cell area, in speficic ocean basin."""
    if basin == 'global':
        return global_mean(da)

    # grid cell area
    area = xr.open_dataset(str(paths.GRIDAREA)).cell_area

    # region mask
    assert basin in ['atlantic', 'pacific', 'indian', 'arctic', 'southern'], basin
    ds_reccap = xr.open_dataset(str(paths.RECCAP2_MASK))
    
    out = da.where(ds_reccap[basin] > 0)
    out = out.weighted(area.fillna(0.)).mean(['lat', 'lon'])
    return out


def load_config(path):
    """Load diffusion model config dict.

    Params:
        path (str): checkpoint directory, e.g.
            str(paths.CHECKPOINTS / 'target-o2_pred-so-thetao_1pctCO2_res-annual_fields-depthint_split-insample_full')
    """
    config_file = Path(path) / 'config.json'
    with open(config_file, "r") as f:
        config = json.load(f)
        
    config['save_dir'] = path
    assert Path(config['save_dir']).exists()
    
    return config


_CKPT_EPOCH_RE = re.compile(r'ckpt_epoch(\d+)\.pt$')


def last_checkpoint(save_dir):
    """Highest-EPOCH checkpoint in ``save_dir``.

    Sorts by parsed epoch number, not by filename (``ckpt_epoch1000.pt`` would
    otherwise sort directly after ``ckpt_epoch100.pt``). Reached only via
    ``load_model(epoch=None)``.
    """
    best, best_ep = None, -1
    for f in Path(save_dir).glob('ckpt_epoch*.pt'):
        m = _CKPT_EPOCH_RE.search(f.name)
        if m and int(m.group(1)) > best_ep:
            best, best_ep = f, int(m.group(1))
    if best is None:
        raise FileNotFoundError(
            f'no ckpt_epoch*.pt in {save_dir} -- nothing to load')
    return best


# ---------------------------------------------------------------------------
# Land-fill convention (train/eval consistency)
# ---------------------------------------------------------------------------
# Two normalization orders exist for LAND (NaN) cells of the conditioning
# channels:
#   OLD ('zero'):    normalize THEN fill land 0
#       -> (x-mean)/std, then land NaN -> 0            => land value = 0
#   NEW ('meanstd'): fill raw 0 THEN normalize
#       -> nan_to_num(raw, 0), then (x-mean)/std        => land value = -mean/std
# Sampling must feed the SAME land values the model was trained with, else the
# conditioning is off-distribution and the generator collapses to flat fields.
# As a fallback the convention is keyed on each checkpoint's saved 'timestamp'.
_LANDFILL_BOUNDARY = '2026-07-03'


def land_fill_mode(timestamp):
    """Return the land-fill convention for a checkpoint's saved ``timestamp``.

    'zero'    -> OLD convention (land cells = 0 in the normalized conditioning)
    'meanstd' -> NEW convention (land cells = -mean/std)
    A missing/unparseable timestamp defaults to 'zero'.
    """
    ts = str(timestamp)[:10] if timestamp else ''
    return 'meanstd' if ts >= _LANDFILL_BOUNDARY else 'zero'


def resolve_land_fill(ckpt, save_dir=None):
    """Land-fill convention for a checkpoint, preferring a RECORDED fact over the
    fragile timestamp guess. Resolution order:

      1. ``ckpt['land_fill']``  -- stamped at train time by the Trainer (new runs);
      2. a ``land_fill`` sidecar file in ``save_dir`` -- written for older runs
         that predate the checkpoint stamp;
      3. ``land_fill_mode(ckpt['timestamp'])`` -- the timestamp heuristic, used only
         when neither of the above exists (a warning is logged).
    """
    v = ckpt.get('land_fill') if isinstance(ckpt, dict) else None
    if v in ('zero', 'meanstd'):
        return v
    if save_dir is not None:
        p = Path(save_dir) / 'land_fill'
        if p.is_file():
            v = p.read_text().strip()
            if v in ('zero', 'meanstd'):
                return v
    conv = land_fill_mode(ckpt.get('timestamp') if isinstance(ckpt, dict) else None)
    import warnings
    warnings.warn(
        f'land_fill not stamped in checkpoint or sidecar; inferring {conv!r} from '
        f'timestamp (fragile -- write a land_fill sidecar)')
    return conv


def normalize_channel(da, mean, std, land_fill='zero'):
    """Normalize one (lat, lon) conditioning channel to the model-input convention.

    Returns a float32 numpy array. ``land_fill`` selects how NaN (land) cells are
    handled so that evaluation matches how the checkpoint was TRAINED:
      'zero'    : (x-mean)/std, then land -> 0            (OLD training order)
      'meanstd' : raw NaN -> 0, then (x-mean)/std => land = -mean/std (NEW order)
    Ocean cells are identical either way.
    """
    if land_fill == 'meanstd':
        raw = np.nan_to_num(da.values.astype(np.float32), nan=0.0)
        return ((raw - mean) / std).astype(np.float32)
    return ((da - mean) / std).fillna(0.).values.astype(np.float32)


def load_model(
    config: dict,
    device: torch.device = None,
    #load_optimizer: bool = False,
    epoch=None,
    _compile=True,
    log=True,
):
    """
    Loads the SimpleUNetCond model + DDPM wrapper (from last available checkpoint).

    Parameters
    ----------
    config : dict
        Training configuration dictionary used to create the model.
    device : torch.device, optional
        Device to load onto. Defaults to CUDA if available.
    load_optimizer : bool
        If True, also returns the optimizer loaded from checkpoint.
    epoch (int, default=None): specific epoch to be loaded.
        if None, load last available checkpoint (highest epoch)

    Returns
    -------
    ddpm : DDPM
        The DDPM wrapper with loaded model inside.
    optimizer or None
        The optimizer (if load_optimizer=True)
    start_epoch : int
        Epoch number to resume training from.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize Unet
    in_ch = len(config["predictors"]) + 1
    model = SimpleUNetCond(
        in_ch=in_ch,
        base_ch=config["base_ch"],
        ch_mults=config["ch_mults"],
        time_emb_dim=config["time_emb_dim"],
        num_res_blocks=config["num_res_blocks"],
    )
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)  # memory format for speedup
    if _compile:
        model = torch.compile(model)  # JIT compile model for speedup

    # initialize diffusion model
    ddpm = DDPM(
        model,
        device=device,
        num_timesteps=config["num_timesteps"],
        penalize_non_negative=config['penalize_non_negative'],
        integral_loss=config['integral_loss'],
        w_integral=config['w_integral'],
        schedule=config['schedule'],
        prediction=config['prediction'],
    )

    # Load the checkpoint
    if epoch is None:
        checkpoint = last_checkpoint(config['save_dir'])
    else:
        checkpoint = Path(config['save_dir']) / f'ckpt_epoch{str(epoch).zfill(3)}.pt'
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Checkpoints are saved from a torch.compile'd model, so their state-dict keys
    # are prefixed "_orig_mod.". When loading WITHOUT compile (e.g. a fast eval
    # that skips the per-model compile cost), strip that prefix so the keys match
    # the eager module.
    def _strip(sd):
        if _compile:
            return sd
        pfx = '_orig_mod.'
        return {(k[len(pfx):] if k.startswith(pfx) else k): v
                for k, v in sd.items()}

    ddpm.model.load_state_dict(_strip(ckpt["model_state"]))

    # load EMA and use correct device + dtype
    ema = EMA(ddpm.model, decay=ckpt["ema_decay"])
    msd = dict(ddpm.model.named_parameters())
    ema.shadow = {
        k: v.to(device=msd[k].device, dtype=msd[k].dtype)
        for k, v in _strip(ckpt["ema_state"]).items()
    }

    if log:
        print(f'Loaded {str(checkpoint)}')

    ddpm.model.to(device)
    ddpm.model.eval()

    scaling = {'means': ckpt['means'], 'stds': ckpt['stds'],
               'land_fill': resolve_land_fill(ckpt, config.get('save_dir'))}

    return ddpm, scaling, ema
