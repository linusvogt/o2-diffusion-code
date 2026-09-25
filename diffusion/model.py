"""Conditional DDPM: dataset wrapper, UNet backbone, diffusion wrapper, EMA and
training loop.

Extracted from ml_framework/diffusion_model/conditional_diffusion.py.
"""
import math
from typing import List, Optional, Tuple, Sequence

import numpy as np
import xarray as xr
from pathlib import Path
import hashlib
import json
import logging
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import paths as _paths


class TracerDataset(Dataset):
    """
    Wrap an xarray.Dataset with dims (sample, lat, lon) and multiple variables.

    Returns dicts with keys:
     - 'y': (1,H,W) normalized target
     - 'cond': (C,H,W) normalized predictors
     - 'mask': (1,H,W) binary float mask with 1 for ocean, 0 for land (same for all samples)

    Preproc cache (``cache_dir`` given)
    -----------------------------------
    When ``cache_dir`` is provided the *raw* (un-normalized) per-channel data is
    cached to disk as one float32 file per (channel, model-member), and each
    sample is normalized on the fly in ``__getitem__`` using this run's own
    train-split stats. Because the raw data is split-independent, one shared
    ``cache_dir`` (keyed by the DATA identity, not the split) serves the
    insample run and every leave-one-model-out run over the same
    predictors/experiment/resolution/field-type — the per-member files are
    populated incrementally and reused across runs, so a full LOMO sweep stores
    the raw data ~once instead of once per held-out model.

    Correctness notes:
      * Stats (means/stds) are computed per run over ITS OWN train members, so
        the held-out model never leaks into normalization.
      * Raw is stored float32 (NOT float16): these are depth-integrated fields
        whose raw magnitudes (e.g. salinity ×2000 m) overflow float16's 65504
        limit; only the O(1) normalized values were float16-safe.
      * Each per-member file name embeds a fingerprint of that member's
        time/exp coords, so changed source data yields a new file rather than a
        stale hit; writes are per-PID-tmp + atomic rename, safe for the parallel
        sweep (many jobs may write the same member concurrently).

    With ``cache_dir=None`` the in-RAM path is used (normalized float16
    buffers).
    """

    # subdir holding raw shared caches, keyed by data identity (see
    # ``shared_cache_dir`` in naming.py); float32 raw per-channel files.
    _RAW_DTYPE = np.float32

    def __init__(self, ds: xr.Dataset, predictors: List[str], target: str,
                 dtype=np.float32, cache_dir=None):
        """Initialize torch Dataset from stacked xarray Dataset.

        Args:
            ds (xr.Dataset): dataset with dims (sample, lat, lon) and several variables
                (both predictors and target)
            predictors (list of str): predictor variables (to condition on in DDPM)
            target (str): target variable
            dtype: default=float32
            cache_dir (str or Path, optional): shared raw-cache directory keyed
                by the DATA identity (NOT the split). If given, raw per-channel
                data is cached here and reused across all splits over the same
                data; normalization happens on the fly with this run's stats.
                Falls back to in-RAM computation if caching fails.
        """

        # make sure all necessary variables are in dataset
        assert target in ds.variables
        for p in predictors:
            assert p in ds.variables
        assert target not in predictors

        # list of all variables
        variables = predictors + [target]

        # check ds dimensions and transpose to correct order
        dims_list = [dim for dim in ds.dims]
        assert set(dims_list) == {'sample', 'lat', 'lon'}, f'{dims_list=}'
        ds = ds.transpose('sample', 'lat', 'lon')

        self.predictors = predictors
        self.target = target
        self.dtype = dtype
        # Land-fill convention this dataset normalizes with. BOTH build paths
        # (_build_inram pass 2 and _norm_onfly) fill raw NaN->0 THEN normalize, so
        # land = -mean/std ('meanstd'). Stamped into each checkpoint by the Trainer
        # so eval reads a recorded fact instead of inferring it from a timestamp
        # (see inference.resolve_land_fill / land_fill_mode).
        self.land_fill = 'meanstd'

        # ocean mask from first sample (True on ocean, False on land)
        self.mask = np.isfinite(ds[target].isel(sample=0).values)

        n_samples = len(ds.sample)
        H, W = self.mask.shape
        var_idx = {v: j for j, v in enumerate(variables)}
        n_vars = len(variables)
        self.n_samples = n_samples

        if cache_dir is not None:
            ok = self._build_raw_shared(
                ds, variables, var_idx, n_vars, n_samples, H, W, cache_dir)
            if not ok:  # any failure → fall back to the in-RAM path
                logging.warning('TracerDataset: raw-cache build failed; '
                                'falling back to in-RAM preproc')
                self._build_inram(ds, variables, var_idx, n_vars,
                                  n_samples, H, W)
        else:
            self._build_inram(ds, variables, var_idx, n_vars,
                              n_samples, H, W)

        self.mask_tensor = torch.from_numpy(self.mask.astype(np.float32))[None, ...]  # (1,H,W)

    # ------------------------------------------------------------------
    # member / channel layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _channel_groups(variables):
        """Group channels by source file prefix (thetao_dlev0..9 → 'thetao')."""
        groups: dict = {}
        for v in variables:
            prefix = v.split('_dlev')[0] if '_dlev' in v else v
            groups.setdefault(prefix, []).append(v)
        return groups

    @staticmethod
    def _member_slices(ds):
        """Contiguous per-(model,member) sample slices. Stacking orders members
        first so all time steps of a member are contiguous in ds.sample."""
        from itertools import groupby as _groupby
        model_coord = ds['model'].values
        slices, i = [], 0
        for _, grp in _groupby(model_coord,
                               key=lambda m: (getattr(m, 'center', ''),
                                              getattr(m, 'model', str(m)),
                                              getattr(m, 'member', ''))):
            cnt = sum(1 for _ in grp)
            slices.append(slice(i, i + cnt))
            i += cnt
        return slices

    @staticmethod
    def _member_key_fp(ds, sl):
        """(filename-safe member key, fingerprint) for a member sample slice.

        The fingerprint hashes the member's identity + its exact sample time/exp
        coords, so a change in the source data (e.g. different NaN-drop, new
        timesteps) produces a different cache-file name instead of a stale hit.
        """
        m = ds['model'].values[sl.start]
        fr = getattr(m, 'fname_repr', None)
        if callable(fr):
            try:
                key = str(fr())
            except Exception:
                key = str(m)
        else:
            key = str(m)
        key = ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in key)

        h = hashlib.sha256()
        h.update(key.encode())
        h.update(np.asarray([sl.stop - sl.start], dtype=np.int64).tobytes())
        for coord in ('year', 'time', 'exp'):
            if coord in ds.coords:
                try:
                    vals = np.asarray(ds[coord].values[sl])
                    h.update(np.ascontiguousarray(vals.astype('S64')).tobytes())
                except Exception:
                    pass
        return key, h.hexdigest()[:12]

    def _channel_path(self, cache_dir, channel, member_key, fp):
        return Path(cache_dir) / channel / f'{member_key}_{fp}.npy'

    @staticmethod
    def _member_stats_path(cache_dir, member_key, fp):
        """Sidecar holding one member's PARTIAL area-weighted sums.

        Keyed exactly like the .npy files -- (member, fingerprint) -- and so
        split-INDEPENDENT: the sums are per member, and a run's stats are just
        the sum over the members it trains on. Every leave-one-out split
        therefore reuses the same sidecars while still averaging only its own
        members, so the held-out model stays out of the normalization. (Keying
        the stats by the whole averaging set instead would give every split a
        different key and share nothing.)
        """
        return Path(cache_dir) / '_stats' / f'{member_key}_{fp}.json'

    @staticmethod
    def _load_member_stats(path, grp_vars):
        """{channel: (sum_wx, sum_wx2, sum_wn)} for grp_vars, or None.

        None whenever the file is missing, corrupt, or lacks a channel, so the
        caller recomputes; a bad sidecar can only cost time, never correctness.
        """
        try:
            with open(path) as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            return None
        out = {}
        for v in grp_vars:
            triple = blob.get(v)
            if not triple or len(triple) != 3:
                return None
            out[v] = tuple(float(x) for x in triple)
        return out

    @staticmethod
    def _update_member_stats(path, new_triples):
        """Merge {channel: (sum_wx, sum_wx2, sum_wn)} into a member's sidecar.

        Read-modify-write with per-PID tmp + atomic rename. Concurrent writers
        can interleave, but every value is a deterministic function of the file
        it summarizes, so the worst case is a lost update -- the next reader
        recomputes that channel. Never fatal: the caller already has the sums.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(path) as fh:
                    blob = json.load(fh)
            except (OSError, ValueError):
                blob = {}
            blob.update({v: list(t) for v, t in new_triples.items()})
            tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
            with open(tmp, 'w') as fh:
                json.dump(blob, fh)
            os.replace(tmp, path)
        except Exception as e:
            logging.warning(f'  could not write stats sidecar {path.name} ({e})')

    def _write_channel(self, path, raw_f32):
        """Atomically write a (T,H,W) float32 raw member-channel file.

        Per-PID tmp + atomic replace so concurrent sweep jobs writing the same
        member never corrupt or half-read the file (content is identical).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.tmp.{os.getpid()}')
        try:
            with open(tmp, 'wb') as fh:
                np.save(fh, np.ascontiguousarray(raw_f32, dtype=self._RAW_DTYPE))
            os.replace(tmp, path)
        except BaseException:
            # A failed write (e.g. disk quota exceeded) must not leave
            # a partial .tmp file behind wasting space. Remove it and re-raise so
            # _build_raw_shared falls back to the in-RAM path.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # build paths
    # ------------------------------------------------------------------
    def _build_raw_shared(self, ds, variables, var_idx, n_vars,
                          n_samples, H, W, cache_dir):
        """Populate/reuse the shared raw cache and compute this run's stats.

        Returns True on success, False if anything went wrong (caller then
        falls back to the in-RAM path). Stores per-member raw memmap handles in
        ``self._raw`` for on-the-fly normalization in ``__getitem__``.
        """
        import time as _time
        try:
            da_area = xr.open_dataset(str(_paths.GRIDAREA))
            da_area = da_area.cell_area.fillna(0.).transpose('lat', 'lon')
            w = da_area.values.astype(np.float64)  # (H, W)

            groups = self._channel_groups(variables)
            slices = self._member_slices(ds)
            n_members = len(slices)

            # per-member raw handles: self._raw[channel][member_k] -> (T,H,W)
            self._raw = {v: [None] * n_members for v in variables}
            self._sample_member = np.empty(n_samples, dtype=np.int64)
            self._sample_local = np.empty(n_samples, dtype=np.int64)

            sum_wx  = np.zeros(n_vars, dtype=np.float64)
            sum_wx2 = np.zeros(n_vars, dtype=np.float64)
            sum_wn  = np.zeros(n_vars, dtype=np.float64)

            logging.info(
                f'TracerDataset: raw shared cache at {cache_dir} — {n_vars} '
                f'vars, {n_samples} samples, {n_members} members, '
                f'{len(groups)} source groups')
            t0 = _time.monotonic()
            n_hit = n_miss = 0

            # Stats sidecars. Even on a full cache hit the area-weighted stats
            # pass would re-read every byte; each member's contribution is
            # cached instead, so a warm run only reads .npy headers.
            member_ids = [self._member_key_fp(ds, sl) for sl in slices]
            n_stats_hit = n_stats_miss = 0

            for k, sl in enumerate(slices):
                self._sample_member[sl] = k
                self._sample_local[sl] = np.arange(sl.stop - sl.start)
                member_key, fp = member_ids[k]

                stats_path = self._member_stats_path(cache_dir, member_key, fp)

                for grp_vars in groups.values():
                    paths = {v: self._channel_path(cache_dir, v, member_key, fp)
                             for v in grp_vars}
                    if all(p.exists() for p in paths.values()):
                        # reuse cached raw (memmap) — no netCDF decode
                        cached = self._load_member_stats(stats_path, grp_vars)
                        for v in grp_vars:
                            arr = np.load(paths[v], mmap_mode='r')
                            self._raw[v][k] = arr
                        if cached is not None:
                            # header reads only: the arrays above are never
                            # materialized, which is the whole saving.
                            for v in grp_vars:
                                j = var_idx[v]
                                sx, sx2, sn = cached[v]
                                sum_wx[j] += sx
                                sum_wx2[j] += sx2
                                sum_wn[j] += sn
                            n_stats_hit += len(grp_vars)
                        else:
                            new = {}
                            for v in grp_vars:
                                new[v] = self._accum_stats(
                                    self._raw[v][k], w, var_idx[v],
                                    sum_wx, sum_wx2, sum_wn)
                            self._update_member_stats(stats_path, new)
                            n_stats_miss += len(grp_vars)
                        n_hit += len(grp_vars)
                    else:
                        # decode this member's group from netCDF, cache, reuse
                        chunk = ds[grp_vars].isel(sample=sl).compute()
                        new = {}
                        for v in grp_vars:
                            raw = chunk[v].values.astype(self._RAW_DTYPE)  # keep NaN
                            self._write_channel(paths[v], raw)
                            self._raw[v][k] = np.load(paths[v], mmap_mode='r')
                            new[v] = self._accum_stats(raw, w, var_idx[v],
                                                       sum_wx, sum_wx2, sum_wn)
                        self._update_member_stats(stats_path, new)
                        del chunk
                        n_miss += len(grp_vars)
                if (k + 1) % 5 == 0 or k == n_members - 1:
                    logging.info(f'  raw cache: {k+1}/{n_members} members, '
                                 f'{n_hit} channel-hits {n_miss} misses '
                                 f'({_time.monotonic()-t0:.1f}s)')

            _safe_wn = np.where(sum_wn > 0, sum_wn, 1.0)
            _mu  = sum_wx / _safe_wn
            _var = np.maximum(sum_wx2 / _safe_wn - _mu ** 2, 0.0)
            self.means = {v: float(_mu[j])           for v, j in var_idx.items()}
            self.stds  = {v: float(np.sqrt(_var[j])) for v, j in var_idx.items()}
            self._onfly = True
            logging.info(f'  raw cache done ({_time.monotonic()-t0:.1f}s); '
                         f'{n_hit} reused, {n_miss} decoded; '
                         f'stats {n_stats_hit} cached, {n_stats_miss} computed')
            return True
        except Exception as e:
            logging.warning(f'TracerDataset: raw shared cache error ({e})')
            return False

    @staticmethod
    def _accum_stats(arr, w, j, sum_wx, sum_wx2, sum_wn):
        """Accumulate area-weighted E[X], E[X²], E[finite] for one member-channel.

        Returns this member-channel's own (sum_wx, sum_wx2, sum_wn) so it can be
        cached in a sidecar -- the totals are a plain sum over members, which is
        what makes the sidecars reusable across splits.
        """
        a = np.asarray(arr, dtype=np.float64)  # (T,H,W); loads memmap if needed
        fin = np.isfinite(a)
        a_f = np.where(fin, a, 0.0)
        sx  = float((a_f * w).sum())
        sx2 = float((a_f ** 2 * w).sum())
        sn  = float((fin.astype(np.float64) * w).sum())
        sum_wx[j]  += sx
        sum_wx2[j] += sx2
        sum_wn[j]  += sn
        return sx, sx2, sn

    def _build_inram(self, ds, variables, var_idx, n_vars, n_samples, H, W):
        """In-RAM path: two passes → normalized float16 buffers (used when no
        cache_dir is given).
        """
        import time as _time

        da_area = xr.open_dataset(str(_paths.GRIDAREA))
        da_area = da_area.cell_area.fillna(0.).transpose('lat', 'lon')
        w = da_area.values.astype(np.float64)  # (H, W)

        groups = self._channel_groups(variables)
        slices = self._member_slices(ds)

        # --- Pass 1: accumulate weighted E[X] and E[X²] per member ---
        logging.info(f'TracerDataset: stats pass — {n_vars} vars, {n_samples} samples, '
                     f'{len(slices)} members, {len(groups)} source groups ...')
        t0 = _time.monotonic()
        sum_wx  = np.zeros(n_vars, dtype=np.float64)
        sum_wx2 = np.zeros(n_vars, dtype=np.float64)
        sum_wn  = np.zeros(n_vars, dtype=np.float64)

        for _k, _sl in enumerate(slices):
            for _grp_vars in groups.values():
                _chunk = ds[_grp_vars].isel(sample=_sl).compute()
                for v in _grp_vars:
                    j = var_idx[v]
                    arr = _chunk[v].values.astype(np.float64)
                    fin = np.isfinite(arr)
                    arr_f = np.where(fin, arr, 0.0)
                    sum_wx[j]  += (arr_f * w).sum()
                    sum_wx2[j] += (arr_f ** 2 * w).sum()
                    sum_wn[j]  += (fin.astype(np.float64) * w).sum()
                del _chunk
            if (_k + 1) % 5 == 0 or _k == len(slices) - 1:
                logging.info(f'  stats: {_k+1}/{len(slices)} members '
                             f'({_time.monotonic()-t0:.1f}s)')

        _safe_wn = np.where(sum_wn > 0, sum_wn, 1.0)
        _mu      = sum_wx / _safe_wn
        _var     = np.maximum(sum_wx2 / _safe_wn - _mu ** 2, 0.0)
        self.means = {v: float(_mu[j])             for v, j in var_idx.items()}
        self.stds  = {v: float(np.sqrt(_var[j]))   for v, j in var_idx.items()}
        logging.info(f'  stats done ({_time.monotonic()-t0:.1f}s)')

        # --- Pass 2: normalize per member → float16 preproc arrays ---
        logging.info(f'  preproc pass — allocating {n_vars} × {n_samples} × {H} × {W} '
                     f'float16 arrays ...')
        self._preproc = {v: np.empty((n_samples, H, W), dtype=np.float16)
                         for v in variables}

        t0 = _time.monotonic()
        for _k, _sl in enumerate(slices):
            for _grp_vars in groups.values():
                _chunk = ds[_grp_vars].isel(sample=_sl).compute()
                for v in _grp_vars:
                    _std = self.stds[v] if self.stds[v] > 0 else 1.0
                    arr = _chunk[v].values.astype(np.float32)
                    np.nan_to_num(arr, copy=False, nan=0.0)
                    arr -= self.means[v]
                    arr /= _std
                    self._preproc[v][_sl] = arr.astype(np.float16)
                del _chunk
            if (_k + 1) % 5 == 0 or _k == len(slices) - 1:
                logging.info(f'  preproc: {_k+1}/{len(slices)} members '
                             f'({_time.monotonic()-t0:.1f}s)')
        logging.info(f'  preproc done ({_time.monotonic()-t0:.1f}s)')
        self._onfly = False

    # ------------------------------------------------------------------
    def __len__(self):
        return self.n_samples

    def _norm_onfly(self, channel, k, lo):
        """Normalize one (1,H,W) raw slice on the fly (raw-cache path)."""
        a = self._raw[channel][k][lo:lo+1].astype(np.float32)  # copy (writable)
        np.nan_to_num(a, copy=False, nan=0.0)
        std = self.stds[channel] if self.stds[channel] > 0 else 1.0
        a -= self.means[channel]
        a /= std
        return a

    def __getitem__(self, idx):
        if self._onfly:
            k = int(self._sample_member[idx])
            lo = int(self._sample_local[idx])
            y = self._norm_onfly(self.target, k, lo)
            cond = np.concatenate(
                [self._norm_onfly(p, k, lo) for p in self.predictors], axis=0)
        else:
            # np.array(copy=True) so slices become writable owned arrays.
            y = np.array(self._preproc[self.target][idx:idx+1])  # (1,H,W)
            cond = np.concatenate(
                [self._preproc[p][idx:idx+1] for p in self.predictors], axis=0)
        return {
            "y": torch.from_numpy(y).float(),
            "cond": torch.from_numpy(cond).float(),
        }


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        timesteps: (B,) long or float
        returns: (B, dim)
        """
        device = timesteps.device
        half = self.dim // 2
        # Avoid div-by-zero if dim=1 (not a realistic setting here, but safe)
        if half <= 1:
            emb = timesteps.float()[:, None]
            return F.pad(emb, (0, self.dim - emb.shape[-1]))

        freq = math.log(10000.0) / (half - 1)
        freq = torch.exp(torch.arange(half, device=device) * -freq)  # (half,)
        t = timesteps.float()[:, None]                                # (B,1)
        emb = t * freq[None, :]                                       # (B,half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)               # (B,2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.act = nn.SiLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)

        # time embedding -> channel-wise bias (FiLM-like shift)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.gn1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.act(self.gn2(self.conv2(h)))
        return h + self.res_conv(x)


class SimpleUNetCond(nn.Module):
    """
    Conditional UNet backbone for DDPM-style diffusion on 2D fields.

    This network predicts a single target channel (e.g. noise ε or velocity v)
    from a multi-channel input consisting of the noisy target and conditioning
    variables. Time is injected through a sinusoidal embedding followed by an MLP
    and applied in every residual block as a channel-wise bias (FiLM-like shift).

    Architecture
    ------------
    - Encoder/decoder UNet with skip connections via **addition** (not concatenation).
    - Each resolution level contains `num_res_blocks` residual blocks.
    - Downsampling is performed with average pooling (factor 2).
    - Upsampling is bilinear followed by a 1×1 projection if channel counts differ.
    - A two-block bottleneck operates at the lowest resolution.
    - Final 1×1 convolution maps features to a single output channel.

    Parameters
    ----------
    in_ch : int
        Number of input channels. Typically:
        in_ch = 1 (noisy target) + N_cond (conditioning variables).
    base_ch : int, default=64
        Base channel width of the UNet. The actual channels at each level are
        `base_ch * ch_mults[level]`. Controls overall model capacity.
    ch_mults : Sequence[int], default=(1, 2, 4)
        Multiplicative factors defining channel width at each resolution level.
        Length of this sequence equals the number of UNet levels (depth).
    time_emb_dim : int, default=256
        Dimensionality of the timestep embedding. Used in all ResBlocks.
    num_res_blocks : int, default=2
        Number of residual blocks per resolution level (both encoder and decoder).
        Increasing this increases model depth and capacity.
    gn_groups : int, default=8
        Number of groups for GroupNorm in residual blocks.

    Notes
    -----
    - Skip connections use **elementwise addition**, so encoder and decoder
      feature maps must share the same channel count at each level.
    - Spatial size must be divisible by 2^(len(ch_mults)) for exact symmetry,
      though minor mismatches are handled via interpolation.
    - Output shape is (B, 1, H, W), matching a single predicted target channel.
    """
    def __init__(
        self,
        in_ch: int,
        base_ch: int = 64,
        ch_mults: Sequence[int] = (1, 2, 4),
        time_emb_dim: int = 256,
        num_res_blocks: int = 2,
        gn_groups: int = 8,
    ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError(f"num_res_blocks must be >= 1, got {num_res_blocks}")

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mults = tuple(ch_mults)
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks

        # time embedding: sinusoidal -> MLP
        self.time_sinu = SinusoidalPosEmb(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # helpers
        def make_res_stack(in_c: int, out_c: int) -> nn.ModuleList:
            blocks = nn.ModuleList()
            blocks.append(ResBlock(in_c, out_c, time_emb_dim, groups=gn_groups))
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(out_c, out_c, time_emb_dim, groups=gn_groups))
            return blocks

        # -------------------------
        # Encoder (down path)
        # -------------------------
        self.down_levels = nn.ModuleList()
        self.skip_channels: Tuple[int, ...] = tuple(base_ch * m for m in self.ch_mults)

        in_c = in_ch
        for out_c in self.skip_channels:
            level = nn.ModuleDict(
                {
                    "blocks": make_res_stack(in_c, out_c),
                    "downsample": nn.AvgPool2d(kernel_size=2),
                }
            )
            self.down_levels.append(level)
            in_c = out_c

        # -------------------------
        # Bottleneck
        # -------------------------
        self.mid = nn.ModuleList(
            [
                ResBlock(in_c, in_c, time_emb_dim, groups=gn_groups),
                ResBlock(in_c, in_c, time_emb_dim, groups=gn_groups),
            ]
        )

        # -------------------------
        # Decoder (up path)
        # -------------------------
        self.up_levels = nn.ModuleList()
        prev_c = in_c
        for skip_c in reversed(self.skip_channels):
            level = nn.ModuleDict(
                {
                    "upsample": nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    "proj": nn.Conv2d(prev_c, skip_c, kernel_size=1) if prev_c != skip_c else nn.Identity(),
                    "blocks": make_res_stack(skip_c, skip_c),
                }
            )
            self.up_levels.append(level)
            prev_c = skip_c

        self.final = nn.Conv2d(prev_c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_ch, H, W) where in_ch = 1 + n_cond
        timesteps: (B,) int tensor of time indices
        returns: (B, 1, H, W)
        """
        # time embedding
        t = self.time_mlp(self.time_sinu(timesteps))

        h = x
        skips = []

        # down path
        for level in self.down_levels:
            for block in level["blocks"]:
                h = block(h, t)
            skips.append(h)
            h = level["downsample"](h)

        # bottleneck
        for block in self.mid:
            h = block(h, t)

        # up path
        for level, skip in zip(self.up_levels, reversed(skips)):
            h = level["upsample"](h)
            h = level["proj"](h)

            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            h = h + skip
            for block in level["blocks"]:
                h = block(h, t)

        return self.final(h)



class EMA:
    """Exponential moving average helper class."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Load EMA weights into model (for sampling / validation)."""
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original training weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        

class DDPM:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        img_size: Tuple[int,int] = (180, 360),
        num_timesteps: int = 1000,
        penalize_non_negative: bool = False,
        integral_loss: bool = False,
        w_integral=0,
        schedule='linear',
        prediction='eps',
    ):
        """
        Denoising Diffusion Probabilistic Model (DDPM) implementation.

        Params:
            model: U-Net to be trained to predict noise
            device: torch.device (use GPU)
            img_size: shape of maps (360x180 for 1deg grid)
            num_timesteps: number of (de)noising time steps
            penalize_non_negative: not implemented; must be False
            integral_loss: whether to add a basic global integral loss term
        """
        if penalize_non_negative:
            raise NotImplementedError('Non-negative constraint not implemented')

        self.model = model
        self.device = device
        self.img_size = img_size
        self.num_timesteps = num_timesteps
        self.integral_loss = integral_loss
        self.w_integral = w_integral

        # grid cell area (for area weighted integral constraint)
        da_area = xr.open_dataset(str(_paths.GRIDAREA))
        da_area = da_area.cell_area.fillna(0.).transpose('lat', 'lon')
        self.area_tensor = torch.tensor(
            da_area.values, dtype=torch.float32).unsqueeze(0).to(self.device)

        # prediction: "eps" or "v"
        if prediction not in ("eps", "v"):
            raise ValueError(f"prediction must be 'eps' or 'v', got {prediction}")
        self.prediction = prediction

        # schedule: "linear" or "cos"
        self.schedule = schedule
        if schedule == "linear":
            beta_start = 1e-4
            beta_end = 2e-2
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        elif schedule == "cos":
            # Cosine schedule via alpha_bar(t)
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps
        
            alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar(0)=1
        
            # Convert alpha_bar -> betas
            betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
            self.betas = betas.clamp(1e-8, 0.999)
        else:
            raise ValueError(f"Unknown schedule '{schedule}', must be 'linear' or 'cos'")
        
        self.alphas = 1.0 - self.betas                                    # (T,)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)            # alpha_bar_t
        
        # alpha_bar_{t-1} with alpha_bar_{-1}=1
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0
        )
        
        # posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        ).clamp(min=1e-20)
        
        # useful precomputed quantities
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)


    def _extract(self, vec: torch.Tensor, t: torch.Tensor, x: torch.Tensor):
        """
        vec: (T,)
        t: (B,)
        returns vec[t] reshaped to (B,1,1,1) for broadcasting over x
        """
        return vec[t].view(-1, 1, 1, 1).to(x.device)
    
    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        # v = sqrt(alpha_bar)*eps - sqrt(1-alpha_bar)*x0
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0)
        return sqrt_ab * noise - sqrt_1mab * x0
    
    def v_to_eps_x0(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        Given x_t and v, return (eps, x0).
        eps = sqrt(1-alpha_bar)*x_t + sqrt(alpha_bar)*v
        x0  = sqrt(alpha_bar)*x_t - sqrt(1-alpha_bar)*v
        """
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        eps = sqrt_1mab * x_t + sqrt_ab * v
        x0 = sqrt_ab * x_t - sqrt_1mab * v
        return eps, x0

    
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion q(x_t | x_0)
        x0: (B,1,H,W)
        t: (B,) long
        noise: (B,1,H,W)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_alphas_cumprod[t].reshape(-1,1,1,1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1,1,1,1)
        return sqrt_acp * x0 + sqrt_om * noise

    def p_losses(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        Computes MSE between true noise (or v) and predicted noise (or v),
        plus optional global-mean loss on reconstructed x0.
    
        Returns:
            (loss_mse, loss_integral) where loss_integral is already weighted
            by self.w_integral when enabled, else None.
        """
        B = x0.shape[0]
        device = x0.device
    
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)
    
        model_in = torch.cat([xt, cond], dim=1)   # (B, 1+C, H, W)
        pred = self.model(model_in, t)            # (B, 1, H, W)
    
        # --- base DDPM loss target ---
        if self.prediction == "eps":
            target = noise
        elif self.prediction == "v":
            target = self.v_target(x0=x0, noise=noise, t=t)
        else:
            raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
        mse = (target - pred) ** 2  # (B,1,H,W)
    
        # --- build mask_b once (used by both losses) ---
        if mask is not None:
            if mask.shape[0] == 1:
                mask_b = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(device)
            else:
                mask_b = mask.to(device)
        else:
            mask_b = None
    
        # --- pixelwise MSE with optional masking ---
        if mask_b is not None:
            mse_w = mse * mask_b
            loss_mse = mse_w.sum() / mask_b.sum().clamp_min(1.0)
        else:
            loss_mse = mse.mean()
    
        # --- optional global-mean loss on reconstructed x0 ---
        if self.integral_loss:
            # 1) reconstruct predicted eps (needed to reconstruct x0_pred robustly)
            if self.prediction == "eps":
                pred_eps = pred
            elif self.prediction == "v":
                pred_eps, _x0_pred_direct = self.v_to_eps_x0(x_t=xt, v=pred, t=t)
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
            # 2) reconstruct x0_pred from (x_t, eps_pred)
            sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, xt)              # (B,1,1,1)
            sqrt_1mab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt)  # (B,1,1,1)
            x0_pred = (xt - sqrt_1mab * pred_eps) / sqrt_ab.clamp_min(1e-12)      # (B,1,H,W)
    
            # 3) compute (area-)weighted ocean mean error of x0
            #    weights w = mask * area  (if mask is None, treat as all-ones)
            if mask_b is None:
                # allow integral loss even without a mask
                mask_b = torch.ones_like(x0_pred, device=device)
    
            area = self.area_tensor.to(device)  # stored as (1,H,W)
            if area.ndim == 3:
                area = area.unsqueeze(1)        # -> (1,1,H,W)
            area_b = area.expand(B, 1, area.shape[-2], area.shape[-1])  # (B,1,H,W)
    
            w = (mask_b * area_b)  # (B,1,H,W)  area-weighted ocean weights
    
            # mean error per sample: μ_b = sum(w * (x0_pred-x0)) / sum(w)
            err_x0 = (x0_pred - x0)
            wsum = w.sum(dim=(2, 3), keepdim=False).clamp_min(1e-12)               # (B,1)
            mean_err = (w * err_x0).sum(dim=(2, 3), keepdim=False) / wsum          # (B,1)
    
            # 4) scale to match per-pixel MSE magnitude:
            #    Var(mean) ~ Var / N_eff  ->  mean^2 * N_eff ~ Var scale
            #    Use Kish effective sample size for weighted averages:
            #       N_eff = (sum w)^2 / sum(w^2)
            wsqsum = (w ** 2).sum(dim=(2, 3), keepdim=False).clamp_min(1e-12)      # (B,1)
            n_eff = (wsum ** 2) / wsqsum                                           # (B,1)
    
            loss_integral = (mean_err ** 2 * n_eff).mean()                         # scalar
            loss_integral = self.w_integral * loss_integral
        else:
            loss_integral = None
    
        return loss_mse, loss_integral
    

    @torch.no_grad()
    def sample(
        self, cond: torch.Tensor, mask: torch.Tensor,
        sampler: str = "ddpm", # or "ddim"
        # below: DDIM args
        steps=100,
        eta=0.1,
        timestep_spacing="uniform",  # or "quadratic"
        target_land=None,
    ):
        if sampler == 'ddpm':
            return self.sample_ddpm(cond, mask, target_land=target_land)
        elif sampler == 'ddim':
            return self.sample_ddim(
                cond, mask,
                steps=steps, eta=eta, timestep_spacing=timestep_spacing,
                target_land=target_land,
            )
        else:
            raise ValueError(f'{sampler=}')

    @torch.no_grad()
    def sample_ddpm(self, cond: torch.Tensor, mask: torch.Tensor,
                    target_land=None):
        """
        Sample x0 given cond.
        cond: (B, C, H, W)
        mask: (1,H,W) or (B,1,H,W)
        target_land: normalized land value of the TARGET channel (float). If None
            (default) land is clamped to 0 each step -- correct for checkpoints
            trained with land=0 (the 'zero' land-fill era). For 'meanstd'-era
            checkpoints the target trained with land = -mean/std, so land must be
            re-injected at each reverse step as its forward-diffused expected
            value sqrt(alpha_bar_{t_next})*target_land (RePaint-style known
            region); clamping to 0 instead is off-distribution and biases the
            ocean via the UNet's convolutions.
        """
        B = cond.shape[0]  # number of samples
        H, W = self.img_size
        x = torch.randn((B, 1, H, W), device=self.device)

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
            model_in = torch.cat([x, cond], dim=1)
            pred = self.model(model_in, t_batch)  # (B,1,H,W)
            
            if self.prediction == "eps":
                pred_eps = pred
                # broadcast scalars
                sqrt_ab = self.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
                sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
                x0_pred = (x - sqrt_1mab * pred_eps) / sqrt_ab
            
            elif self.prediction == "v":
                pred_v = pred
                # directly recover x0 (more stable) and eps (needed for mean formula if you prefer)
                pred_eps, x0_pred = self.v_to_eps_x0(x_t=x, v=pred_v, t=t_batch)
            
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")

            # posterior update
            if t > 0:
                beta_t = self.betas[t]
                alpha_cum_t = self.alphas_cumprod[t]
                alpha_cum_prev = self.alphas_cumprod_prev[t]
            
                coef1 = (beta_t * torch.sqrt(alpha_cum_prev)) / (1.0 - alpha_cum_t)
                coef2 = ((1.0 - alpha_cum_prev) * torch.sqrt(1.0 - beta_t)) / (1.0 - alpha_cum_t)
                mean = coef1 * x0_pred + coef2 * x
            
                var = self.posterior_variance[t]
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = x0_pred

            # enforce land in normalized space; x is now x_{t-1} (or x0 at t==0),
            # so the value fed to the model next is at timestep t_next = t-1.
            if mask is not None:
                if mask.shape[0] == 1:
                    m = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(x.device)
                else:
                    m = mask.to(x.device)
                if target_land is None:
                    x = x * m
                else:
                    scale = self.sqrt_alphas_cumprod[t - 1] if t > 0 else 1.0
                    x = x * m + (target_land * scale) * (1.0 - m)

        return x  # normalized-space prediction

    @torch.no_grad()
    def sample_ddim(
        self,
        cond: torch.Tensor,
        mask: torch.Tensor,
        steps: int = 100,
        eta: float = 0.0,
        timestep_spacing: str = "uniform",  # or "quadratic"
        target_land=None,
    ):
        """
        DDIM sampling (Song et al.) with optional stochasticity via `eta`.

        Usage:
        x = ddpm.sample_ddim(cond, mask, steps=50, eta=0.0)   # deterministic
        x = ddpm.sample_ddim(cond, mask, steps=50, eta=0.1)   # slightly stochastic
    
        Parameters
        ----------
        cond : torch.Tensor
            Conditioning tensor of shape (B, C, H, W).
        mask : torch.Tensor
            Land/sea mask of shape (1, H, W) or (B, 1, H, W). Applied each step to
            enforce land=0 in normalized space.
        steps : int
            Number of DDIM steps S (<< num_timesteps). Typical values: 25, 50, 100.
        eta : float
            DDIM stochasticity. eta=0 gives deterministic sampling. eta>0 adds noise.
            Typical values: 0.0, 0.1, 0.2.
        timestep_spacing : str
            "uniform" uses uniformly spaced timesteps.
            "quadratic" allocates more steps near t=0 (often slightly better).
    
        Returns
        -------
        x0 : torch.Tensor
            Generated samples in normalized space, shape (B, 1, H, W).
        """
        B = cond.shape[0]
        H, W = self.img_size
        device = self.device
    
        # initial noise
        x = torch.randn((B, 1, H, W), device=device)
    
        # build timestep schedule: a decreasing list of indices in [0, T-1]
        T = self.num_timesteps
        steps = int(steps)
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")
        if steps > T:
            raise ValueError(f"steps must be <= num_timesteps ({T}), got {steps}")
    
        if timestep_spacing == "uniform":
            t_seq = torch.linspace(T - 1, 0, steps, device=device)
        elif timestep_spacing == "quadratic":
            # more density near 0
            t_seq = (torch.linspace(0, 1, steps, device=device) ** 2) * (T - 1)
            t_seq = torch.flip(t_seq, dims=[0])
        else:
            raise ValueError("timestep_spacing must be 'uniform' or 'quadratic'")
    
        t_seq = t_seq.round().long().clamp(0, T - 1)
        # ensure strictly non-increasing and unique-ish (remove duplicates while preserving order)
        t_list = []
        last = None
        for ti in t_seq.tolist():
            if last is None or ti != last:
                t_list.append(ti)
                last = ti
        if t_list[-1] != 0:
            t_list.append(0)
    
        # prepare mask (broadcast once)
        if mask is not None:
            if mask.shape[0] == 1:
                m = mask.expand(B, 1, mask.shape[-2], mask.shape[-1]).to(device)
            else:
                m = mask.to(device)
        else:
            m = None
    
        for idx in range(len(t_list) - 1):
            t = t_list[idx]
            t_prev = t_list[idx + 1]  # smaller index (closer to 0)
    
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
    
            # model prediction at timestep t
            model_in = torch.cat([x, cond], dim=1)
            pred = self.model(model_in, t_batch)  # (B,1,H,W)
    
            # get x0_pred and eps_pred depending on parameterization
            if self.prediction == "eps":
                eps = pred
                sqrt_ab_t = self.sqrt_alphas_cumprod[t].view(1, 1, 1, 1)
                sqrt_1mab_t = self.sqrt_one_minus_alphas_cumprod[t].view(1, 1, 1, 1)
                x0_pred = (x - sqrt_1mab_t * eps) / sqrt_ab_t
    
            elif self.prediction == "v":
                eps, x0_pred = self.v_to_eps_x0(x_t=x, v=pred, t=t_batch)
    
            else:
                raise RuntimeError(f"Unknown prediction type: {self.prediction}")
    
            # DDIM update x_t -> x_{t_prev}
            a_t = self.alphas_cumprod[t].view(1, 1, 1, 1)         # alpha_bar_t
            a_prev = self.alphas_cumprod[t_prev].view(1, 1, 1, 1) # alpha_bar_{t_prev}
    
            # sigma_t (DDIM): controls added noise
            # sigma = eta * sqrt((1-a_prev)/(1-a_t)) * sqrt(1 - a_t/a_prev)
            sigma = eta * torch.sqrt((1.0 - a_prev) / (1.0 - a_t)) * torch.sqrt(1.0 - (a_t / a_prev))
    
            # direction term
            # x_{t_prev} = sqrt(a_prev)*x0 + sqrt(1-a_prev-sigma^2)*eps + sigma*z
            c = torch.sqrt((1.0 - a_prev) - sigma ** 2).clamp_min(0.0)
    
            noise = torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(a_prev) * x0_pred + c * eps + sigma * noise
    
            # enforce land after each step; x is now x_{t_prev}, so the value fed
            # to the model next is at timestep t_prev (alpha_bar = a_prev). See
            # sample_ddpm docstring for the target_land rationale.
            if m is not None:
                if target_land is None:
                    x = x * m
                else:
                    x = x * m + (target_land * torch.sqrt(a_prev)) * (1.0 - m)

        # final x is x0 estimate at t=0
        return x


class Trainer:
    def __init__(
            self,
            ddpm: DDPM,
            dataset: Dataset,
            save_dir: str,
            save_every: int = 25,
            batch_size: int = 16,
            lr: float = 1e-4,
            epochs: int = 1000,
            num_workers: int = 4,
            # Optional DataLoader throughput knobs (off by default; they raise
            # peak memory).
            persistent_workers: bool = False,
            prefetch_factor: int | None = None,
            grad_accum: int = 1,
            amp: bool = True,
            device=None,
        ):
        self.ddpm = ddpm
        self.dataset = dataset
        self.save_dir = save_dir
        self.save_every = save_every
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.grad_accum = grad_accum
        self.amp = amp
        self.device = device if device is not None else ddpm.device

        # Both are meaningless (and rejected by older torch) at num_workers=0, so
        # they are only passed when there is a worker pool to apply them to.
        _worker_kw = {}
        if num_workers > 0:
            if persistent_workers:
                _worker_kw['persistent_workers'] = True
            if prefetch_factor is not None:
                _worker_kw['prefetch_factor'] = prefetch_factor
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, drop_last=True, pin_memory=True,
                             **_worker_kw)
        self.opt = torch.optim.Adam(self.ddpm.model.parameters(), lr=lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=amp)

        self.ema = EMA(self.ddpm.model, decay=0.9999)

    def train(self, resume_from: str = None):
        model = self.ddpm.model
        device = self.device
        model.train()

        # load mask here, not in each epoch/batch
        mask = self.dataset.mask_tensor.to(device)  # (1,H,W)

        use_cuda_timing = (device.type == "cuda")
        
        logging.info('Start training')
        
        start_epoch = 0
        global_step = 0
        if resume_from is not None:
            start_epoch, global_step = self.load_checkpoint(resume_from)
            logging.info(f"Resuming from {resume_from}: {start_epoch=}, {global_step=}")

        losses_mse, losses_integral = [], []

        for epoch in range(start_epoch, self.epochs + 1):

            # Epoch wall-clock (no synchronize)
            epoch_t0 = time.perf_counter()

            # Data timing (CPU wall-clock for data loader wait + launch of H2D copies)
            data_time_total = 0.0

            # time spent waiting for the *next* batch (and doing H2D enqueue)
            t_data_start = time.perf_counter()

            measure_every = 20  # or 50
            evt_pairs = []
            comp_ms_total = 0.0
            
            for i, batch in enumerate(self.dl):
                # ---- data time (no GPU sync; measures host-side wait/enqueue) ----
                y = batch["y"].to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                cond = batch["cond"].to(
                    device,
                    non_blocking=True,
                    memory_format=torch.channels_last
                )
                data_time_total += (time.perf_counter() - t_data_start)

                B = y.shape[0]
                t = torch.randint(0, self.ddpm.num_timesteps, (B,), device=device).long()

                do_measure = use_cuda_timing and (i % measure_every == 0)
                if do_measure:
                    start_evt = torch.cuda.Event(enable_timing=True)
                    end_evt = torch.cuda.Event(enable_timing=True)
                    start_evt.record()

                with torch.amp.autocast("cuda", enabled=self.amp):
                    loss_mse, loss_integral = self.ddpm.p_losses(y, cond, mask, t)

                    losses_mse.append(float(loss_mse.detach()))
                    if loss_integral is not None:
                        loss = loss_mse + loss_integral
                        losses_integral.append(float(loss_integral.detach()))
                    else:
                        loss = loss_mse
                    loss = loss / self.grad_accum
                        
                self.scaler.scale(loss).backward()

                if (i + 1) % self.grad_accum == 0:
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)
                    self.ema.update(model)
                    global_step += 1  # update optimizer step count

                if do_measure:
                    end_evt.record()
                    evt_pairs.append((start_evt, end_evt))

                # start timing the wait for the next batch
                t_data_start = time.perf_counter()

            logging.info(f'\tMSE loss: {np.nanmean(losses_mse):.2e}, std={np.nanstd(losses_mse):.2e}')
            if loss_integral is not None:
                logging.info(f'\tIntegral loss: {np.nanmean(losses_integral):.2e}, std={np.nanstd(losses_integral):.2e}')
            
            # end epoch
            if use_cuda_timing:
                torch.cuda.synchronize(device)
                measured_ms = sum(s.elapsed_time(e) for s, e in evt_pairs)
                # scale up to approximate full epoch compute time
                comp_ms_total = measured_ms * (len(self.dl) / max(len(evt_pairs), 1))
            compute_time = comp_ms_total / 1000.0

            epoch_time = time.perf_counter() - epoch_t0
            data_time = data_time_total

            logging.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"time={epoch_time:.1f}s ({epoch_time/60:.2f}m) | "
                f"data={data_time:.1f}s ({100*data_time/max(epoch_time,1e-9):.0f}%) | "
                f"compute={compute_time:.2f}s ({100*compute_time/max(epoch_time,1e-9):.0f}%)"
            )

            # save checkpoint every N epochs, + last epoch
            if epoch % self.save_every == 0 or epoch == self.epochs:
                ckpt = {
                    "epoch": epoch + 1,
                    "global_step": global_step,            # optimizer-step count
                    "model_state": model.state_dict(),
                    "opt_state": self.opt.state_dict(),
                    "scaler_state": self.scaler.state_dict(),
                    "ema_state": {k: v.detach().cpu() for k, v in self.ema.shadow.items()},
                    "ema_decay": float(self.ema.decay),
                    "means": self.dataset.means,
                    "stds": self.dataset.stds,
                    "n_samples": len(self.dataset),
                    "loss_history_mse": np.array(losses_mse),
                    "loss_history_integral": np.array(losses_integral),
                    "timestamp": datetime.today().strftime('%Y-%m-%d %H:%M'),
                    # recorded land-fill convention -> eval reads this instead of
                    # guessing from the timestamp (inference.resolve_land_fill).
                    "land_fill": getattr(self.dataset, 'land_fill', 'meanstd'),
                }

                # save to disk
                path_save = self.checkpoint_path(epoch)
                tmp = path_save.with_suffix(".pt.tmp")
                try:
                    torch.save(ckpt, tmp)
                    tmp.replace(path_save)
                except Exception as e:
                    # A failed checkpoint write (e.g. disk quota exceeded) must
                    # not crash the run: drop the partial .pt.tmp, log loudly,
                    # and keep training so a later interval can checkpoint once
                    # space is freed.
                    try:
                        tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
                    logging.error(
                        f"Checkpoint save FAILED at epoch {epoch} "
                        f"({type(e).__name__}: {e}). Likely cause: disk "
                        f"quota exceeded. Training continues but NO checkpoint "
                        f"was written this interval -- free disk space so a "
                        f"later save (or resume) can persist progress. "
                        f"Save dir: {self.save_dir}")

        logging.info("Training finished.")

    
    def checkpoint_path(self, epoch):
        """Path where checkpoint .pt file is saved for a specific epoch."""
        return Path(self.save_dir) / f"ckpt_epoch{str(epoch).zfill(3)}.pt"

    
    @staticmethod
    def _stats_mismatch(old, new, rtol=1e-9):
        """Descriptions of per-channel stat differences beyond `rtol`, [] if none.

        A differing channel SET is always a mismatch; values are compared
        relatively so float summation order can't fail a legitimate resume.
        """
        bad = []
        if set(old) != set(new):
            return [f'channel set {sorted(set(old) ^ set(new))}']
        for k in old:
            a, b = float(old[k]), float(new[k])
            scale = max(abs(a), abs(b))
            if abs(a - b) > rtol * scale and not (a == 0.0 and b == 0.0):
                bad.append(f'{k}: {a!r} vs {b!r}')
        return bad

    def load_checkpoint(self, path):
        """Load previously saved checkpoint to continue training from."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        # Make sure the training data (and its normalization) is unchanged since
        # this checkpoint was written. This only trips when the input files
        # themselves changed, in which case the old checkpoint's normalization
        # no longer matches the data and resuming would be silently wrong.
        if ckpt['n_samples'] != len(self.dataset):
            raise RuntimeError(
                f"Cannot resume from {path}: checkpoint was trained on "
                f"n_samples={ckpt['n_samples']} but the current dataset has "
                f"{len(self.dataset)} samples. The input data changed since "
                f"this checkpoint was written (new/updated source files). The "
                f"checkpoint's normalization no longer matches the data. To "
                f"restart from scratch on the new data, remove the old "
                f"checkpoints (and preproc_cache.*) in {self.save_dir}; to "
                f"resume, restore the original input files.")
        # Compared with a relative tolerance, not exact equality: the same data
        # yields stats that differ in the last few ulps depending on summation
        # order (in-RAM vs shared-cache path). Genuinely different data moves
        # these by far more than 1e-9.
        _bad = self._stats_mismatch(ckpt['means'], self.dataset.means)
        _bad += self._stats_mismatch(ckpt['stds'], self.dataset.stds)
        if _bad:
            raise RuntimeError(
                f"Cannot resume from {path}: normalization stats (means/stds) "
                f"differ from the current dataset despite matching sample count "
                f"— the input data content changed. Remove old checkpoints and "
                f"preproc_cache.* in {self.save_dir} to restart on the new data. "
                f"Mismatches: {'; '.join(_bad)}")

        # UNet and optimizer state
        self.ddpm.model.load_state_dict(ckpt["model_state"])
        self.opt.load_state_dict(ckpt["opt_state"])
    
        # AMP scaler
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
    
        # EMA (move back to device)
        if "ema_state" in ckpt:
            self.ema.decay = float(ckpt.get("ema_decay", self.ema.decay))
            # ensure device
            self.ema.shadow = {k: v.to(self.device) for k, v in ckpt["ema_state"].items()}
    
        start_epoch = int(ckpt['epoch'])
        global_step = int(ckpt['global_step'])
    
        return start_epoch, global_step
