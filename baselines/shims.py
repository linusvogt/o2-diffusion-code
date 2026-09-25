"""The two seams that let a baseline reuse the diffusion pipeline unchanged.

Extracted from general/baselines/shims.py. Nothing scientific happens here -- these are adapters whose entire job is to make the
existing ``Trainer`` and the existing ``figures.eval_common`` evaluation path accept a
non-diffusion model without either of them being edited.

Seam 1 -- training
------------------
``Trainer.train`` touches its ``ddpm`` argument in exactly three places:

    self.ddpm.num_timesteps            to draw t
    self.ddpm.model                    .train(), .state_dict(), .parameters()
    self.ddpm.p_losses(y, cond, mask, t) -> (loss_mse, loss_integral)

(plus ``ddpm.device`` as a fallback in ``__init__``). So an object exposing
those runs through ``Trainer`` unchanged and inherits AMP, gradient
accumulation, EMA, per-epoch timing, the checkpoint format -- including
``means``/``stds``/``land_fill``/``n_samples`` -- the resume-compatibility
assertion, and the atomic-write-on-quota-failure handling. None of that is
re-implemented.

Seam 2 -- evaluation
--------------------
``eval_common.predict_ensemble`` reaches the model through

    run.ddpm.model.eval()
    run.ddpm.sample(cond, mask, sampler=, eta=, steps=, target_land=)

so a ``.sample()`` with that signature makes the ENTIRE evaluation path
byte-identical to the diffusion model's: same normalization, same land-fill
convention, same ocean mask, same ``materialize_window``/``select_case`` time
handling, same ``model_timemean`` averaging order, same ``ec.model_mean_abs``.
That identity is the deliverable -- a second implementation of the protocol
would be a second place for it to drift, and the comparison would stop being a
comparison.
"""
from __future__ import annotations

import torch


class _NoOpEMA:
    """An EMA that does nothing, for a run whose weights are already final.

    ``predict_ensemble`` calls ``run.ema.apply_shadow(model)`` / ``restore``
    around every sample when ``sample_cfg.apply_ema`` is set, which it is by
    default and which the diffusion caches were produced with. A baseline
    loaded from a checkpoint that HAS EMA weights uses the real
    ``diffusion.model.EMA``; this exists only for the paths that do not
    (a smoke run, or a deliberately non-EMA readout), so the call sites stay
    identical rather than growing an ``if``.
    """

    def apply_shadow(self, model):
        return None

    def restore(self, model):
        return None


class DeterministicRegressor:
    """A ``DDPM``-shaped wrapper around a plain regression network (the U-Net baseline).

    Implements both seams. The network is called ONCE, on the conditioning
    channels alone; there is no noise, no reverse chain and no timestep, so
    ``sample`` ignores ``sampler``/``eta``/``steps`` -- they are accepted only
    so the call site in ``predict_ensemble`` needs no branch.

    ⚠️ ``num_timesteps = 1`` so ``Trainer``'s ``torch.randint(0,
    num_timesteps, ...)`` yields all-zero ``t``. The backbone still takes a
    timestep argument (it is the unmodified ``SimpleUNetCond``), and a constant
    t=0 turns its FiLM time bias into a constant learned bias. That is
    harmless, and keeps the backbone byte-identical to the diffusion model's.
    """

    def __init__(self, model, device, img_size, prediction='mean'):
        self.model = model
        self.device = device
        self.img_size = img_size
        #: recorded so a checkpoint cannot be mistaken for a diffusion one
        self.prediction = prediction
        self.num_timesteps = 1

    # -- Seam 1 ------------------------------------------------------------
    def p_losses(self, x0, cond, mask, t):
        """Masked MSE between the prediction and the true target.

        ⚠️ The mask weighting mirrors ``DDPM.p_losses`` EXACTLY:
        ``(mse * mask_b).sum() / mask_b.sum().clamp_min(1)``, i.e. a mean over
        **ocean cells only**. A plain ``.mean()`` over all 180x360 cells would
        train this network to fit the land-fill constant, which is most of the
        domain -- silently changing what is optimized and understating the
        baseline, in exactly the direction that would make the comparison
        unpersuasive. Land contributes zero by construction.

        Returns ``(loss, None)``: the second slot is ``DDPM``'s optional
        integral loss, which this baseline does not have. ``Trainer`` handles
        None there already (the diffusion runs are all ``integral_loss=False``).
        """
        pred = self.model(cond, self._t_like(x0, t))
        mse = (x0 - pred) ** 2
        mask_b = self._broadcast_mask(mask, x0)
        if mask_b is None:
            return mse.mean(), None
        return (mse * mask_b).sum() / mask_b.sum().clamp_min(1.0), None

    # -- Seam 2 ------------------------------------------------------------
    @torch.no_grad()
    def sample(self, cond, mask, sampler=None, eta=None, steps=None,
               timestep_spacing=None, target_land=None):
        """One forward pass, land-handled exactly as the reverse chain ends.

        ``DDPM.sample_ddim``'s final step applies
        ``x = x * m`` (or ``x * m + target_land * sqrt(a_prev) * (1 - m)``,
        where ``a_prev`` is 1 at the last step), so land leaves the sampler at
        0 or at the trained land value. This reproduces that, which matters
        because ``predict_ensemble`` denormalizes the whole array before
        masking it -- a different land value there would not change the ocean,
        but it would change what a land cell contains in any downstream code
        that forgets to mask.
        """
        x = self.model(cond, self._t_like(cond, None))
        m = self._broadcast_mask(mask, x)
        if m is not None:
            x = x * m if target_land is None else x * m + target_land * (1.0 - m)
        return x

    # -- helpers -----------------------------------------------------------
    def _t_like(self, ref, t=None):
        """The all-zero timestep the backbone still expects.

        Always BUILT from ``ref``, never derived from the ``t`` handed in.
        Two reasons, both load-bearing:

        * ``sample`` has no ``t`` at all, and train and inference must agree
          exactly or they differ by a FiLM bias.
        * the incoming ``t`` may be on a different device. ``Trainer`` creates
          it on the GPU, but a direct caller may create
          it on the CPU, and ``torch.zeros_like`` would inherit that and fail
          inside the time MLP. Taking the device from the data instead makes
          the shim correct for any caller.
        """
        return torch.zeros(ref.shape[0], dtype=torch.long, device=ref.device)

    @staticmethod
    def _broadcast_mask(mask, ref):
        """``DDPM.p_losses``' mask handling, verbatim: (1,H,W) -> (B,1,H,W)."""
        if mask is None:
            return None
        if mask.shape[0] == 1:
            return mask.expand(ref.shape[0], 1, mask.shape[-2],
                               mask.shape[-1]).to(ref.device)
        return mask.to(ref.device)


class QuantileSampler:
    """A ``DDPM``-shaped wrapper around a fitted quantile model (the QRF baseline).

    Seam 2 only -- a forest is not trained through ``Trainer``. ``sample``
    draws one member of the predictive distribution per call, which is what
    makes ``predict_ensemble``'s ``n_samples`` loop produce a genuine ensemble
    rather than n copies of one field, so the spread comparison is
    measured on the same footing as the diffusion model's.

    ``predict_fn(cond, mask)`` -> ``(1, 1, H, W)`` tensor is supplied by
    ``qrf.py``; keeping the draw there rather than here means this class never
    needs to know how the quantiles are stored, and passing the mask through
    means the forest is traversed for the ~40k ocean cells rather than all
    64800 (land is overwritten below in any case).

    ⚠️ The ensemble here is only as meaningful as ``predict_fn``'s draw rule,
    which is a scientific choice recorded in ``qrf.py`` and stamped into every
    evaluation cache -- see its module docstring.
    """

    def __init__(self, predict_fn, device, img_size):
        self.model = _EvalOnlyModel()
        self.predict_fn = predict_fn
        self.device = device
        self.img_size = img_size
        self.num_timesteps = 1

    @torch.no_grad()
    def sample(self, cond, mask, sampler=None, eta=None, steps=None,
               timestep_spacing=None, target_land=None):
        x = self.predict_fn(cond, mask)
        m = DeterministicRegressor._broadcast_mask(mask, x)
        if m is not None:
            x = x * m if target_land is None else x * m + target_land * (1.0 - m)
        return x


class _EvalOnlyModel:
    """Stands in for ``run.ddpm.model`` where there is no torch module.

    ``predict_ensemble`` calls ``run.ddpm.model.eval()`` unconditionally. A
    forest has nothing to put in eval mode, and the alternative -- branching at
    that call site -- would mean editing ``eval_common`` for a baseline, which is
    the thing this module exists to avoid.
    """

    def eval(self):
        return self

    def train(self, mode=True):
        return self
