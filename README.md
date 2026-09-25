# o2-diffusion-code

This repository contains the code behind "Multi-model ocean oxygen fields predicted by conditional diffusion models" (L. Vogt and L. Zanna, 2026, preprint).
It includes training code for a conditional denoising diffusion model (DDPM) that generates maps of
0–2000 m depth-integrated ocean oxygen (O₂) from temperature and salinity (and optionally other
predictors) of CMIP6 Earth system models. It also includes the scripts behind the manuscript's main-text figures.

## Model in brief

- **Data.** Each sample is one year (or month) of one CMIP6 model member, regridded to 1°
  (180×360). The target is depth-integrated O₂. The predictors are depth-integrated temperature
  `thetao` and salinity `so`, optionally with DIC, wind stress or mixed-layer depth, or the same
  tracers on σ₁ density layers. Normalisation is area-weighted and computed on the training split
  only. Land cells are filled and masked.
- **Network.** `SimpleUNetCond` is a residual U-Net. Its input is the noisy target concatenated
  with the predictor channels. Skip connections are additive, and the timestep enters every
  ResBlock as a channel bias.
- **Diffusion.** 1000 steps, cosine schedule, v-prediction. At sampling time the network uses its
  EMA weights, with a 100-step DDIM sampler, and the land mask is reapplied after every step.
- **Splits.**
  - *In-sample*: train on years 1–80 and 100–150, test on years 80–100 of the same models.
  - *Leave-one-model-out (LOMO)*: one CMIP6 model is held out entirely
    (`diffusion/models.txt`).
- **Ensembles.** Every figure draws 5 samples per case (Fig. 2 draws 10). All figures use the
  **epoch-500** checkpoint.

Training hyperparameters (`diffusion/train.py: config`):

| setting | value |
|---|---|
| learning rate | 1e-4 (Adam) |
| batch size | 16 |
| epochs | 500 |
| AMP | on |
| U-Net | `base_ch` 32, `ch_mults` (1, 2, 4), 2 ResBlocks per level, time embedding 256 |
| diffusion | 1000 timesteps, `cos` schedule, `v` prediction |

## Layout

```
paths.py                 all data / output locations (placeholders)
diffusion/
  model.py               U-Net, DDPM (training loss, DDPM and DDIM samplers), EMA, Trainer, TracerDataset
  train.py               training entry point (in-sample or leave-one-model-out)
  data_loading.py        builds the (sample, lat, lon) training set for each resolution and field type
  cmip_io.py             CMIP6 file discovery / loading helpers and train-test splits
  naming.py              checkpoint and cache directory names
  inference.py           load a trained run and normalise inputs
  obs.py                 loaders for the observational products (WOA23, GOBAI-O2, GLODAP, ECCO, ERA5, ...)
  calibration.py         spatial variance-inflation factor (used for Fig. 10)
  models.txt             the CMIP6 models held out in the LOMO experiments
vendored/ocean_utils_min.py   the parts of the author's utility library that are used (CMIP6 model IDs, map plotting)
baselines/               U-Net (MSE), quantile regression forest, pointwise linear, climatology (see baselines/README.md)
figures/
  eval_common.py         shared evaluation library: load runs, sample ensembles, error metrics, map panels
  render_common.py       final figure styling and saving
  fig01_unet_schematic/  TikZ source of the U-Net schematic
  fig02_... fig10_...    one script per figure
```

## Training

```bash
export PYTHONPATH=$PWD            # repo root
# in-sample, T+S predictors, annual depth-integrated, 1pctCO2
python -m diffusion.train -p "thetao so" -t o2 -m insample -w 0.0 -r annual -f depthint -e 1pctCO2
# leave-one-model-out: hold out one model
python -m diffusion.train -p "thetao so" -t o2 -m ACCESS-ESM1-5 -w 0.0 -r annual -f depthint -e 1pctCO2
```

The other configurations in the manuscript use the following flags. The resulting runs are
named by `diffusion/naming.py`.

| axis | flag | values used |
|---|---|---|
| predictor set | `-p` | `thetao so`, plus `dissic`, `tauuo tauvo` or MLD |
| resolution | `-r` | `annual`, `monthly` |
| field type | `-f` | `depthint`, `density` |
| experiment | `-e` | `1pctCO2`, `abrupt-4xCO2` |
| held-out model | `-m` | `insample`, or any model in `models.txt` |

Training requires a GPU. Checkpoints are written every 25 epochs, and a run resumes from the
latest checkpoint when relaunched.

## Figures

Every figure script has two steps:

- `compute` samples the trained model(s) and writes an evaluation cache (netCDF). It needs a GPU.
- `plot` redraws the figure from that cache without a GPU.

Only the main-text version of each figure is included.

```bash
python -m figures.fig06_lomo_error compute --epoch 500
python -m figures.fig06_lomo_error plot
```

| Fig. | script | content | `compute` settings of the published cache |
|---|---|---|---|
| 1 | `fig01_unet_schematic/unet_flat.tex` | U-Net schematic (`pdflatex unet_flat.tex`) | — |
| 2 | `fig02_predictor_set.py` | relative error of the five predictor sets (in-sample, common model fleet) | 10 samples, years 81–100 |
| 3 | `fig03_predictor_boxplot.py` | per-model global-mean relative bias for each predictor set | 5 samples |
| 4 | `fig04_seasonal_cycle.py` | regional seasonal O₂ cycle, monthly runs, depth-integrated vs density predictors | 5 samples, every 5th year |
| 5 | `fig05_scenario_transfer.py` | error when training and testing on 1pctCO2 / abrupt-4xCO2, and transfer minus native | 5 samples |
| 6 | `fig06_lomo_error.py` | leave-one-model-out error: signed, absolute and cross-model coherence | 5 samples, 14 held-out models |
| 7 | `fig07_obs_validation.py` | O₂ generated from WOA23 T/S vs observed WOA23 O₂ | 5 samples, 2004–2017 |
| 8 | `fig08_baseline_grid.py` | diffusion model vs baselines, in-sample and LOMO (unsigned, %) | caches from `baselines/evaluate.py`, plus Fig. 6's `compute` with `--split oos` and `--split insample` |
| 9 | `fig09_extrapolation.py` | O₂ generated for CMIP6 models without O₂ output: the resulting correction to the multi-model mean | 5 samples, every 2nd year |
| 10 | `fig10_extrapolation_omz.py` | the same correction in the eastern tropical Pacific OMZ, and the ensemble distribution | uses Fig. 9's cache |

## Environment

The manuscript runs used Python 3.12 with the package versions pinned in `requirements.txt`
(PyTorch 2.9.1, xarray 2025.12, ultraplot 1.70, cartopy 0.25). `quantile-forest` and
`scikit-learn` are needed only for the baselines.

## License

MIT (see `LICENSE`).
