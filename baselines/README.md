# Baselines (Fig. 8)

Four baselines, each trained on the same data, split and normalization as the
diffusion model (T+S, annual, depth-integrated, 1pctCO2):

- `unet-mse`: the diffusion model's U-Net backbone, trained with a masked MSE through the stock `Trainer` (`unet_mse.py`, `shims.py`). GPU.
- `qrf`: a quantile regression forest over pointwise rows (predictors at a cell plus sin/cos lat/lon). `min_samples_leaf` is selected on validation pinball loss (`qrf.py`). CPU.
- `pointwise-linear`: per-cell OLS on that cell's predictors, closed form (`pointwise_linear.py`). CPU.
- `climatology`: the intercept-only submodel of the linear fit. Written in the same pass.

Checkpoints go to `paths.BASELINE_ROOT/baseline-<kind>/`, using the same directory names as the diffusion runs (`naming.py`).

## Order of operations
`<M>` is each of the 14 models in `T-S_ann_int_oos.nc`, and `insample` is the single in-sample run. Run every step for both.

    # 1. U-Net, trained to 1000 epochs with checkpoints every 25
    python -m baselines.train_baseline --baseline unet-mse -m <M|insample> \
        -p "thetao so" -t o2 -r annual -f depthint -e 1pctCO2 --epochs 1000 --save-every 25
    # 2. U-Net epoch selection on held-in models' held-out years (writes epoch_selection.json)
    python -m baselines.select_epoch --model <M>        # or --insample
    # 3. QRF with min_samples_leaf selection (val years 81 100 for LOMO, 61 80 in-sample)
    python -m baselines.train_baseline --baseline qrf -m <M|insample> -p "thetao so" -t o2 \
        -r annual -f depthint -e 1pctCO2 --n-cells 400 --n-estimators 100 \
        --min-samples-leaf 20 --max-samples-leaf 40 --qrf-seed 0 --n-jobs 4 \
        --select-min-samples-leaf 5 10 20 --val-years 81 100
    # 4. linear + climatology from one set of normal equations
    python -m baselines.train_baseline --baseline pointwise-linear --also climatology \
        -m <M|insample> -p "thetao so" -t o2 -r annual -f depthint -e 1pctCO2 --min-samples 50
    # 5. evaluation parts, then one cache per kind (see figures/fig08_baseline_grid.py compute)
    python -m baselines.evaluate --baseline <kind> --split <oos|insample> [--model <M>]
    python -m baselines.evaluate --baseline <kind> --split <oos|insample> --aggregate

None of the training steps pass `--use-cache`. All four baselines therefore read the same in-RAM preprocessing path. `<kind>` covers the four baselines and `diffusion`; step 5 also reruns the diffusion model through the same per-model loop. The QRF uses `--draw per-cell --seed 0`.

## Which epoch Fig. 8 shows
`evaluate.py` takes its fleet, epoch (500), ensemble size, sampler and year window from the diffusion `error_maps` caches `T-S_ann_int_{oos,insample}.nc` in `paths.EVAL_DIR/error_maps/cache/`. Fig. 8 reads the default `--epoch pinned` caches, so its U-Net column is the epoch-500 checkpoint, the same epoch as the diffusion model. The epoch chosen in step 2 is used only by `--epoch selected`, which writes a separate `*_selected.nc` cache that the figure does not read.

Only the oos pin cache has a producer in this release (`figures/fig06_lomo_error.py compute`, which writes to that directory). The in-sample `T-S_ann_int_insample.nc` came from the in-sample path of `general/eval/error_maps.py`, which is not included. Without it, `evaluate --split insample` and the in-sample crosscheck in Fig. 8 cannot run.
