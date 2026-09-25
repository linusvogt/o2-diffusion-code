"""Filesystem locations used throughout the code.

The processed data are not distributed with this repository; these paths only
document the layout the code expects. Set ``O2DIFF_ROOT`` (data, checkpoints,
caches) and ``O2DIFF_FIG_DIR`` (figure output) to point them elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('O2DIFF_ROOT', '/path/to/scratch'))
FIG_DIR = Path(os.environ.get('O2DIFF_FIG_DIR', '/path/to/figures'))

# --- CMIP6 inputs, regridded to 1 degree (r360x180) ---------------------------
REGRID_YEARLY = ROOT / 'cmip_regrid_yearly'            # annual 0-2000 m integrals
REGRID_MONTHLY = ROOT / 'cmip_regrid_monthly'          # monthly 0-2000 m integrals
DENSITY_SURFACES = ROOT / 'cmip_density_surfaces'      # tracers on sigma_1 layers
CMIP6_REGRIDDED = ROOT / 'cmip6_data_regridded' / 'r360x180'  # wind stress
MLD_DIR = ROOT / 'mld' / 'r360x180'                    # monthly mixed-layer depth

# --- static grid files --------------------------------------------------------
GRIDAREA = ROOT / 'gridarea_r360x180.nc'               # cdo gridarea
BATHY_MASK = ROOT / 'bathymetry_mask_r360x180.nc'
REGION_MASKS = ROOT / 'region_masks'
RECCAP2_MASK = REGION_MASKS / 'RECCAP2_region_masks_all_v20221025_remapnn_r360x180.nc'

# --- observational products (regridded to r360x180) ---------------------------
OBS = ROOT / 'obs'
WOA_DIR = OBS / 'woa'
WOA_O2_FILE = WOA_DIR / 'oxygen' / 'r360x180' / 'woa23_all_o00_01_r360x180.nc'
GOBAI_O2_FILE = OBS / 'gobai_o2' / 'GOBAI-O2-ann-v2.3_r360x180.nc'
GLODAP_DIR = OBS / 'glodap'
ECCO_TS_FILE = OBS / 'ecco' / 'ds_ts_levint.nc'
ECCO_MLD_DIR = OBS / 'ecco' / 'mld' / 'r360x180'
ECCO_MLD_FROM_TS_FILE = OBS / 'ecco' / 'mld_from_ts.nc'
ERA5_TAU_FILE = OBS / 'era5' / 'windstress_era5_r360x180_remapcon.nc'
MOBO_DIC_FILE = OBS / 'dic' / 'MPI_MOBO-DIC_2004-2019_v2.nc'
WANG2025_DIR = OBS / 'wang_2025'
ROACH_BINDOFF_DIR = OBS / 'roach_bindoff_2023'

# --- outputs ------------------------------------------------------------------
CHECKPOINTS = ROOT / 'diffusion_model' / 'checkpoints'
PREPROC_CACHE = ROOT / 'diffusion_model' / 'preproc_cache_shared'
BASELINE_ROOT = ROOT / 'diffusion_model' / 'baselines'
EVAL_DIR = FIG_DIR / 'general_eval'                    # eval caches + figures
MANUSCRIPT_DIR = FIG_DIR / 'manuscript'                # final figures

# --- small metadata tables shipped with the repo ------------------------------
REPO = Path(__file__).resolve().parent
CMIP6_CENTERS_CSV = REPO / 'vendored' / 'cmip6_centers_for_each_model.csv'
MODELS_TXT = REPO / 'diffusion' / 'models.txt'         # the LOMO fleet
DENSITY_QC_CSV = REPO / 'diffusion' / 'density_data_qc.csv'
