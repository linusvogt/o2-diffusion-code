"""Observational-product loaders, regridded to r360x180 and depth-integrated
over 0-2000 m on the same ``2000 * (thickness-weighted column mean)``
convention as the CMIP training fields.

Extracted from ml_framework/diffusion_model/util_obs.py.
"""
import numpy as np
import xarray as xr

import paths


## ---- shared helpers ---- ##


# Any r360x180 CMIP field serves as the ocean/land mask for products delivered on
# a full global grid (ERA5, Roach & Bindoff); it is time-invariant, so which
# member/step it comes from does not matter.
_CMIP_MASK_FILE = str(paths.REGRID_YEARLY / 'mld' / '1pctCO2' / 'mean' /
                      'mld_1pctCO2_BCC_BCC-CSM2-MR_r1i1p1f1_yearmean.nc')


def _cmip_ocean_mask():
    return xr.open_dataset(_CMIP_MASK_FILE).isel(time=0).mld.drop_vars('time')


def split_year_month(ds, time='time'):
    """Monthly ``time`` axis -> separate ``year`` and ``month`` dimensions.

    What the *monthly* observational products return, so that a caller can pick
    one (year, month) map the same way it picks one year from an annual product.

    ``month`` is **0-based** (0 = January). That is the convention the whole
    evaluation side uses -- ``eval_common.MONTH_NAMES``, ``select_case``'s
    ``time % 12`` match against the CMIP sample index, and the ``--months`` CLI
    flag -- so month ``m`` of an observational product is month ``m`` of a
    monthly CMIP run. Off-by-one here would be invisible in a 12-month average
    and wrong in every per-month selection.

    Requires a gap-free record: ``unstack`` fills any missing (year, month) with
    NaN, which would otherwise pass silently as a blank map.
    """
    ds = ds.assign_coords(year=(time, ds[time].dt.year.values),
                          month=(time, ds[time].dt.month.values - 1))
    ds = ds.set_index({time: ('year', 'month')}).unstack(time)
    return ds.reset_coords(drop=True)


## ---- GOBAI-O2 ---- ##


def load_gobai_levmean():
    """Depth-integrated GOBAI-O2 tracers: T, S, O2, annual from 2004-2024."""
    
    ds_gobai = xr.open_dataset(str(paths.GOBAI_O2_FILE), chunks=dict(time=1))
    ds_gobai = ds_gobai.rename(pres='lev')
    
    # manually determined lev bounds
    lev_bnds = [0] + list(range(5, 175+5, 10)) + list(range(190, 440+20, 20)) + list(range(475, 1350+50, 50)) + list(range(1450, 1950+50, 100)) + [2000]
    
    # level thickness
    thick = np.diff(lev_bnds)
    thick = xr.DataArray(thick, dims=('lev', ))
    
    #ds_int = (ds_gobai * thick).sum('lev')
    ds_int = 2000 * ds_gobai.weighted(thick).mean('lev')  # depth-integral
    ds_int['oxy'] /= 1000  # convert from umol/kg to mol/m3

    # integer year coordinate
    ds_int['time'] = ds_int.time.astype(int)
    ds_int = ds_int.rename(time='year')

    ds_int = ds_int.rename({'temp': 'thetao', 'sal': 'so', 'oxy': 'o2'})
    
    return ds_int


## ---- WOA ---- ##


def load_woa_levmean():
    decade_dict = {
        '5564': 1960,
        '6574': 1970,
        '7584': 1980,
        '8594': 1990,
        '95A4': 2000,
        'A5B4': 2010,
        'B5C2': 2020,
    }

    datasets = {}
    for var in ['temperature', 'salinity']:
        data = []
        path = paths.WOA_DIR / var / 'r360x180'
        for file in path.glob('*.nc'):
            ds = xr.open_dataset(file, decode_times=False)
            decade_str = str(file.stem).split('_')[1]
            ds = ds.assign_coords(time=[decade_dict[decade_str]])
            ds = ds.rename(time='year')
            ds = ds.rename(depth='lev')
        
            varname = f'{var[0]}_an'
            da = ds[varname]
            data.append(da)
        ds_var = xr.concat(data, 'year')
        datasets[var] = ds_var

    da_o2 = xr.open_dataset(
        str(paths.WOA_O2_FILE),
        decode_times=False
    ).o_an
    da_o2 = da_o2.drop_vars('time').squeeze()
    da_o2 = da_o2.rename(depth='lev')
    da_o2 /= 1000  # convert from umol/kg to mol/m3
        
    ds = xr.Dataset({'thetao': datasets['temperature'], 'so': datasets['salinity'], 'o2': da_o2, 'lev_bnds': ds.depth_bnds})

    thick = ds.lev_bnds.values[:, 1] - ds.lev_bnds.values[:, 0]
    thick = xr.DataArray(thick, dims=('lev', ))
    thick = thick.assign_coords(lev=ds.lev)

    ds_int = 2000 * ds.sel(lev=slice(None, 2000)).weighted(thick.sel(lev=slice(None, 2000))).mean('lev')  # depth-integral
    ds_int = ds_int.drop_vars('lev_bnds')

    ds_int = ds_int.where(~np.isnan(ds.sel(lev=2000).isel(year=0)))

    return ds_int


## ---- ECCO ---- ##


def load_ecco_levmean():
    ds_int = xr.open_dataset(str(paths.ECCO_TS_FILE))
    ds_int = ds_int.drop_vars(['lev_bnds', 'time_bnds'])
    ds_int = ds_int.drop_vars('lev')
    
    return ds_int


def load_mld_ecco(from_ts=True):

    if from_ts:
        ds = xr.open_dataset(str(paths.ECCO_MLD_FROM_TS_FILE)).drop_vars('lev')
    else:
        path = paths.ECCO_MLD_DIR
        files = sorted(list(path.glob('*.nc')))
        ds = xr.open_mfdataset(files).rename(MXLDEPTH='mld')

    return ds


def load_ecco(mld_from_ts=True):
    ds_ts_int = load_ecco_levmean()
    ds_mld = load_mld_ecco(from_ts=mld_from_ts)

    ds = xr.Dataset({
        'thetao': ds_ts_int.thetao.groupby('time.year').mean(),
        'so': ds_ts_int.so.groupby('time.year').mean(),
        'mld_max': ds_mld.mld.groupby('time.year').max(),
        'mld_mean': ds_mld.mld.groupby('time.year').mean(),
    })

    return ds


def load_ecco_monthly(mld_from_ts=True):
    """ECCO V4r5 on a (year, month) axis: depth-integrated T and S, plus MLD.

    The monthly counterpart of ``load_ecco``, and the only observational product
    that supplies predictors at monthly resolution -- which is what lets the
    monthly-trained runs be validated against observations at all. Source is the
    same ECCO V4r5 state estimate, 336 monthly steps covering 1992-2019 (so the
    2004-2017 obs window is complete), already regridded to r360x180 under
    ``obs/ecco/temperature_salinity/r360x180/``.

    **No unit conversion, and none is needed.** ECCO reports ``THETA`` in
    degree_C and ``SALT`` on the practical salinity scale (units ``1e-3``),
    exactly the CMIP ``thetao``/``so`` convention; MLD is in metres like CMIP
    ``mlotst``.

    Depth integration is the shared ``2000 * (thickness-weighted column mean)``
    convention of every other loader here, applied to the 37 ECCO levels at or
    above 2000 m (deepest cell edge 1993.6 m) and restricted to columns that
    actually reach that depth -- 32587 of 41169 wet cells, the same deep-ocean
    restriction ``load_woa_levmean`` and ``load_glodap_levmean`` apply. It is
    read from the precomputed ``ds_ts_levint.nc``, which was verified
    bit-identical to recomputing that integral from the regridded monthly files.

    Against the stored normalization of the monthly in-sample checkpoints, the
    2004-2017 mean sits well inside the training distribution: z = -0.35
    (1pctCO2) / -0.65 (abrupt-4xCO2) for T -- the CMIP runs are warming
    scenarios, so a real ocean colder than their mean is expected, and the
    accepted *annual* ECCO cells sit at the identical -0.35 / -0.64 -- and
    z ~ 0.00 for S.

    ``mld_from_ts=True`` (the default, matching ``load_ecco``) takes MLD from
    the density criterion applied to ECCO's own T/S rather than ECCO's native
    ``MXLDEPTH`` diagnostic. Both are usable, the T/S-derived one is the closer
    match to the CMIP ``mlotst`` the runs trained on: z = +0.05 / +0.10 against
    the monthly T+S+MLD checkpoints, versus +0.29 / +0.40 for the native field
    (which runs ~40% deeper in the global mean, 83.8 m vs 59.7 m).

    Only ``mld`` is returned, not the ``mld_max``/``mld_mean`` of ``load_ecco``:
    those are annual *reductions* over the 12 months and exist only because an
    annual run has to summarize a seasonal cycle. A monthly run conditions on
    the month's MLD directly.
    """
    ds_ts_int = load_ecco_levmean()
    ds_mld = load_mld_ecco(from_ts=mld_from_ts)

    ds = xr.Dataset({'thetao': ds_ts_int.thetao, 'so': ds_ts_int.so,
                     'mld': ds_mld.mld})
    return split_year_month(ds)


## ---- Wang et al. (2025) ---- ##


def load_wang_2025():
    ds = xr.open_dataset(str(paths.WANG2025_DIR / 'ds_wang_2025_levint.nc'))

    return ds


## ---- ERA5 ---- ##


def _load_era5_tau_raw():
    """ERA5 surface wind stress on its native MONTHLY axis, r360x180 (N/m^2)."""
    ds = xr.open_dataset(str(paths.ERA5_TAU_FILE), chunks='auto')
    return ds.rename({'valid_time': 'time', 'avg_iews': 'tauuo', 'avg_inss': 'tauvo'})


def load_era5_tau():
    ds = _load_era5_tau_raw().groupby('time.year').mean()
    ds = ds.where(~np.isnan(_cmip_ocean_mask()))
    return ds


def load_era5_tau_monthly():
    """ERA5 wind stress on a (year, month) axis -- the monthly counterpart of
    ``load_era5_tau``.

    The source file is monthly already (1032 steps, 1940-2025); the annual
    loader is the one that reduces it. Same variables, same masking, so a
    monthly cell's tau channel differs from an annual cell's only in not having
    been averaged over the seasonal cycle.
    """
    ds = split_year_month(_load_era5_tau_raw())
    ds = ds.where(~np.isnan(_cmip_ocean_mask()))
    return ds


## ---- Roach & Bindoff (2023) ---- ##


def load_roach_bindoff():
    data = []
    for var in ['o2', 'so', 'thetao']:
        file_name = {'o2': 'Oxygen_concentration', 'so': 'Salinity', 'thetao': 'Temperature'}[var]
        var_name = {'o2': 'oxygen_concentration', 'so': 'Salinity', 'thetao': 'Temperature'}[var]
        
        ds = xr.open_dataset(str(paths.ROACH_BINDOFF_DIR / f'{file_name}_anomaly_and_background_r360x180.nc'), chunks='auto')
        ds = ds.rename({'depth': 'lev', f'{var_name}_anomaly': f'{var}_anom', f'{var_name}_background': f'{var}_background'})
        ds = ds.drop_vars([f'{var_name}_anomaly_standard_deviation', f'{var_name}_background_standard_deviation'])
        ds[var] = ds[f'{var}_background'] + ds[f'{var}_anom']  # full time-varying O2 field
        
        data.append(ds)

    ds = xr.merge(data)

    for var in ds.keys():
        if 'o2' in var:
            ds[var] = ds[var] / 1000

    ds = ds.rename(time='year')
    ds['year'] = np.arange(1960, 2017+1)

    thick = [2.5] + list(np.diff(ds.lev.values))
    thick = xr.DataArray(thick, dims=('lev', ))
    thick['lev'] = ds.lev

    ds_int = 2000 * ds.sel(lev=slice(None, 2000)).weighted(thick.sel(lev=slice(None, 2000))).mean('lev')  # depth-integral

    ds_int = ds_int.where(~np.isnan(_cmip_ocean_mask()))

    return ds_int


## ---- MOBO-DIC (Keppler et al.) ---- ##


def load_mobo_dic():
    """Depth-integrated MOBO-DIC: observational `dissic`, annual 2004-2019.

    Mapped Observation-Based Oceanic DIC (v2, 2023), monthly on 28 levels from
    2.5 m to 1500 m. Returned on the same convention as every other
    depth-integrated loader here -- ``2000 * (depth-weighted column mean)`` --
    so it is directly comparable with the CMIP `dissic` the runs were trained
    on (fleet global mean ~4400-4560; this product 4426).

    Two things this file needs that the others do not:

    - Its ``missing_value`` attribute is the STRING '-999.9', so xarray does not
      auto-mask it and the fill value lands in the data as an ordinary number.
      It is masked explicitly below; without that the column mean is garbage
      (global mean 1074 instead of 4426, with negative DIC).
    - The column stops at 1500 m (last cell edge 1550 m), not 2000 m, so the
      convention above extrapolates the 0-1550 m mean over the full 2000 m.
      That is a good approximation for DIC specifically, which is near-uniform
      below ~1000 m: filling 1550-2000 m with the deepest level instead moves
      the global mean by 0.2% (4419 vs 4426). What remains is a ~150 mol/m^2
      (~1.1 sigma) low bias against the CMIP training distribution -- real, and
      worth reading the error maps with in mind.

    umol/kg -> mol/m^3 uses the /1000 convention of the other loaders here (an
    implicit rho = 1000 kg/m^3, ~2.5% low against rho = 1025).
    """
    ds = xr.open_dataset(str(paths.MOBO_DIC_FILE),
                         decode_times=False, chunks={'juld': 48})
    da = ds['DIC']
    da = da.where(da > -999) / 1000       # mask the string-typed fill; -> mol/m3

    lev = da['depth'].values
    mids = (lev[:-1] + lev[1:]) / 2
    outer = np.concatenate([[0.], mids, [lev[-1] + (lev[-1] - lev[-2]) / 2]])
    thick = xr.DataArray(np.diff(outer), dims=('depth',),
                         coords={'depth': da['depth']})
    da_int = 2000 * da.weighted(thick).mean('depth')   # skipna: partial columns

    # 'months since 2004-01-01', 1..192 -> integer year, then annual means
    years = 2004 + (da_int['juld'].values.astype(int) - 1) // 12
    da_int = da_int.assign_coords(juld=('juld', years)).rename(juld='year')
    da_int = da_int.groupby('year').mean()

    # -179.5..179.5 -> 0.5..359.5 (the caller relabels onto the 0..359 reference
    # grid, same 0.5 deg labelling offset as Wang 2025)
    da_int = da_int.assign_coords(lon=(da_int['lon'] % 360)).sortby('lon')

    return xr.Dataset({'dissic': da_int})


## ---- GLODAP ---- ##


GLODAP_ROOT = paths.GLODAP_DIR

# GLODAPv2 is reported in micro-mol kg-1 for the two carbon-system tracers; T is
# degrees Celsius and S practical salinity, which is what CMIP uses already.
GLODAP_PERMASS_VARS = ('o2', 'dissic')


def load_glodap_levmean(variables=('thetao', 'so', 'o2', 'dissic')):
    """Depth-integrated GLODAPv2 T, S, O2 and DIC -- the only observational
    product here that supplies *all four* tracers on one consistent grid.

    Returned on the same convention as every other depth-integrated loader in
    this file -- ``2000 * (depth-weighted column mean)`` -- so it is directly
    comparable with the CMIP fields the runs were trained on. Against the T+S+DIC
    annual/depthint checkpoint's stored normalization it sits well inside the
    training distribution on every channel (z = -0.02 for S, -0.04 for the O2
    target, -0.33 for T, -0.87 for DIC).

    Source levels are the 33 standard GLODAP surfaces (0..5500 m); only the 26
    at or above 2000 m enter the integral. Cell thicknesses come from the level
    midpoints with the outermost edges clamped to 0 and 2000 m, so they sum to
    exactly 2000 m. Columns that do not reach the 2000 m level are dropped
    (``ds.sel(lev=2000)`` is the mask), the same deep-ocean restriction
    ``load_woa_levmean`` applies -- otherwise a shelf column's 0-100 m mean would
    be scaled up as though it were a full 2000 m column.

    Caveats worth carrying into any figure made from this:

    - **It is a climatology, not a time series.** The mapping used measurements
      from 1972-2013 for all surfaces, and the DIC field was normalized to the
      year 2002 with TTD-derived anthropogenic carbon. There is no ``year``
      dimension, so the obs sweep treats it as constant in time and says so in
      the log. Scoring it against generations averaged over 2004-2017 therefore
      carries an era offset against a real deoxygenation trend -- the same class
      of caveat as using WOA as a climatology.
    - umol/kg -> mol/m^3 uses the /1000 convention of the other loaders here (an
      implicit rho = 1000 kg/m^3, ~2.5% low against rho = 1025).
    """
    data = {}
    mask = None
    for var in variables:
        ds = xr.open_dataset(GLODAP_ROOT / f'{var}_obs_GLODAP_r360x180.nc',
                             decode_times=False)
        da = ds[var].rename(depth='lev')
        if mask is None:
            # drop=True: without it the scalar `lev` coord rides along through
            # `.where(mask)` into every channel and on into the cached netCDF
            mask = np.isfinite(da.sel(lev=2000, drop=True))
        da = da.sel(lev=slice(None, 2000))

        lev = da['lev'].values
        mids = (lev[:-1] + lev[1:]) / 2
        outer = np.concatenate([[0.], mids, [2000.]])   # sums to exactly 2000 m
        thick = xr.DataArray(np.diff(outer), dims=('lev',),
                             coords={'lev': da['lev']})

        da_int = 2000 * da.weighted(thick).mean('lev')  # skipna: partial columns
        if var in GLODAP_PERMASS_VARS:
            da_int = da_int / 1000                      # umol/kg -> mol/m^3
        data[var] = da_int.reset_coords(drop=True)

    return xr.Dataset(data).where(mask)


## ---- observations on density (sigma) layers ---- ##


DENSITY_ROOT = paths.DENSITY_SURFACES


def load_obs_density(density_var='sigma_1', product='WOA23',
                     variables=('thetao', 'so')):
    """Observed tracers on sigma layers, as ``<var>_dlev<i>`` channels.

    The observational counterpart of the per-model density files that
    ``diffusion/data_loading.py`` feeds the density runs: same r360x180 grid, same
    number of sigma layers, and one channel per layer named exactly as the
    training channels are.

    **No unit conversion happens here, and none should.** Unlike the
    depth-integrated files -- where GLODAP's O2 and DIC arrive in umol/kg and
    ``load_glodap_levmean`` divides by 1000 -- the sigma-layer files are written
    in mol/m^3 already (DIC ~2.3, O2 ~0.18), matching the per-model density
    files channel for channel. A future product delivered on sigma layers in
    umol/kg would need converting *before* it reaches this function.

    Level index ``i`` means "the i-th volume-weighted density quantile of THIS
    dataset", not a fixed sigma value -- that is the design of
    ``density_surfaces.compute_quantile_density_levels``, which gives every
    model its own levels so index i is the same fraction of ocean volume
    everywhere. The observational levels are likewise the product's own. They
    are returned in the ``density_levels`` attribute so a caller can record
    which sigma values it actually conditioned on.

    Returned as a climatology (no ``year`` dimension): the source files carry a
    single time step. If time-resolved observational density files land later,
    give them a ``year`` dimension here and the rest of the obs sweep follows.
    """
    root = DENSITY_ROOT / density_var
    channels, levels = {}, None
    for var in variables:
        path = root / var / 'obs' / f'{var}_obs_{product}_{density_var}.nc'
        da = xr.open_dataset(path, decode_times=False)[var]
        if levels is None and density_var in da.coords:
            levels = np.asarray(da[density_var].values, dtype=float)
        if 'time' in da.dims:
            da = da.mean('time')            # single-step climatology
        # drop the sigma_1 coord before splitting: it becomes a differing
        # scalar on each layer and would collide when the channels are merged
        da = da.reset_coords(drop=True)
        for i in range(da.sizes['lev']):
            channels[f'{var}_dlev{i}'] = da.isel(lev=i, drop=True)
    ds = xr.Dataset(channels)
    ds.attrs['density_var'] = density_var
    if levels is not None:
        ds.attrs['density_levels'] = ' '.join(f'{v:.4f}' for v in levels)
    return ds


def load_woa_density(density_var='sigma_1'):
    """WOA23 T and S on sigma layers. No O2 or DIC exist for this product."""
    return load_obs_density(density_var, 'WOA23', ('thetao', 'so'))


def load_glodap_density(density_var='sigma_1'):
    """GLODAPv2 T, S and DIC on sigma layers.

    ``o2_obs_GLODAP_sigma_1.nc`` exists beside these and is deliberately NOT
    loaded: the diffusion target is always single-channel *depth-integrated* O2
    whatever the predictors are, so a
    density cell is scored against depth-integrated observed O2 -- which for
    GLODAP comes from ``load_glodap_levmean``. The sigma-layer O2 file has no
    consumer in this pipeline; it was not overlooked.
    """
    return load_obs_density(density_var, 'GLODAP', ('thetao', 'so', 'dissic'))