"""CMIP6 map loading and train/test split helpers.

Collects the subsets of three original helper modules that the training and
evaluation code use, under their original names:

  * ml_framework/util.py           -- file paths and loaders for the regridded,
                                      depth-integrated maps
  * ml_framework/cnn_2d/data_2d_new.py -- ``wrap_load_data`` (annual,
                                      depth-integrated loader)
  * ml_framework/pred_0d_1d/util.py -- ``split_train_test``,
                                      ``split_train_test_insample``,
                                      ``filter_first_model``
"""
import re
import warnings
from pathlib import Path

import ultraplot as pplt
import numpy as np
import xarray as xr
from xmip.preprocessing import cmip6_renaming_dict

import paths
from vendored.ocean_utils_min import ModelCMIP6, centers_from_modelname_dict


# ----------------------------------------------------------------------------
# file paths and loaders (from ml_framework/util.py)
# ----------------------------------------------------------------------------
def member_key(member_id):
    """Return r/i/p/f numbers from member_id string like r1i1p1f1.

    This is useful for sorting a list of models by their modelname and
    member ID.
    """
    pattern = 'r([0-9]+)i([0-9]+)p([0-9]+)f([0-9]+)'
    m = re.search(pattern, member_id)
    return tuple(int(m.group(i)) for i in range(1, 4+1))  # (r, i, p, f)


def depth_range_str(depth):
    if depth is None:  # e.g. tas
        return 'no-depth-level'
    elif type(depth) is slice:
        return f'{depth.start}m-{depth.stop}m'
    elif type(depth) is int:
        return f'{depth}m'
    else:
        raise ValueError(f'{type(depth)=}')


def regrid_yearly_outpath(
        experiment_id, institution_id, source_id, member_id, variable_id,
        depth):

    p = Path(paths.REGRID_YEARLY)
    p /= variable_id
    p /= experiment_id
    p /= depth_range_str(depth)
    p /= (f'{variable_id}_{experiment_id}_{institution_id}_'
          + f'{source_id}_{member_id}_{depth_range_str(depth)}.nc')
    return p


def tau_regrid_yearly_outpath(
        experiment_id, institution_id, source_id, member_id, variable_id,
        depth):

    # tauuo/tauvo (annual + monthly) live in the unified regridded tree.
    # Filename convention is unchanged: <var>_<exp>_<model>_<member>_annual.nc
    # (no modelling-center field).
    p = Path(paths.CMIP6_REGRIDDED)
    p /= f'{experiment_id}/Omon/{variable_id}/annual/'
    p /= f'{variable_id}_{experiment_id}_{source_id}_{member_id}_annual.nc'
    return p


def mld_regrid_yearly_outpath(
        experiment_id, institution_id, source_id, member_id, which):
    assert which in ['max', 'mean']
    p = Path(paths.REGRID_YEARLY) / 'mld'
    p /= f'{experiment_id}/{which}/'
    p /= f'mld_{experiment_id}_{institution_id}_{source_id}_{member_id}_year{which}.nc'
    return p


def ts_single_level_regrid_yearly_outpath(
        var, exp, institution_id, source_id, member_id):
    p = Path(paths.REGRID_YEARLY) / 'ts_single_levels'
    p /= f'{var}/{exp}/'
    p /= f'{var}_{exp}_{institution_id}_{source_id}_{member_id}.nc'
    return p


def load_regrid_yearly(
        experiment_id, institution_id, source_id, member_id, variable_id,
        depth):

    if variable_id in ['tauuo', 'tauvo']:
        p = tau_regrid_yearly_outpath(
            experiment_id=experiment_id, institution_id=institution_id,
            source_id=source_id, member_id=member_id, variable_id=variable_id,
            depth=depth
        )
    elif variable_id in ['mld_max', 'mld_mean']:
        which = variable_id.split('_')[1]
        p = mld_regrid_yearly_outpath(
            experiment_id=experiment_id, institution_id=institution_id,
            source_id=source_id, member_id=member_id,
            which=which
        )
    elif 'thetao_' in variable_id or 'so_' in variable_id:
        varname = variable_id.split('_')[0]
        p = ts_single_level_regrid_yearly_outpath(
            varname, experiment_id, institution_id, source_id, member_id)
    else:
        p = regrid_yearly_outpath(
            experiment_id=experiment_id, institution_id=institution_id,
            source_id=source_id, member_id=member_id, variable_id=variable_id,
            depth=depth
        )

    varname = 'mld' if 'mld' in variable_id else variable_id
    if 'thetao_' in variable_id or 'so_' in variable_id:
        varname = variable_id.split('_')[0]

    warnings.filterwarnings("ignore", category=FutureWarning)
    da = xr.open_dataset(p, use_cftime=True)[varname].squeeze()
    da['time'] = np.array([tt.year for tt in da.time.values])
    da = da.rename(time='year')

    return da


def regrid_monthly_outpath(
        experiment_id, institution_id, source_id, member_id, variable_id,
        depth):
    """Path for monthly depth-integrated maps (the monthly analogue of
    regrid_yearly_outpath).
    """
    p = Path(paths.REGRID_MONTHLY)
    p /= variable_id
    p /= experiment_id
    p /= depth_range_str(depth)
    p /= (f'{variable_id}_{experiment_id}_{institution_id}_'
          + f'{source_id}_{member_id}_{depth_range_str(depth)}.nc')
    return p


def load_regrid_monthly(
        experiment_id, institution_id, source_id, member_id, variable_id,
        depth):
    """Load a monthly depth-integrated map, keeping the monthly `time` axis.

    Unlike load_regrid_yearly (which collapses time to integer `year`), this
    preserves the monthly cftime `time` coordinate so that monthly samples can
    be used directly. Only the plain depth-integrated variables (e.g. o2,
    thetao, so) are supported here -- the tau/mld/single-level special cases of
    load_regrid_yearly are not needed for the monthly setup.
    """
    p = regrid_monthly_outpath(
        experiment_id=experiment_id, institution_id=institution_id,
        source_id=source_id, member_id=member_id, variable_id=variable_id,
        depth=depth
    )

    warnings.filterwarnings("ignore", category=FutureWarning)
    # chunks keeps the data lazy (dask) so that xr.concat across 20+ models
    # doesn't pull everything into RAM before TracerDataset runs
    da = xr.open_dataset(p, use_cftime=True, chunks={'time': 120})[variable_id].squeeze()

    return da


# ----------------------------------------------------------------------------
# annual depth-integrated loader (from ml_framework/cnn_2d/data_2d_new.py)
# ----------------------------------------------------------------------------
def stack(ds):
    """Create single `sample` dimension by stacking the model, basin, exp, year dimensions."""
    # .transpose('sample', 'lev') needed for 1D CNN
    coords = ['model', 'exp', 'year']
    ds = ds.squeeze()
    coords = tuple(coord for coord in coords if coord in ds.coords and ds[coord].size != 1)
    ds = ds.stack(sample=coords)
    #ds = ds.reset_index('sample')  # optional
    return ds


def get_models(exp, variables, depth):
    """List of models with preprocessed 2D data available."""

    # recursive search for hist+SSP combindation
    if 'hist-ssp' in exp:
        ssp = exp.split('-')[1]
        models_hist = get_models('historical', variables, depth)
        models_ssp = get_models(ssp, variables, depth)
        models = set.intersection(set(models_hist), set(models_ssp))
        models = sorted(list(models), key=lambda m: (m.model, member_key(m.member)))
        return models
        
    mdict = {}
    for var in variables:
        if var in ['tauuo', 'tauvo']:
            p = tau_regrid_yearly_outpath(
                experiment_id=exp,
                institution_id=None, source_id=None, member_id=None,
                variable_id=var, depth=depth
            )
            models = [ModelCMIP6(None, *str(f.stem).split('_')[2:4])
                      for f in p.parent.glob('*.nc') if 'TMP' not in f.stem]
            # add modeling center
            models = [ModelCMIP6(centers_from_modelname_dict()[m.model][0],
                                 m.model, m.member)
                      for m in models]
        elif var in ['mld_max', 'mld_mean']:
            which = var.split('_')[1]
            p = mld_regrid_yearly_outpath(
                experiment_id=exp,
                institution_id=None, source_id=None, member_id=None,
                which=which
            )
            models = [ModelCMIP6(*str(f.stem).split('_')[2:5])
                      for f in p.parent.glob('*.nc') if 'TMP' not in f.stem]
        elif 'thetao_' in var or 'so_' in var:
            varname = var.split('_')[0]
            p = ts_single_level_regrid_yearly_outpath(
                var=varname, exp=exp,
                institution_id=None, source_id=None, member_id=None,
            )
            models = [ModelCMIP6(*str(f.stem).split('_')[2:5])
                      for f in p.parent.glob('*.nc') if 'TMP' not in f.stem]
        else:
            p = regrid_yearly_outpath(
                experiment_id=exp,
                institution_id=None, source_id=None, member_id=None,
                variable_id=var, depth=depth
            )
            models = [ModelCMIP6(*str(f.stem).split('_')[2:5])
                      for f in p.parent.glob('*.nc') if 'TMP' not in f.stem]
        mdict[var] = models

    # set intersection (retain only models with all variables)
    models = set.intersection(*[set(models) for models in mdict.values()])

    # sort
    models = sorted(list(models), key=lambda m: (m.model, member_key(m.member)))
    
    return models


def _load_exp(exp, var, model, depth):
    """Load CMIP6 2D (+time) array for single experiment."""
    
    # exp = "hist-sspXXX": historical + ssp
    if 'hist-ssp' in exp:
        ssp = exp.split('-')[1]
        da_hist = _load_exp('historical', var, model, depth)
        da_ssp = _load_exp(ssp, var, model, depth)
        da = xr.concat([da_hist, da_ssp], 'time')
        da = da.assign_coords(exp=exp)
        da = da.sel(time=slice(None, '2100'))
        return da
        
    da = load_regrid_yearly(
        experiment_id=exp,
        institution_id=model.center,
        source_id=model.model, member_id=model.member,
        variable_id=var, depth=depth
    )

    if exp in ['1pctCO2', 'abrupt-4xCO2']:
        da['year'] = np.arange(len(da.year))
    elif exp == 'historical':
        da = da.sel(year=slice(None, 2014))
        da['year'] = np.arange(1850, 2014+1)
    elif 'ssp' in exp:
        da = da.sel(year=slice(None, 2100))
        da['year'] = np.arange(2015, 2100+1)

    # assign coordinates
    da = da.assign_coords(model=model, exp=exp)

    if ((var not in ['tauuo', 'tauvo', 'mld_max', 'mld_mean'])
        and 'thetao_' not in var and 'so_' not in var):
        depth_name = f'{depth.start}m-{depth.stop}m'
        da = da.assign_coords(depth=depth_name)

    if 'thetao_' in var or 'so_' in var:
        lev = int(var.split('_')[1])
        for wrong in cmip6_renaming_dict()['lev']:
            if wrong in da.coords:
                da = da.rename({wrong: 'lev'})
        da = da.sel(lev=lev)

    # drop depth coord
    for coord in ['lev', 'olevel']:
        try:
            da = da.drop(coord)
        except Exception:
            continue
            
    return da


def load_data(variables, experiments, depth, require_vars=None,
              filter_first=False):
    """Load preprocessed 2D fields for all variables & experiments.

    Params:
        variables (list of str)
        experiments (list of str)
        depth (slice)
        require_vars (list of str, default=None): list of variables to require
            (i.e., need to be available), but not load into memory. Useful
            for loading only data from models for which a larger set of
            variables is available in principle.

    Returns:
        ddict (dict): access with ddict[exp][model], contains xr.Dataset
        mdict (dict): access with mdict[exp], contains list of ModelCMIP6
        color_dict (dict): access with color_dict[model], contains color values for mpl
    """

    # dictionaries holding data and model lists
    ddict = {exp: {} for exp in experiments}
    mdict = {exp: [] for exp in experiments}
    
    bad_models = []

    for exp in experiments:
        print(exp)
    
        # get list of models with necessary files (all variables)
        models = get_models(exp, variables, depth)

        if require_vars is not None:
            all_vars = list(set.union(set(variables), set(require_vars)))
            models_with_required = get_models(exp, all_vars, depth)
            models = [m for m in models if m in models_with_required]

        if filter_first:
            models = filter_first_model(models)

        mdict[exp] = models

        ddict_exp = {}
        for model in models:
            try:
                data_vars = {}
                for var in variables:
                    da = _load_exp(exp=exp, var=var, model=model, depth=depth)
                    data_vars[var] = da
                ds_model = xr.Dataset(data_vars=data_vars)
                ddict_exp[model] = ds_model
            except Exception:  # skip model
                bad_models.append(model)
                continue

        # ddict_exp = {
        #     model: xr.Dataset(
        #         data_vars={
        #             var: _load_exp(exp=exp, var=var, model=model, depth=depth)
        #             for var in variables
        #         }
        #     )
        #     for model in models
        # }

        ddict[exp] = ddict_exp
    
    # intersection of model lists
    mdict['int'] = sorted(list(set.intersection(*[set(mdict[exp]) for exp in experiments])), key=lambda m: m.model)
    
    # union of model lists (for color dict)
    mdict['all'] = sorted(list(set.union(*[set(mdict[exp]) for exp in experiments])), key=lambda m: m.model)
    
    # remove datasets/models with all NaNs
    # bad_models = []
    for exp in ddict.keys():
        for model, ds in ddict[exp].items():
            for var in variables:
                if np.isnan(ds[var]).all():
                    bad_models.append(model)
    bad_models = sorted(list(set(bad_models)), key=lambda m: m.model)
    ddict = {exp: {model: ds for model, ds in ddict[exp].items() if model not in bad_models} for exp in ddict.keys()}
    mdict = {exp: [m for m in mdict[exp] if m not in bad_models] for exp in mdict.keys()}
    print(f'Dropped samples from {len(bad_models)} models that were all NaN')
    
    # dictionary of colors for each model for plotting
    colors = (pplt.get_colors('tab20') + pplt.get_colors('tab20')[1:])[::2]
    color_dict = dict(zip(mdict['all'], colors))
    #color_dict = {model: ('tab20', mm) for mm, model in enumerate(mdict['all'])}

    return ddict, mdict, color_dict


def construct_ds(ddict, mdict, experiments, _anom=True):
    """Create annual mean xr.Dataset from ddict dictionary.

    Params:
        ddict: nested dictionary with keys `ddict[exp][][model]`
            containing xr.Dataset with all variables.
        _anom (bool, default=True): whether to compute anomalies wrt. piControl

    Returns:
        ds (xr.Dataset): dataset with coordinates exp, model, depth (+ lev, year).
    """

    if 'piControl' in experiments:
        raise ValueError('piControl should not be part of `experiments` (cannot compute anomaly)')
    
    # list of models
    experiments_req = experiments
    models = sorted(list(set.intersection(*[set(mdict[exp]) for exp in experiments_req])), key=lambda m: m.model)

    data_exps = []
    bad_models = []
    for exp in experiments:
        print(exp)
        data_models = []
        
        #for model in models:  # use intersection of models
        for model in mdict[exp]:
            #print(f'        {model}')
            ds = ddict[exp][model]

            # compute anomaly
            if _anom:
                #ds = anom(ds, ds_pi, exp, model, method='linear', time_dim='year', path=path_exp)
                ds = ds - ds.isel(year=slice(None, 20)).mean('year')
            
            # annual mean
            #ds = ds.groupby('time.year').mean()

            # select years (for models with longer runs)
            if exp in ['abrupt-4xCO2', '1pctCO2']:
                ds = ds.isel(year=slice(None, 150))
            elif 'hist-ssp' in exp:
                ds = ds.isel(year=slice(None, 251))
                
            # if 'hist' in exp:
            #     ds = ds.assign_coords(year=np.arange(1850, 1850+len(ds.year)))
            # else:
                # ds = ds.assign_coords(year=np.arange(len(ds.year)))

            #depth_name = f'{depth.start}m-{depth.stop}m'
            ds = ds.assign_coords(exp=exp, model=model)
            data_models.append(ds)
            
        ds_exp = xr.concat(data_models, 'model')
        data_exps.append(ds_exp)
        
    ds = xr.concat(data_exps, 'exp')
    mdict = {exp: [m for m in mlist if m not in bad_models] for exp, mlist in mdict.items()}
    
    return ds


def wrap_load_data(variables, experiments, depth, anom=True, require_vars=None,
                   filter_first=False):
    # load individual profiles into dictionary
    print('Loading data...')
    ddict, mdict, color_dict = load_data(
        variables=variables,
        experiments=experiments,
        depth=depth,
        require_vars=require_vars,
        filter_first=filter_first
    )

    # print number of models
    print('\nAll data loaded. Number of models:')
    for exp, models in mdict.items():
        print(exp, len(models))

    # construct dataset
    print('\nConstructing xr.Dataset...')
    ds = construct_ds(
        ddict, mdict, experiments=[exp for exp in experiments if exp != 'piControl'],
        _anom=anom
    )

    # stack to create "sample" dimension
    ds_stacked = stack(ds.squeeze())
    len_orig = len(ds_stacked.sample)
    
    # remove invalid samples (only where ALL values are NaN, because continents are always NaN)
    ds_stacked = ds_stacked.dropna('sample', how='all')
    
    len_new = len(ds_stacked.sample)
    print('\nDataset constructed and stacked along sample dimension.')
    print(f'Dropped {len_orig - len_new} samples that were NaN')
    print(f'Have {len_new} samples in total')

    # for var in variables:
    #     assert not np.any(np.isnan(ds_stacked[var].values))

    return ds_stacked, ddict, mdict, color_dict


# ----------------------------------------------------------------------------
# train/test splits (from ml_framework/pred_0d_1d/util.py)
# ----------------------------------------------------------------------------
def filter_first_model(models):
    """Select first member of each model."""
    modelnames = sorted(list(set([m.model for m in models])))
    out = []
    for modelname in modelnames:
        out.append(sorted([m for m in models if m.model == modelname], key=lambda m: member_key(m.member))[0])
    return out


def split_train_test(
    ds,
    split_by,
    training_set, test_set,
    assert_all_samples=True
):
    """Split dataset into train and test parts.

    Params:
        ds (xr.Dataset): dataset with `sample` dimension and exp/basin/model coords
    """
    from collections.abc import Sequence
    
    # split train/test set along single coordinate
    if type(split_by) is str:  # if split_by in ['exp', 'model', 'basin', 'year']
        ds_train = ds.where(ds[split_by].isin(training_set), drop=True)
        ds_test = ds.where(ds[split_by].isin(test_set), drop=True)
        assert set(ds_train[split_by].values).isdisjoint(set(ds_test[split_by].values))
        
    # split along multiple coordinates at the same time
    elif isinstance(split_by, Sequence):
        assert len(split_by) == len(training_set) == len(test_set)
        train_conditions = np.logical_and.reduce([ds[coord].isin(train) for coord, train in zip(split_by, training_set)])
        test_conditions = np.logical_and.reduce([ds[coord].isin(test) for coord, test in zip(split_by, test_set)])

        train_conditions = xr.DataArray(train_conditions, dims='sample')
        test_conditions = xr.DataArray(test_conditions, dims='sample')
        
        ds_train = ds.where(train_conditions, drop=True)
        ds_test = ds.where(test_conditions, drop=True)
            
    # no other option implemented
    else:
        raise NotImplementedError(f'{split_by=}')

    # make sure all samples are used
    if assert_all_samples:
        assert len(ds_train.sample) + len(ds_test.sample) == len(ds.sample)
        
    return ds_train, ds_test


def split_train_test_insample(ds, exp):
    """Train on years 1-80 and 100-150, test on years 80-100.

    Args:
        ds (xr.Dataset): dataset with `year` and `exp` coordinates
        exp (str): which experiment to select for both training & testing
    """
    # handle case of several experiments recursively
    if type(exp) == list:
        train, test = [], []
        for experiment in exp:
            ds_train, ds_test = split_train_test_insample(ds, experiment)
            train.append(ds_train.assign_coords(exp=experiment))
            test.append(ds_test.assign_coords(exp=experiment))
        ds_train = xr.concat(train, 'sample')
        ds_test = xr.concat(test, 'sample')
        return ds_train, ds_test
    
    # # allow exp=None for the case where only one exp was loaded
    # if exp is None:
    #     if not ds.exp.size == 1:
    #         raise ValueError(f"got exp=None but {ds.exp.size=}")
    #     ds = ds.copy()
    #     exp = ds.exp.item()
    # else:
    #     ds = ds.copy().sel(exp=exp)

    try:
        ds = ds.copy().sel(exp=exp)
    except Exception:
        pass
        
    if exp in ['1pctCO2', 'abrupt-4xCO2']:
        years_train = list(range(80+1)) + list(range(100+1, 150+1))
        years_test = list(range(80+1, 100+1))
    elif 'hist-ssp' in exp:
        years_train = list(range(1850, 1950)) + list(range(2050, 2100+1))
        years_test = list(range(1950, 2050))
    elif exp == 'historical':
        years_train = list(range(1850, 1975)) + list(range(2000, 2014+1))
        years_test = list(range(1975, 2000))
    else:
        raise ValueError(exp)
    ds_train = ds.where(ds.year.isin(years_train), drop=True)
    ds_test = ds.where(ds.year.isin(years_test), drop=True)
    #ds_test = ds.where(np.logical_not(ds.year.isin(years_train)), drop=True)
    return ds_train, ds_test
