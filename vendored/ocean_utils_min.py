"""Vendored subset of ocean_utils (author's utility library), included so this release is self-contained.

Function bodies are unchanged; only what this release uses is kept, from
ocean_utils/cmip6.py (ModelCMIP6, without its site-specific methods;
centers_from_modelname_dict), ocean_utils/general.py (nice_title, log_setup,
regrid, add_cyclic) and ocean_utils/plotting.py (fix_orca, contourf_plot).
Heavy optional dependencies (cartopy, xesmf, scipy) are imported inside the
functions that need them.
"""
import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

import paths


# ---------------------------------------------------------------------------
# CMIP6 model identity (ocean_utils/cmip6.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCMIP6:
    """Represents a CMIP6 model by its institution, name, and member ID."""

    center: str
    model: str
    member: str

    @classmethod
    def from_fname_repr(cls, fname_repr: str):
        """Construct a ModelCMIP6 object from a string center_model_member"""
        assert fname_repr.count('_') == 2
        center, model, member = fname_repr.split('_')
        return cls(center, model, member)

    def __eq__(self, other):
        if isinstance(other, ModelCMIP6):
            return self.__key() == other.__key()
        return NotImplemented

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __gt__(self, other):
        return (other.center, other.model, other.member) > (
            other.center, other.model, other.member)

    def __key(self):
        return (self.center, self.model, self.member)

    def __hash__(self):
        return hash(self.__key())

    def __repr__(self):
        r = f'<ModelCMIP6: {self.center} | {self.model} | {self.member}>'
        return r

    def fname_repr(self) -> str:
        return f'{self.center}_{self.model}_{self.member}'

    def center_model_repr(self) -> str:
        return f'{self.center}-{self.model}'

    def center_model_member_repr(self) -> str:
        return f'{self.center}-{self.model}-{self.member}'

    def model_member_repr(self) -> str:
        return f'{self.model}_{self.member}'

    def human_repr(self) -> str:
        return self.center_model_member_repr()


def centers_from_modelname_dict():
    out = {}
    fname = str(paths.CMIP6_CENTERS_CSV)
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        for rr, row in enumerate(reader):
            if rr == 0:
                continue
            model, centers = row
            centers = centers.split(':')
            out[model] = centers
    return out


# ---------------------------------------------------------------------------
# logging (ocean_utils/general.py)
# ---------------------------------------------------------------------------
def nice_title(msg, for_logfile=True):
    """Transform a string into a nice title to be used for a logging file.

    TODO: wrap to 80 characters
    """
    import time
    from pathlib import Path

    import __main__

    try:
        script_name_str = (
            f"Logfile for {Path(__main__.__file__).name}\n" if for_logfile else ""
        )
    except:
        script_name_str = ""
    ct = str(time.ctime()) + "\n"
    underline = "-" * max([len(msg), len(ct), len(script_name_str)]) + "\n"
    msg = underline + script_name_str + ct + underline + msg + "\n" + underline
    if for_logfile:
        msg = "\n" + msg
    return msg


def log_setup(name: str, level="info", reverse=False):
    import logging

    p = Path(".")
    if "/" in name:
        spl = name.split("/")
        p /= "/".join(spl[:-1])
        name = spl[-1]
    rev_str = "_r" if reverse else ""
    filename = p / f"LOG_{name}{rev_str}.log"
    level_obj = {"info": logging.INFO, "debug": logging.DEBUG}[level]
    logging.basicConfig(
        filename=filename,
        filemode="w",
        level=level_obj,
        format="> %(message)s",
        force=True,
    )
    logging.info(nice_title("Start."))
    print(f"logfile created: {str(filename)}", flush=True)


# ---------------------------------------------------------------------------
# plotting (ocean_utils/general.py, ocean_utils/plotting.py)
# ---------------------------------------------------------------------------
def regrid(da, xres=1, yres=0.5, ignore_degenerate=True, periodic=True):
    """Regrid a dataset to a globally uniform lat-lon grid."""
    import xesmf as xe

    da_name = "var" if da.name is None else da.name
    ds = da.to_dataset(name=da_name)

    target_grid = xe.util.grid_global(d_lon=xres, d_lat=yres)

    regridder = xe.Regridder(
        ds,
        target_grid,
        "bilinear",
        periodic=periodic,
        ignore_degenerate=ignore_degenerate,
    )

    ds_out = regridder(ds)
    da_out = ds_out[da_name]
    return da_out


def add_cyclic(da, lon):
    from cartopy.util import add_cyclic_point

    try:
        da = da.transpose("lat", "lon")
    except Exception as e:
        try:
            da = da.transpose("latitude", "longitude")
        except Exception:
            da = da.transpose("y", "x")

    cyc_da, cyc_lon = add_cyclic_point(da, coord=lon)
    try:
        da = xr.DataArray(cyc_da, coords=dict(lon=cyc_lon, lat=da.lat))
    except Exception:
        da = xr.DataArray(cyc_da.T, coords=dict(lon=cyc_lon, lat=da.lat)).T
    return da, da.lon


# ocean_utils/plotting.py imports these two as private aliases, because
# contourf_plot's keyword arguments of the same name shadow them.
_regrid = regrid
_add_cyclic = add_cyclic


def fix_orca(lon: np.ndarray, lat: np.ndarray, data: np.ndarray):
    """Fix ORCA grid data by projecting onto regular lon-lat grid.

    Linearly interpolate longitude, latitude and data values onto a global
    regular 1-degree grid.
    Note: currently works only for 2d data, no time-dependence.

    From: ajheaps.github.io/cf-plot/irregular.html

    Args:
        lon (ndarray): 2D longitudes on original grid (e.g. ORCA)
        lat (ndarray): 2D latitudes on original grid (e.g. ORCA)
        data (ndarray): 2D data values on original grid (e.g. ORCA)

    Returns:
        lon_new (ndarray): 2D longitudes on new regular grid
        lat_new (ndarray): 2D latitudes on new regular grid
        data_new (ndarray): 2D data values on new regular grid

    TODO: add support for arrays with time dimension (broadcast over time)
    """
    from scipy.interpolate import griddata

    assert data.ndim == 2, f'data must be 2D, got {data.ndim=}'

    # convert to numpy array to allow flattening
    lon, lat, data = np.array(lon), np.array(lat), np.array(data)

    lon, lat, data = lon.flatten(), lat.flatten(), data.flatten()
    pts = np.squeeze(np.where(lon < -150))
    lon = np.append(lon, lon[pts]+360)
    lat = np.append(lat, lat[pts])
    data = np.append(data, data[pts])

    pts = np.squeeze(np.where(lon > 150))
    lon = np.append(lon, lon[pts]-360)
    lat = np.append(lat, lat[pts])
    data = np.append(data, data[pts])

    # target grid: global 1 degree
    xpts = np.arange(-180, 180.25, 1)
    ypts = np.arange(-90, 90.25, 1)
    lon_new, lat_new = np.meshgrid(xpts, ypts)

    data_new = griddata((lon, lat), data, (lon_new, lat_new),
                        method='linear')

    return lon_new, lat_new, data_new


def contourf_plot(
    ax, da,
    format_kw=None, contourf_kw=None,
    regrid=False, add_cyclic=False,
    lon=None, lat=None, fix_grid=False,
    southern_ocean=False, boundinglat=-30,
    cbar=True, cbar_label=None,
    cbar_kw=None,
    stipling=None, stipling_style=None, stipling_color='k',
    fill=True
):
    """Plot a contour+contourf map of a 2D array.
    Includes gridlines, coastlines, and black land mask.

    Parameters:
        ax: proplot axis with (e.g.) proj='robin'
        da (xr.DataArray): 2D array to plot
        contourf_kw:  dict with keys 'cmap', 'levels', and 'extend'
        stipling (xr.DataArray): 2D array with 1=stipling, else no stipling.
            same shape as `da`

    Returns:
        c -- (QuadContourSet) return value of ax.contourf() call, to be used
             for e.g. colorbar
    """
    try:
        da = da.compute()
    except Exception:
        pass

    if lat is None:
        lat = da.lat
    if lon is None:
        lon = da.lon

    orig_lon = lon
    if add_cyclic:
        da, lon = _add_cyclic(da, lon)

    if regrid:
        da = _regrid(da)
        lon = da.lon
        lat = da.lat

    if southern_ocean:
        da = da.where(da.lat <= boundinglat)
        ax.format(boundinglat=boundinglat)

    if fix_grid:
        lon, lat, da = fix_orca(lon, lat, da)

    contour_func = ax.contourf if fill else ax.contour

    if contourf_kw is None:
        im = contour_func(lon, lat, da)
    else:
        im = contour_func(lon, lat, da, **contourf_kw)

    if cbar:
        if cbar_label is not None and cbar_kw is not None:
            try:
                cbar = ax.colorbar(im, label=cbar_label, **cbar_kw)
            except TypeError:
                msg = f'got {cbar_label=} and {cbar_kw["label"]=}, '
                msg += 'disregarding cbar_label'
                warnings.warn(msg)
                cbar = ax.colorbar(im, **cbar_kw)
        elif cbar_kw is not None:
            cbar = ax.colorbar(im, **cbar_kw)
        elif cbar_label is not None:
            cbar = ax.colorbar(im, label=cbar_label)
        else:
            cbar = ax.colorbar(im)

    format_kw_default = dict(land=True, labels=False, gridlinewidth=0)
    if format_kw is not None:
        format_kw_default.update(format_kw)
        format_kw = format_kw_default
    else:
        format_kw = format_kw_default
    if not format_kw['land']:
        format_kw['coast'] = True
    ax.format(**format_kw)

    # TODO don't use recursion
    if stipling is not None:
        if stipling_style is None:
            stipling_style = '....'
        contourf_plot(ax=ax, da=stipling,
                      contourf_kw=dict(levels=[0.8, 1.2], hatches=[
                                       stipling_style], hatchcolor=stipling_color),
                      cbar=False, southern_ocean=southern_ocean,
                      format_kw=format_kw,
                      regrid=regrid,
                      lon=orig_lon, lat=lat,
                      fix_grid=fix_grid, cbar_label=cbar_label,
                      cbar_kw=cbar_kw, add_cyclic=add_cyclic,
                      boundinglat=boundinglat)

    return im
