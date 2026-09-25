"""Fig. 10: the eastern tropical Pacific oxygen minimum zone under
extrapolation -- a) the CMIP6 ensemble-mean O2 offset against WOA23, b) the
correction contributed by the models without oxygen, c) the offset after that
correction, d) the distribution of OMZ-mean O2 across models and (calibrated)
ensemble members for the three populations behind a)-c).

Extracted from general/eval/extrapolation.py (plot). It draws the same cache
as Fig. 9, whose compute step (figures/fig09_extrapolation.py) produces it; the
plot step also re-reads the raw WOA23, GOBAI-O2, RECCAP2 region-mask and
grid-area files.

    python -m figures.fig09_extrapolation compute [--epoch 500]   # GPU
    python -m figures.fig10_extrapolation_omz plot                # no GPU
"""
from __future__ import annotations

from figures import render_common   # first: forces the headless Agg backend

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from figures import eval_common as ec
from figures.eval_common import DEPTH_INT
from figures import fig09_extrapolation as ex
from figures.fig09_extrapolation import (
    CELL, OMZ_REF_LABEL, WOA_O2_FILE, _OMZ_CACHE, cell_slug, load_cache,
    cache_path, omz_obs, omz_mean)


# ---------------------------------------------------------------------------
# pinned manuscript constants (general/manuscript/figures.py, main variant)
# ---------------------------------------------------------------------------
# Panel d) is drawn from the CALIBRATED (variance-inflated) ensemble. Inflation
# preserves the pointwise ensemble mean exactly, so a)-c) are identical either
# way and only the WIDTH of the two generated curves moves.
OMZ_CALIBRATED = True


# ===========================================================================
# from general/eval/extrapolation.py
# ===========================================================================
OUT_DIR = ex.OUT_DIR

OMZ_BIAS_LIMIT = 0.04    # mol/m^3, symmetric, shared by all three maps
# gaussian_kde bw_method, shared by all three curves. A FACTOR on each curve's
# own std, so a narrow population is still resolved rather than smoothed away --
# what it removes is the n-dependence: Scott/Silverman scale as n^(-1/5), and
# truth's ~16 points against the generated curves' ~80 would render truth
# smoother for no reason but sample size.
OMZ_KDE_BW = 0.4

# Which statistic the panel's markers report. The MEDIAN is the default:
# bandwidth-free, and the most consistent candidate across the four
# configurations (|median - obs| 3.0-8.4 mol/m^2 against the mode's 0.6-19.2).
OMZ_STAT = 'median'
# Iglewicz-Hoaglin cut for naming a model an outlier on the rug.
OMZ_Z_THRESHOLD = 3.5


def _woa_o2_stat_field(name):
    """0-2000 m column mean of one WOA oxygen statistic, mol/m^3, cached.

    Same thickness weighting and the same 0-2000 m slice as
    ``obs.load_woa_levmean`` applies to ``o_an``, so a statistic reduced
    here is on the footing of the observed line it annotates. WOA stores
    umol/kg, hence the /1000 -- identical to the loader's conversion.
    """
    key = f'woa_{name}'
    if key not in _OMZ_CACHE:
        ds = xr.open_dataset(WOA_O2_FILE, decode_times=False)
        if name not in ds:
            raise KeyError(f'{WOA_O2_FILE} has no {name!r} '
                           f'(has {list(ds.data_vars)})')
        thick = xr.DataArray(ds.depth_bnds.values[:, 1] - ds.depth_bnds.values[:, 0],
                             dims=('lev',)).assign_coords(lev=ds.depth.values)
        da = (ds[name].drop_vars('time').squeeze().rename(depth='lev')
              .assign_coords(lev=ds.depth.values)) / 1000.0
        sl = slice(None, 2000)
        _OMZ_CACHE[key] = (da.sel(lev=sl)
                           .weighted(thick.sel(lev=sl)).mean('lev').load())
    return _OMZ_CACHE[key]


def omz_obs_band(ds, name='o_se'):
    """Half-width of the observational uncertainty band, mol/m^3.

    **This is the SPATIALLY CORRELATED bound, and the assumption is the whole
    story.** ``o_se`` is a per-cell, per-level standard error, and the drawn
    value is an area-weighted mean over ~4000 OMZ cells. Treating those errors
    as independent divides the band by sqrt(n_eff) and gives **+/-0.057
    mol/m^2**; treating them as fully correlated gives **+/-2.54** -- a factor
    of **44**. WOA's objective analysis has a correlation length of a few
    hundred km, i.e. far more than one 1-degree cell, so the independent bound
    is not credible and this returns the correlated one. It is the bound that
    does not overstate WOA's precision, and it lands within 20% of the
    independent estimate from the inter-product difference (WOA 148.84 vs
    GOBAI 144.68 mol/m^2, half-range +/-2.08).

    Reduced over the cell's own common ocean mask (``isfinite(truth)``), the
    same footprint as the ``omz_obs`` attr it brackets.
    """
    se = _woa_o2_stat_field(name)
    if 'truth' in ds:
        se = se.where(np.isfinite(ds['truth']))
    return float(omz_mean(se))


def omz_obs_other(ds, source):
    """The OMZ-mean observed value from a DIFFERENT product, mol/m^3.

    For the second tick on the observational marker: the band comes from the
    reference product's own error estimate, and this shows where an
    independent product falls, through the identical mask, column integration
    and weighting. Not a substitute for the band -- two products are a range of
    two, not a distribution.
    """
    da = omz_obs(source)
    if 'truth' in ds:
        da = da.where(np.isfinite(ds['truth']))
    return float(omz_mean(da))


# ---------------------------------------------------------------------------
# the three OMZ maps and the PDF in ONE figure
# ---------------------------------------------------------------------------
#: Tighter zoom than the whole mask (lon 138-289): the far-western third, where
#: the mask is empty, is dropped so the eastern-Pacific signal fills the panels.
OMZ_COMBINED_LONLIM = (200, 300)
OMZ_COMBINED_LATLIM = (-35, 35)

_OMZ_COMBINED_TITLES = {
    'bias': 'Ensemble mean {o2} bias\n(simulated by CMIP6 models)',
    'correction': 'Generated {o2} correction\nfrom additional CMIP6 models',
    'corrected': 'Ensemble mean {o2} bias after correction',
}


def omz_curves(ds, calibrated=True):
    """The three KDE curve specs ``(key, suffix, colour, label)``.

    ``calibrated`` selects the variance-inflated series (``omz_*_cal``).
    Inflation preserves the pointwise ensemble MEAN exactly, so this changes
    only the WIDTH of the two generated curves.
    """
    suffix = '_cal' if calibrated else ''
    return [
        ('omz_truth', '', 'C2', f'Ground truth {ec.O2} from CMIP6 models with BGC'),
        ('omz_tso', suffix, 'C1', f'Generated {ec.O2} for CMIP6 models with BGC'),
        ('omz_ts', suffix, 'C0', f'Generated {ec.O2} after correction'),
    ]


def omz_series_check(slug, ds, calibrated=True):
    """Raise unless the cache carries the series the PDF panel needs."""
    missing = [k + sfx for k, sfx, _, _ in omz_curves(ds, calibrated)
               if k + sfx not in ds]
    if missing:
        raise KeyError(
            f'{slug}: cache has no {missing} -- it predates '
            f'{"the calibrated series" if calibrated else "the OMZ series"}. '
            f'Re-run the compute step of this cell.')


# --- per-model attribution of the cached OMZ cloud ----------------------------
def curve_models(ds, curve='omz_ts_cal'):
    """Model names behind ``curve``, in the order its points are concatenated.

    ``omz_ts`` is ``concat(res_tso + res_extra)`` and ``omz_tso`` is
    ``concat(res_tso)``, so the name list is the stamped ``models_tso`` with
    ``models_extra`` appended for the all-model curve. ``models_skipped`` are
    already absent from both.
    """
    tso = str(ds.attrs.get('models_tso', '')).split()
    extra = str(ds.attrs.get('models_extra', '')).split()
    return tso + extra if curve.startswith('omz_ts_') or curve == 'omz_ts' else tso


def per_model(ds, curve='omz_ts_cal'):
    """``(names, values (n_models, n_samples) in mol/m^2)`` for one curve.

    Reshapes the flat cached cloud back onto its models. The row count is
    checked against the stamped model list rather than inferred, because a
    silent off-by-one here would attribute every point to the wrong model and
    still produce a plausible-looking answer.
    """
    v = ds[curve].values
    v = v[np.isfinite(v)] * DEPTH_INT          # cached as mol/m^3 column means
    names = curve_models(ds, curve)
    n = len(names)
    if n == 0 or v.size % n:
        raise ValueError(
            f'{ds.attrs.get("cell", "?")}: {curve} has {v.size} points which do '
            f'not divide by {n} models -- the cache and its model list disagree')
    return names, v.reshape(n, v.size // n)


def modified_z(x):
    """Iglewicz-Hoaglin modified z-score, ``0.6745*(x-med)/MAD``.

    MAD-based on purpose: an outlier cannot inflate the scale it is tested
    against, which is the masking failure a standard-deviation z-score has.
    Returns zeros when the MAD is zero (a degenerate population).
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def _draw_top_marks(ax, marks, xlim, fontsize=6.0, key=True, frame=True):
    """Colour-coded ticks on the TOP axis plus a value key.

    In-place labels cannot work here: the four statistics sit within
    ~130-160 mol/m^2 on an axis that must span 0-470 to contain the outlier
    models, so any text anchored at its marker's x overlaps its neighbours. So
    the marker keeps its position on the axis and the label moves to a key,
    which also lets it carry the VALUE.
    """
    # local import: the plotting stack is only needed once a figure exists.
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker

    tr = ax.get_xaxis_transform()               # x in data, y in axes fraction
    for m in marks:
        if not np.isfinite(m['x']):      # key-only entry: label, no tick
            continue
        ax.plot([m['x'], m['x']], [0.945, 1.0], transform=tr, color=m['color'],
                lw=1.4, clip_on=False, zorder=5, solid_capstyle='butt')
    if not key:
        return
    # Upper-RIGHT, right-aligned: the curves rise steeply on the left and peak
    # at or left of centre, while the right tail decays, so the top-right corner
    # is free. Right alignment also columnises the trailing numbers. ONE rounded
    # opaque panel behind the whole key (an offsetbox sizes itself to the
    # text). Anchored at y=0.94, below the top-axis ticks at 0.945-1.0.
    #
    # The rows fall into blocks -- the statistics read off the ensembles, then
    # the observational references -- separated by a wider sep. A mark opens a
    # block by carrying 'gap', since which rows are present varies per cell.
    blocks, cur = [], []
    for m in marks:
        if m.get('gap') and cur:
            blocks.append(cur)
            cur = []
        cur.append(m)
    blocks.append(cur)

    def _rows(block):
        # sep is in points
        return VPacker(align='right', pad=0, sep=1.5, children=[
            TextArea(m['label'],
                     textprops=dict(color=m['color'], size=fontsize))
            for m in block])

    child = (_rows(blocks[0]) if len(blocks) == 1 else
             VPacker(children=[_rows(b) for b in blocks],
                     align='right', pad=0, sep=4.5))
    # `frame` off drops the opaque panel and leaves the bare rows (for narrow
    # panels, where an opaque box would hide the curves); unframed there is no
    # frame for `pad` to clear, so it goes to 0.
    key_box = AnchoredOffsetbox(
        loc='upper right', pad=(0.3 if frame else 0.0),
        borderpad=(0.25 if frame else 0.05), frameon=frame,
        bbox_to_anchor=(1.0, 0.94), bbox_transform=ax.transAxes, child=child)
    if frame:
        key_box.patch.set(boxstyle='round,pad=0.15,rounding_size=0.35',
                          facecolor='w', edgecolor='0.75', linewidth=0.6,
                          alpha=1.0)
    key_box.set_zorder(6)
    ax.add_artist(key_box)


def draw_omz_pdf(ax, ds, calibrated=True, bw=OMZ_KDE_BW, legend=True,
                 title=None, stat=OMZ_STAT, rug=True, band=True,
                 mark_outliers=True, fontsize=6.0, key_frame=True):
    """Draw the OMZ ensemble KDE (panel d) into ONE axes.

    Returns ``dict(handles, vals, obs, curves, stats, outliers, band)`` -- the
    caller needs ``vals`` and ``obs`` to log the per-curve statistics.

    The x range is computed from THIS cache's own values: two OMZ cells
    extrapolate over different model populations and are not directly
    comparable as drawn, so marks, rug and band are all computed PER PANEL.

    * ``stat`` selects which statistic the markers report (default the
      **median**). The label names the statistic.
    * ``rug`` draws one tick per MODEL (its mean over samples), and
      ``mark_outliers`` names those exceeding |modified z| = 3.5 on the
      corrected curve. Where any are found, a second marker reports the
      statistic recomputed WITHOUT them.
    * ``band`` shades WOA's own standard error of the analysed mean,
      propagated through this cell's mask and weighting, plus a tick for GOBAI.
    """
    import scipy.stats

    stat_fn = {'median': np.median, 'mean': np.mean}[stat]
    curves = omz_curves(ds, calibrated)
    omz_series_check(str(ds.attrs.get('cell', '?')), ds, calibrated)
    # mol/m^2 over the 0-2000 m column: the depth INTEGRAL, which is what the
    # model actually carries. The cached series are column means (mol/m^3).
    vals = {k + sfx: ds[k + sfx].values[np.isfinite(ds[k + sfx].values)] * DEPTH_INT
            for k, sfx, _, _ in curves}
    obs = float(ds.attrs['omz_obs']) * DEPTH_INT

    # --- per-model attribution, for the rug and the outlier sensitivity ------
    corr_key = curves[-1][0] + curves[-1][1]        # the all-model curve
    model_vals, outliers, stat_excl = None, [], None
    try:
        names, per = per_model(ds, corr_key)
        model_vals = per.mean(axis=1)               # one value per model
        if mark_outliers:
            z = modified_z(model_vals)
            outliers = [(names[k], float(model_vals[k]), float(z[k]))
                        for k in np.where(np.abs(z) > OMZ_Z_THRESHOLD)[0]]
    except (ValueError, KeyError) as exc:           # cache/model-list mismatch
        logging.warning('  omz pdf: no per-model rug (%s)', exc)
    if outliers:
        drop = {n for n, _, _ in outliers}
        keep = np.array([n not in drop for n in names])
        stat_excl = float(stat_fn(per[keep].ravel()))

    obs_lo = obs_hi = None
    if band:
        try:
            half = float(omz_obs_band(ds)) * DEPTH_INT
            obs_lo, obs_hi = obs - half, obs + half
        except Exception as exc:                    # missing WOA stats fields
            logging.warning('  omz pdf: no observational band (%s)', exc)

    pool = [np.concatenate(list(vals.values())), [obs]]
    if obs_lo is not None:
        pool.append([obs_lo, obs_hi])
    pool = np.concatenate(pool)
    pad = 0.25 * max(pool.max() - pool.min(), 1e-6)
    xvals = np.linspace(pool.min() - pad, pool.max() + pad, 1000)
    xlim = (xvals[0], xvals[-1])

    # --- observational uncertainty, behind everything -----------------------
    if obs_lo is not None:
        ax.axvspan(obs_lo, obs_hi, color='k', alpha=0.10, lw=0, zorder=0)

    # back to front (ts, tso, truth) so the reference curve stays visible where
    # it coincides with the reconstruction -- what a good run produces.
    handles, stats, marks = {}, {}, []
    for key, sfx, color, label in reversed(curves):
        v = vals[key + sfx]
        lw = 2.2 if key == 'omz_truth' else 1.4
        pdf = scipy.stats.gaussian_kde(v, bw_method=bw).pdf(xvals)
        handles[key] = ax.plot(xvals, pdf, color=color, lw=lw, label=label)[0]
        ax.fill_between(xvals, np.zeros_like(xvals), pdf, color=color,
                        alpha=0.15)
        stats[key] = float(stat_fn(v))
    # marks in reading order (truth, tso, ts); the key carries the VALUE and
    # each label names its statistic.
    short = {'omz_truth': 'truth', 'omz_tso': 'generated', 'omz_ts': 'corrected'}
    for key, sfx, color, _ in curves:
        marks.append(dict(x=stats[key], color=color,
                          label=f'{stat} {short[key]}  {stats[key]:.0f}'))
    if stat_excl is not None:
        marks.append(dict(x=stat_excl, color=curves[-1][2],
                          label=f'{stat} corrected, no outliers  '
                                f'{stat_excl:.0f}'))
    obs_label = f'{OMZ_REF_LABEL}  {obs:.0f}'
    if obs_lo is not None:
        obs_label += f' $\\pm$ {0.5 * (obs_hi - obs_lo):.0f}'
    marks.append(dict(x=obs, color='k', label=obs_label, gap=True))
    if band and obs_lo is not None:
        try:
            gob = float(omz_obs_other(ds, 'gobai')) * DEPTH_INT
            marks.append(dict(x=gob, color='0.45', label=f'GOBAI-O2  {gob:.0f}'))
        except Exception as exc:
            logging.warning('  omz pdf: no GOBAI tick (%s)', exc)

    obs_h = ax.axvline(obs, color='k', ls='--', lw=1.2,
                       label=f'Observed ({OMZ_REF_LABEL})')
    handles['obs'] = obs_h
    _draw_top_marks(ax, marks, xlim, fontsize=fontsize, frame=key_frame)

    # --- rug: one tick per model, outliers picked out ------------------------
    if rug and model_vals is not None:
        tr = ax.get_xaxis_transform()
        out_x = {v for _, v, _ in outliers}
        # Outliers are the corrected curve's OWN colour, like every other rug
        # tick -- they are members of that population, not a separate series.
        # Height, width and opacity pick them out.
        for x in model_vals:
            is_out = x in out_x
            ax.plot([x, x], [0.0, 0.045 if is_out else 0.028], transform=tr,
                    color=curves[-1][2],
                    lw=(1.4 if is_out else 0.9),
                    alpha=(1.0 if is_out else 0.55), zorder=4, clip_on=False)
        # The labels sit just above the corrected curve's own right-hand bump;
        # they are staggered because neighbouring names overlap horizontally.
        for k, (name, x, _) in enumerate(sorted(outliers, key=lambda t: t[1])):
            ax.text(x, 0.085 + 0.036 * k, name, transform=tr,
                    color=curves[-1][2],
                    fontsize=fontsize, ha='center', va='bottom', zorder=5)

    ax.format(xlim=xlim, ylim=(0, None),
              xlabel=(f'OMZ-mean {ec.O2}, 0-2000 m column integral '
                      f'[mol m$^{{-2}}$]'),
              ylabel='Probability density',
              title=(f'Ensemble distribution of {ec.O2}\n'
                     f'in the tropical eastern Pacific OMZ') if title is None
              else title,
              gridminor=False, grid=True)
    if legend:
        ax.legend([handles[k] for k, _, _, _ in curves] + [obs_h], loc='b',
                  ncols=1)
    return dict(handles=handles, vals=vals, obs=obs, curves=curves,
                stats=stats, outliers=outliers, stat_excl=stat_excl,
                stat=stat, band=(obs_lo, obs_hi))


def plot_cell_omz_combined(cell, ds, limit=None, lonlim=None, latlim=None,
                           calibrated=True, bw=OMZ_KDE_BW, save_path=None):
    """a-c) the three OMZ maps, d) the ensemble PDF -- one figure.

    Panel a) mixes eras: the truth is the years 81-100 mean while the
    observations are present-day, so a) and c) are model-minus-observation
    OFFSETS carrying forced deoxygenation on top of the model bias; b) lies
    within one window.

    ``calibrated`` draws panel d from the variance-inflated ensemble
    (``omz_*_cal``, see ``fig09_extrapolation.omz_calibrated``). Inflation
    preserves the pointwise ensemble MEAN exactly, so a-c are identical either
    way and only the WIDTH of the two generated curves moves.

    One shared ``bw`` for all three curves -- with SciPy's per-dataset default
    the truth curve (one point per model) would get a much wider kernel than the
    generated ones (n_samples x models), and its width would read as a property
    of the ensemble rather than of the sample size.
    """
    pplt, contourf_plot = ec._pplt(), ec._contourf()

    obs_field = omz_obs()
    bias = ds['truth'] - obs_field
    correction = ds['gen_ts'] - ds['gen_tso']
    maps = dict(bias=bias, correction=correction, corrected=bias + correction)
    keys = ('bias', 'correction', 'corrected')
    if limit is None:
        limit = OMZ_BIAS_LIMIT
    lonlim = tuple(lonlim or OMZ_COMBINED_LONLIM)
    latlim = tuple(latlim or OMZ_COMBINED_LATLIM)
    omz_series_check(cell_slug(cell), ds, calibrated)

    # 4 = the PDF, spanning both rows on the right; the maps sit in an L-shape.
    axgrid = [[1, 1, 2, 2, 4, 4, 4],
              [0, 3, 3, 0, 4, 4, 4]]
    fig, axs = pplt.subplots(
        axgrid, refwidth='52mm', share=0,
        proj={1: 'pcarree', 2: 'pcarree', 3: 'pcarree', 4: 'cartesian'},
        proj_kw=dict(lon_0=float(np.mean(lonlim))))
    for ax in list(axs)[:3]:
        ax.format(lonlim=lonlim, latlim=latlim)
    kw = dict(vmin=-float(limit), vmax=float(limit), extend='both', cmap='Div')
    for i, key in enumerate(keys):
        ax = axs[i]
        contourf_plot(ax=ax, da=maps[key], contourf_kw=kw,
                      cbar=(i == len(keys) - 1), cbar_label=ec.CBAR_LABEL)
        ax.format(title=_OMZ_COMBINED_TITLES[key].format(o2=ec.O2))

    got = draw_omz_pdf(axs[3], ds, calibrated=calibrated, bw=bw)
    vals, obs, curves = got['vals'], got['obs'], got['curves']
    # abcloc='ul' (inside the axes), NOT the default placement: the three map
    # titles are two-line and wider than the 52 mm axes, so the default puts
    # the letter ON the title.
    axs.format(abc='a)', abcloc='ul')
    save_path = save_path or OUT_DIR / f'{cell_slug(cell)}_omz_combined.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.save(save_path, dpi=250)
    logging.info(f'  saved {save_path}')
    for key, sfx, _, _ in curves:
        v = vals[key + sfx]
        logging.info(f'    {key + sfx:<14} n={v.size:>4}  mean={v.mean():.2f}  '
                     f'std={v.std(ddof=0):.2f} mol/m^2')
    logging.info(f'    {"observed":<14} {"":>6}  mean={obs:.2f}')
    return fig


# ===========================================================================
# entry points
# ===========================================================================
def compute(epoch=ex.DEFAULT_EPOCH):
    """Fig. 10 draws Fig. 9's cache; this is ``fig09_extrapolation.compute``."""
    return ex.compute(epoch)


def plot():
    ds = load_cache(CELL)
    if ds is None:
        raise SystemExit(f'no cache at {cache_path(CELL)}; run '
                         f'`python -m figures.fig09_extrapolation compute` first')
    fig = plot_cell_omz_combined(CELL, ds, calibrated=OMZ_CALIBRATED)
    # abc off: plot_cell_omz_combined letters its own axes.
    render_common.finish(fig, abc=False)
    return render_common.save(fig, 'extrapolation_omz')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=['compute', 'plot'])
    p.add_argument('--epoch', default=str(ex.DEFAULT_EPOCH),
                   help="checkpoint epoch, or 'latest' (default 500)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if args.step == 'compute':
        compute('latest' if args.epoch.lower() == 'latest' else int(args.epoch))
    else:
        plot()
