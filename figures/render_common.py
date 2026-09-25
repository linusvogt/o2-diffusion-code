"""Final styling and saving of a manuscript figure.

Extracted from general/manuscript/render.py (``render_variant`` /
``save_figure``). Every figure is drawn by an eval plotter, which returns the
ultraplot figure unclosed; this module then

  1. strips the diagnostic suptitle and optionally adds panel letters
     (:func:`finish`), and
  2. saves it to ``paths.MANUSCRIPT_DIR/<name>/<name>.<fmt>`` plus a ``.png``
     companion with identical save kwargs (:func:`save`).

Import this module BEFORE anything that imports matplotlib: it forces the
headless ``Agg`` backend (an explicit ``MPLBACKEND`` still wins).

The eval plotters also save a PNG under their own cell slug into their
module's ``OUT_DIR`` as a side effect. Point that elsewhere for the duration of
the call (``mod.set_out_dir(staging)``, restoring ``OUT_DIR`` *and*
``CACHE_DIR`` afterwards) so stray slug-named files do not land next to the
real output and the next ``load_cache`` still finds its cache.
"""
import os
from pathlib import Path

# Headless always: this code only ever writes files.
os.environ.setdefault('MPLBACKEND', 'Agg')

import paths


def finish(fig, abc=False):
    """Strip the eval suptitle and, if ``abc``, stamp panel letters a), b), ...

    The eval suptitles are diagnostic (fleet size, epoch, year label); that
    information belongs in the caption. Letters sit inside the axes at the
    upper left.
    """
    fmt_kw = {'suptitle': ''}
    if abc:
        fmt_kw.update(abc='a)', abcloc='ul')
    fig.format(**fmt_kw)
    return fig


def save_figure(mpl_fig, path, **save_kw):
    """Save one figure to ``path`` AND, unless that already is one, to a ``.png``
    beside it, with **identical** kwargs.

    Returns the primary path.
    """
    path = Path(path)
    mpl_fig.save(path, **save_kw)
    if path.suffix.lower() != '.png':
        mpl_fig.save(path.with_suffix('.png'), **save_kw)
    return path


def save(fig, name, fmt='pdf'):
    """Save to ``paths.MANUSCRIPT_DIR/<name>/<name>.<fmt>`` (+ .png) at 300 dpi,
    then close the figure. Returns the primary path."""
    import matplotlib.pyplot as plt
    fig_dir = Path(paths.MANUSCRIPT_DIR) / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, fig_dir / f'{name}.{fmt}', dpi=300)
    plt.close(fig)
    return path
