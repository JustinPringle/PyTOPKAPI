"""plot_flow_precip.py

Diagnostic hydrograph + hyetograph plotting for PyTOPKAPI results.

Supersedes ``plot_Qsim_Qobs_Rain.py``. The differences all matter for
debugging a run that produced zero flow:

  * The outlet is located from the network topology (``cell_down < 0``),
    not hard-coded as column 0. Column 0 is a corner cell in the canonical
    W->E, N->S ordering and is almost never the outlet; ``Qc_out`` is set to
    0 for every non-channel cell, so plotting column 0 shows a flat zero
    even when the solver routed water perfectly.
  * The time axis is derived from the simulation clock (``Dt`` from
    ``global_param.dat`` + a start datetime), so observed flow is *optional*.
    Ungauged catchments (e.g. the Ohlanga) plot fine with no ``Qobs`` file.
  * A short flow summary is printed so "zero everywhere" is diagnosed at the
    source: it says whether the outlet series is genuinely zero, or whether
    the previous tool was simply reading the wrong cell.
  * Guards for all-zero rainfall, NaN/inf, and a degenerate (zero-variance)
    Nash denominator.

Pure functions do the work; ``run(ini_file)`` is a thin config wrapper that
mirrors the rest of ``results_analysis``.
"""

import datetime as dt
import warnings
from configparser import ConfigParser

import h5py
import numpy as np

import pytopkapi.utils as ut
import pytopkapi.pretreatment as pm


# --------------------------------------------------------------------------
# readers / topology helpers  (reused by the other diagnostic plots)
# --------------------------------------------------------------------------

def read_dt_seconds(file_global_param):
    """Timestep length ``Dt`` (seconds) from ``global_param.dat``."""
    _, Dt, _, _, _, _, _, _ = pm.read_global_parameters(file_global_param)
    return float(Dt)


def find_outlet_index(file_cell_param):
    """Row index of the catchment outlet in ``cell_param.dat``.

    The outlet is the single cell whose downstream neighbour is off-map
    (``cell_down < 0``). Result columns are written in ``cell_param`` row
    order (model.py writes ``dset[t+1, cell]``), so this index selects the
    correct column of every result array.
    """
    params = np.loadtxt(file_cell_param)
    cell_down = params[:, 14].astype(int)
    outlets = np.where(cell_down < 0)[0]

    if len(outlets) == 0:
        raise ValueError(
            "No outlet found: no cell has cell_down < 0 in %s. "
            "The network is not terminated correctly." % file_cell_param)
    if len(outlets) > 1:
        warnings.warn(
            "%d cells have cell_down < 0 (expected 1). Using the first, "
            "row %d. A multi-outlet parameter file usually means the mask "
            "or flow-direction raster leaks off the catchment."
            % (len(outlets), outlets[0]))
    return int(outlets[0])


def build_timeline(start, dt_seconds, n_steps):
    """Interval-ending datetimes: ``start + Dt, ..., start + n*Dt``.

    Matches the forcing convention (each value is the accumulation/mean over
    the preceding ``Dt``) and the result slicing below, which drops the
    initial-condition row.
    """
    step = dt.timedelta(seconds=dt_seconds)
    return np.array([start + (i + 1) * step for i in range(n_steps)])


def outlet_channel_flow(file_sim, outlet_index):
    """Simulated channel outflow at the outlet, shape ``(n_t,)`` in m3/s.

    Drops row 0 (the initial condition) so it aligns with the forcing clock.
    """
    Qc = ut.read_one_array_hdf(file_sim, 'Channel', 'Qc_out')
    return Qc[1:, outlet_index]


def mean_catchment_rain(file_rain, group_name):
    """Spatial-mean rainfall per timestep, shape ``(n_t,)`` in mm."""
    with h5py.File(file_rain, 'r') as h5:
        rain = h5['/%s/rainfall' % group_name][...]
    return rain.mean(axis=1)


def read_observed_flow(file_name):
    """Observed flow from a ``year month day hour minute flow`` ASCII file."""
    date = np.loadtxt(file_name, dtype=np.int32, usecols=(0, 1, 2, 3, 4))
    date = np.atleast_2d(date)
    dates = np.array([dt.datetime(*row) for row in date])
    Q = np.loadtxt(file_name, usecols=(5,))
    return dates, np.atleast_1d(Q)


def align_observed(sim_dates, obs_dates, obs_Q):
    """Observed flow resampled onto ``sim_dates`` by timestamp match.

    Returns an array the length of ``sim_dates`` with NaN where there is no
    matching observation, so the two series never silently misalign by index.
    """
    lookup = {d: q for d, q in zip(obs_dates, obs_Q)}
    return np.array([lookup.get(d, np.nan) for d in sim_dates])


# --------------------------------------------------------------------------
# flow summary  (the "zero everywhere" check)
# --------------------------------------------------------------------------

def flow_summary(ar_Qsim, outlet_index, verbose=True):
    """Cheap sanity stats on the simulated outlet series.

    Returns a dict; prints a one-block report when ``verbose``. This is the
    first thing to read when the hydrograph looks flat: it distinguishes
    "the solver produced no flow" from "the plot read the wrong cell".
    """
    finite = np.isfinite(ar_Qsim)
    stats = {
        'outlet_index': outlet_index,
        'n_steps': int(ar_Qsim.size),
        'n_nonfinite': int((~finite).sum()),
        'n_positive': int((ar_Qsim[finite] > 0).sum()),
        'max': float(np.nanmax(ar_Qsim)) if finite.any() else np.nan,
        'mean': float(np.nanmean(ar_Qsim)) if finite.any() else np.nan,
    }
    if verbose:
        print("--- outlet flow summary ---")
        print("  outlet cell (col) : %d" % stats['outlet_index'])
        print("  timesteps         : %d" % stats['n_steps'])
        print("  non-finite values : %d" % stats['n_nonfinite'])
        print("  steps with Q > 0  : %d" % stats['n_positive'])
        print("  peak / mean (m3/s): %.4g / %.4g" % (stats['max'], stats['mean']))
        if stats['n_positive'] == 0:
            print("  >> outlet flow is zero for the whole run. If the outlet "
                  "cell above is a corner/headwater, the network termination "
                  "is wrong; if it is the true mouth, look upstream with "
                  "audit_stores() -- storage may never be generating runoff.")
    return stats


# --------------------------------------------------------------------------
# the plot
# --------------------------------------------------------------------------

def plot_hydrograph(ar_date, ar_Qsim, ar_rain=None, ar_Qobs=None,
                    title='', ax=None, image_out=None):
    """Simulated hydrograph with an inverted rainfall hyetograph.

    ``ar_Qobs`` and ``ar_rain`` are optional. Returns the primary axis so the
    figure can be composed into a larger diagnostic panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    lines, labels = [], []
    if ar_Qobs is not None and np.isfinite(ar_Qobs).any():
        lines += ax.plot(ar_date, ar_Qobs, 'r-', lw=1.0)
        labels.append('Observed')
    lines += ax.plot(ar_date, ar_Qsim, 'k-', lw=1.0)
    labels.append('Model')

    ax.set_xlim(ar_date[0], ar_date[-1])
    ax.set_ylabel(r'$Q\ (\mathrm{m^3/s})$', fontsize=13)
    ax.set_title(title)

    # Nash only where both series are finite and Qobs actually varies
    if ar_Qobs is not None:
        m = np.isfinite(ar_Qobs) & np.isfinite(ar_Qsim)
        if m.sum() >= 2 and np.var(ar_Qobs[m]) > 0:
            eff = ut.Nash(ar_Qsim[m], ar_Qobs[m])
            labels.append('Eff = %.3f' % eff)
            lines += ax.plot(ar_date[:1], ar_Qsim[:1], 'w:', alpha=0)

    if ar_rain is not None:
        ax2 = ax.twinx()
        if len(ar_date) > 1:
            width = date2num(ar_date[1]) - date2num(ar_date[0])
        else:
            width = 1.0
        ax2.bar(ar_date, ar_rain, width=width, color='b', alpha=0.4,
                edgecolor='none')
        ax2.set_ylabel(r'$Rainfall\ (\mathrm{mm})$', fontsize=13, color='b')
        # invert so rain hangs from the top; guard the all-zero case
        rmax = float(np.nanmax(ar_rain)) if np.isfinite(ar_rain).any() else 0.0
        ax2.set_ylim(max(rmax * 2.0, 1.0), 0.0)
        ax2.tick_params(axis='y', colors='b')

    ax.legend(lines, labels, loc='upper right', framealpha=0.75)

    if image_out is not None:
        ut.check_file_exist(image_out)
        ax.figure.autofmt_xdate()
        ax.figure.savefig(image_out, dpi=130, bbox_inches='tight')
    return ax

def plot_hyetograph(ar_date, ar_rain=None, ar_Qobs=None,
                    title='', ax=None, image_out=None):
    """Simulated hyetograph.

    ``ar_Qobs`` and ``ar_rain`` are optional. Returns the primary axis so the
    figure can be composed into a larger diagnostic panel.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import date2num

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    lines, labels = [], []
    if ar_Qobs is not None and np.isfinite(ar_Qobs).any():
        lines += ax.plot(ar_date, ar_Qobs, 'r-', lw=1.0)
        labels.append('Observed')
    # lines += ax.plot(ar_date, ar_Qsim, 'k-', lw=1.0)
    # labels.append('Model')

    ax.set_xlim(ar_date[0], ar_date[-1])
    ax.set_ylabel(r'$Q\ (\mathrm{m^3/s})$', fontsize=13)
    ax.set_title(title)

    # Nash only where both series are finite and Qobs actually varies
    if ar_Qobs is not None:
        m = np.isfinite(ar_Qobs) & np.isfinite(ar_Qsim)
        if m.sum() >= 2 and np.var(ar_Qobs[m]) > 0:
            eff = ut.Nash(ar_Qsim[m], ar_Qobs[m])
            labels.append('Eff = %.3f' % eff)
            lines += ax.plot(ar_date[:1], ar_Qsim[:1], 'w:', alpha=0)

    if ar_rain is not None:
        ax2 = ax.twinx()
        if len(ar_date) > 1:
            width = date2num(ar_date[1]) - date2num(ar_date[0])
        else:
            width = 1.0
        ax2.bar(ar_date, ar_rain, width=width, color='b', alpha=0.4,
                edgecolor='none')
        ax2.set_ylabel(r'$Rainfall\ (\mathrm{mm})$', fontsize=13, color='b')
        # invert so rain hangs from the top; guard the all-zero case
        rmax = float(np.nanmax(ar_rain)) if np.isfinite(ar_rain).any() else 0.0
        ax2.set_ylim(max(rmax * 2.0, 1.0), 0.0)
        ax2.tick_params(axis='y', colors='b')

    ax.legend(lines, labels, loc='upper right', framealpha=0.75)

    if image_out is not None:
        ut.check_file_exist(image_out)
        ax.figure.autofmt_xdate()
        ax.figure.savefig(image_out, dpi=130, bbox_inches='tight')
    return ax

# --------------------------------------------------------------------------
# config-driven entry point
# --------------------------------------------------------------------------

def run(ini_file='plot_flow_precip.ini'):
    """Config wrapper. Only ``file_sim``, ``file_cell_param``,
    ``file_global_param`` and ``group_name`` are required; observed flow and
    rainfall are drawn only if their flags are set.
    """
    config = ConfigParser()
    config.read(ini_file)
    print('Read the file ', ini_file)

    file_sim = config.get('files', 'file_Qsim')
    file_cell_param = config.get('files', 'file_cell_param')
    file_global_param = config.get('files', 'file_global_param')
    image_out = config.get('files', 'image_out')
    rain_image_out = config.get('files', 'rain_image_out')
    group_name = config.get('groups', 'group_name')

    want_rain = config.getboolean('flags', 'Pobs', fallback=False)
    want_qobs = config.getboolean('flags', 'Qobs', fallback=False)

    # start of the simulation, for the time axis
    start = None
    if config.has_option('timeline', 'start_datetime'):
        start = dt.datetime.fromisoformat(
            config.get('timeline', 'start_datetime'))

    # optional explicit outlet override (e.g. plot at a gauge, not the mouth)
    if config.has_option('files', 'outlet_cell'):
        outlet_index = config.getint('files', 'outlet_cell')
    else:
        outlet_index = find_outlet_index(file_cell_param)

    dt_seconds = read_dt_seconds(file_global_param)
    ar_Qsim = outlet_channel_flow(file_sim, outlet_index)

    flow_summary(ar_Qsim, outlet_index)

    if start is not None:
        ar_date = build_timeline(start, dt_seconds, ar_Qsim.size)
    else:
        warnings.warn("No [timeline] start_datetime; using integer step axis.")
        ar_date = np.arange(1, ar_Qsim.size + 1)

    ar_rain = None
    if want_rain:
        file_rain = config.get('files', 'file_rain')
        ar_rain = mean_catchment_rain(file_rain, group_name)

    ar_Qobs = None
    if want_qobs:
        file_Qobs = config.get('files', 'file_Qobs')
        obs_dates, obs_Q = read_observed_flow(file_Qobs)
        if start is not None:
            ar_Qobs = align_observed(ar_date, obs_dates, obs_Q)
        elif obs_Q.size == ar_Qsim.size:
            ar_Qobs = obs_Q
        else:
            warnings.warn("Cannot align Qobs without a start_datetime and "
                          "with mismatched length; skipping observed flow.")

    # plot_hyetograph(ar_date, ar_rain=ar_rain, ar_Qobs=ar_Qobs,
    #                     title=group_name, image_out=rain_image_out)
    plot_hydrograph(ar_date, ar_Qsim, ar_rain=ar_rain, ar_Qobs=ar_Qobs,
                    title=group_name, image_out=image_out)


if __name__ == '__main__':
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else 'plot_flow_precip.ini')
