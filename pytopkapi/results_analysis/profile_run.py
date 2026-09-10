"""Measure where a PyTOPKAPI run spends its time.

Step 1 of the speed-up: find the hot spots before changing anything.  The
numbers this prints decide what is worth optimising and what is not.

Everything here runs the solver **serially**.  Parallel routing sums a
cell's inflows in thread-completion order, which moves the answer at the
1e-05 level, and that noise would sit underneath every later measurement.
Serial is the invariant for this exercise.

Four subcommands:

``time``
    Wall clock, no instrumentation.  The honest baseline.

``profile``
    cProfile.  Which functions cost what, by total and cumulative time.
    Writes a ``.prof`` file that ``snakeviz`` can draw.

``tally``
    A census of the solver's paths: how many cell-solves, and which branch
    of ``solve_storage_eq`` each one took.  The quasi-analytical solution
    is cheap and the Runge-Kutta-Fehlberg fallback is not, so the split
    between them matters more than any single function's cost.

``lines``
    line_profiler on the four functions the plan names, if it is
    installed.  Line-by-line cost inside ``_solve_cell`` and
    ``solve_storage_eq``.

Why the ``--steps`` option exists
---------------------------------
cProfile charges roughly a microsecond per function call.  A full run makes
hundreds of millions of calls, so profiling one would take hours.  Instead
this truncates the forcing to the first N timesteps and profiles that.  The
*shape* of the profile is what we need, and it does not depend on run
length.  Twenty-four steps is usually plenty.

Truncation writes a temporary copy of the rainfall and ET files and a
temporary ini pointing at them, in a scratch directory.  Nothing in the
project is touched, and the reference run is never overwritten.

Usage
-----
::

    python -m pytopkapi.results_analysis.profile_run time \\
        --ini projects/umhlanga/model-simulation-60.ini --steps 24 --repeat 3

    python -m pytopkapi.results_analysis.profile_run profile \\
        --ini projects/umhlanga/model-simulation-60.ini --steps 24 \\
        --out profile_baseline.prof

    python -m pytopkapi.results_analysis.profile_run tally \\
        --ini projects/umhlanga/model-simulation-60.ini --steps 24

Paths inside the ini are used as the solver uses them, relative to the
working directory, so run this from wherever you normally run the model.

"""
import argparse
import cProfile
import os
import pstats
import shutil
import sys
import tempfile
import time
from configparser import ConfigParser

import numpy as np
import h5py

from .. import model
from .. import ode


# ---------------------------------------------------------------------
# truncating the forcing so a profile finishes in a minute
# ---------------------------------------------------------------------

def _copy_truncated(src, dst, datasets, n_step):
    """Copy an HDF5 forcing file, keeping only the first n_step rows."""
    with h5py.File(src, 'r') as fin, h5py.File(dst, 'w') as fout:
        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        for name in datasets:
            fout.create_dataset(name, data=fin[name][:n_step, :])


def truncate_project(ini_path, n_step, scratch):
    """Write a scratch ini whose forcing is cut to n_step timesteps.

    Returns the path to the new ini.  The output file is redirected into
    the scratch directory and append_output is forced off, so the
    reference run cannot be overwritten.
    """
    config = ConfigParser()
    if not config.read(ini_path):
        raise SystemExit('could not read %s' % ini_path)

    group = config.get('groups', 'group_name')
    file_rain = config.get('input_files', 'file_rain')
    file_ET = config.get('input_files', 'file_ET')

    rain_out = os.path.join(scratch, 'rain_truncated.h5')
    et_out = os.path.join(scratch, 'ET_truncated.h5')
    _copy_truncated(file_rain, rain_out, ['/%s/rainfall' % group], n_step)
    _copy_truncated(file_ET, et_out,
                    ['/%s/ETr' % group, '/%s/ETo' % group], n_step)

    with h5py.File(rain_out, 'r') as handle:
        kept = handle['/%s/rainfall' % group].shape[0]
    if kept < n_step:
        print('note: forcing holds only %d steps, using all of them' % kept)

    config.set('input_files', 'file_rain', rain_out)
    config.set('input_files', 'file_ET', et_out)
    config.set('output_files', 'file_out',
               os.path.join(scratch, 'simulation_profile.h5'))
    config.set('output_files', 'append_output', 'False')

    ini_out = os.path.join(scratch, 'model-simulation-profile.ini')
    with open(ini_out, 'w') as handle:
        config.write(handle)
    return ini_out


class scratch_project(object):
    """Context manager yielding an ini, truncated if n_step is given."""

    def __init__(self, ini_path, n_step=None, keep=False):
        self.ini_path = ini_path
        self.n_step = n_step
        self.keep = keep
        self.directory = None

    def __enter__(self):
        if not self.n_step:
            return self.ini_path
        self.directory = tempfile.mkdtemp(prefix='topkapi_profile_')
        return truncate_project(self.ini_path, self.n_step, self.directory)

    def __exit__(self, *exc):
        if self.directory and not self.keep:
            shutil.rmtree(self.directory, ignore_errors=True)
        return False


def _run_once(ini_path, quiet=True):
    """One serial solve.  Parallel is deliberately not an option here."""
    model.run(ini_file=ini_path, parallel_exec=False, quiet=quiet)


def _run_size(ini_path):
    """Cells and timesteps for the run this ini describes."""
    config = ConfigParser()
    config.read(ini_path)
    group = config.get('groups', 'group_name')
    with h5py.File(config.get('input_files', 'file_rain'), 'r') as handle:
        n_step, n_cell = handle['/%s/rainfall' % group].shape
    return int(n_cell), int(n_step)


# ---------------------------------------------------------------------
# time
# ---------------------------------------------------------------------

def timed(ini_path, repeat=1):
    """Wall-clock a run, repeat times.  Returns the list of seconds."""
    seconds = []
    for index in range(repeat):
        start = time.perf_counter()
        _run_once(ini_path)
        seconds.append(time.perf_counter() - start)
        print('  run %d: %8.2f s' % (index + 1, seconds[-1]))
    return seconds


# ---------------------------------------------------------------------
# tally: which path does each cell-solve take
# ---------------------------------------------------------------------

class _Census(object):
    """Counting wrappers around the solver's leaf routines.

    Installed by monkeypatching the module attributes the solver looks up
    at call time, so no source file is edited.  The counts are exact; the
    timings taken under a census are not, because each wrapper costs a
    fraction of a microsecond.
    """

    NAMES = ('cell_solve', 'storage_eq_call', 'input_zero', 'coefb_zero',
             'qas_call', 'qas_failed', 'rkf_solve', 'rkf_substep')

    def __init__(self):
        self.count = dict.fromkeys(self.NAMES, 0)
        self._saved = []

    def _patch(self, module, name, factory):
        original = getattr(module, name)
        self._saved.append((module, name, original))
        setattr(module, name, factory(original))

    def __enter__(self):
        count = self.count

        def counted(key):
            def wrap(original):
                def wrapper(*args, **kwargs):
                    count[key] += 1
                    return original(*args, **kwargs)
                return wrapper
            return wrap

        def qas_wrap(original):
            def wrapper(*args, **kwargs):
                count['qas_call'] += 1
                result = original(*args, **kwargs)
                if result is None:
                    count['qas_failed'] += 1
                return result
            return wrapper

        def step_wrap(original):
            def wrapper(self, f, x, t, delta_t=0):
                # step() recurses: the outer call carries a delta_t, the
                # inner substeps do not.  Counting them apart gives the
                # average number of substeps per fallback.
                if delta_t:
                    count['rkf_solve'] += 1
                else:
                    count['rkf_substep'] += 1
                return original(self, f, x, t, delta_t)
            return wrapper

        # the serial path calls the positional core; the dict-keyed
        # _solve_cell is only an adapter for the parallel path
        self._patch(model, '_solve_cell_core', counted('cell_solve'))
        self._patch(ode, 'solve_storage_eq', counted('storage_eq_call'))
        self._patch(ode, 'input_zero_solution', counted('input_zero'))
        self._patch(ode, 'coefb_zero_solution', counted('coefb_zero'))
        self._patch(ode, 'qas', qas_wrap)
        self._patch(ode.RKF, 'step', step_wrap)
        return self

    def __exit__(self, *exc):
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        return False


def _report_census(count, n_cell, n_step, seconds):
    solves = count['cell_solve']
    stores = count['storage_eq_call']
    print('\ncell-solves            %12d  (%d cells x %d steps)'
          % (solves, n_cell, n_step))
    print('storage equations      %12d  (%.2f per cell-solve)'
          % (stores, stores / max(solves, 1)))

    print('\nhow each storage equation was solved')
    print('  zero inflow          %12d  %6.2f %%'
          % (count['input_zero'], 100.0 * count['input_zero'] / max(stores, 1)))
    print('  zero b coefficient   %12d  %6.2f %%'
          % (count['coefb_zero'], 100.0 * count['coefb_zero'] / max(stores, 1)))
    solved_by_qas = count['qas_call'] - count['qas_failed']
    print('  quasi-analytical     %12d  %6.2f %%'
          % (solved_by_qas, 100.0 * solved_by_qas / max(stores, 1)))
    print('  Runge-Kutta fallback %12d  %6.2f %%'
          % (count['rkf_solve'], 100.0 * count['rkf_solve'] / max(stores, 1)))

    if count['rkf_solve']:
        print('\n  RKF substeps         %12d  (%.1f per fallback)'
              % (count['rkf_substep'],
                 count['rkf_substep'] / count['rkf_solve']))

    print('\ninstrumented wall clock %11.2f s  (counters add overhead;'
          ' use the time subcommand for the real figure)' % seconds)


# ---------------------------------------------------------------------
# line profiling
# ---------------------------------------------------------------------

LINE_TARGETS = (('pytopkapi.model', '_solve_cell_core'),
                ('pytopkapi.ode', 'solve_storage_eq'),
                ('pytopkapi.ode', 'qas'),
                ('pytopkapi.ode', 'input_zero_solution'))


def _line_profile(ini_path):
    try:
        from line_profiler import LineProfiler
    except ImportError:
        raise SystemExit(
            'line_profiler is not installed.\n'
            '    conda install -c conda-forge line_profiler\n'
            'The profile and tally subcommands need no extra package.')

    profiler = LineProfiler()
    for module_name, function_name in LINE_TARGETS:
        module = sys.modules[module_name]
        profiler.add_function(getattr(module, function_name))

    profiler.enable_by_count()
    try:
        _run_once(ini_path)
    finally:
        profiler.disable_by_count()
    profiler.print_stats()


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------

def _add_common(parser):
    parser.add_argument('--ini', required=True,
                        help='the simulation ini file')
    parser.add_argument('--steps', type=int, default=None,
                        help='truncate the forcing to this many timesteps')
    parser.add_argument('--keep-scratch', action='store_true',
                        help='leave the truncated project on disk')


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m pytopkapi.results_analysis.profile_run',
        description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    wall = sub.add_parser('time', help='wall clock, no instrumentation')
    _add_common(wall)
    wall.add_argument('--repeat', type=int, default=1)

    prof = sub.add_parser('profile', help='cProfile the run')
    _add_common(prof)
    prof.add_argument('--out', default=None, metavar='PROF',
                      help='save the raw profile for snakeviz')
    prof.add_argument('--top', type=int, default=25)

    tally = sub.add_parser('tally', help='census of the solver paths')
    _add_common(tally)

    lines = sub.add_parser('lines', help='line_profiler on the hot functions')
    _add_common(lines)

    args = parser.parse_args(argv)

    with scratch_project(args.ini, args.steps, args.keep_scratch) as ini:
        n_cell, n_step = _run_size(ini)
        print('serial run: %d cells x %d steps = %d cell-solves\n'
              % (n_cell, n_step, n_cell * n_step))

        if args.command == 'time':
            if args.steps:
                print('caution: a truncated run still pays the full setup'
                      ' cost (reading the parameters, building the cell'
                      ' hierarchy).\n         For the baseline, time the'
                      ' whole run with no --steps.\n')
            seconds = timed(ini, args.repeat)
            best = min(seconds)
            print('\nbest %.2f s  =  %.1f us per cell-solve'
                  % (best, 1e6 * best / (n_cell * n_step)))
            return 0

        if args.command == 'profile':
            profiler = cProfile.Profile()
            profiler.runcall(_run_once, ini)

            stats = pstats.Stats(profiler)
            stats.strip_dirs()
            print('\n=== by time in the function itself ===')
            stats.sort_stats('tottime').print_stats(args.top)
            print('\n=== by time including everything it calls ===')
            stats.sort_stats('cumtime').print_stats(args.top)
            if args.out:
                profiler.dump_stats(args.out)
                print('raw profile written to %s\n'
                      '  view it with:  snakeviz %s' % (args.out, args.out))
            return 0

        if args.command == 'tally':
            with _Census() as census:
                start = time.perf_counter()
                _run_once(ini)
                seconds = time.perf_counter() - start
            _report_census(census.count, n_cell, n_step, seconds)
            return 0

        _line_profile(ini)
        return 0


if __name__ == '__main__':
    sys.exit(main())
