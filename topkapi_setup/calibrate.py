"""Run many PyTOPKAPI simulations at once and keep only the hydrographs.

Calibration evaluates hundreds of parameter sets that do not depend on
each other, so the cores are better spent running whole simulations side
by side than splitting one simulation across threads. Each run here is
serial and therefore reproducible; the parallelism is one process per
parameter set.

Three things this handles that a shell loop does not:

**Isolation.** Every run gets its own ini and its own output file in a
scratch directory. Without that, concurrent runs overwrite each other's
results and the answers are silently mixed.

**Disk.** A full results file is nine datasets of `(n_t, n_cell)`. Three
hundred of them is tens of gigabytes. Calibration wants one column, so
each worker extracts the outlet series and deletes the file before
moving on.

**Failure.** A calibration will hand the solver parameter sets that make
it fall over. A failed run is recorded with its traceback and its
hydrograph filled with NaN; the batch carries on.

The four multipliers are the ones PyTOPKAPI already reads from the ini's
``[calib_params]`` section: ``fac_L`` on soil depth, ``fac_Ks`` on
saturated conductivity, ``fac_n_o`` and ``fac_n_c`` on overland and
channel Manning.

Outlet
------
The outlet is found from topology, the cell whose ``cell_down`` is
negative, never by assuming a column index. Total discharge there is
``Qc_out + Q_down``: the channel store's outflow plus the soil and
overland flow leaving the same cell. Both are returned separately so the
choice stays yours; ``Q_outlet`` in the output file is their sum.

Usage
-----
::

    python -m topkapi_setup.calibrate run \\
        --ini      model-simulation.ini \\
        --factors  factor_sets.csv \\
        --out      batch_hydrographs.h5 \\
        --workers  4

``factor_sets.csv`` is a header row of ``fac_L,fac_Ks,fac_n_o,fac_n_c``
and one row per parameter set.

Paths inside the ini are used as the solver uses them, relative to the
working directory, so run this from wherever you normally run the model.

"""
import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from configparser import ConfigParser
from contextlib import redirect_stderr
from datetime import datetime, timezone
from io import StringIO
import multiprocessing as mp

import numpy as np
import h5py

from pytopkapi.results_analysis.fingerprint import (fingerprint_bytes,
                                                    outlet_index, _git_state)

FACTOR_NAMES = ('fac_L', 'fac_Ks', 'fac_n_o', 'fac_n_c')


# ---------------------------------------------------------------------
# one run, executed in a worker process
# ---------------------------------------------------------------------

def _run_one(job):
    """Run a single simulation and return the outlet series.

    Module level and taking one picklable argument, because macOS spawns
    workers rather than forking them.
    """
    index, base_ini, factors, outlet, scratch_root, keep = job
    started = time.perf_counter()

    directory = os.path.join(scratch_root, 'run_%05d' % index)
    os.makedirs(directory, exist_ok=True)
    results = os.path.join(directory, 'simulation.h5')
    noise = StringIO()

    try:
        config = ConfigParser()
        if not config.read(base_ini):
            raise RuntimeError('could not read %s' % base_ini)
        for name, value in zip(FACTOR_NAMES, factors):
            config.set('calib_params', name, repr(float(value)))
        config.set('output_files', 'file_out', results)
        config.set('output_files', 'append_output', 'False')

        ini = os.path.join(directory, 'model-simulation.ini')
        with open(ini, 'w') as handle:
            config.write(handle)

        # imported here so the worker pays the cost once, after spawning
        from pytopkapi import model
        with redirect_stderr(noise):
            model.run(ini_file=ini, parallel_exec=False, quiet=True)

        with h5py.File(results, 'r') as h5file:
            channel = h5file['Channel/Qc_out'][:, outlet].astype('f8')
            down = h5file['Q_down'][:, outlet].astype('f8')

        return {'index': index, 'ok': True, 'Qc_out': channel,
                'Q_down': down, 'seconds': time.perf_counter() - started,
                'error': None}

    except BaseException:
        return {'index': index, 'ok': False, 'Qc_out': None, 'Q_down': None,
                'seconds': time.perf_counter() - started,
                'error': traceback.format_exc() + noise.getvalue()}
    finally:
        if not keep:
            shutil.rmtree(directory, ignore_errors=True)


# ---------------------------------------------------------------------
# the batch
# ---------------------------------------------------------------------

def run_size(base_ini):
    """Cells and timesteps the ini describes, without running anything."""
    config = ConfigParser()
    if not config.read(base_ini):
        raise SystemExit('could not read %s' % base_ini)
    group = config.get('groups', 'group_name')
    with h5py.File(config.get('input_files', 'file_rain'), 'r') as handle:
        n_step, n_cell = handle['/%s/rainfall' % group].shape
    return int(n_cell), int(n_step)


def cell_param_path(base_ini):
    config = ConfigParser()
    config.read(base_ini)
    return config.get('input_files', 'file_cell_param')


def run_batch(base_ini, factor_sets, outlet=None, workers=None,
              scratch=None, keep_results=False, progress=True):
    """Run one simulation per factor set and collect the outlet series.

    Parameters
    ----------
    base_ini : str
        The simulation ini to vary. Its `[calib_params]` are replaced per
        run; everything else is inherited.
    factor_sets : array-like, shape (n_sets, 4)
        Rows of (fac_L, fac_Ks, fac_n_o, fac_n_c).
    outlet : int, optional
        Outlet cell index. Derived from the ini's cell_param file by
        topology when omitted.
    workers : int, optional
        Worker processes. Defaults to half the cores, which leaves room
        for the roughly 200 MB of forcing each worker holds.
    keep_results : bool
        Leave each run's full results file on disk. Off by default: the
        point of this module is not to write tens of gigabytes.

    Returns
    -------
    dict with keys `factors`, `Qc_out`, `Q_down`, `seconds`, `failures`,
    `outlet`, `n_t`, `wall_seconds`. Failed runs hold NaN.

    """
    factor_sets = np.atleast_2d(np.asarray(factor_sets, dtype='f8'))
    if factor_sets.shape[1] != len(FACTOR_NAMES):
        raise ValueError('expected %d factors per set, got %d'
                         % (len(FACTOR_NAMES), factor_sets.shape[1]))
    n_set = factor_sets.shape[0]

    if outlet is None:
        outlet = outlet_index(cell_param_path(base_ini))
    n_cell, n_step = run_size(base_ini)
    n_t = n_step + 1                       # the solver writes an initial row

    if workers is None:
        workers = max(1, mp.cpu_count() // 2)
    workers = max(1, min(workers, n_set))

    owned_scratch = scratch is None
    if owned_scratch:
        scratch = tempfile.mkdtemp(prefix='topkapi_batch_')
    else:
        os.makedirs(scratch, exist_ok=True)

    channel = np.full((n_set, n_t), np.nan)
    down = np.full((n_set, n_t), np.nan)
    seconds = np.full(n_set, np.nan)
    failures = {}

    base_ini = os.path.abspath(base_ini)
    jobs = [(index, base_ini, factor_sets[index], outlet, scratch,
             keep_results) for index in range(n_set)]

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=n_set, ascii=True, unit=' run',
                       desc='%d runs on %d workers' % (n_set, workers))
        except ImportError:
            bar = None

    started = time.perf_counter()
    context = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=context) as pool:
            futures = [pool.submit(_run_one, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                index = result['index']
                seconds[index] = result['seconds']
                if result['ok']:
                    channel[index] = result['Qc_out']
                    down[index] = result['Q_down']
                else:
                    failures[index] = result['error']
                if bar is not None:
                    bar.update()
    finally:
        if bar is not None:
            bar.close()
        if owned_scratch and not keep_results:
            shutil.rmtree(scratch, ignore_errors=True)

    return {'factors': factor_sets,
            'Qc_out': channel,
            'Q_down': down,
            'seconds': seconds,
            'failures': failures,
            'outlet': int(outlet),
            'n_cell': n_cell,
            'n_t': n_t,
            'workers': workers,
            'wall_seconds': time.perf_counter() - started}


# ---------------------------------------------------------------------
# reading and writing
# ---------------------------------------------------------------------

def read_factor_sets(path):
    """Read a CSV of factor sets. Header names may be in any order."""
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        missing = set(FACTOR_NAMES) - set(reader.fieldnames or ())
        if missing:
            raise SystemExit('%s is missing column(s): %s'
                             % (path, ', '.join(sorted(missing))))
        rows = [[float(row[name]) for name in FACTOR_NAMES]
                for row in reader]
    if not rows:
        raise SystemExit('%s holds no factor sets' % path)
    return np.array(rows, dtype='f8')


def write_batch(batch, path, base_ini):
    """Write the hydrographs to HDF5 and a manifest beside it."""
    with h5py.File(path, 'w') as h5file:
        h5file.attrs['outlet_cell'] = batch['outlet']
        h5file.attrs['n_cell'] = batch['n_cell']
        h5file.attrs['n_t'] = batch['n_t']
        h5file.attrs['base_ini'] = os.path.abspath(base_ini)
        h5file.attrs['factor_names'] = list(FACTOR_NAMES)
        h5file.create_dataset('factors', data=batch['factors'])
        h5file.create_dataset('Qc_out', data=batch['Qc_out'])
        h5file.create_dataset('Q_down', data=batch['Q_down'])
        h5file.create_dataset('Q_outlet',
                              data=batch['Qc_out'] + batch['Q_down'])
        h5file.create_dataset('seconds', data=batch['seconds'])

    config = ConfigParser()
    config.read(base_ini)
    inputs = {}
    for key in ('file_cell_param', 'file_global_param'):
        try:
            target = config.get('input_files', key)
        except Exception:
            continue
        inputs[os.path.basename(target)] = fingerprint_bytes(target)

    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(
                    timespec='seconds'),
                'base_ini': os.path.abspath(base_ini),
                'base_ini_sha256': fingerprint_bytes(base_ini)['sha256'],
                'inputs': inputs,
                'git': _git_state(os.path.dirname(os.path.abspath(base_ini))
                                  or '.'),
                'outlet_cell': batch['outlet'],
                'n_set': int(batch['factors'].shape[0]),
                'n_t': batch['n_t'],
                'workers': batch['workers'],
                'wall_seconds': round(batch['wall_seconds'], 2),
                'seconds_per_run': {
                    'min': _round(np.nanmin(batch['seconds'])),
                    'median': _round(np.nanmedian(batch['seconds'])),
                    'max': _round(np.nanmax(batch['seconds']))},
                'failures': {str(index): message.strip().splitlines()[-1]
                             for index, message in
                             batch['failures'].items()}}

    manifest_path = os.path.splitext(path)[0] + '_manifest.json'
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write('\n')
    return manifest_path


def _round(value):
    return None if not np.isfinite(value) else round(float(value), 2)


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m topkapi_setup.calibrate',
        description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='run a batch of parameter sets')
    run.add_argument('--ini', required=True, help='the base simulation ini')
    run.add_argument('--factors', required=True, metavar='CSV')
    run.add_argument('--out', required=True, metavar='H5')
    run.add_argument('--workers', type=int, default=None)
    run.add_argument('--keep-results', action='store_true',
                     help='leave every full results file on disk')
    run.add_argument('--quiet', action='store_true')

    args = parser.parse_args(argv)

    factor_sets = read_factor_sets(args.factors)
    outlet = outlet_index(cell_param_path(args.ini))
    n_cell, n_step = run_size(args.ini)
    print('%d parameter sets, %d cells x %d steps, outlet cell %d'
          % (factor_sets.shape[0], n_cell, n_step, outlet))

    batch = run_batch(args.ini, factor_sets, outlet=outlet,
                      workers=args.workers, keep_results=args.keep_results,
                      progress=not args.quiet)

    manifest = write_batch(batch, args.out, args.ini)
    median = _round(np.nanmedian(batch['seconds']))
    print('\n%d runs on %d workers in %.1f s  (median %.1f s per run,'
          ' %.1fx over running them one at a time)'
          % (factor_sets.shape[0], batch['workers'], batch['wall_seconds'],
             median or float('nan'),
             (median or 0) * factor_sets.shape[0] /
             max(batch['wall_seconds'], 1e-9)))
    if batch['failures']:
        print('%d run(s) failed:' % len(batch['failures']))
        for index in sorted(batch['failures'])[:5]:
            factors = ', '.join('%s=%g' % pair for pair in
                                zip(FACTOR_NAMES, batch['factors'][index]))
            print('  set %d (%s)' % (index, factors))
            print('    %s' %
                  batch['failures'][index].strip().splitlines()[-1])
        if len(batch['failures']) > 5:
            print('  ... %d more, all recorded in the manifest'
                  % (len(batch['failures']) - 5))
    print('\nhydrographs -> %s\nmanifest    -> %s' % (args.out, manifest))
    return 1 if batch['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
