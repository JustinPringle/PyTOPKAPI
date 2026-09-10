"""Fingerprint a PyTOPKAPI run so speed work can be proved harmless.

The optimisation work changes *how* the solver computes, never *what* it
computes.  This module makes that claim testable: it records a SHA-256 of
every dataset in a simulation file (plus the inputs that produced it) and
can re-check a later run against that record.

Three subcommands:

``write``
    Take a reference run and write ``fingerprint.json``.

``check``
    Recompute the fingerprint of a new run and compare it to the manifest.
    Exit status 0 means bit-identical, 1 means something moved.

``compare``
    When ``check`` fails, open both files side by side and report the
    largest absolute and relative difference per dataset, plus the outlet
    column of ``Channel/Qc_out`` on its own.  This is the fallback bar for
    changes where floating-point reordering is unavoidable.

What is hashed, and what is not
-------------------------------
The hash covers the *decoded values* of each dataset — its dtype, shape and
bytes — not the raw bytes of the HDF5 container.  Compression level, chunk
layout and file metadata may therefore change freely; only the numbers are
the invariant.  This matters because buffering the writes will change the
chunk layout on disk while leaving every number alone.

The file attributes (``pytopkapi_git_revision`` and friends) are recorded
for provenance but deliberately excluded from the comparison: they change
on every commit by construction.

Usage
-----
::

    python -m pytopkapi.results_analysis.fingerprint write \\
        --results projects/umhlanga/results_60/simulation_20250101_20250131.h5 \\
        --input   projects/umhlanga/cell_param_60.dat \\
        --input   projects/umhlanga/global_param.dat \\
        --input   projects/umhlanga/forcing/rainfields_60.h5 \\
        --out     projects/umhlanga/results_60/fingerprint_reference.json

    python -m pytopkapi.results_analysis.fingerprint check \\
        --results projects/umhlanga/results_60/simulation_20250101_20250131.h5 \\
        --input   projects/umhlanga/cell_param_60.dat \\
        --input   projects/umhlanga/global_param.dat \\
        --input   projects/umhlanga/forcing/rainfields_60.h5 \\
        --manifest projects/umhlanga/results_60/fingerprint_reference.json

    python -m pytopkapi.results_analysis.fingerprint compare \\
        --reference reference.h5 --candidate candidate.h5 \\
        --cell-param projects/umhlanga/cell_param_60.dat

"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import h5py

SCHEMA_VERSION = 1
CHUNK_ROWS = 128           # rows read at a time; bounds memory on big runs
CELL_DOWN_COL = 14         # cell_param.dat column holding the downstream label


# ---------------------------------------------------------------------
# hashing primitives
# ---------------------------------------------------------------------

def _iter_blocks(dset, chunk_rows=CHUNK_ROWS):
    """Yield contiguous row-blocks of a dataset, in order."""
    if dset.shape == ():
        yield np.ascontiguousarray(dset[()])
        return
    n_row = dset.shape[0]
    for start in range(0, n_row, chunk_rows):
        yield np.ascontiguousarray(dset[start:start + chunk_rows])


def _summarise_dataset(dset, chunk_rows=CHUNK_ROWS):
    """SHA-256 and descriptive statistics for one HDF5 dataset.

    The digest covers dtype (with byte order), shape and the value bytes.
    The statistics are for human eyes only and are never compared.
    """
    digest = hashlib.sha256()
    digest.update(dset.dtype.str.encode('ascii'))
    digest.update(repr(tuple(dset.shape)).encode('ascii'))

    numeric = (np.issubdtype(dset.dtype, np.floating) or
               np.issubdtype(dset.dtype, np.integer))
    vmin = vmax = None
    total = 0.0
    n_nonfinite = 0
    n_value = 0

    for block in _iter_blocks(dset, chunk_rows):
        digest.update(block.tobytes())
        if not numeric or block.size == 0:
            continue
        wide = block.astype('f8', copy=False)
        n_value += wide.size
        good = np.isfinite(wide)
        n_nonfinite += int(wide.size - good.sum())
        if good.any():
            kept = wide[good]
            block_min = float(kept.min())
            block_max = float(kept.max())
            vmin = block_min if vmin is None else min(vmin, block_min)
            vmax = block_max if vmax is None else max(vmax, block_max)
            total += float(kept.sum())

    entry = {'sha256': digest.hexdigest(),
             'shape': list(dset.shape),
             'dtype': dset.dtype.str}
    if numeric:
        entry['stats'] = {'min': vmin, 'max': vmax, 'sum': total,
                          'n_value': n_value, 'n_nonfinite': n_nonfinite}
    return entry


def _dataset_names(h5file):
    """Every dataset path in the file, leading slash, sorted."""
    names = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            names.append('/' + name)

    h5file.visititems(visit)
    return sorted(names)


def fingerprint_hdf5(path, chunk_rows=CHUNK_ROWS):
    """Per-dataset fingerprint of an HDF5 file, plus its root attributes."""
    with h5py.File(path, 'r') as h5file:
        datasets = {name: _summarise_dataset(h5file[name], chunk_rows)
                    for name in _dataset_names(h5file)}
        attrs = {key: _plain(value) for key, value in h5file.attrs.items()}
    return {'kind': 'hdf5', 'datasets': datasets, 'attrs': attrs}


def fingerprint_bytes(path):
    """SHA-256 of a whole file, for ASCII inputs such as cell_param.dat."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return {'kind': 'bytes',
            'sha256': digest.hexdigest(),
            'n_byte': os.path.getsize(path)}


def _plain(value):
    """Coerce an HDF5 attribute to something json can hold."""
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    return value


def fingerprint_file(path, chunk_rows=CHUNK_ROWS):
    """Dispatch on extension: HDF5 by dataset, everything else by bytes."""
    if os.path.splitext(path)[1].lower() in ('.h5', '.hdf5', '.he5'):
        return fingerprint_hdf5(path, chunk_rows)
    return fingerprint_bytes(path)


# ---------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------

def _git_state(start_dir):
    """Best-effort record of the working tree, for provenance only."""
    def run(*args):
        out = subprocess.run(('git',) + args, cwd=start_dir,
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip())
        return out.stdout.strip()

    try:
        return {'commit': run('rev-parse', 'HEAD'),
                'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
                'dirty': bool(run('status', '--porcelain'))}
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return {'commit': None, 'branch': None, 'dirty': None}


def build_manifest(results, inputs=(), chunk_rows=CHUNK_ROWS, label=None):
    """Fingerprint a run and return the manifest as a dictionary.

    Parameters
    ----------
    results : str
        Path to the simulation HDF5 file.
    inputs : sequence of str
        Paths to the files that drove the run — cell_param.dat,
        global_param.dat, the forcing files.  Recorded so a check cannot
        silently pass against different inputs.
    label : str, optional
        Free text describing the run.

    Returns
    -------
    dict

    """
    entries = {}
    entries['results'] = dict(fingerprint_file(results, chunk_rows),
                              path=results, role='results')
    for path in inputs:
        name = os.path.basename(path)
        if name in entries:
            raise ValueError('two inputs share the basename %r; rename one '
                             'so the check can pair them up' % name)
        entries[name] = dict(fingerprint_file(path, chunk_rows),
                             path=path, role='input')

    return {'schema_version': SCHEMA_VERSION,
            'label': label,
            'created_utc': datetime.now(timezone.utc).isoformat(
                timespec='seconds'),
            'git': _git_state(os.path.dirname(os.path.abspath(results)) or '.'),
            'runtime': {'python': sys.version.split()[0],
                        'numpy': np.__version__,
                        'h5py': h5py.__version__},
            'entries': entries}


def check_manifest(manifest, results, inputs=(), chunk_rows=CHUNK_ROWS):
    """Compare a fresh fingerprint against a stored manifest.

    Returns a list of human-readable problems.  Empty means identical.
    Paths are not compared; only content is.
    """
    fresh = build_manifest(results, inputs, chunk_rows)
    problems = []
    old, new = manifest['entries'], fresh['entries']

    for name in sorted(set(old) - set(new)):
        problems.append('%s: in the manifest but not offered for checking'
                        % name)
    for name in sorted(set(new) - set(old)):
        problems.append('%s: offered for checking but not in the manifest'
                        % name)

    for name in sorted(set(old) & set(new)):
        problems.extend(_compare_entry(name, old[name], new[name]))
    return problems


def _compare_entry(name, old, new):
    problems = []
    if old['kind'] != new['kind']:
        return ['%s: was %s, now %s' % (name, old['kind'], new['kind'])]

    if old['kind'] == 'bytes':
        if old['sha256'] != new['sha256']:
            problems.append('%s: file contents changed' % name)
        return problems

    old_d, new_d = old['datasets'], new['datasets']
    for path in sorted(set(old_d) - set(new_d)):
        problems.append('%s%s: dataset missing' % (name, path))
    for path in sorted(set(new_d) - set(old_d)):
        problems.append('%s%s: unexpected new dataset' % (name, path))

    for path in sorted(set(old_d) & set(new_d)):
        was, now = old_d[path], new_d[path]
        if was['shape'] != now['shape']:
            problems.append('%s%s: shape %s -> %s'
                            % (name, path, was['shape'], now['shape']))
        elif was['dtype'] != now['dtype']:
            problems.append('%s%s: dtype %s -> %s'
                            % (name, path, was['dtype'], now['dtype']))
        elif was['sha256'] != now['sha256']:
            problems.append('%s%s: values changed' % (name, path))
    return problems


# ---------------------------------------------------------------------
# numeric comparison, for when the hashes differ
# ---------------------------------------------------------------------

def outlet_index(cell_param_path):
    """Row index of the outlet cell: the one with cell_down < 0."""
    cell_down = np.loadtxt(cell_param_path)[:, CELL_DOWN_COL].astype(int)
    found = np.flatnonzero(cell_down < 0)
    if found.size != 1:
        raise ValueError('expected one outlet in %s, found %d: %s'
                         % (cell_param_path, found.size, found))
    return int(found[0])


def _block_difference(ref, cand):
    """Largest absolute and relative gap between two blocks.

    Two NaNs in the same slot count as agreement; a NaN facing a number
    does not.  The relative measure divides by the larger magnitude of the
    pair, so it is symmetric and safe near zero.
    """
    ref = ref.astype('f8', copy=False)
    cand = cand.astype('f8', copy=False)

    both_nan = np.isnan(ref) & np.isnan(cand)
    gap = np.abs(np.where(both_nan, 0.0, ref - cand))
    gap = np.where(np.isnan(gap), np.inf, gap)

    scale = np.maximum(np.abs(ref), np.abs(cand))
    rel = np.where(scale > 0, gap / np.where(scale > 0, scale, 1.0), 0.0)
    return gap, rel


def compare_hdf5(reference, candidate, cell_param=None,
                 chunk_rows=CHUNK_ROWS):
    """Dataset-by-dataset numeric comparison of two simulation files."""
    report = {}
    with h5py.File(reference, 'r') as ref_h5, \
            h5py.File(candidate, 'r') as cand_h5:
        ref_names = set(_dataset_names(ref_h5))
        cand_names = set(_dataset_names(cand_h5))
        report['only_in_reference'] = sorted(ref_names - cand_names)
        report['only_in_candidate'] = sorted(cand_names - ref_names)

        outlet = outlet_index(cell_param) if cell_param else None
        datasets = {}
        for name in sorted(ref_names & cand_names):
            ref_d, cand_d = ref_h5[name], cand_h5[name]
            if ref_d.shape != cand_d.shape:
                datasets[name] = {'shape_mismatch':
                                  [list(ref_d.shape), list(cand_d.shape)]}
                continue

            max_abs = max_rel = 0.0
            n_differ = 0
            worst_row = None
            for start in range(0, ref_d.shape[0] or 1, chunk_rows):
                stop = start + chunk_rows
                gap, rel = _block_difference(ref_d[start:stop],
                                             cand_d[start:stop])
                if gap.size == 0:
                    continue
                n_differ += int((gap > 0).sum())
                if gap.max() > max_abs:
                    max_abs = float(gap.max())
                    worst_row = start + int(np.unravel_index(
                        int(np.argmax(gap)), gap.shape)[0])
                max_rel = max(max_rel, float(rel.max()))

            entry = {'max_abs': max_abs, 'max_rel': max_rel,
                     'n_differ': n_differ, 'worst_row': worst_row}
            if outlet is not None and name.lower().endswith('qc_out'):
                gap, rel = _block_difference(ref_d[:, outlet],
                                             cand_d[:, outlet])
                entry['outlet'] = {'index': outlet,
                                   'max_abs': float(gap.max()),
                                   'max_rel': float(rel.max())}
            datasets[name] = entry
        report['datasets'] = datasets
    return report


# ---------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------

def _print_manifest(manifest):
    print('label      %s' % (manifest.get('label') or '-'))
    print('created    %s' % manifest['created_utc'])
    git = manifest['git']
    print('git        %s on %s%s' % (git['commit'], git['branch'],
                                     '  (DIRTY)' if git['dirty'] else ''))
    for name, entry in manifest['entries'].items():
        if entry['kind'] == 'bytes':
            print('\n%-12s %s  %d bytes'
                  % (name, entry['sha256'][:16], entry['n_byte']))
            continue
        print('\n%s  (%s)' % (name, entry['path']))
        for path, dset in entry['datasets'].items():
            shape = 'x'.join(str(n) for n in dset['shape'])
            print('  %-20s %-14s %-6s %s'
                  % (path, shape, dset['dtype'], dset['sha256'][:16]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m pytopkapi.results_analysis.fingerprint',
        description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)

    write = sub.add_parser('write', help='fingerprint a reference run')
    write.add_argument('--results', required=True)
    write.add_argument('--input', action='append', default=[],
                       dest='inputs', metavar='PATH')
    write.add_argument('--out', required=True, metavar='JSON')
    write.add_argument('--label', default=None)

    check = sub.add_parser('check', help='verify a run against a manifest')
    check.add_argument('--results', required=True)
    check.add_argument('--input', action='append', default=[],
                       dest='inputs', metavar='PATH')
    check.add_argument('--manifest', required=True, metavar='JSON')

    diff = sub.add_parser('compare', help='numeric difference of two runs')
    diff.add_argument('--reference', required=True)
    diff.add_argument('--candidate', required=True)
    diff.add_argument('--cell-param', default=None,
                      help='cell_param.dat, to isolate the outlet column')
    diff.add_argument('--rtol', type=float, default=0.0,
                      help='pass if every relative difference is below this')

    args = parser.parse_args(argv)

    if args.command == 'write':
        manifest = build_manifest(args.results, args.inputs, label=args.label)
        with open(args.out, 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write('\n')
        _print_manifest(manifest)
        print('\nwritten to %s' % args.out)
        return 0

    if args.command == 'check':
        with open(args.manifest) as handle:
            manifest = json.load(handle)
        if manifest.get('schema_version') != SCHEMA_VERSION:
            raise SystemExit('manifest schema %s, this code writes %d'
                             % (manifest.get('schema_version'),
                                SCHEMA_VERSION))
        problems = check_manifest(manifest, args.results, args.inputs)
        if not problems:
            print('IDENTICAL  %d entries match %s'
                  % (len(manifest['entries']), args.manifest))
            return 0
        print('CHANGED    %d problem(s) against %s'
              % (len(problems), args.manifest))
        for problem in problems:
            print('  ' + problem)
        print('\nRun the compare subcommand to size the difference.')
        return 1

    report = compare_hdf5(args.reference, args.candidate, args.cell_param)
    for side in ('only_in_reference', 'only_in_candidate'):
        for name in report[side]:
            print('%s: %s' % (side, name))

    print('%-22s %12s %12s %12s' % ('dataset', 'max abs', 'max rel', 'n differ'))
    print('-' * 62)
    worst = 0.0
    for name, entry in report['datasets'].items():
        if 'shape_mismatch' in entry:
            print('%-22s shape %s vs %s'
                  % (name, *entry['shape_mismatch']))
            worst = np.inf
            continue
        print('%-22s %12.4g %12.4g %12d'
              % (name, entry['max_abs'], entry['max_rel'], entry['n_differ']))
        worst = max(worst, entry['max_rel'])
        if 'outlet' in entry:
            out = entry['outlet']
            print('%-22s %12.4g %12.4g   (outlet cell %d)'
                  % ('  ^ outlet column', out['max_abs'], out['max_rel'],
                     out['index']))

    print('\nlargest relative difference anywhere: %.4g' % worst)
    if args.rtol > 0:
        verdict = 'WITHIN' if worst <= args.rtol else 'OUTSIDE'
        print('%s the %g tolerance' % (verdict, args.rtol))
        return 0 if worst <= args.rtol else 1
    return 0 if worst == 0.0 else 1


if __name__ == '__main__':
    sys.exit(main())
