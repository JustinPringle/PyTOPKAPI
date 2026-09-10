"""Tests for pytopkapi.results_analysis.fingerprint.

Synthetic fixtures only: a three-cell, four-step simulation file with the
same dataset layout that utils.open_simulation_file writes, and a matching
cell_param.dat with a single outlet.

"""
import json
import os

import numpy as np
import h5py
import pytest

from pytopkapi.results_analysis import fingerprint as fp

DATASETS = ('Soil/Qs_out', 'Soil/V_s',
            'Overland/Qo_out', 'Overland/V_o',
            'Channel/Qc_out', 'Channel/V_c', 'Channel/Ec_out',
            'ET_out', 'Q_down')

N_STEP, N_CELL = 5, 3
OUTLET = 1


def _write_results(path, seed=0, compression='gzip'):
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as h5file:
        h5file.attrs['title'] = 'PyTOPKAPI simulation'
        h5file.attrs['pytopkapi_git_revision'] = 'deadbeef'
        for name in DATASETS:
            values = rng.random((N_STEP, N_CELL)).astype('f4')
            h5file.create_dataset(name, data=values, compression=compression)
    return path


def _write_cell_param(path, n_cell=N_CELL, outlet=OUTLET):
    table = np.zeros((n_cell, 21))
    table[:, 0] = np.arange(n_cell)
    table[:, fp.CELL_DOWN_COL] = np.arange(1, n_cell + 1)
    table[outlet, fp.CELL_DOWN_COL] = -9999
    np.savetxt(path, table)
    return path


@pytest.fixture
def scene(tmp_path):
    """A reference run, its cell_param file, and a written manifest."""
    results = _write_results(str(tmp_path / 'sim.h5'))
    cell_param = _write_cell_param(str(tmp_path / 'cell_param.dat'))
    manifest = fp.build_manifest(results, [cell_param], label='fixture')
    return {'dir': str(tmp_path), 'results': results,
            'cell_param': cell_param, 'manifest': manifest}


# ---- the fingerprint itself -----------------------------------------

def test_manifest_covers_every_dataset(scene):
    entry = scene['manifest']['entries']['results']
    assert entry['kind'] == 'hdf5'
    assert set(entry['datasets']) == {'/' + name for name in DATASETS}
    for dset in entry['datasets'].values():
        assert dset['shape'] == [N_STEP, N_CELL]
        assert dset['dtype'] == '<f4'
        assert len(dset['sha256']) == 64


def test_unchanged_run_checks_clean(scene):
    problems = fp.check_manifest(scene['manifest'], scene['results'],
                                 [scene['cell_param']])
    assert problems == []


def test_manifest_survives_a_json_round_trip(scene, tmp_path):
    path = str(tmp_path / 'fingerprint.json')
    with open(path, 'w') as handle:
        json.dump(scene['manifest'], handle)
    with open(path) as handle:
        reloaded = json.load(handle)
    assert fp.check_manifest(reloaded, scene['results'],
                             [scene['cell_param']]) == []


def test_one_changed_value_is_caught(scene):
    with h5py.File(scene['results'], 'r+') as h5file:
        h5file['Channel/Qc_out'][3, 2] += np.float32(1e-6)
    problems = fp.check_manifest(scene['manifest'], scene['results'],
                                 [scene['cell_param']])
    assert problems == ['results/Channel/Qc_out: values changed']


def test_container_layout_is_not_part_of_the_invariant(scene, tmp_path):
    """Re-storing the same numbers uncompressed must still check clean.

    This is the whole point: buffering the writes will change the chunk
    layout on disk without touching a single number.
    """
    rewritten = str(tmp_path / 'sim_uncompressed.h5')
    with h5py.File(scene['results'], 'r') as src, \
            h5py.File(rewritten, 'w') as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for name in DATASETS:
            dst.create_dataset(name, data=src[name][...], compression=None)

    problems = fp.check_manifest(scene['manifest'], rewritten,
                                 [scene['cell_param']])
    assert problems == []


def test_git_revision_attribute_is_not_compared(scene, tmp_path):
    moved = str(tmp_path / 'sim_next_commit.h5')
    with h5py.File(scene['results'], 'r') as src, \
            h5py.File(moved, 'w') as dst:
        dst.attrs['title'] = src.attrs['title']
        dst.attrs['pytopkapi_git_revision'] = 'cafef00d'
        for name in DATASETS:
            dst.create_dataset(name, data=src[name][...])
    assert fp.check_manifest(scene['manifest'], moved,
                             [scene['cell_param']]) == []


def test_changed_input_file_is_caught(scene):
    with open(scene['cell_param'], 'a') as handle:
        handle.write('# a stray edit\n')
    problems = fp.check_manifest(scene['manifest'], scene['results'],
                                 [scene['cell_param']])
    assert problems == ['cell_param.dat: file contents changed']


def test_missing_input_is_reported(scene):
    problems = fp.check_manifest(scene['manifest'], scene['results'], [])
    assert len(problems) == 1
    assert 'cell_param.dat' in problems[0]


def test_shape_change_beats_the_value_check(scene, tmp_path):
    shorter = str(tmp_path / 'sim_short.h5')
    with h5py.File(scene['results'], 'r') as src, \
            h5py.File(shorter, 'w') as dst:
        for name in DATASETS:
            dst.create_dataset(name, data=src[name][:-1, :])
    problems = fp.check_manifest(scene['manifest'], shorter,
                                 [scene['cell_param']])
    assert len(problems) == len(DATASETS)
    assert all('shape' in problem for problem in problems)


def test_chunk_size_does_not_change_the_digest(scene):
    coarse = fp.fingerprint_hdf5(scene['results'], chunk_rows=1024)
    fine = fp.fingerprint_hdf5(scene['results'], chunk_rows=1)
    assert coarse['datasets'] == fine['datasets']


# ---- outlet topology -------------------------------------------------

def test_outlet_comes_from_topology(scene):
    assert fp.outlet_index(scene['cell_param']) == OUTLET


def test_two_outlets_raise(tmp_path):
    path = str(tmp_path / 'two_outlets.dat')
    table = np.zeros((4, 21))
    table[:, fp.CELL_DOWN_COL] = [-1, 2, 3, -1]
    np.savetxt(path, table)
    with pytest.raises(ValueError, match='found 2'):
        fp.outlet_index(path)


# ---- numeric comparison ---------------------------------------------

def test_compare_of_a_file_with_itself_is_all_zero(scene):
    report = fp.compare_hdf5(scene['results'], scene['results'],
                             scene['cell_param'])
    assert report['only_in_reference'] == []
    for entry in report['datasets'].values():
        assert entry['max_abs'] == 0.0
        assert entry['max_rel'] == 0.0
        assert entry['n_differ'] == 0
    assert report['datasets']['/Channel/Qc_out']['outlet']['index'] == OUTLET


def test_compare_sizes_a_known_perturbation(scene, tmp_path):
    nudged = str(tmp_path / 'nudged.h5')
    with h5py.File(scene['results'], 'r') as src, \
            h5py.File(nudged, 'w') as dst:
        for name in DATASETS:
            dst.create_dataset(name, data=src[name][...])
        base = float(dst['Channel/Qc_out'][2, OUTLET])
        dst['Channel/Qc_out'][2, OUTLET] = np.float32(base * (1 + 1e-3))

    report = fp.compare_hdf5(scene['results'], nudged, scene['cell_param'])
    channel = report['datasets']['/Channel/Qc_out']
    assert channel['n_differ'] == 1
    assert channel['worst_row'] == 2
    # the denominator is the larger of the pair, so the reported figure is
    # 1e-3 / (1 + 1e-3), a hair under the nominal nudge
    assert channel['max_rel'] == pytest.approx(1e-3, rel=1e-2)
    assert channel['outlet']['max_rel'] == pytest.approx(1e-3, rel=1e-2)
    assert report['datasets']['/Soil/V_s']['max_rel'] == 0.0


def test_matching_nan_counts_as_agreement(tmp_path):
    paths = []
    for name in ('a.h5', 'b.h5'):
        path = str(tmp_path / name)
        with h5py.File(path, 'w') as h5file:
            values = np.array([[1.0, np.nan], [3.0, 4.0]], dtype='f4')
            h5file.create_dataset('Channel/Qc_out', data=values)
        paths.append(path)
    report = fp.compare_hdf5(paths[0], paths[1])
    assert report['datasets']['/Channel/Qc_out']['max_abs'] == 0.0


def test_nan_facing_a_number_is_an_infinite_gap(tmp_path):
    good = str(tmp_path / 'good.h5')
    bad = str(tmp_path / 'bad.h5')
    with h5py.File(good, 'w') as h5file:
        h5file.create_dataset('Channel/Qc_out',
                              data=np.array([[1.0, 2.0]], dtype='f4'))
    with h5py.File(bad, 'w') as h5file:
        h5file.create_dataset('Channel/Qc_out',
                              data=np.array([[1.0, np.nan]], dtype='f4'))
    report = fp.compare_hdf5(good, bad)
    assert np.isinf(report['datasets']['/Channel/Qc_out']['max_abs'])


# ---- command line ----------------------------------------------------

def test_cli_write_then_check_round_trip(scene, tmp_path, capsys):
    manifest_path = str(tmp_path / 'fingerprint_reference.json')
    write = ['write', '--results', scene['results'],
             '--input', scene['cell_param'],
             '--out', manifest_path, '--label', 'reference']
    assert fp.main(write) == 0
    assert os.path.exists(manifest_path)

    check = ['check', '--results', scene['results'],
             '--input', scene['cell_param'],
             '--manifest', manifest_path]
    assert fp.main(check) == 0
    assert 'IDENTICAL' in capsys.readouterr().out

    with h5py.File(scene['results'], 'r+') as h5file:
        h5file['ET_out'][0, 0] += np.float32(1.0)
    assert fp.main(check) == 1
    assert 'CHANGED' in capsys.readouterr().out


def test_cli_compare_exit_codes(scene, tmp_path):
    same = ['compare', '--reference', scene['results'],
            '--candidate', scene['results'],
            '--cell-param', scene['cell_param']]
    assert fp.main(same) == 0

    nudged = str(tmp_path / 'nudged.h5')
    with h5py.File(scene['results'], 'r') as src, \
            h5py.File(nudged, 'w') as dst:
        for name in DATASETS:
            dst.create_dataset(name, data=src[name][...])
        dst['Q_down'][1, 0] = np.float32(float(dst['Q_down'][1, 0]) * 1.0001)

    differ = ['compare', '--reference', scene['results'],
              '--candidate', nudged, '--cell-param', scene['cell_param']]
    assert fp.main(differ) == 1
    assert fp.main(differ + ['--rtol', '1e-3']) == 0
    assert fp.main(differ + ['--rtol', '1e-12']) == 1
