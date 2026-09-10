"""Tests for topkapi_setup.calibrate.

These drive the real solver on PyTOPKAPI's shipped four-cell continuity
fixture. It runs in a fraction of a second, so the batch machinery is
exercised end to end -- worker processes, isolation, extraction, cleanup
-- rather than against a mock.

"""
import json
import os
import shutil
from configparser import ConfigParser

import h5py
import numpy as np
import pytest

from pytopkapi.results_analysis.fingerprint import outlet_index
from topkapi_setup import calibrate as cal

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(cal.__file__))), 'pytopkapi', 'tests', 'test_continuity')


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A private copy of the four-cell project, with cwd moved into it."""
    shutil.copytree(os.path.join(FIXTURE, '4cells'), str(tmp_path / '4cells'))
    shutil.copy(os.path.join(FIXTURE, '4cells.ini'), str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    return '4cells.ini'


def _reference(ini):
    """The unmodified run, all four multipliers at one."""
    from pytopkapi import model
    config = ConfigParser()
    config.read(ini)
    results = config.get('output_files', 'file_out')
    model.run(ini_file=ini, parallel_exec=False, quiet=True)
    outlet = outlet_index(config.get('input_files', 'file_cell_param'))
    with h5py.File(results, 'r') as handle:
        return (handle['Channel/Qc_out'][:, outlet].astype('f8'),
                handle['Q_down'][:, outlet].astype('f8'))


ONES = [1.0, 1.0, 1.0, 1.0]


def _same(left, right):
    """Bit-for-bit equality, so a NaN matches the identical NaN.

    Row zero of every results dataset holds the solver's no_data marker,
    and np.array_equal would call two identical NaNs different.
    """
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    return (left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes() == right.tobytes())


# ---- the invariant ---------------------------------------------------

def test_unit_factors_reproduce_the_plain_run_exactly(project):
    """A batch run of all-ones must be bit-identical to running it directly.

    If this drifts, every calibration result is measured against a
    different model than the one that was verified.
    """
    want_channel, want_down = _reference(project)
    batch = cal.run_batch(project, [ONES], workers=1, progress=False)
    assert batch['failures'] == {}
    assert _same(batch['Qc_out'][0], want_channel)
    assert _same(batch['Q_down'][0], want_down)


def test_the_same_set_run_twice_gives_the_same_answer(project):
    batch = cal.run_batch(project, [ONES, ONES], workers=2, progress=False)
    assert _same(batch['Qc_out'][0], batch['Qc_out'][1])


def test_workers_do_not_contaminate_each_other(project):
    """Distinct factor sets, run concurrently, must keep their identities.

    Four runs across four workers, each also run alone; the answers must
    match set for set. A shared output path or a shared ini would show up
    here as rows landing under the wrong parameters.
    """
    sets = [[f, 1.0, 1.0, 1.0] for f in (0.6, 1.0, 1.4, 2.0)]
    together = cal.run_batch(project, sets, workers=4, progress=False)
    assert together['failures'] == {}
    for index, one in enumerate(sets):
        alone = cal.run_batch(project, [one], workers=1, progress=False)
        assert _same(together['Qc_out'][index], alone['Qc_out'][0])


def test_factors_actually_change_the_answer(project):
    """A guard against the multipliers being silently ignored."""
    batch = cal.run_batch(project, [ONES, [1.0, 4.0, 1.0, 1.0]],
                          workers=1, progress=False)
    assert not _same(batch['Qc_out'][0], batch['Qc_out'][1])


# ---- housekeeping ----------------------------------------------------

def test_row_order_follows_the_input_not_completion_order(project):
    sets = [[f, 1.0, 1.0, 1.0] for f in (2.0, 0.5, 1.0, 1.5)]
    batch = cal.run_batch(project, sets, workers=3, progress=False)
    assert np.array_equal(batch['factors'], np.array(sets))
    single = cal.run_batch(project, [sets[1]], workers=1, progress=False)
    assert _same(batch['Qc_out'][1], single['Qc_out'][0])


def test_results_files_are_deleted(project, tmp_path):
    scratch = str(tmp_path / 'scratch')
    cal.run_batch(project, [ONES, ONES], workers=1, scratch=scratch,
                  progress=False)
    leftovers = [name for _, _, files in os.walk(scratch) for name in files]
    assert leftovers == []


def test_keep_results_leaves_them(project, tmp_path):
    scratch = str(tmp_path / 'scratch')
    cal.run_batch(project, [ONES], workers=1, scratch=scratch,
                  keep_results=True, progress=False)
    found = [name for _, _, files in os.walk(scratch)
             for name in files if name.endswith('.h5')]
    assert len(found) == 1


def test_a_failing_set_is_recorded_and_the_batch_continues(project):
    """A negative depth multiplier should break one run, not the batch."""
    batch = cal.run_batch(project, [ONES, [-1.0, 1.0, 1.0, 1.0], ONES],
                          workers=1, progress=False)
    assert set(batch['failures']) == {1}
    assert np.isnan(batch['Qc_out'][1]).all()
    assert np.isfinite(batch['Qc_out'][0]).any()
    assert _same(batch['Qc_out'][0], batch['Qc_out'][2])


def test_outlet_comes_from_topology(project):
    batch = cal.run_batch(project, [ONES], workers=1, progress=False)
    assert batch['outlet'] == outlet_index('4cells/cell_parameters.dat')


def test_wrong_factor_count_is_refused(project):
    with pytest.raises(ValueError, match='4 factors'):
        cal.run_batch(project, [[1.0, 1.0, 1.0]], workers=1, progress=False)


# ---- reading and writing --------------------------------------------

def test_read_factor_sets_ignores_column_order(tmp_path):
    path = str(tmp_path / 'sets.csv')
    with open(path, 'w') as handle:
        handle.write('fac_n_c,fac_L,fac_n_o,fac_Ks\n0.4,0.1,0.3,0.2\n')
    assert np.array_equal(cal.read_factor_sets(path),
                          np.array([[0.1, 0.2, 0.3, 0.4]]))


def test_read_factor_sets_names_the_missing_column(tmp_path):
    path = str(tmp_path / 'sets.csv')
    with open(path, 'w') as handle:
        handle.write('fac_L,fac_Ks,fac_n_o\n1,1,1\n')
    with pytest.raises(SystemExit, match='fac_n_c'):
        cal.read_factor_sets(path)


def test_write_batch_stores_hydrographs_and_a_manifest(project, tmp_path):
    batch = cal.run_batch(project, [ONES, ONES], workers=1, progress=False)
    out = str(tmp_path / 'hydrographs.h5')
    manifest_path = cal.write_batch(batch, out, project)

    with h5py.File(out, 'r') as handle:
        assert handle['Qc_out'].shape == (2, batch['n_t'])
        assert handle['factors'].shape == (2, 4)
        assert np.allclose(handle['Q_outlet'][...],
                           handle['Qc_out'][...] + handle['Q_down'][...],
                           equal_nan=True)
        assert handle.attrs['outlet_cell'] == batch['outlet']

    with open(manifest_path) as handle:
        manifest = json.load(handle)
    assert manifest['n_set'] == 2
    assert manifest['failures'] == {}
    assert 'cell_parameters.dat' in manifest['inputs']


def test_cli_round_trip(project, tmp_path, capsys):
    sets = str(tmp_path / 'sets.csv')
    with open(sets, 'w') as handle:
        handle.write('fac_L,fac_Ks,fac_n_o,fac_n_c\n1,1,1,1\n1,2,1,1\n')
    out = str(tmp_path / 'batch.h5')
    assert cal.main(['run', '--ini', project, '--factors', sets,
                     '--out', out, '--workers', '2', '--quiet']) == 0
    assert os.path.exists(out)
    assert os.path.exists(str(tmp_path / 'batch_manifest.json'))
    assert 'parameter sets' in capsys.readouterr().out


def test_cli_exits_nonzero_when_a_run_fails(project, tmp_path):
    sets = str(tmp_path / 'sets.csv')
    with open(sets, 'w') as handle:
        handle.write('fac_L,fac_Ks,fac_n_o,fac_n_c\n1,1,1,1\n-1,1,1,1\n')
    out = str(tmp_path / 'batch.h5')
    assert cal.main(['run', '--ini', project, '--factors', sets,
                     '--out', out, '--workers', '1', '--quiet']) == 1
