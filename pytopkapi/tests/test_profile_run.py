"""Tests for pytopkapi.results_analysis.profile_run.

These drive the real solver on the shipped four-cell continuity fixture,
which runs in a fraction of a second, so the harness is exercised
end-to-end rather than against a mock.

"""
import os
import shutil
from configparser import ConfigParser

import h5py
import pytest

from pytopkapi import model
from pytopkapi import ode
from pytopkapi.results_analysis import profile_run as pr

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'test_continuity')
N_CELL = 4


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A private copy of the four-cell project, with cwd moved into it."""
    shutil.copytree(os.path.join(FIXTURE, '4cells'), str(tmp_path / '4cells'))
    shutil.copy(os.path.join(FIXTURE, '4cells.ini'), str(tmp_path))
    monkeypatch.chdir(str(tmp_path))
    return '4cells.ini'


def _forcing_rows(ini_path):
    config = ConfigParser()
    config.read(ini_path)
    group = config.get('groups', 'group_name')
    with h5py.File(config.get('input_files', 'file_rain'), 'r') as handle:
        return handle['/%s/rainfall' % group].shape[0]


# ---- truncation ------------------------------------------------------

def test_truncation_cuts_the_forcing(project, tmp_path):
    scratch = str(tmp_path / 'scratch')
    os.makedirs(scratch)
    cut = pr.truncate_project(project, 7, scratch)
    assert _forcing_rows(cut) == 7
    assert pr._run_size(cut) == (N_CELL, 7)


def test_truncation_cuts_et_as_well(project, tmp_path):
    scratch = str(tmp_path / 'scratch')
    os.makedirs(scratch)
    cut = pr.truncate_project(project, 5, scratch)
    config = ConfigParser()
    config.read(cut)
    group = config.get('groups', 'group_name')
    with h5py.File(config.get('input_files', 'file_ET'), 'r') as handle:
        assert handle['/%s/ETr' % group].shape[0] == 5
        assert handle['/%s/ETo' % group].shape[0] == 5


def test_truncation_cannot_overwrite_the_original_output(project, tmp_path):
    scratch = str(tmp_path / 'scratch')
    os.makedirs(scratch)
    cut = pr.truncate_project(project, 5, scratch)
    config = ConfigParser()
    config.read(cut)
    assert config.get('output_files', 'file_out').startswith(scratch)
    assert config.getboolean('output_files', 'append_output') is False


def test_scratch_project_cleans_up_after_itself(project):
    with pr.scratch_project(project, 5) as cut:
        directory = os.path.dirname(cut)
        assert os.path.exists(directory)
    assert not os.path.exists(directory)


def test_scratch_project_passes_the_ini_through_untouched(project):
    with pr.scratch_project(project, None) as cut:
        assert cut == project


# ---- the census ------------------------------------------------------

def test_census_counts_every_cell_solve(project):
    with pr.scratch_project(project, 6) as cut:
        with pr._Census() as census:
            pr._run_once(cut)
    assert census.count['cell_solve'] == N_CELL * 6


def test_census_accounts_for_every_storage_equation(project):
    with pr.scratch_project(project, 6) as cut:
        with pr._Census() as census:
            pr._run_once(cut)
    count = census.count
    solved_by_qas = count['qas_call'] - count['qas_failed']
    branches = (count['input_zero'] + count['coefb_zero'] +
                solved_by_qas + count['rkf_solve'])
    assert branches == count['storage_eq_call']


def test_census_restores_the_solver(project):
    before = (model._solve_cell_core, ode.solve_storage_eq, ode.qas,
              ode.input_zero_solution, ode.coefb_zero_solution, ode.RKF.step)
    with pr.scratch_project(project, 3) as cut:
        with pr._Census():
            pr._run_once(cut)
    after = (model._solve_cell_core, ode.solve_storage_eq, ode.qas,
             ode.input_zero_solution, ode.coefb_zero_solution, ode.RKF.step)
    assert before == after


def test_census_survives_an_exception(project):
    before = model._solve_cell_core
    with pytest.raises(ValueError):
        with pr._Census():
            raise ValueError('boom')
    assert model._solve_cell_core is before


# ---- timing and the command line -------------------------------------

def test_timed_runs_and_writes_output(project):
    with pr.scratch_project(project, 5, keep=True) as cut:
        seconds = pr.timed(cut, repeat=2)
        config = ConfigParser()
        config.read(cut)
        results = config.get('output_files', 'file_out')
        assert os.path.exists(results)
        with h5py.File(results, 'r') as handle:
            assert handle['Channel/Qc_out'].shape == (6, N_CELL)
    assert len(seconds) == 2
    assert all(second > 0 for second in seconds)


def test_cli_time(project, capsys):
    assert pr.main(['time', '--ini', project, '--steps', '5']) == 0
    assert 'per cell-solve' in capsys.readouterr().out


def test_cli_tally(project, capsys):
    assert pr.main(['tally', '--ini', project, '--steps', '5']) == 0
    out = capsys.readouterr().out
    assert 'cell-solves' in out
    assert 'Runge-Kutta fallback' in out


def test_cli_profile_writes_a_prof_file(project, tmp_path, capsys):
    target = str(tmp_path / 'baseline.prof')
    assert pr.main(['profile', '--ini', project, '--steps', '5',
                    '--top', '5', '--out', target]) == 0
    assert os.path.exists(target)
    out = capsys.readouterr().out
    assert 'by time in the function itself' in out
    assert 'by time including everything it calls' in out


def test_cli_runs_serially(project, monkeypatch):
    """The harness must never hand parallel_exec a True."""
    seen = {}
    original = model.run

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(model, 'run', spy)
    pr.main(['time', '--ini', project, '--steps', '3'])
    assert seen['parallel_exec'] is False
