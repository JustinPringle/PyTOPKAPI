"""The two ways into a cell solve must agree exactly.

`_serial_execute` calls `_solve_cell_core` positionally; the parallel path
calls `_solve_cell` with a dict. They are the same calculation and must
stay that way, so this pins the adapter's key-to-position mapping. Get one
pair transposed — say `b_o` and `b_c`, or `Vo0` and `Vc0` — and the model
still runs, and is quietly wrong.

"""
import numpy as np
import pytest

from pytopkapi.model import _solve_cell, _solve_cell_core

# the argument order of _solve_cell_core, paired with the dict keys
# _solve_cell reads. Two names differ between the two forms, which is
# exactly the sort of thing worth writing down.
ORDER = (('Dt', 'Dt'),
         ('rain_depth', 'rain_depth'),
         ('psi', 'psi'),
         ('eff_theta', 'eff_theta'),
         ('eff_sat', 'eff_sat'),
         ('Ks', 'Ks'),
         ('X', 'X'),
         ('soil_upstream_inflow', 'soil_upstream_inflow'),
         ('b_s', 'b_s'),
         ('alpha_s', 'alpha_s'),
         ('Vs0', 'Vs0'),
         ('solve_s', 'solve_s'),
         ('Vsm', 'Vsm'),
         ('b_o', 'b_o'),
         ('alpha_o', 'alpha_o'),
         ('Vo0', 'Vo0'),
         ('solve_o', 'solve_o'),
         ('channel_flag', 'channel_flag'),
         ('W', 'W'),
         ('Xc', 'Xc'),
         ('channel_upstream_inflow', 'chan_up_inflow'),
         ('Kc', 'Kc'),
         ('ETr', 'ETr'),
         ('b_c', 'b_c'),
         ('alpha_c', 'alpha_c'),
         ('Vc0', 'Vc0'),
         ('solve_c', 'solve_c'),
         ('ET0', 'ET0'),
         ('external_flow_flag', 'external_flow_flag'),
         ('external_flow', 'external_flow'))


def _cell(rng, channel, external):
    """A plausible parameter set for one cell of a 60 m grid."""
    Vsm = float(rng.uniform(50.0, 1000.0))
    n_up = int(rng.integers(0, 4))          # headwater cells have none
    return {'Dt': 3600.0,
            'rain_depth': float(rng.uniform(0.0, 20.0)),
            'psi': float(rng.uniform(0.05, 0.6)),
            'eff_theta': float(rng.uniform(0.2, 0.45)),
            'eff_sat': float(rng.uniform(0.05, 0.99)),
            'Ks': float(rng.uniform(1e-4, 6e-2)),
            'X': 60.0,
            # upstream inflows arrive as arrays, one entry per upstream
            # cell: the solver indexes them with li_cell_up and sums
            'soil_upstream_inflow': rng.uniform(0.0, 0.5, size=n_up),
            'b_s': float(rng.uniform(1e-6, 1e-3)),
            'alpha_s': 2.5,
            'Vs0': float(rng.uniform(0.0, Vsm)),
            'solve_s': 1,
            'Vsm': Vsm,
            'b_o': float(rng.uniform(1e-4, 1e-1)),
            'alpha_o': 1.6666667,
            'Vo0': float(rng.uniform(0.0, 50.0)),
            'solve_o': 1,
            'channel_flag': 1 if channel else 0,
            'W': float(rng.uniform(0.5, 8.0)),
            'Xc': 60.0,
            'chan_up_inflow': rng.uniform(0.0, 5.0, size=n_up),
            'Kc': 1.0,
            'ETr': float(rng.uniform(0.0, 0.5)),
            'b_c': float(rng.uniform(1e-3, 1.0)),
            'alpha_c': 1.6666667,
            'Vc0': float(rng.uniform(0.0, 100.0)),
            'solve_c': 1,
            'ET0': float(rng.uniform(0.0, 0.5)),
            'external_flow_flag': external,
            'external_flow': float(rng.uniform(0.0, 2.0)) if external else None}


def _positional(params):
    return [params[key] for _, key in ORDER]


@pytest.mark.parametrize('channel', [False, True])
@pytest.mark.parametrize('external', [False, True])
def test_adapter_matches_the_core(channel, external):
    rng = np.random.default_rng(20260910)
    for _ in range(500):
        params = _cell(rng, channel, external)
        by_dict = _solve_cell(params)
        by_position = _solve_cell_core(*_positional(params))
        assert len(by_dict) == 9
        for left, right in zip(by_dict, by_position):
            assert repr(left) == repr(right)


def test_the_core_takes_exactly_the_arguments_the_adapter_supplies():
    """A guard on the count, so an added argument cannot go unnoticed."""
    code = _solve_cell_core.__code__
    assert code.co_argcount == len(ORDER)
    assert tuple(code.co_varnames[:code.co_argcount]) == \
        tuple(name for name, _ in ORDER)


def test_zero_rainfall_on_a_dry_hillslope_loses_nothing_downstream():
    """A hand-checkable case: no rain, no inflow, empty stores, no flow."""
    rng = np.random.default_rng(1)
    params = _cell(rng, channel=False, external=False)
    params.update({'rain_depth': 0.0,
                   'soil_upstream_inflow': np.zeros(0),
                   'chan_up_inflow': np.zeros(0),
                   'Vs0': 0.0, 'Vo0': 0.0, 'ETr': 0.0, 'ET0': 0.0})
    Qs_out, Qo_out, Qc_out, Q_down = _solve_cell(params)[:4]
    assert Qs_out == 0.0
    assert Qo_out == 0.0
    assert Qc_out == 0.0
    assert Q_down == 0.0
