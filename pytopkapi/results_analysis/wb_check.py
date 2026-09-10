"""Scratch water-balance + parameter-range check for a PyTOPKAPI run.

Usage:
    python wb_check.py results.h5 cell_param.dat global_param.dat rainfields.h5 GROUP

Reads the solver's own outputs and closes the catchment books:

    rain  -  ET  -  outflow  -  dS   =  residual

Outlet is found from topology (cell_down < 0), never assumed.
"""
import sys
import numpy as np
import h5py

M3_TO_MM = None  # set once catchment area is known


def main(f_res, f_cell, f_glob, f_rain, group):
    # ---- static -------------------------------------------------------
    cp = np.loadtxt(f_cell)
    gp = np.loadtxt(f_glob, skiprows=1)
    X, Dt = float(gp[0]), float(gp[1])
    area_cell = X ** 2
    n_cell = cp.shape[0]
    area = n_cell * area_cell

    cell_down = cp[:, 14].astype(int)
    outlet = np.flatnonzero(cell_down < 0)
    if outlet.size != 1:
        raise SystemExit(f'expected one outlet, found {outlet.size}: {outlet}')
    outlet = int(outlet[0])

    def mm(v):
        return v / area * 1e3

    # ---- forcing ------------------------------------------------------
    with h5py.File(f_rain, 'r') as h:
        rain = h[f'/{group}/rainfall'][...]          # (n_t, n_cell), mm
    V_rain = rain.sum() * 1e-3 * area_cell

    # ---- state and fluxes ---------------------------------------------
    with h5py.File(f_res, 'r') as h:
        Vs = h['Soil/V_s'][...]                      # (n_t+1, n_cell), m3
        Vo = h['Overland/V_o'][...]
        Vc = h['Channel/V_c'][...]
        Ec = h['Channel/Ec_out'][1:, :]              # m3
        ET = h['ET_out'][1:, :]                      # mm
        Qc = h['Channel/Qc_out'][1:, outlet]         # m3/s
        Qd = h['Q_down'][1:, outlet]                 # m3/s

    S = np.nansum(Vs, axis=1) + np.nansum(Vo, axis=1) + np.nansum(Vc, axis=1)
    dS = S[-1] - S[0]

    V_et = np.nansum(ET) * 1e-3 * area_cell + np.nansum(Ec)
    V_out = np.nansum(Qc + Qd) * Dt

    resid = V_rain - V_et - V_out - dS
    scale = max(abs(V_rain), abs(dS), abs(V_out), 1.0)

    print(f'\ncells {n_cell}   X {X:g} m   Dt {Dt:g} s   steps {rain.shape[0]}')
    print(f'area {area/1e6:.2f} km2   outlet cell {outlet}\n')
    print(f'{"term":<22}{"m3":>16}{"mm":>12}')
    print('-' * 50)
    print(f'{"rain in":<22}{V_rain:16.1f}{mm(V_rain):12.2f}')
    print(f'{"ET out":<22}{-V_et:16.1f}{mm(-V_et):12.2f}')
    print(f'{"outlet out":<22}{-V_out:16.1f}{mm(-V_out):12.2f}')
    print(f'{"storage change":<22}{-dS:16.1f}{mm(-dS):12.2f}')
    print('-' * 50)
    print(f'{"residual":<22}{resid:16.1f}{mm(resid):12.2f}'
          f'   ({100*resid/scale:+.3f}% of largest term)')
    print(f'\nstorage start {S[0]:.3e} m3   end {S[-1]:.3e} m3'
          f'   ({100*(S[-1]-S[0])/S[0]:+.1f}%)')
    nan_frac = np.isnan(Vs).mean() + np.isnan(Vc).mean()
    if nan_frac:
        print(f'WARNING: NaNs present in state arrays (frac {nan_frac:.4f})')

    # ---- parameter ranges ---------------------------------------------
    cols = {9: ('Ks  mm/s', 1e-4, 6e-2),
            8: ('L    m', 0.1, 5.0),
            11: ('theta_s', 0.3, 0.6),
            10: ('theta_r', 0.0, 0.15),
            6: ('tan_beta', 1e-4, 2.0),
            7: ('tan_beta_c', 1e-4, 2.0),
            13: ('n_c', 0.01, 0.15),
            15: ('pVs_t0 %', 0.0, 100.0)}
    print(f'\n{"col":<5}{"name":<12}{"min":>12}{"p50":>12}{"max":>12}  flag')
    print('-' * 60)
    for c, (name, lo, hi) in cols.items():
        v = cp[:, c]
        q = np.nanpercentile(v, [0, 50, 100])
        bad = (v < lo) | (v > hi) | ~np.isfinite(v)
        flag = f'{bad.sum()} outside [{lo:g}, {hi:g}]' if bad.any() else ''
        print(f'{c:<5}{name:<12}{q[0]:12.4g}{q[1]:12.4g}{q[2]:12.4g}  {flag}')

    ks = cp[:, 9]
    print(f'\nKs median {np.nanmedian(ks):.4g} mm/s'
          f'  = {np.nanmedian(ks)*360:.4g} cm/h')
    print('  Rawls range in mm/s is ~1e-4 (clay) to ~6e-2 (sand).')
    print('  If the median lands near 1-100 mm/s, the column holds cm/h.')


if __name__ == '__main__':
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
