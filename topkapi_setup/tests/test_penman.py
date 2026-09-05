"""Tests for penman.py.

Every intermediate and the two reference-ET methods are pinned to published
FAO-56 numbers (Allen et al., 1998), so a regression shows up as a disagreement
with the book, not with our own arithmetic. The chain culminates in the FAO-56
worked example: Rs=22.07, Tmax=21.5, Tmin=12.3, ea=1.409, lat=50.8, z=100,
doy=187 -> Rn=13.28 -> ET0=3.9 mm/day.
"""

import numpy as np
import pandas as pd
import pytest

from topkapi_setup.forcing import penman as pm


# --------------------------------------------------------------------------
# vapour pressure, slope, pressure, gamma
# --------------------------------------------------------------------------

def test_svp_and_mean_svp():
    assert pm.svp(25) == pytest.approx(3.168, abs=1e-3)
    assert pm.svp(18) == pytest.approx(2.064, abs=1e-3)
    assert pm.mean_svp(25, 18) == pytest.approx(2.616, abs=1e-3)


def test_svp_slope():
    assert pm.svp_slope(16.9) == pytest.approx(0.122, abs=1e-3)


def test_pressure_and_psychrometric_constant():
    assert pm.atm_pressure(100) == pytest.approx(100.1, abs=0.1)
    assert pm.atm_pressure(1800) == pytest.approx(81.8, abs=0.1)     # FAO Ex. 2
    assert pm.psychrometric_constant(elevation=100) == pytest.approx(0.0666, abs=1e-4)
    assert pm.psychrometric_constant(pressure=81.8) == pytest.approx(0.054, abs=1e-3)


# --------------------------------------------------------------------------
# actual vapour pressure routes
# --------------------------------------------------------------------------

def test_ea_from_tdew_and_rhmean():
    # tdew route: ea = e0(tdew)
    assert pm.ea_from_tdew(17.0) == pytest.approx(pm.svp(17.0))
    # rhmean route (FAO Eq. 19): 50% of es
    ea = pm.ea_from_rhmean(50.0, 25, 18)
    assert ea == pytest.approx(0.5 * pm.mean_svp(25, 18), abs=1e-6)


# --------------------------------------------------------------------------
# solar geometry / radiation
# --------------------------------------------------------------------------

def test_extraterrestrial_radiation_matches_fao_examples():
    assert pm.extraterrestrial_radiation(-20, 246) == pytest.approx(32.2, abs=0.1)
    assert pm.extraterrestrial_radiation(50.8, 187) == pytest.approx(41.1, abs=0.2)


def test_daylight_hours():
    assert pm.daylight_hours(-20, 246) == pytest.approx(11.7, abs=0.1)


def test_rs_from_wm2_conversion():
    assert pm.rs_from_wm2(100.0) == pytest.approx(8.64, abs=1e-6)
    # the example's Rs=22.07 corresponds to a daily mean near 255.4 W/m^2
    assert pm.rs_from_wm2(255.4) == pytest.approx(22.07, abs=0.02)


def test_net_radiation_matches_fao_example():
    rn = pm.net_radiation(rs=22.07, tmax=21.5, tmin=12.3, ea=1.409,
                          latitude=50.8, elevation=100, doy=187)
    assert rn == pytest.approx(13.28, abs=0.05)


def test_net_longwave_cloud_factor_clipped():
    # Rs above Rso must not make the cloudiness factor exceed 1 (would flip Rnl)
    rnl_hi = pm.net_longwave(30, 20, 1.5, rs=40.0, rso=30.0)
    rnl_eq = pm.net_longwave(30, 20, 1.5, rs=30.0, rso=30.0)
    assert rnl_hi == pytest.approx(rnl_eq)


# --------------------------------------------------------------------------
# wind adjustment
# --------------------------------------------------------------------------

def test_wind_speed_2m():
    assert pm.wind_speed_2m(3.0, measured_height=2.0) == 3.0        # no change
    # 10 m -> 2 m reduces the speed (FAO Eq. 47)
    assert pm.wind_speed_2m(3.0, measured_height=10.0) == pytest.approx(2.244, abs=1e-3)


# --------------------------------------------------------------------------
# the reference ET0 combination and the full path
# --------------------------------------------------------------------------

def test_penman_monteith_combine_matches_worked_example():
    et0 = pm.penman_monteith_combine(delta=0.122, rn=13.28, g=0.0,
                                     gamma=0.0666, tmean=16.9, u2=2.078,
                                     es_minus_ea=0.589)
    assert et0 == pytest.approx(3.88, abs=0.02)


def test_et0_penman_monteith_full_chain():
    et0 = pm.et0_penman_monteith(
        tmax=21.5, tmin=12.3, rs=22.07, u2=2.078, ea=1.409,
        elevation=100, latitude=50.8, doy=187, tmean=16.9)
    assert et0 == pytest.approx(3.9, abs=0.05)


def test_et0_hargreaves_reasonable_and_formula():
    # hand value for the same case: 0.0023*(16.9+17.8)*sqrt(9.2)*0.408*Ra
    ra = pm.extraterrestrial_radiation(50.8, 187)
    expect = 0.0023 * (16.9 + 17.8) * np.sqrt(9.2) * 0.408 * ra
    got = pm.et0_hargreaves(21.5, 12.3, 187, 50.8, tmean=16.9)
    assert got == pytest.approx(expect, rel=1e-6)
    assert 3.0 < got < 5.0          # same ballpark as PM, typically a touch high


def test_hargreaves_guards_bad_temperature_range():
    # Tmax < Tmin must not raise (sqrt of negative) -- clipped to 0
    assert pm.et0_hargreaves(10.0, 15.0, 187, 50.8) == 0.0


# --------------------------------------------------------------------------
# open water
# --------------------------------------------------------------------------

def test_open_water_positive_and_exceeds_grass():
    common = dict(tmax=21.5, tmin=12.3, rs=22.07, u2=2.078, ea=1.409,
                  elevation=100, latitude=50.8, doy=187)
    e0 = pm.e0_open_water(**common)
    et0 = pm.et0_penman_monteith(**common, tmean=16.9)
    assert e0 > 0
    assert e0 > et0                 # open water evaporates faster than grass


def test_open_water_increases_with_wind():
    base = dict(tmax=30, tmin=20, rs=25.0, ea=1.5, elevation=50,
                latitude=-29.8, doy=15)
    calm = pm.e0_open_water(u2=1.0, **base)
    windy = pm.e0_open_water(u2=5.0, **base)
    assert windy > calm


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------

def test_reference_et0_picks_penman_when_data_present():
    row = pd.Series({"tmax": 21.5, "tmin": 12.3, "tmean": 16.9,
                     "solar": 255.4, "wind": 2.078, "rh": 50.0, "tdew": np.nan,
                     "elevation_m": 100.0, "date": pd.Timestamp("2001-07-06")})
    et0, method = pm.reference_et0(row, latitude=50.8)
    assert method == "penman_monteith"
    assert 3.0 < et0 < 5.0


def test_reference_et0_falls_back_to_hargreaves_without_humidity():
    # no rh, no tdew -> Hargreaves even though solar/wind exist
    row = pd.Series({"tmax": 30.0, "tmin": 20.0, "tmean": 25.0,
                     "solar": 300.0, "wind": 2.0, "rh": np.nan, "tdew": np.nan,
                     "elevation_m": 50.0, "date": pd.Timestamp("2025-01-15")})
    et0, method = pm.reference_et0(row, latitude=-29.8)
    assert method == "hargreaves"
    assert et0 > 0


def test_reference_et0_nan_without_temperature():
    row = pd.Series({"tmax": np.nan, "tmin": np.nan, "tmean": np.nan,
                     "solar": 300.0, "wind": 2.0, "rh": 60.0, "tdew": np.nan,
                     "elevation_m": 50.0, "date": pd.Timestamp("2025-01-15")})
    et0, method = pm.reference_et0(row, latitude=-29.8)
    assert np.isnan(et0) and method == "none"
