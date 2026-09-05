"""Reference and open-water evaporation formulas for the ET forcing.

Pure functions, no IO: they take numbers (scalars or arrays) and return numbers,
so every one is checkable against the FAO-56 worked examples (Allen et al.,
1998).  The ET build stage calls these per station per day on the daily table
:mod:`met` produces, then disaggregates and interpolates.

Three outputs matter:

* :func:`et0_penman_monteith` -- FAO-56 reference-crop ET (grass), the ``ETr``
  the solver reads.  This is ``ET0`` in FAO notation.
* :func:`et0_hargreaves` -- the temperature-only fallback, for a station with no
  humidity/wind/radiation.  Keeps a first run possible on thin data.
* :func:`e0_open_water` -- Penman (1948) open-water evaporation, the ``ETo`` the
  channel loses at.  Its fallback, when radiation is absent, is ``ETo = ETr``
  (channel evaporation is a sub-percent term in the water balance, so this is
  cheap and defensible -- see the ET design note).

**All three return mm/day.**  The split into mm-per-``Dt`` depths, with a
clear-sky diurnal weight, is ``gauges.disaggregate``'s job downstream, not this
module's.  A value left in mm/day would over-evaporate the catchment by a factor
of ``86400/Dt`` -- the units trap the design note names.

Inputs and their units
-----------------------
======================  ======  =========================================
temperature             degC     ``tmax``, ``tmin``, ``tmean``, ``tdew``
relative humidity       %        ``rh`` (0-100)
wind speed              m/s      ``u`` (adjusted to 2 m by :func:`wind_speed_2m`)
solar radiation         W/m^2    daily-mean (:func:`rs_from_wm2` -> MJ/m^2/day)
elevation               m        station height (for air pressure -> gamma)
latitude                degrees  north positive, south negative
day of year             1-366    (:func:`day_of_year`)
======================  ======  =========================================

Solar is the feed's confirmed unit (W/m^2, daily-mean); :func:`rs_from_wm2`
converts it to the MJ/m^2/day the radiation balance needs.

References: Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998), FAO-56;
Hargreaves & Samani (1985); Penman (1948).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "GSC", "SIGMA", "ALBEDO_GRASS", "ALBEDO_WATER", "WM2_TO_MJ_DAY",
    "svp", "svp_slope", "mean_svp",
    "atm_pressure", "psychrometric_constant",
    "ea_from_tdew", "ea_from_rhmean", "ea_from_rhminmax",
    "day_of_year",
    "extraterrestrial_radiation", "daylight_hours", "clear_sky_radiation",
    "rs_from_wm2", "net_shortwave", "net_longwave", "net_radiation",
    "wind_speed_2m",
    "penman_monteith_combine", "et0_penman_monteith",
    "et0_hargreaves", "e0_open_water",
    "reference_et0",
]

#: Solar constant (MJ m^-2 min^-1), FAO-56 Eq. 28.
GSC = 0.0820
#: Stefan-Boltzmann (MJ K^-4 m^-2 day^-1), FAO-56 Eq. 39.
SIGMA = 4.903e-9
#: Reference-grass and open-water albedos.
ALBEDO_GRASS = 0.23
ALBEDO_WATER = 0.08
#: Daily-mean W/m^2 -> MJ/m^2/day (86400 s / 1e6).
WM2_TO_MJ_DAY = 0.0864


# ---------------------------------------------------------------------------
# Vapour pressure and its slope
# ---------------------------------------------------------------------------

def svp(t):
    """Saturation vapour pressure e0(T) [kPa] at temperature ``t`` [degC].

    FAO-56 Eq. 11.  ``svp(25) == 3.168``, ``svp(18) == 2.064``.
    """
    t = np.asarray(t, dtype=float)
    return 0.6108 * np.exp(17.27 * t / (t + 237.3))


def svp_slope(t):
    """Slope of the SVP curve Delta [kPa/degC] at ``t`` [degC].  FAO-56 Eq. 13.

    ``svp_slope(16.9) == 0.122``.
    """
    t = np.asarray(t, dtype=float)
    return 4098.0 * svp(t) / (t + 237.3) ** 2


def mean_svp(tmax, tmin):
    """Saturation vapour pressure es [kPa] = mean of e0(Tmax), e0(Tmin).

    FAO-56 Eq. 12 -- the mean of the two, **not** e0 of the mean temperature,
    which would bias es low.  ``mean_svp(25, 18) == 2.616``.
    """
    return (svp(tmax) + svp(tmin)) / 2.0


# ---------------------------------------------------------------------------
# Pressure and the psychrometric constant
# ---------------------------------------------------------------------------

def atm_pressure(elevation):
    """Atmospheric pressure P [kPa] at ``elevation`` [m].  FAO-56 Eq. 7.

    ``atm_pressure(100) == 100.1``; ``atm_pressure(1800) == 81.8``.
    """
    z = np.asarray(elevation, dtype=float)
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26


def psychrometric_constant(elevation=None, pressure=None):
    """Psychrometric constant gamma [kPa/degC].  FAO-56 Eq. 8.

    Give either ``elevation`` (m) or ``pressure`` (kPa).  At 100 m,
    gamma == 0.0666.
    """
    if pressure is None:
        if elevation is None:
            raise ValueError("give elevation or pressure")
        pressure = atm_pressure(elevation)
    return 0.665e-3 * np.asarray(pressure, dtype=float)


# ---------------------------------------------------------------------------
# Actual vapour pressure ea (three routes, best data first)
# ---------------------------------------------------------------------------

def ea_from_tdew(tdew):
    """Actual vapour pressure ea [kPa] from dew point [degC].  FAO-56 Eq. 14.

    The most reliable route when a dew point (or a humidity that converts to
    one) is measured.
    """
    return svp(tdew)


def ea_from_rhmean(rhmean, tmax, tmin):
    """ea [kPa] from mean relative humidity [%].  FAO-56 Eq. 19.

    ``ea = RHmean/100 * (e0(Tmax) + e0(Tmin)) / 2``.  The route used for the
    eThekwini feed, which reports a mean humidity (``relativeHumidity``).
    """
    rhmean = np.asarray(rhmean, dtype=float)
    return rhmean / 100.0 * mean_svp(tmax, tmin)


def ea_from_rhminmax(rhmax, rhmin, tmax, tmin):
    """ea [kPa] from RHmax and RHmin [%].  FAO-56 Eq. 17 (most accurate)."""
    rhmax = np.asarray(rhmax, dtype=float)
    rhmin = np.asarray(rhmin, dtype=float)
    return (svp(tmin) * rhmax / 100.0 + svp(tmax) * rhmin / 100.0) / 2.0


# ---------------------------------------------------------------------------
# Solar geometry and radiation
# ---------------------------------------------------------------------------

def day_of_year(date):
    """Day of year (1-366) from a date, datetime, or array of them."""
    return pd.DatetimeIndex(pd.to_datetime(np.atleast_1d(date))).dayofyear.to_numpy()


def _sun_geometry(latitude, doy):
    """Return ``(phi, dr, decl, ws)`` for FAO-56 Eqs. 22-25."""
    phi = np.radians(np.asarray(latitude, dtype=float))
    j = np.asarray(doy, dtype=float)
    dr = 1.0 + 0.033 * np.cos(2 * np.pi * j / 365.0)          # Eq. 23
    decl = 0.409 * np.sin(2 * np.pi * j / 365.0 - 1.39)       # Eq. 24
    # clip guards the poles/solstice where |tan phi tan decl| can exceed 1
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(decl), -1.0, 1.0))   # Eq. 25
    return phi, dr, decl, ws


def extraterrestrial_radiation(latitude, doy):
    """Extraterrestrial radiation Ra [MJ/m^2/day].  FAO-56 Eq. 21.

    Depends only on latitude and day of year -- pure solar geometry, no
    measurement.  ``Ra(-20 deg, 246) == 32.2``; ``Ra(50.8 deg, 187) == 41.1``.
    """
    phi, dr, decl, ws = _sun_geometry(latitude, doy)
    ra = (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(decl)
        + np.cos(phi) * np.cos(decl) * np.sin(ws))
    return ra


def daylight_hours(latitude, doy):
    """Daylight hours N.  FAO-56 Eq. 34.  ``N(-20 deg, 246) == 11.7``."""
    _, _, _, ws = _sun_geometry(latitude, doy)
    return 24.0 / np.pi * ws


def clear_sky_radiation(ra, elevation):
    """Clear-sky shortwave Rso [MJ/m^2/day].  FAO-56 Eq. 37."""
    z = np.asarray(elevation, dtype=float)
    return (0.75 + 2e-5 * z) * np.asarray(ra, dtype=float)


def rs_from_wm2(mean_wm2):
    """Daily-mean solar irradiance [W/m^2] -> Rs [MJ/m^2/day].

    ``Rs = mean_wm2 * 86400 / 1e6``.  The feed's confirmed unit is W/m^2, so a
    day's mean of, say, 255.6 W/m^2 is 22.1 MJ/m^2/day.
    """
    return np.asarray(mean_wm2, dtype=float) * WM2_TO_MJ_DAY


def net_shortwave(rs, albedo=ALBEDO_GRASS):
    """Net shortwave Rns [MJ/m^2/day].  FAO-56 Eq. 38."""
    return (1.0 - albedo) * np.asarray(rs, dtype=float)


def net_longwave(tmax, tmin, ea, rs, rso):
    """Net longwave Rnl [MJ/m^2/day] (outgoing positive).  FAO-56 Eq. 39.

    The ``Rs/Rso`` cloudiness factor is clipped to <= 1: a sensor reading
    slightly above the clear-sky estimate must not flip the term negative.
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    ea = np.asarray(ea, dtype=float)
    tmax_k4 = (tmax + 273.16) ** 4
    tmin_k4 = (tmin + 273.16) ** 4
    cloud = 1.35 * np.clip(np.asarray(rs, dtype=float) / rso, None, 1.0) - 0.35
    return (SIGMA * (tmax_k4 + tmin_k4) / 2.0
            * (0.34 - 0.14 * np.sqrt(np.clip(ea, 0, None))) * cloud)


def net_radiation(rs, tmax, tmin, ea, latitude, elevation, doy,
                  albedo=ALBEDO_GRASS):
    """Net radiation Rn [MJ/m^2/day].  FAO-56 Eq. 40 (Rn = Rns - Rnl).

    With the FAO-56 example inputs (Rs=22.07, Tmax=21.5, Tmin=12.3, ea=1.409,
    lat=50.8, z=100, doy=187) this returns 13.28.
    """
    ra = extraterrestrial_radiation(latitude, doy)
    rso = clear_sky_radiation(ra, elevation)
    rns = net_shortwave(rs, albedo)
    rnl = net_longwave(tmax, tmin, ea, rs, rso)
    return rns - rnl


# ---------------------------------------------------------------------------
# Wind
# ---------------------------------------------------------------------------

def wind_speed_2m(u, measured_height=2.0):
    """Wind speed adjusted to 2 m [m/s].  FAO-56 Eq. 47.

    ``measured_height`` is the anemometer height (m).  The eThekwini height is
    not documented in the feed; leave the default 2.0 (no adjustment) until it
    is confirmed, and set it (often 10.0 for an AWS) once known -- a 10 m speed
    is ~1.2x the 2 m speed, a real bias on the aerodynamic term.
    """
    u = np.asarray(u, dtype=float)
    h = float(measured_height)
    if h == 2.0:
        return u
    return u * 4.87 / np.log(67.8 * h - 5.42)


# ---------------------------------------------------------------------------
# Penman-Monteith reference ET
# ---------------------------------------------------------------------------

def penman_monteith_combine(delta, rn, g, gamma, tmean, u2, es_minus_ea):
    """The FAO-56 Eq. 6 combination, given its assembled terms.

    Split out so the arithmetic can be checked directly against the published
    worked example: ``delta=0.122, rn=13.28, g=0, gamma=0.0666, tmean=16.9,
    u2=2.078, es_minus_ea=0.589`` -> ``3.88`` mm/day.
    """
    delta = np.asarray(delta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    u2 = np.asarray(u2, dtype=float)
    num = (0.408 * delta * (np.asarray(rn, dtype=float) - g)
           + gamma * (900.0 / (np.asarray(tmean, dtype=float) + 273.0))
           * u2 * np.asarray(es_minus_ea, dtype=float))
    den = delta + gamma * (1.0 + 0.34 * u2)
    return num / den


def et0_penman_monteith(tmax, tmin, rs, u2, ea, elevation, latitude, doy,
                        tmean=None, g=0.0):
    """FAO-56 reference-crop ET0 [mm/day] from daily met.  FAO-56 Eq. 6.

    ``rs`` is in MJ/m^2/day (use :func:`rs_from_wm2` on the W/m^2 feed); ``u2``
    at 2 m (use :func:`wind_speed_2m`); ``ea`` from :func:`ea_from_rhmean` or
    :func:`ea_from_tdew`.  ``tmean`` defaults to (Tmax+Tmin)/2, the FAO-56
    convention.  ``g`` (soil heat flux) is ~0 for a daily step.

    With the FAO-56 worked-example inputs this returns 3.9 mm/day.
    """
    if tmean is None:
        tmean = (np.asarray(tmax, dtype=float) + np.asarray(tmin, dtype=float)) / 2.0
    delta = svp_slope(tmean)
    gamma = psychrometric_constant(elevation=elevation)
    es = mean_svp(tmax, tmin)
    rn = net_radiation(rs, tmax, tmin, ea, latitude, elevation, doy,
                       albedo=ALBEDO_GRASS)
    return penman_monteith_combine(delta, rn, g, gamma, tmean, u2, es - ea)


def et0_hargreaves(tmax, tmin, doy, latitude, tmean=None):
    """Hargreaves-Samani ET0 [mm/day] -- the temperature-only fallback.

    Hargreaves & Samani (1985): ``ET0 = 0.0023 (Tmean+17.8) sqrt(Tmax-Tmin)
    Ra_mm``, with Ra converted from MJ/m^2/day to mm/day by 0.408.  Needs only
    temperature plus solar geometry (Ra), so it runs on a station with no
    humidity, wind, or radiation.
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    if tmean is None:
        tmean = (tmax + tmin) / 2.0
    ra_mm = 0.408 * extraterrestrial_radiation(latitude, doy)
    dtr = np.clip(tmax - tmin, 0.0, None)      # guard a bad Tmax < Tmin day
    return 0.0023 * (tmean + 17.8) * np.sqrt(dtr) * ra_mm


# ---------------------------------------------------------------------------
# Open-water evaporation (ETo)
# ---------------------------------------------------------------------------

def e0_open_water(tmax, tmin, rs, u2, ea, elevation, latitude, doy, tmean=None):
    """Penman (1948) open-water evaporation E0 [mm/day] -- the channel's ``ETo``.

    Same energy + drying-power split as Penman-Monteith but for a free water
    surface: open-water albedo (0.08, so a little more net radiation than
    grass) and Penman's classic wind function in the aerodynamic term,
    ``Ea = 0.26 (1 + 0.54 u2)(es - ea)`` [mm/day].  Combined as
    ``E0 = (Delta*Rn_mm + gamma*Ea) / (Delta + gamma)``, with
    ``Rn_mm = 0.408*Rn``.

    Open water evaporates a little faster than grass, so E0 > ET0 under the same
    weather -- a useful sanity check.  When radiation is unavailable, the design
    note's fallback is ``ETo = ETr`` (channel evaporation is a sub-percent term),
    handled by the caller, not here.
    """
    tmax = np.asarray(tmax, dtype=float)
    tmin = np.asarray(tmin, dtype=float)
    if tmean is None:
        tmean = (tmax + tmin) / 2.0
    delta = svp_slope(tmean)
    gamma = psychrometric_constant(elevation=elevation)
    es = mean_svp(tmax, tmin)
    rn = net_radiation(rs, tmax, tmin, ea, latitude, elevation, doy,
                       albedo=ALBEDO_WATER)
    rn_mm = 0.408 * rn
    ea_term = 0.26 * (1.0 + 0.54 * np.asarray(u2, dtype=float)) * (es - ea)
    return (delta * rn_mm + gamma * ea_term) / (delta + gamma)


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def reference_et0(row, latitude, measured_wind_height=2.0):
    """ET0 [mm/day] for one daily record, choosing the method from the data.

    ``row`` is a mapping/Series with the :mod:`met` daily-table fields
    (``tmax, tmin, tmean, date`` always; ``solar, wind, rh, tdew`` where the
    station has them).  Full Penman-Monteith is used when radiation **and** a
    humidity route (``rh`` or ``tdew``) are present; otherwise Hargreaves.

    Returns ``(et0, method)`` where method is ``"penman_monteith"`` or
    ``"hargreaves"``.  ``et0`` is NaN when even Tmax/Tmin are missing.
    """
    tmax, tmin = row.get("tmax"), row.get("tmin")
    tmean = row.get("tmean")
    doy = day_of_year(row["date"])[0]

    if pd.isna(tmax) or pd.isna(tmin):
        return float("nan"), "none"

    solar, wind = row.get("solar"), row.get("wind")
    rh, tdew = row.get("rh"), row.get("tdew")
    have_humidity = not pd.isna(rh) or not pd.isna(tdew)
    have_energy = not pd.isna(solar) and not pd.isna(wind)

    if have_energy and have_humidity:
        ea = (ea_from_tdew(tdew) if not pd.isna(tdew)
              else ea_from_rhmean(rh, tmax, tmin))
        rs = rs_from_wm2(solar)
        u2 = wind_speed_2m(wind, measured_wind_height)
        et0 = et0_penman_monteith(tmax, tmin, rs, u2, ea,
                                  row["elevation_m"], latitude, doy, tmean=tmean)
        return float(et0), "penman_monteith"

    et0 = et0_hargreaves(tmax, tmin, doy, latitude, tmean=tmean)
    return float(et0), "hargreaves"


# ---------------------------------------------------------------------------
# CLI -- daily ET0/E0 per station from a met daily table
# ---------------------------------------------------------------------------

def et0_eto_frame(daily, manifest, *, wind_height=2.0, method="auto"):
    """Compute daily ``et0`` (ETr) and ``eto`` (E0 open water) per station-day.

    ``daily`` is the :func:`met.daily_table` output (a DataFrame or its CSV path);
    ``manifest`` is the weather manifest (DataFrame or path) -- it supplies each
    station's ``elevation_m`` and, via its coordinates, latitude.  ``eto`` is the
    open-water Penman evaporation where the full energy balance is available and
    falls back to ``eto = et0`` otherwise (the design-note channel-evaporation
    shortcut).  Returns a frame ``station_id, date, et0, eto, method``.
    """
    from pyproj import Transformer
    from . import met as _met

    if not isinstance(daily, pd.DataFrame):
        daily = pd.read_csv(daily, parse_dates=["date"])
    daily = daily.copy()
    daily["station_id"] = daily["station_id"].astype(str).str.strip()

    man = manifest if isinstance(manifest, pd.DataFrame) else _met.read_manifest(manifest)
    # latitude (degrees) per station: reproject the model-CRS coordinates back
    # to lon/lat, so it is correct whatever CRS the manifest declared.
    tx = Transformer.from_crs(str(man["crs"].iloc[0]), "EPSG:4326", always_xy=True)
    _, lat = tx.transform(man["x"].to_numpy(float), man["y"].to_numpy(float))
    lat_of = dict(zip(man.index, lat))
    elev_of = man["elevation_m"].to_dict()

    rows = []
    for rec in daily.to_dict("records"):
        sid = rec["station_id"]
        if sid not in lat_of:
            continue
        rec["elevation_m"] = elev_of.get(sid, np.nan)
        latitude = lat_of[sid]

        forced_harg = method == "hargreaves"
        et0, used = reference_et0(rec, latitude, measured_wind_height=wind_height)
        if forced_harg and not pd.isna(rec.get("tmax")) and not pd.isna(rec.get("tmin")):
            doy = day_of_year(rec["date"])[0]
            et0 = float(et0_hargreaves(rec["tmax"], rec["tmin"], doy, latitude,
                                       tmean=rec.get("tmean")))
            used = "hargreaves"

        # open-water ETo: full Penman where energy + humidity exist, else = ETr
        eto = et0
        if used == "penman_monteith":
            ea = (ea_from_tdew(rec["tdew"]) if not pd.isna(rec.get("tdew"))
                  else ea_from_rhmean(rec["rh"], rec["tmax"], rec["tmin"]))
            eto = float(e0_open_water(
                rec["tmax"], rec["tmin"], rs_from_wm2(rec["solar"]),
                wind_speed_2m(rec["wind"], wind_height), ea,
                rec["elevation_m"], latitude, day_of_year(rec["date"])[0],
                tmean=rec.get("tmean")))

        rows.append({"station_id": sid, "date": rec["date"],
                     "et0": et0, "eto": eto, "method": used})

    return pd.DataFrame(rows, columns=["station_id", "date", "et0", "eto", "method"])


def _cli(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Daily reference ET0 (ETr) and open-water E0 (ETo) per "
                    "station from a `met` daily table.",
    )
    p.add_argument("--daily", required=True,
                   help="weather_daily.csv from `met` (station_id, date, tmax, …)")
    p.add_argument("--manifest", required=True,
                   help="weather_manifest.csv (elevation_m + coordinates)")
    p.add_argument("--out", default=None, help="write per-station-day ET0/ETo CSV")
    p.add_argument("--wind-height", type=float, default=2.0,
                   help="anemometer height (m) for the 2 m adjustment "
                        "(default 2.0; set 10 for a typical AWS once confirmed)")
    p.add_argument("--method", choices=["auto", "hargreaves"], default="auto",
                   help="'auto' uses Penman-Monteith where radiation+humidity "
                        "allow, else Hargreaves; 'hargreaves' forces the "
                        "temperature-only method everywhere")
    args = p.parse_args(argv)

    out = et0_eto_frame(args.daily, args.manifest,
                        wind_height=args.wind_height, method=args.method)
    good = out[out["et0"].notna()]

    print("=== method used (station-days) ===")
    print(out["method"].value_counts().to_string())
    print("\n=== ET0 (mm/day) per station ===")
    summ = (good.groupby("station_id")["et0"]
            .agg(["count", "min", "mean", "max"]).round(2))
    print(summ.to_string())
    print("\n=== head ===")
    print(out.head(10).to_string(index=False))

    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\n{len(out)} station-days -> {args.out}")


if __name__ == "__main__":
    _cli()
