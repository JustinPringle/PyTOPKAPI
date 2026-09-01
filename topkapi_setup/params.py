"""Parameter-raster generation for PyTOPKAPI (toolkit stage 4.2, milestone M2).

``terrain.py`` (M1) produces five of the twelve rasters that
``pytopkapi.parameter_utils.create_file.generate_param_file`` reads: the DEM
(the user's input), ``mask``, ``slope`` (its ``hillslope``), ``network`` and
``flowdir``. This module produces the remaining **seven**, every one snapped to
the terrain ``mask`` grid:

===========================  ======================  ==================  ============
create_file ``.ini`` key      cell_param column       physical quantity   this module
===========================  ======================  ==================  ============
``soil_depth_fname``          8  (``L``)               soil depth (m)      soil table
``conductivity_fname``        9  (``Ks``)              sat. K (mm/s)       soil table
``resid_moisture_..._fname``  10 (``theta_r``)         residual moisture   soil table
``sat_moisture_..._fname``    11 (``theta_s``)         sat. moisture       soil table
``overland_manning_fname``    12 (``n_o``)             overland Manning    land cover
``bubbling_pressure_fname``   19 (``psi_b``)           bubbling head (mm)  soil table
``pore_size_dist_fname``      20 (``lambda``)          pore-size index     soil table
===========================  ======================  ==================  ============

Design mirrors ``terrain.py``: pure functions with a thin CLI, every output
snapped to ``mask.tif`` (bilinear for continuous inputs, nearest for classes),
range-validated before write, with a ``params_manifest.json`` that ``config.py``
(M4) will consume.

Two default lookup tables let a catchment run *before* site-specific data exist
(section 4.2 of the project instructions):

* :data:`RAWLS_BROOKS_COREY` -- Brooks-Corey hydraulics by USDA texture class
  (Rawls, Brakensiek & Saxton 1982; Rawls & Brakensiek 1985). Values are stored
  in literature units (``psi_b`` in **metres**); :func:`brooks_corey_from_texture`
  converts ``psi_b`` to **millimetres** on write, because the solver's Green-Ampt
  path (``model.py``: ``psi = psi_b / eff_sat**(1/lambda)``) works in mm-depth.
  The shipped reference ``cell_param.dat`` carries ``psi_b`` ~332 (mm), which is
  how this unit was pinned down.
* :data:`SANLC_N_O` -- overland Manning ``n_o`` by SANLC 2020 class group.

The soil source is pluggable, exactly as the DEM was for terrain: pass a
soil-form (or texture-class) raster and a crosswalk, **or** a single
``--uniform-texture`` for a first pass. Likewise land cover.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from .terrain import MASK_IN, read_raster, write_raster

# ---------------------------------------------------------------------------
# Default lookup tables (approved: Rawls texture classes + SANLC groups)
# ---------------------------------------------------------------------------

#: Brooks-Corey hydraulic parameters by USDA texture class.
#: ``psi_b`` in **metres** (converted to mm on write); ``Ks`` in **mm/s**.
#: Sources: Rawls, Brakensiek & Saxton (1982); Rawls & Brakensiek (1985).
RAWLS_BROOKS_COREY: dict[str, dict[str, float]] = {
    "sand":       {"theta_r": 0.020, "theta_s": 0.417, "psi_b_m": 0.073,
                   "lambda_pore": 0.592, "Ks_mm_s": 5.83e-2},
    "sandy_loam": {"theta_r": 0.041, "theta_s": 0.453, "psi_b_m": 0.147,
                   "lambda_pore": 0.322, "Ks_mm_s": 1.20e-2},
    "loam":       {"theta_r": 0.027, "theta_s": 0.463, "psi_b_m": 0.112,
                   "lambda_pore": 0.220, "Ks_mm_s": 3.67e-3},
    "clay_loam":  {"theta_r": 0.075, "theta_s": 0.464, "psi_b_m": 0.259,
                   "lambda_pore": 0.194, "Ks_mm_s": 6.39e-4},
    "clay":       {"theta_r": 0.090, "theta_s": 0.475, "psi_b_m": 0.373,
                   "lambda_pore": 0.165, "Ks_mm_s": 1.67e-4},
}

#: Default soil depth ``L`` (m) by texture class. Soil depth is *not* a
#: Brooks-Corey output; these are pragmatic defaults and are the weakest part of
#: the soil table -- override with a measured depth raster or ``--soil-depth``
#: once SA Land Type / SoilGrids depth-to-restriction data are on hand.
DEFAULT_SOIL_DEPTH_M: dict[str, float] = {
    "sand": 0.8, "sandy_loam": 1.0, "loam": 1.2,
    "clay_loam": 1.0, "clay": 0.8,
}

#: Overland Manning ``n_o`` by SANLC 2020 class *group* (Chow-type roughness).
SANLC_N_O: dict[str, float] = {
    "water_wetland":  0.030,
    "bare_eroded":    0.050,
    "cultivated":     0.100,
    "grassland":      0.150,
    "bush_thicket":   0.250,
    "forest":         0.400,
    "built_up":       0.015,
}

#: Default SANLC 2020 raster-code -> group crosswalk. SANLC 2020 has ~73 classes;
#: this maps the standard top-level groupings and is meant to be *edited* to the
#: exact codes of the tile in hand. Unmapped codes fall back to ``grassland``
#: with a warning. (Codes below follow the common SANLC 2020 legend ordering;
#: confirm against the specific product/version delivered.)
DEFAULT_SANLC_CROSSWALK: dict[int, str] = {
    1: "forest", 2: "forest", 3: "forest",            # indigenous / plantation forest
    4: "bush_thicket", 5: "bush_thicket",             # woodland / thicket
    6: "grassland",                                   # grassland
    7: "water_wetland", 8: "water_wetland",           # water bodies / wetlands
    9: "bare_eroded", 10: "bare_eroded",              # bare / eroded ground
    11: "cultivated", 12: "cultivated",               # commercial / subsistence crops
    13: "built_up", 14: "built_up", 15: "built_up",   # urban / built / hardened
}

#: Default SA Land Type soil-form -> USDA texture crosswalk. The texture side is
#: citable literature; *this* mapping is the local piece and should be tuned to
#: the catchment's dominant soil forms. Keyed on an integer soil-form code in
#: the soil raster; unmapped codes fall back to ``loam`` with a warning.
DEFAULT_SOILFORM_TEXTURE: dict[int, str] = {
    1: "sand", 2: "sandy_loam", 3: "loam", 4: "clay_loam", 5: "clay",
}

# Physically plausible ranges, used by :func:`validate_ranges`.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "soil_depth":  (0.1, 5.0),      # m
    "Ks":          (1e-6, 1.0),     # mm/s
    "theta_r":     (0.0, 0.20),
    "theta_s":     (0.30, 0.55),
    "psi_b":       (10.0, 2000.0),  # mm
    "lambda_pore": (0.05, 1.0),
    "n_o":         (0.01, 0.6),
}

# The seven raster keys this module emits, in create_file order.
RASTER_KEYS = ("soil_depth", "conductivity", "resid_moisture_content",
               "sat_moisture_content", "overland_manning",
               "bubbling_pressure", "pore_size_dist")

NODATA = np.float32(-9999.0)


# ---------------------------------------------------------------------------
# Grid definition (everything snaps to the terrain mask)
# ---------------------------------------------------------------------------

@dataclass
class GridSpec:
    """The model grid, taken from ``mask.tif`` so params align with terrain."""
    shape: tuple[int, int]
    transform: object
    crs: object
    mask: np.ndarray            # bool, True inside the catchment


def grid_from_mask(mask_path: str) -> GridSpec:
    """Read the terrain mask and return the grid every param raster snaps to."""
    arr, transform, crs, _ = read_raster(mask_path)
    return GridSpec(shape=arr.shape, transform=transform, crs=crs,
                    mask=(arr == MASK_IN))


def resample_to_grid(src_path: str, grid: GridSpec, *, continuous: bool) -> np.ndarray:
    """Reproject/resample a source raster onto ``grid``.

    ``continuous=True`` uses bilinear (elevation-like fields); ``False`` uses
    nearest, correct for integer class rasters (soil form, land cover).
    """
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_crs = src.crs
        src_transform = src.transform
        src_nodata = src.nodata
    dtype = "float32" if continuous else "int32"
    dst = np.zeros(grid.shape, dtype=dtype)
    reproject(
        source=src_arr.astype(dtype),
        destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=grid.transform, dst_crs=grid.crs,
        src_nodata=src_nodata,
        resampling=Resampling.bilinear if continuous else Resampling.nearest,
    )
    return dst


# ---------------------------------------------------------------------------
# Class -> property mapping
# ---------------------------------------------------------------------------

def _map_classes(class_arr, lookup, default, what):
    """Map an integer class raster to floats via ``lookup``; warn on fallback."""
    out = np.full(class_arr.shape, np.nan, dtype="float32")
    seen_unmapped = set()
    for code in np.unique(class_arr):
        code = int(code)
        val = lookup.get(code)
        if val is None:
            seen_unmapped.add(code)
            val = default
        out[class_arr == code] = val
    if seen_unmapped:
        warnings.warn(
            f"{what}: codes {sorted(seen_unmapped)} not in crosswalk; "
            f"fell back to default ({default}). Edit the crosswalk for this tile.",
            stacklevel=2,
        )
    return out


def brooks_corey_from_texture(texture_code_arr, table=RAWLS_BROOKS_COREY,
                              crosswalk=DEFAULT_SOILFORM_TEXTURE,
                              depth_table=DEFAULT_SOIL_DEPTH_M):
    """Return the six soil rasters from an integer soil-form/texture raster.

    Returns a dict with ``soil_depth`` (m), ``conductivity`` (mm/s),
    ``resid_moisture_content``, ``sat_moisture_content``,
    ``bubbling_pressure`` (**mm**), ``pore_size_dist``.
    """
    # code -> texture name (via crosswalk); codes already naming a texture pass through
    texture_of = {}
    for code in np.unique(texture_code_arr):
        code = int(code)
        texture_of[code] = crosswalk.get(code, "loam")

    def field(param_key, scale=1.0):
        lut = {c: table[t][param_key] * scale for c, t in texture_of.items()}
        return _map_classes(texture_code_arr, lut, table["loam"][param_key] * scale,
                            f"soil form ({param_key})")

    depth_lut = {c: depth_table[t] for c, t in texture_of.items()}
    return {
        "soil_depth": _map_classes(texture_code_arr, depth_lut,
                                   depth_table["loam"], "soil form (depth)"),
        "conductivity":            field("Ks_mm_s"),
        "resid_moisture_content":  field("theta_r"),
        "sat_moisture_content":    field("theta_s"),
        "bubbling_pressure":       field("psi_b_m", scale=1000.0),   # m -> mm
        "pore_size_dist":          field("lambda_pore"),
    }


def manning_from_landcover(lc_code_arr, groups=SANLC_N_O,
                           crosswalk=DEFAULT_SANLC_CROSSWALK):
    """Return the overland Manning ``n_o`` raster from a SANLC class raster."""
    lut = {code: groups[group] for code, group in crosswalk.items()}
    return _map_classes(lc_code_arr, lut, groups["grassland"], "SANLC")


# ---------------------------------------------------------------------------
# HWSD soil-depth helper
# ---------------------------------------------------------------------------

def _read_smu_depth_csv(csv_path, units="cm"):
    """Read an SMU -> depth lookup CSV into ``{code: depth_m}``.

    Two columns, header row: the first is the SMU code, the second the depth.
    ``units`` converts the depth column to metres (``cm`` -> /100, ``m`` -> as-is).
    """
    import csv as _csv
    scale = 0.01 if units == "cm" else 1.0
    out = {}
    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        next(reader, None)                       # skip header
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            out[int(float(row[0]))] = float(row[1]) * scale
    return out


def depth_from_smu(smu_raster_path, smu_to_depth_m, out_path,
                   default_depth_m=None):
    """Reclass an HWSD SMU-code raster into a soil-depth raster (metres).

    HWSD v2.0 is an SMU-code raster linked to an attribute database; the rootable
    soil depth lives in that table, not the raster. Export an ``SMU, depth``
    lookup once from the HWSD ``.mdb`` (HWSD viewer or ``mdbtools``), then this
    turns the SMU raster into a depth raster in a single call. Feed the result to
    :func:`build_params` via ``soil_depth_path`` / ``--soil-depth``; it is
    resampled onto the model grid there, so this output keeps HWSD's own CRS.

    Parameters
    ----------
    smu_raster_path : path to the HWSD SMU-code GeoTIFF.
    smu_to_depth_m : ``{smu_code: depth_m}`` dict, or a path to a ``SMU, depth``
        CSV (assumed centimetres; see :func:`_read_smu_depth_csv`).
    out_path : where to write the depth GeoTIFF (float32, metres).
    default_depth_m : depth for SMU codes not in the lookup. ``None`` writes
        nodata there (and warns), so gaps surface rather than hide.
    """
    if isinstance(smu_to_depth_m, (str, Path)):
        smu_to_depth_m = _read_smu_depth_csv(smu_to_depth_m)

    smu, transform, crs, _ = read_raster(str(smu_raster_path))
    smu = np.asarray(smu)
    depth = np.full(smu.shape, NODATA, dtype="float32")
    missing = set()
    for code in np.unique(smu):
        code = int(code)
        val = smu_to_depth_m.get(code, default_depth_m)
        if val is None:
            missing.add(code)
            continue
        depth[smu == code] = val
    if missing:
        warnings.warn(
            f"depth_from_smu: {len(missing)} SMU code(s) not in the lookup "
            f"(e.g. {sorted(missing)[:5]}); left as nodata. Extend the SMU->depth "
            f"table or pass default_depth_m.", stacklevel=2)
    return write_raster(str(out_path), depth, transform, crs,
                        nodata=float(NODATA), dtype="float32")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ranges(rasters: dict, mask: np.ndarray) -> list[str]:
    """Check in-mask values against :data:`PARAM_RANGES`. Returns problem strings.

    NaNs inside the mask are always a problem (they become garbage in
    ``cell_param``); out-of-range values are flagged so implausible defaults are
    caught before the solver sees them.
    """
    # map raster key -> range key
    range_of = {
        "soil_depth": "soil_depth", "conductivity": "Ks",
        "resid_moisture_content": "theta_r", "sat_moisture_content": "theta_s",
        "overland_manning": "n_o", "bubbling_pressure": "psi_b",
        "pore_size_dist": "lambda_pore",
    }
    problems = []
    for key, arr in rasters.items():
        inside = arr[mask]
        if np.isnan(inside).any():
            problems.append(f"{key}: {int(np.isnan(inside).sum())} NaN cell(s) inside mask")
        lo, hi = PARAM_RANGES[range_of[key]]
        finite = inside[np.isfinite(inside)]
        if finite.size and (finite.min() < lo or finite.max() > hi):
            problems.append(
                f"{key}: values [{finite.min():.4g}, {finite.max():.4g}] "
                f"outside plausible [{lo:g}, {hi:g}]")
    return problems


# ---------------------------------------------------------------------------
# Result + orchestrator
# ---------------------------------------------------------------------------

@dataclass
class ParamResult:
    """Paths and provenance emitted by :func:`build_params`."""
    soil_depth: str
    conductivity: str
    resid_moisture_content: str
    sat_moisture_content: str
    overland_manning: str
    bubbling_pressure: str
    pore_size_dist: str
    crs: str
    n_cells: int
    soil_source: str
    landcover_source: str
    depth_source: str = "texture-default"
    warnings: list = field(default_factory=list)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def _uniform_class_raster(grid: GridSpec, code: int) -> np.ndarray:
    return np.full(grid.shape, code, dtype="int32")


def build_params(
    mask_path: str,
    out_dir: str,
    *,
    soil_form_path: str | None = None,
    landcover_path: str | None = None,
    soil_depth_path: str | None = None,
    uniform_texture: str | None = None,
    uniform_landcover: str | None = None,
    validate: bool = True,
) -> ParamResult:
    """Build the seven parameter rasters, snapped to ``mask_path``.

    Provide either a raster or a ``uniform_*`` fallback for each of soil and land
    cover. Uniform fills give a defensible first run before site data exist.

    ``soil_depth_path`` is an optional continuous depth raster **in metres** (e.g.
    from HWSD via :func:`depth_from_smu`, or SoilGrids). When given it overrides
    the per-texture default depth and is resampled bilinearly onto the grid.
    """
    grid = grid_from_mask(mask_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- soil source -> texture-code raster ---
    if soil_form_path:
        soil_codes = resample_to_grid(soil_form_path, grid, continuous=False)
        soil_source = f"raster:{soil_form_path}"
    elif uniform_texture:
        if uniform_texture not in RAWLS_BROOKS_COREY:
            raise ValueError(f"uniform_texture must be one of {list(RAWLS_BROOKS_COREY)}")
        # invert the crosswalk so the texture name maps back to a code
        code = next((c for c, t in DEFAULT_SOILFORM_TEXTURE.items()
                     if t == uniform_texture), None)
        if code is None:      # texture not in default crosswalk: synthesise a code
            code = max(DEFAULT_SOILFORM_TEXTURE) + 1
            DEFAULT_SOILFORM_TEXTURE[code] = uniform_texture
        soil_codes = _uniform_class_raster(grid, code)
        soil_source = f"uniform:{uniform_texture}"
    else:
        raise ValueError("Provide soil_form_path or uniform_texture")

    # --- land-cover source -> SANLC-code raster ---
    if landcover_path:
        lc_codes = resample_to_grid(landcover_path, grid, continuous=False)
        landcover_source = f"raster:{landcover_path}"
    elif uniform_landcover:
        if uniform_landcover not in SANLC_N_O:
            raise ValueError(f"uniform_landcover must be one of {list(SANLC_N_O)}")
        code = next((c for c, g in DEFAULT_SANLC_CROSSWALK.items()
                     if g == uniform_landcover), None)
        if code is None:
            code = max(DEFAULT_SANLC_CROSSWALK) + 1
            DEFAULT_SANLC_CROSSWALK[code] = uniform_landcover
        lc_codes = _uniform_class_raster(grid, code)
        landcover_source = f"uniform:{uniform_landcover}"
    else:
        raise ValueError("Provide landcover_path or uniform_landcover")

    # --- map to properties ---
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rasters = brooks_corey_from_texture(soil_codes)
        rasters["overland_manning"] = manning_from_landcover(lc_codes)
        warn_msgs = [str(w.message) for w in caught]

    # zero out (nodata) outside the mask so downstream reads are unambiguous
    for key in rasters:
        rasters[key] = np.where(grid.mask, rasters[key], np.nan).astype("float32")

    # optional measured soil-depth raster (HWSD/SoilGrids) overrides the default
    depth_source = f"texture-default ({soil_source})"
    if soil_depth_path:
        depth = resample_to_grid(soil_depth_path, grid, continuous=True)
        rasters["soil_depth"] = np.where(grid.mask, depth, np.nan).astype("float32")
        depth_source = f"raster:{soil_depth_path}"

    # --- validate before writing ---
    if validate:
        problems = validate_ranges(rasters, grid.mask)
        if problems:
            raise ValueError("Parameter rasters failed validation:\n  - "
                             + "\n  - ".join(problems))

    # --- write (nodata outside mask) ---
    paths = {}
    for key in RASTER_KEYS:
        arr = np.where(grid.mask, rasters[key], NODATA).astype("float32")
        paths[key] = write_raster(str(out / f"{key}.tif"), arr,
                                  grid.transform, grid.crs,
                                  nodata=float(NODATA), dtype="float32")

    crs_str = str(grid.crs)
    result = ParamResult(
        soil_depth=paths["soil_depth"], conductivity=paths["conductivity"],
        resid_moisture_content=paths["resid_moisture_content"],
        sat_moisture_content=paths["sat_moisture_content"],
        overland_manning=paths["overland_manning"],
        bubbling_pressure=paths["bubbling_pressure"],
        pore_size_dist=paths["pore_size_dist"],
        crs=crs_str, n_cells=int(grid.mask.sum()),
        soil_source=soil_source, landcover_source=landcover_source,
        depth_source=depth_source, warnings=warn_msgs,
    )
    result.to_json(out / "params_manifest.json")
    return result


def check_params(result: ParamResult, mask_path: str) -> dict:
    """Re-read the written rasters and assert they align with the mask.

    A cheap post-write guard: same shape/CRS as the mask, and no NaN/nodata
    inside the catchment. Mirrors ``terrain.check_terrain``.
    """
    grid = grid_from_mask(mask_path)
    for key in RASTER_KEYS:
        arr, _, crs, nodata = read_raster(getattr(result, key))
        if arr.shape != grid.shape:
            raise ValueError(f"{key}: shape {arr.shape} != mask {grid.shape}")
        inside = arr[grid.mask]
        bad = np.isnan(inside)
        if nodata is not None:
            bad = bad | (inside == nodata)
        if bad.any():
            raise ValueError(f"{key}: {int(bad.sum())} nodata/NaN cell(s) inside mask")
    return {"n_cells": int(grid.mask.sum()), "rasters": len(RASTER_KEYS)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(
        prog="python -m topkapi_setup.params",
        description="Generate the 7 soil/land-cover parameter rasters "
                    "generate_param_file needs, snapped to a terrain mask.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the 7 parameter rasters")
    b.add_argument("--mask", required=True,
                   help="mask.tif from terrain.py (defines the model grid)")
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--soil-form",
                   help="integer soil-form / texture-class raster")
    b.add_argument("--uniform-texture", choices=list(RAWLS_BROOKS_COREY),
                   help="fill the whole catchment with one texture (first-pass)")
    b.add_argument("--landcover", help="SANLC 2020 class raster")
    b.add_argument("--uniform-landcover", choices=list(SANLC_N_O),
                   help="fill the whole catchment with one land-cover group")
    b.add_argument("--soil-depth",
                   help="continuous soil-depth raster in METRES (HWSD/SoilGrids); "
                        "overrides the per-texture default depth")
    b.add_argument("--no-validate", action="store_true",
                   help="skip the pre-write range/NaN validation")

    h = sub.add_parser(
        "hwsd-depth",
        help="reclass an HWSD SMU-code raster into a soil-depth raster (metres)")
    h.add_argument("--smu", required=True, help="HWSD SMU-code GeoTIFF")
    h.add_argument("--table", required=True,
                   help="CSV lookup: 'SMU,depth' (header row)")
    h.add_argument("--units", choices=["cm", "m"], default="cm",
                   help="units of the depth column (default cm; HWSD is cm)")
    h.add_argument("--default-depth-m", type=float, default=None,
                   help="depth (m) for unmapped SMU codes (default: nodata)")
    h.add_argument("--out", required=True, help="output depth GeoTIFF")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    if args.cmd == "hwsd-depth":
        table = _read_smu_depth_csv(args.table, units=args.units)
        path = depth_from_smu(args.smu, table, args.out,
                              default_depth_m=args.default_depth_m)
        print(f"Wrote soil-depth raster (m) -> {path}")
        print("Feed it to `params build --soil-depth`.")
        return

    result = build_params(
        args.mask, args.out,
        soil_form_path=args.soil_form, landcover_path=args.landcover,
        soil_depth_path=args.soil_depth,
        uniform_texture=args.uniform_texture,
        uniform_landcover=args.uniform_landcover,
        validate=not args.no_validate,
    )
    info = check_params(result, args.mask)
    print(f"Wrote {info['rasters']} parameter rasters for {info['n_cells']} "
          f"cells -> {args.out}")
    print(f"  soil depth from: {result.depth_source}")
    if result.warnings:
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")
    print(f"Manifest: {Path(args.out) / 'params_manifest.json'}")


if __name__ == "__main__":
    main()
