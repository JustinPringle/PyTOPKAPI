"""Tests for topkapi_setup.params (M2).

Uses the synthetic valley DEM to make a real single-outlet mask, then drives
params.py against uniform and multi-class synthetic soil / land-cover rasters.
No data licence, deterministic, and it exercises the full snap-map-validate-write
path the same way a real catchment would.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio

from topkapi_setup import terrain as T
from topkapi_setup import params as P
from topkapi_setup.tests._synthetic import valley_dem, outlet_coord, write_dem


@pytest.fixture
def mask_path(tmp_path):
    """Build a terrain mask from the synthetic valley and return its path."""
    dem, transform, crs = valley_dem()
    dem_path = tmp_path / "dem.tif"
    write_dem(str(dem_path), dem, transform, crs)
    result = T.build_terrain(
        dem_path=str(dem_path),
        outlet_xy=outlet_coord(dem, transform),
        out_dir=str(tmp_path / "terrain"),
        a_thres_m2=5000.0, validate=False,
    )
    return result.mask


def _write_class_raster(path, mask_path, code_array):
    """Write an int class raster on the same grid as the mask."""
    with rasterio.open(mask_path) as src:
        profile = src.profile
    profile.update(dtype="int32", nodata=0, count=1)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(code_array.astype("int32"), 1)


def test_uniform_run_emits_seven_rasters(mask_path, tmp_path):
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="loam", uniform_landcover="grassland",
    )
    for key in P.RASTER_KEYS:
        assert rasterio.open(getattr(result, key))  # opens without error
    info = P.check_params(result, mask_path)
    assert info["rasters"] == 7
    assert info["n_cells"] == result.n_cells > 0


def test_psi_b_written_in_mm(mask_path, tmp_path):
    """Loam psi_b is 0.112 m in the table; the raster must be 112 mm."""
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="loam", uniform_landcover="grassland",
    )
    arr, _, _, _ = T.read_raster(result.bubbling_pressure)
    grid = P.grid_from_mask(mask_path)
    inside = arr[grid.mask]
    assert np.allclose(inside, 112.0, atol=1e-2)


def test_theta_and_ks_match_table(mask_path, tmp_path):
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="clay", uniform_landcover="forest",
    )
    grid = P.grid_from_mask(mask_path)
    ks, *_ = T.read_raster(result.conductivity)
    ts, *_ = T.read_raster(result.sat_moisture_content)
    no, *_ = T.read_raster(result.overland_manning)
    assert np.allclose(ks[grid.mask], P.RAWLS_BROOKS_COREY["clay"]["Ks_mm_s"])
    assert np.allclose(ts[grid.mask], P.RAWLS_BROOKS_COREY["clay"]["theta_s"])
    assert np.allclose(no[grid.mask], P.SANLC_N_O["forest"])


def test_nodata_outside_mask(mask_path, tmp_path):
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="sandy_loam", uniform_landcover="cultivated",
    )
    grid = P.grid_from_mask(mask_path)
    arr, _, _, nodata = T.read_raster(result.soil_depth)
    outside = arr[~grid.mask]
    assert np.all(outside == nodata)          # outside is nodata
    assert not np.isnan(arr[grid.mask]).any() # inside is clean


def test_multiclass_soil_maps_per_texture(mask_path, tmp_path):
    """A soil raster with two form codes must give two distinct Ks values."""
    grid = P.grid_from_mask(mask_path)
    # Split on the mask's own column midpoint so both textures fall inside it
    # (the synthetic catchment is asymmetric and does not span the full grid).
    _, cc = np.where(grid.mask)
    split = int(np.median(cc))
    col_idx = np.arange(grid.shape[1])[None, :]
    codes = np.where(col_idx <= split, 1, 5)   # 1 -> sand, 5 -> clay
    codes = np.broadcast_to(codes, grid.shape).copy()
    soil_path = tmp_path / "soilform.tif"
    _write_class_raster(str(soil_path), mask_path, codes)
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        soil_form_path=str(soil_path), uniform_landcover="grassland",
    )
    ks, *_ = T.read_raster(result.conductivity)
    vals = np.unique(np.round(ks[grid.mask], 8))
    vals = vals[vals != P.NODATA]
    assert P.RAWLS_BROOKS_COREY["sand"]["Ks_mm_s"] in vals
    assert P.RAWLS_BROOKS_COREY["clay"]["Ks_mm_s"] in vals


def test_validation_rejects_bad_theta():
    """An implausible sat-moisture must be caught before write."""
    mask = np.ones((4, 4), bool)
    rasters = {
        "soil_depth": np.full((4, 4), 1.0, "float32"),
        "conductivity": np.full((4, 4), 1e-3, "float32"),
        "resid_moisture_content": np.full((4, 4), 0.05, "float32"),
        "sat_moisture_content": np.full((4, 4), 0.99, "float32"),  # too high
        "overland_manning": np.full((4, 4), 0.15, "float32"),
        "bubbling_pressure": np.full((4, 4), 112.0, "float32"),
        "pore_size_dist": np.full((4, 4), 0.22, "float32"),
    }
    problems = P.validate_ranges(rasters, mask)
    assert any("sat_moisture_content" in p for p in problems)


def test_unmapped_landcover_warns(mask_path, tmp_path):
    grid = P.grid_from_mask(mask_path)
    codes = np.full(grid.shape, 999, "int32")   # not in crosswalk
    lc_path = tmp_path / "lc.tif"
    _write_class_raster(str(lc_path), mask_path, codes)
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="loam", landcover_path=str(lc_path),
    )
    assert any("SANLC" in w for w in result.warnings)
    # unmapped falls back to grassland
    no, *_ = T.read_raster(result.overland_manning)
    assert np.allclose(no[grid.mask], P.SANLC_N_O["grassland"])


def test_soil_depth_raster_overrides_default(mask_path, tmp_path):
    """A supplied depth raster (metres) must override the per-texture default."""
    grid = P.grid_from_mask(mask_path)
    depth = np.full(grid.shape, 2.5, "float32")   # 2.5 m everywhere
    depth_path = tmp_path / "depth.tif"
    with rasterio.open(mask_path) as src:
        profile = src.profile
    profile.update(dtype="float32", nodata=-9999.0, count=1)
    with rasterio.open(depth_path, "w", **profile) as dst:
        dst.write(depth, 1)
    result = P.build_params(
        mask_path, str(tmp_path / "params"),
        uniform_texture="loam", uniform_landcover="grassland",
        soil_depth_path=str(depth_path),
    )
    arr, *_ = T.read_raster(result.soil_depth)
    assert np.allclose(arr[grid.mask], 2.5, atol=1e-3)
    assert result.depth_source.startswith("raster:")
    # loam default depth is 1.2 m, so the override genuinely changed it
    assert not np.isclose(P.DEFAULT_SOIL_DEPTH_M["loam"], 2.5)


def test_depth_from_smu_reclass_cm_to_m(tmp_path):
    """HWSD SMU codes reclass to depth in metres; unmapped -> nodata + warn."""
    from rasterio.transform import from_origin
    smu = np.array([[10, 10, 20], [20, 30, 30]], dtype="int32")  # 30 unmapped
    smu_path = tmp_path / "smu.tif"
    profile = {"driver": "GTiff", "height": 2, "width": 3, "count": 1,
               "dtype": "int32", "nodata": 0,
               "transform": from_origin(300000, 6800000, 1000, 1000),
               "crs": rasterio.crs.CRS.from_epsg(32736)}
    with rasterio.open(smu_path, "w", **profile) as dst:
        dst.write(smu, 1)
    out = tmp_path / "depth.tif"
    with pytest.warns(UserWarning, match="not in the lookup"):
        P.depth_from_smu(str(smu_path), {10: 1.0, 20: 1.5}, str(out))  # metres
    arr, _, _, nodata = T.read_raster(str(out))
    assert np.isclose(arr[0, 0], 1.0) and np.isclose(arr[0, 2], 1.5)
    assert arr[1, 1] == nodata            # code 30 was unmapped


def test_depth_from_smu_csv_lookup(tmp_path):
    """The CSV lookup path converts a cm depth column to metres."""
    csv = tmp_path / "smu_depth.csv"
    csv.write_text("SMU,depth_cm\n10,100\n20,150\n")
    table = P._read_smu_depth_csv(str(csv), units="cm")
    assert np.isclose(table[10], 1.0) and np.isclose(table[20], 1.5)
