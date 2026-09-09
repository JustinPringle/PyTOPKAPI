"""Guards on the preflight outlet: units, bounds, and how far it may snap.

Synthetic DEM only -- these tests exercise the checks, not the routing, so they
need no conditioning and run in milliseconds.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topkapi_setup import preflight as P


@pytest.fixture
def dem_utm(tmp_path):
    """A 100 x 100 cell, 30 m DEM in UTM36S, origin at a plausible KZN corner."""
    path = tmp_path / "dem.tif"
    arr = np.zeros((100, 100), dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=1,
        dtype="float32", crs="EPSG:32736",
        transform=from_origin(300000.0, 6720000.0, 30.0, 30.0),
    ) as dst:
        dst.write(arr, 1)
    return str(path)


def test_accepts_an_outlet_inside_the_dem(dem_utm):
    assert P.check_outlet_in_dem(dem_utm, (301000, 6719000)) == (301000.0, 6719000.0)


def test_rejects_decimal_degrees_on_a_projected_dem(dem_utm):
    """The uMhlanga failure: lat/lon read as metres snaps to the wrong river."""
    with pytest.raises(ValueError, match="decimal degrees"):
        P.check_outlet_in_dem(dem_utm, (-29.701982, 31.098846))


def test_rejects_an_outlet_outside_the_dem(dem_utm):
    with pytest.raises(ValueError, match="outside the DEM"):
        P.check_outlet_in_dem(dem_utm, (500000, 6719000))


def test_acc_threshold_in_km2_is_resolution_independent(tmp_path):
    """5 km^2 is 5 km^2 whether the cells are 30 m or 60 m."""
    paths = []
    for res in (30.0, 60.0):
        path = tmp_path / f"dem_{int(res)}.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=10, width=10, count=1,
            dtype="float32", crs="EPSG:32736",
            transform=from_origin(300000.0, 6720000.0, res, res),
        ) as dst:
            dst.write(np.zeros((10, 10), dtype="float32"), 1)
        paths.append(str(path))

    n30 = P.acc_cells_from_km2(paths[0], 5.0)
    n60 = P.acc_cells_from_km2(paths[1], 5.0)
    assert n30 == 5556 and n60 == 1389          # 5e6 / 900 and 5e6 / 3600
    assert n30 * 900 == pytest.approx(n60 * 3600, rel=1e-3)


def test_guards_have_one_definition():
    """preflight re-exports terrain's validators; they must not fork."""
    from topkapi_setup import terrain

    assert P.check_outlet_in_dem is terrain.check_outlet_in_dem
    assert P.acc_cells_from_km2 is terrain.acc_cells_from_km2
