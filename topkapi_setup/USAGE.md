# Using the toolkit

Task-oriented guide to `topkapi_setup`. Each stage is a pure library function
with a thin CLI wrapper, so you can script it or run it from the shell. **The
CLI `--help` is the authoritative reference** for every flag; this file is the
map of *which* command to reach for and *why*.

```
raw DEM ──preflight──▶ clean UTM36S DEM ──terrain──▶ mask/flowdir/network/slope
                                                          │
                                              params ◀────┘ (snaps to the mask)
                                                          │
                                            (M3) forcing, (M4) config + run …
```

Convention throughout: **projected CRS in metres (UTM Zone 36S, EPSG:32736)**,
30 m cells for a first pass. Everything downstream snaps to the terrain mask.

---

## 0. One-time setup

```bash
conda env create -f conda-env.yaml      # builds the `topkapi` env (GDAL from conda-forge)
conda activate topkapi
pip install -e .                         # editable install of topkapi_setup + pytopkapi
pytest topkapi_setup/tests -q            # sanity: the full suite should be green
```

---

## 1. Prepare a raw DEM  —  `preflight`

Turns a raw download (often lat/lon SRTM) into a clean, projected DEM and helps
you land the outlet on the real channel. Five composable steps plus a `run-all`.

```bash
# Inspect what you downloaded (CRS, size, nodata) before doing anything:
python -m topkapi_setup.preflight inspect  --dem ohlanga_srtm_4326.tif

# Clip in the raw CRS, reproject to UTM36S at 30 m, then reveal the rivers:
python -m topkapi_setup.preflight clip      --dem ohlanga_srtm_4326.tif \
    --bbox <minx> <miny> <maxx> <maxy> --out clipped.tif
python -m topkapi_setup.preflight reproject --dem clipped.tif \
    --epsg 32736 --res 30 --out dem_utm36s.tif
python -m topkapi_setup.preflight reveal    --dem dem_utm36s.tif --out rivers.png

# Check where a candidate outlet snaps before committing to a full delineation:
python -m topkapi_setup.preflight preview-snap --dem dem_utm36s.tif \
    --outlet <easting> <northing> --min-acc-cells 5000
```

Why it matters: a raw SRTM tile is EPSG:4326 (degrees) with millions of cells,
and a hand-picked mouth coordinate often lands a pixel or two off the channel on
a low-accumulation bank cell. `reveal` (flow accumulation + hillshade) shows you
the true main stem; `preview-snap` confirms the outlet snaps onto it. See
`--help` on any subcommand for the full flag list, and `run-all` to chain them.

---

## 2. Delineate terrain and network  —  `terrain`

From a projected DEM and a snapped outlet, produce the rasters
`generate_param_file` needs: `mask`, `flowdir`, `network`, `slope` (+
`accumulation` for QC) and a `terrain_manifest.json`.

```bash
python -m topkapi_setup.terrain \
    --dem     projects/umhlanga/dem_utm36s.tif \
    --outlet  <easting> <northing> \
    --a-thres 1000000 \
    --out     projects/umhlanga/terrain
```

* `--a-thres` is the channel-initiation area in **m²** (start ~1e6 ≈ 1100
  30 m cells). It sets how dense the channel network is.
* `terrain.py` auto-extracts the catchment from a larger DEM — no manual QGIS
  clipping. It snaps the outlet onto the drainage line (`--min-acc-cells`).
* The final step drives the **real** `create_file.cell_connectivity` to assert a
  single outlet before anything reaches the solver.

**If it raises "outlet is a channel confluence":** `--a-thres` is too small — two
channel cells drain into the outlet and `create_file`'s Strahler ordering only
follows one branch. Raise `--a-thres` until the outlet reach is single-thread.

Three contracts it honours automatically (documented in `README.md`):
flow-direction codes are ArcGIS D8 (`flowdir_source = ArcGIS`); the channel
raster is **inverted** (channel = 1, background = 255); flow direction is
**masked** to the catchment so the one draining cell is the sole outlet. Slope is
in **degrees**, floored above zero.

### Look at it  —  `viz`

The viewer auto-detects whether a directory is terrain or params output (from
its manifest). A combined panel by default; `--each` writes one PNG per raster.

```bash
# terrain panel (hillshade + elevation need the DEM):
python -m topkapi_setup.viz projects/umhlanga/terrain \
    --dem projects/umhlanga/dem_utm36s.tif \
    --out projects/umhlanga/terrain/quicklook.png

# every terrain layer on its own, into a folder:
python -m topkapi_setup.viz projects/umhlanga/terrain --each --out figs/terrain
```

Multi-panel figure: overview, elevation, slope, D8 direction, accumulation. Read
the *accumulation* panel to tune `--a-thres` — channels sprouting in every hollow
means it is too small; a stubby main stem means it is too large.

---

## 3. Generate parameter rasters  —  `params`  (M2)

Produces the **seven** remaining rasters `generate_param_file` reads, each
snapped to the terrain mask: soil depth `L`, `Ks`, `theta_r`, `theta_s`,
`psi_b`, `lambda` (all from a soils source) and overland Manning `n_o` (from land
cover). Values are range-validated before writing; a `params_manifest.json`
records paths and provenance for M4's config generator.

**First pass, before site data exist** — fill the catchment uniformly:

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --uniform-texture loam --uniform-landcover grassland \
    --out  projects/umhlanga/params
```

**With real layers** — pass an integer soil-form/texture raster and a SANLC 2020
land-cover raster:

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --soil-form projects/umhlanga/soilform.tif \
    --landcover projects/umhlanga/sanlc2020.tif \
    --out  projects/umhlanga/params
```

### Soil depth from HWSD (FAO)

Soil depth `L` is *not* a Brooks-Corey output; the default per-texture value is
the weakest number in the table. Replace it with rootable soil depth from HWSD
v2.0. HWSD ships an SMU-code raster linked to a Microsoft Access (`.mdb`)
attribute table — the depth lives in the table, not the raster — at 1 km
(30 arc-second) resolution. Two steps:

```bash
# 1. Export an "SMU,depth" lookup once from the HWSD .mdb (HWSD viewer or
#    mdbtools), then reclass the SMU raster to a depth raster (cm -> m):
python -m topkapi_setup.params hwsd-depth \
    --smu   hwsd2_smu.tif --table smu_depth.csv --units cm \
    --out   projects/umhlanga/soil_depth_hwsd.tif

# 2. Feed it to build; it resamples onto the model grid and overrides the default:
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --uniform-texture loam --uniform-landcover grassland \
    --soil-depth projects/umhlanga/soil_depth_hwsd.tif \
    --out  projects/umhlanga/params
```

`--soil-depth` takes any continuous depth GeoTIFF **in metres**, so SoilGrids
depth-to-bedrock (250 m, continuous, no join) works the same way. At 1 km, HWSD
gives only ~80 cells across the Ohlanga — fine for a slowly-varying field, but
SoilGrids is finer if you want it. The `build` run prints which source the depth
came from and records it in `params_manifest.json`.

### View the parameter rasters

```bash
# all seven in one panel:
python -m topkapi_setup.viz projects/umhlanga/params \
    --out projects/umhlanga/params/quicklook.png

# each raster on its own, into a folder:
python -m topkapi_setup.viz projects/umhlanga/params --each --out figs/params
```

### The default tables

Two documented defaults live at the top of `params.py` and are meant to be read
and edited:

* **`RAWLS_BROOKS_COREY`** — Brooks-Corey hydraulics by USDA texture class (Rawls,
  Brakensiek & Saxton 1982; Rawls & Brakensiek 1985). Stored in literature units
  (`psi_b` in **metres**); written to raster as **millimetres**, because the
  solver's Green-Ampt path works in mm-depth (the shipped reference
  `cell_param.dat` carries `psi_b` ≈ 332 mm, which is how the unit was pinned).
* **`SANLC_N_O`** — overland Manning `n_o` by SANLC 2020 class group.

Two crosswalks are the **local** piece to tune per catchment:
`DEFAULT_SOILFORM_TEXTURE` (SA Land Type soil-form → texture class) and
`DEFAULT_SANLC_CROSSWALK` (SANLC raster code → class group). Codes not in a
crosswalk fall back to `loam` / `grassland` with a warning — check the run's
warning list against the codes actually present in your tile.

`--uniform-texture` accepts any key of `RAWLS_BROOKS_COREY`; `--uniform-landcover`
any key of `SANLC_N_O`. Full flag list: `python -m topkapi_setup.params build --help`.

---

## Writing new stages

Keep the contract every module here follows, so this file and `--help` stay the
authoritative docs:

1. Pure functions do the work; the CLI is a thin `argparse` wrapper.
2. A module docstring with a runnable example at the top of the file.
3. `argparse` help text on every flag.
4. A `*_manifest.json` recording output paths, CRS, cell count and provenance,
   for the next stage to consume.
5. Tests against the synthetic valley fixture (`tests/_synthetic.py`) — no data
   licence, deterministic.

A module is not "done" until (1)–(5) hold and the suite is green.

---

## Milestone status

| # | Stage | Command | State |
|---|---|---|---|
| M0 | env + housekeeping | `conda env create` | done |
| M1 | preflight / terrain / viz | `preflight`, `terrain`, `viz` | done |
| M2 | parameter rasters | `params` | done |
| M3 | forcing builder | `forcing` | to do |
| M4 | config + run (`--check`) | `config`, `run` | to do |
| M5 | calibration | `calibrate` | to do |
