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
`psi_b`, `lambda` (from a soils source) and overland Manning `n_o` (from land
cover). Values are range-validated before writing; a `params_manifest.json`
records paths and provenance for M4's config generator.

The soils source is the **SA Land Type** survey, following the SA TOPKAPI lineage
(Vischel et al. 2008; Sinclair & Pegram 2010): soil depth `L` and texture come
from each land type's memoir, and texture drives `theta_r/Ks/psi_b/lambda`
through the Rawls/Maidment table. `params` rasterises the Land Type **shapefile
directly** — no pre-rasterising — and reads a per-land-type attribute CSV keyed
by the (alphanumeric) code (e.g. `Fa491`). Two helpers in `soil_table` build that
CSV from the memoirs, so the code list and the soil numbers come from the data,
not a hand-typed spreadsheet.

### First pass, before any data — uniform fill

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --uniform-texture loam --uniform-landcover grassland \
    --out  projects/umhlanga/params
```

### 3.1 List the land types in the catchment  —  `soil_table list`

Clip the national Land Type layer to the mask and write a fetch manifest of the
codes present with their memoir URLs.

```bash
python -m topkapi_setup.soil_table list \
    --shp  projects/umhlanga/data/soil/landtype.shp \
    --mask projects/umhlanga/terrain/mask.tif \
    --out  projects/umhlanga/data/soil/lt_list.csv
```

`lt_list.csv` gives `land_type, area_km2, n_polys, objectid, url, pdf_name`,
sorted by area. The `url` column is the direct memoir link
(`http://www.agis.agric.za/memoir/pdfs/<code>.pdf`) carried in the shapefile's
`website` field. Deriving the code list from the clipped geometry — rather than
reading it off a map — is what keeps a mistyped `Fa41` for `Fa491`, or `Aa91`
for `Aa9`, out of the pipeline. Download each `<code>.pdf` into one folder,
named exactly `<code>.pdf`.

### 3.2 Parse the memoirs into the attribute CSV  —  `soil_table build`

```bash
python -m topkapi_setup.soil_table build \
    --pdf-dir   projects/umhlanga/data/soil/pdfs \
    --from-list projects/umhlanga/data/soil/lt_list.csv \
    --out       projects/umhlanga/data/soil/land_type_attrs.csv
```

Per land type the parser area-weights every constituent soil series by its
*Total %*:

| column | meaning |
|---|---|
| `L_m` | mean soil depth, range midpoints, **capped at 1.2 m** (root-restricting layer) |
| `clay_pct` | mean **A-horizon (topsoil)** clay % |
| `sand_pct` | `100 − clay − 20` (the silt assumption `params` uses internally) |
| `texture` | area-weighted **dominant memoir texture** (surveyed field texture) — drives `params` |
| `texture_triangle` | clay+sand → USDA triangle, as a cross-check (audit only) |
| `soil_coverage_pct` / `non_soil_pct` | share that is soil vs Rock / stream beds (excluded from weighting) |

`theta_s` is left blank, so `params` falls back to the texture porosity; drop in
Schulze per-land-type porosity here when you have it. To drive texture purely
from the clay+sand triangle instead of the surveyed class, delete the `texture`
column — `params` then uses `clay_pct` with the same `sand = 100 − clay − 20`.

### 3.3 Build the seven rasters  —  `params build`

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --out  projects/umhlanga/soil \
    --land-type projects/umhlanga/data/soil/landtype.shp --land-type-field landtype \
    --land-type-table projects/umhlanga/data/soil/land_type_attrs.csv \
    --landcover projects/umhlanga/data/landuse/SA_NLC_2020_GEO.tif
```

**Checks worth a glance.** A high `non_soil_pct` (Fa491 is 18.6 % stream beds)
means the land type is largely channel/alluvium — sane at a valley bottom, a flag
upslope. A non-empty `notes` field means the surveyed texture and the triangle
disagreed; open that memoir. Any code in `lt_list.csv` without a matching
`<code>.pdf` is skipped and reported by `build`, and would otherwise be
loam-defaulted by `params` with a warning.

### 3.4 Fallback — an already-rasterised soil source

If a soils layer arrives *already* rasterised, skip the Land Type path and pass
an integer soil-form / texture-class raster; codes map to texture via
`DEFAULT_SOILFORM_TEXTURE`.

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --soil-form projects/umhlanga/soilform.tif \
    --landcover projects/umhlanga/sanlc2020.tif \
    --out  projects/umhlanga/params
```

`L` from a soil-form raster is only the weak per-texture default, so override it
with a continuous depth raster in **metres** — SoilGrids depth-to-bedrock (250 m,
continuous, no join) works directly:

```bash
python -m topkapi_setup.params build \
    --mask projects/umhlanga/terrain/mask.tif \
    --soil-form projects/umhlanga/soilform.tif \
    --soil-depth projects/umhlanga/soilgrids_depth_m.tif \
    --landcover projects/umhlanga/sanlc2020.tif \
    --out  projects/umhlanga/params
```

The Land Type memoir path (3.1–3.3) already carries a real per-land-type depth,
so `--soil-depth` is only needed on this fallback. The `build` run prints which
source the depth came from and records it in `params_manifest.json`.

### View the parameter rasters

```bash
# all seven in one panel:
python -m topkapi_setup.viz projects/umhlanga/soil \
    --out projects/umhlanga/soil/quicklook.png

# each raster on its own, into a folder:
python -m topkapi_setup.viz projects/umhlanga/soil --each --out figs/params
```

### The default tables

Two documented defaults live at the top of `params.py` and are meant to be read
and edited:

* **`RAWLS_BROOKS_COREY`** — Brooks-Corey hydraulics by USDA texture class (Rawls
  & Brakensiek 1985; Maidment 1993). `psi_b` is stored and written in
  **millimetres** and `Ks` in **mm/s**, matching the solver's Green-Ampt path
  (mm-depth) — the shipped reference `cell_param.dat` carries `psi_b` ≈ 332 mm,
  which is how the unit was pinned. `theta_s` is total porosity.
* **`SANLC_N_O`** — overland Manning `n_o` by SANLC 2020 class group.

`DEFAULT_SANLC_CROSSWALK` (SANLC raster code → class group) is the **local** piece
to tune per catchment for `n_o`. `DEFAULT_SOILFORM_TEXTURE` (soil-form → texture
class) only bites on the `--soil-form` fallback of 3.4 — the Land Type path reads
texture straight from the memoir CSV. Codes not in a crosswalk fall back to
`loam` / `grassland` with a warning — check the run's warning list against the
codes actually present in your tile.

`--uniform-texture` accepts any key of `RAWLS_BROOKS_COREY`; `--uniform-landcover`
any key of `SANLC_N_O`. Full flag list: `python -m topkapi_setup.params build --help`.

---
## Creating the cell_params.dat file

```
python -m topkapi_setup.params cell-param \
    --terrain projects/umhlanga/terrain \
    --params  projects/umhlanga/soil \
    --dem     projects/umhlanga/data/dem_utm36s.tif \
    --out     projects/umhlanga/cell_param.dat
```
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
   licence, deterministic. (The memoir parser additionally carries two real
   `<code>.pdf` fixtures, since its whole job is reading that format.)

A module is not "done" until (1)–(5) hold and the suite is green.

---

## Milestone status

| # | Stage | Command | State |
|---|---|---|---|
| M0 | env + housekeeping | `conda env create` | done |
| M1 | preflight / terrain / viz | `preflight`, `terrain`, `viz` | done |
| M2 | parameter rasters | `params`, `soil_table` | done |
| M3 | forcing builder | `forcing` | to do |
| M4 | config + run (`--check`) | `config`, `run` | to do |
| M5 | calibration | `calibrate` | to do |
