# Using the toolkit

Task-oriented guide to `topkapi_setup`. Each stage is a pure library function
with a thin CLI wrapper, so you can script it or run it from the shell. **The
CLI `--help` is the authoritative reference** for every flag; this file is the
map of *which* command to reach for and *why*.

```
raw DEM ──preflight──▶ clean UTM36S DEM ──terrain──▶ mask/flowdir/network/slope
                                                          │
                                              params ◀────┤ (snaps to the mask)
                                                          │
                                             forcing ◀────┘ (same cell order)
                                                          │
                                              (M4) config + run …
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

## 4. Build the rainfall forcing  —  `forcing`  (M3)

Produces `rainfields.h5`: one array of shape `(n_timesteps, n_cells)` holding
rainfall depth in mm for every cell at every step, in the **same cell order**
`cell_param.dat` uses. That file plus `ET.h5` and `global_param.dat` is what the
solver runs on.

> **Partial CLI.** The gauge sources in 4.1 (`gauge_manifest`, `ethekwini_fews`)
> are CLIs. The field builder itself — turning the two input files into
> `rainfields.h5` — is still library functions only; there is no
> `python -m topkapi_setup.forcing`, so drive 4.3 from a script or notebook. That
> CLI and the `forcing_manifest.json` remain outstanding against the contract in
> *Writing new stages*.

```
manifest.csv ─┐
              ├─▶ W (n_cells × n_gauges)  ─┐
   mask.tif ──┘      build once             ├─▶ field ─▶ rainfields.h5
                                            │
measurements.csv ─▶ readings (n_t × n_gauges)┘
                    + availability
```

The one idea worth holding: rainfall on a cell is a **weighted blend of the
gauge readings**, and the weights depend only on where the cell sits relative to
the gauges. That geometry never changes, so the weight table `W` is built once
and reused for every timestep. Choosing an interpolation method means choosing
how `W` is filled — nothing downstream changes.

### 4.1 Sourcing gauges from eThekwini FEWS

The two input files in 4.2 can be handed in directly, or built from the
eThekwini FEWS network with two adapters under `topkapi_setup/forcing/sources/`.
Both are thin CLIs; run them in order.

```
gauges_raw.json ──gauge_manifest──▶ manifest.csv ──ethekwini_fews──▶ measurements.csv
```

**Scope the network to the catchment — `gauge_manifest`.** Reprojects every
station to the mask's CRS, keeps those within a buffer of the delineated
catchment, and writes the manifest.

```bash
python -m topkapi_setup.forcing.sources.gauge_manifest \
    --network projects/umhlanga/raw/gauges_raw.json \
    --mask    projects/umhlanga/terrain/mask.tif \
    --out     projects/umhlanga/data/manifest.csv \
    --buffer-km 20
```

`--network` is the FEWS station dump saved verbatim (the API returns a
Python-literal string, not strict JSON; both parse). `--mask` is the `terrain.py`
mask and must be projected in metres — the buffer distance and the output
coordinates are metres in its CRS. The output carries two extra columns beyond
4.2's contract: `device` (the instrument serial, provenance only) and `in_mask`.

Keep the buffer. A gauge just outside the divide still constrains the field at
the edge, where the constraint is most useful; `in_mask` records whether a gauge
is strictly inside the catchment or doing that boundary work. Both belong in the
manifest, so this is the coarse scoping step — not `select_gauges` (4.4), which
is a finer per-run choice layered on top.

**Pull the series — `ethekwini_fews`.** Fetches each gauge in the manifest over a
date window and writes `measurements.csv`.

```bash
export ETHEKWINI_FEWS_KEY=…            # Authorization header, never the URL
python -m topkapi_setup.forcing.sources.ethekwini_fews \
    --manifest  projects/umhlanga/data/manifest.csv \
    --start 20250101 --end 20250131 \
    --out       projects/umhlanga/data/measurements.csv \
    --cache-dir projects/umhlanga/raw/fews
```

| Flag | Default | Purpose |
|---|---|---|
| `--cache-dir` | none | raw JSON per gauge-window; provenance, and resumes a run |
| `--step` | `1h` | emit grid; no coarser than `Dt` |
| `--gap-threshold` | `12h` | a longer silence is a gap, a shorter one is dry; `none` disables |
| `--min-interval` | `1.0` | seconds between requests; raise it if `429` persists |
| `--auth-scheme` | `bearer` | Authorization header scheme |

Three things about this feed, all handled by the adapter:

- **Auth is `Bearer <key>`** from `$ETHEKWINI_FEWS_KEY`. A `401` means the key is
  absent in the shell you ran from.
- **It rate-limits.** Requests are paced `--min-interval` apart and a `429`/`503`
  is retried with backoff, honouring `Retry-After`; with `--cache-dir` set a
  throttled run resumes rather than restarting.
- **It reports by exception** — a row per tip during rain, only a sparse
  heartbeat when dry. The collector fills dry hours the gauge was demonstrably
  alive through with `0` and reserves a gap for silences past `--gap-threshold`,
  so a dry hour is never mistaken for missing data. (Left raw, every dry hour
  would read as a gap and the field would refuse to assemble — see the last
  gotcha in 4.8.)

Because the collector emits on a regular `--step` grid, these gauges' native step
*is* that grid (`1h` by default), so they need no `native_step` entry in 4.5.

**Timezone — confirm before calibration.** Stamps come from the feed's `tstr`
field, the UTC rendering of its epoch. Whether that wall-clock is genuinely UTC
or a local (SAST) clock labelled as UTC is not knowable from the data. Stamps are
emitted naive — set the zone once on `Timeline` (`tz="Africa/Johannesburg"` if
local) and pin it against a storm's known onset. A silent 2 h offset is invisible
in totals and wrong against tides and DWS flow.

### 4.2 The two input files

Coordinates live apart from the series, mirroring the CWQM river contract. Build
them with 4.1, or supply them yourself in these formats.

**`manifest.csv`** — one row per gauge:

```
gauge_id,x,y,crs,name,source,native_step
0241078,31.05,-29.72,EPSG:4326,uMhlanga,SAWS,5min
0241512,316000,6712000,EPSG:32736,Ohlanga mouth,DWS,1h
```

`crs` is per row and mandatory. SAWS and DWS coordinates often arrive in lat/lon;
declaring the CRS makes the reprojection to UTM36S deterministic rather than
guessed, and a manifest mixing lat/lon and UTM rows works as-is.

`native_step` is optional but **set it whenever you know the resolution** — see
4.5. Row order here fixes the column order of `W` and of `readings`. Extra
columns are carried through untouched, which is how the 4.1 adapter's `device`
and `in_mask` ride along without disturbing anything downstream.

**`measurements.csv`** — long format, one row per reading:

```
datetime,gauge_id,rainfall_mm
2024-01-01 00:05,0241078,0.2
2024-01-01 00:10,0241078,0.0
```

Long rather than one column per gauge, because gauges rarely share a clock.
Ragged records and gaps cost nothing in this shape. A blank `rainfall_mm` is a
**gap**, not a dry reading, and is carried as such; a negative value is rejected
outright, since it is almost always a `-9999` no-data sentinel.

### 4.3 End to end

```python
import numpy as np
from topkapi_setup.forcing import gauges as gg, interpolate as ip, rainfields as rf

TERRAIN = "projects/umhlanga/terrain"
MASK    = f"{TERRAIN}/mask.tif"
PARAM   = "projects/umhlanga/cell_param.dat"

# 1. Gauges, reprojected to the model CRS.
man  = gg.read_manifest("projects/umhlanga/data/manifest.csv")
meas = gg.read_measurements("projects/umhlanga/data/measurements.csv")

# 2. The clock. Interval-ending: the value at 08:00 covers (07:00, 08:00].
tl = gg.Timeline("2024-01-01 01:00", "2024-02-01 00:00", dt_seconds=3600)

# 3. Every gauge onto that clock, with gaps flagged.
readings, available = gg.align_to_clock(
    meas, tl, man.index,
    native_steps=man["native_step"].dropna().to_dict(),
)
print(gg.coverage(available, man.index, tl))     # look at this before going on

# 4. The weight table, built once from the catchment geometry.
cell_xy = np.column_stack(ip.catchment_cell_xy(MASK))
W = ip.build_weights(cell_xy, gg.gauge_xy(man), method="idw")

# 5. Field to disk, with the cell-order guard armed.
rf.build_and_write_rainfields(
    "projects/umhlanga/forcing/rainfields.h5", W, readings, available,
    group_name="ohlanga_jan2024",
    mask_path=MASK, cell_param_path=PARAM, timeline=tl,
)
```

`group_name` must match `group_name` in the simulation `.ini`; the solver reads
`/{group_name}/rainfall`.

Several events can share one file, but the two writers differ on this and the
difference is sharp. `build_and_write_rainfields` always appends: it replaces
its own group and leaves the others alone. `write_rainfields` defaults to
`overwrite=True`, which opens the file in `"w"` mode and **truncates it** — a
second call with a new `group_name` destroys the first group without a word.
Pass `overwrite=False` when adding a group beside an existing one.

### 4.4 Choosing an interpolation method

| Method | How it fills `W` | When |
|---|---|---|
| `mean` | every cell takes `1/n` from every gauge | sanity baseline; a single gauge |
| `thiessen` | each cell takes 100% from its nearest gauge | quick and robust, few gauges |
| `idw` | weights fall off with distance | **default** for a typical network |
| `kriging` | weights from a fitted variogram | well-gauged catchments |

```python
W = ip.build_weights(cell_xy, gg.gauge_xy(man), method="idw", power=2.0)
W = ip.build_weights(cell_xy, gg.gauge_xy(man), method="thiessen")
```

**Kriging** needs roughly 15–30 gauges in range for a stable variogram — rare on
a small coastal catchment, normal on the large inland ones. A variogram is a
property of the rainfall field, not of the geometry, so it cannot come from
coordinates alone: pass `sample_values`, one representative reading per gauge
(a wet-period mean), and the model is fitted once and held fixed for the whole
record so `W` stays a build-once table.

```python
sample = readings[readings.sum(axis=1) > 0].mean(axis=0)     # wet-period mean
W = ip.build_weights(cell_xy, gg.gauge_xy(man), method="kriging",
                     sample_values=sample)                    # needs pykrige
```

Without `sample_values` you get an explicit spherical model (`range_m`, `sill`)
— a smooth distance-decay surface, honestly not a fitted one, and it runs with
no `pykrige` installed. Kriging weights are legitimately negative for screened
gauges, which would give negative rainfall, so they are clipped and renormalised
by default; pass `non_negative=False` for the exact solution.

**Isohyetal** is not a fifth method. Drawn faithfully by machine, those smooth
contours *are* the IDW or kriging surface, so it belongs in the plotter as
contours over the field, not as a separate `W`-builder.

**Out-of-catchment gauges.** Do not clip gauges to the catchment boundary — the
ones just outside constrain the field exactly where the constraint is most
useful. `W` has a row per in-mask cell and a column per gauge within a buffer,
so the result is already masked. The buffer only stops a gauge 200 km away from
dragging on the fit. If the manifest was built with 4.1, this scoping is already
applied; `select_gauges` re-runs it at interpolation time, tighter if you want.

```python
keep = ip.select_gauges(gg.gauge_xy(man), cell_xy, buffer_m=30_000)
man = man.iloc[keep]
```

### 4.5 The clock, and two traps in it

`Timeline` is the single clock for rainfall, ET and point inflows, so the three
forcing files cannot silently disagree. Its convention is **interval-ending**:
the value stamped `t` is the accumulation over `(t - Dt, t]`. `Dt` is a fixed
number of seconds, because `global_param.dat`'s `Dt` is — calendar months are
not expressible, so "monthly" means a 30-day step and February is short.

Each gauge is resampled from its own native step: finer than `Dt` aggregates,
coarser disaggregates. Two things bite here.

**Declare the native step.** Inference reads the modal spacing of the stamps,
and a gappy record defeats it: an hourly gauge that reported only at 01:00 and
03:00 looks two-hourly, and its totals get *spread* across the missing hour
instead of that hour being a gap. Declaring the step keeps it honest.

```python
gg.align_to_clock(meas, tl, man.index, native_steps={"0241078": "5min"})
```

**Partial bins.** Going from 5-minute ticks to an hourly `Dt`, an hour holding
only 1 of its 12 ticks is not a light hour — it is a gap that happens to contain
a reading, and summing it is a silent under-catch spread through the record. A
bin must hold `min_coverage` (default 0.8) of its expected readings or it
becomes a gap. Partial bins are dropped rather than scaled up: rainfall is
intermittent, so the minutes that recorded are not a fair sample of the ones
that did not.

```python
gg.align_to_clock(meas, tl, man.index, min_coverage=0.8)   # 0.0 disables
```

Gaps are handled as a column operation: the offline gauge's weight is zeroed and
the surviving gauges' weights renormalised for those steps. Nothing else in the
pipeline special-cases a gap.

**Disaggregation is a modelling decision, not a resample.** It only arises when
a record is coarser than `Dt` — with 5-minute Ohlanga data at hourly `Dt` it
never does. Splitting a daily total across the hours uniformly conserves mass
but flattens the peak, which on a flashy catchment is the quantity of interest,
so it warns. Pass a fine-resolution `shape` series — IMERG half-hourly is the
intended source — and the gauge sets the volume while the satellite sets the
timing.

```python
gg.align_to_clock(meas, tl, man.index, shape=imerg_hourly)   # imerg_hourly: a Series on tl.times
```

### 4.6 The cell-order guard — do not skip it

Column `j` of `rainfields.h5` is the cell on line `j` of `cell_param.dat`.
Nothing in the file records that, and the solver cannot check it: a permuted
field runs to completion and produces a plausible hydrograph that is wrong
everywhere. Same failure class as the parallel-routing race and the CWQM
`read_river` column bug.

Passing `mask_path` and `cell_param_path` to either writer compares the
mask-derived coordinates against columns 1–2 of `cell_param.dat` and refuses to
write on any mismatch — permutation, reversal, an x/y swap, a count mismatch. It
costs about 0.3 s on the 90,770-cell Ohlanga catchment. There is no good reason
to omit it.

```python
rf.check_cell_order(MASK, PARAM)     # returns the cell count, or raises
```

If it raises a count mismatch, the mask and `cell_param.dat` came from different
terrain runs; rebuild `cell_param.dat` from this mask.

### 4.7 Size, and which writer to use

The field is large, and it scales with cells × timesteps. Measured on the real
Ohlanga catchment (90,770 cells at 30 m, hourly):

| Record | On disk (`float32`) |
|---|---|
| 1 month | 270 MB |
| 1 year | 3.2 GB |

Roughly double that in RAM while it is computed in double precision. Two
writers:

```python
rf.write_rainfields(path, field, ...)              # short events, testing
rf.build_and_write_rainfields(path, W, readings, available, ...)   # anything longer
```

`build_and_write_rainfields` streams in time blocks and never holds the whole
array — peak 0.55 GB on a month of real geometry, output byte-identical to the
in-memory path. Use it beyond a few months.

Compression is off by default and is not worth turning on: IDW puts a little
rain in nearly every cell, so the field is dense float noise rather than sparse.
On the real month, gzip recovered 15% for a 3× slower write. If the files are
still too big, the lever is a coarser grid or a longer `Dt`, not compression —
both cut the array itself.

Read it back the way the solver does:

```python
field = rf.read_rainfields("…/rainfields.h5", group_name="ohlanga_jan2024")
```

### 4.8 Gotchas

- **Manifest row order is load-bearing.** It fixes the columns of `W`,
  `readings` and `available`. Take all three from the same frame — pass
  `man.index` to `align_to_clock` — and they cannot drift. A gauge with no data
  stays as an all-unavailable column rather than vanishing, for this reason.
- **A file that changes datetime format mid-record** loses rows silently in
  pandas, which infers one format and coerces the rest to `NaT`. The reader
  falls back to per-value parsing when it sees this.
- **A clock not starting on a round boundary** used to wipe the record: pandas
  anchors resample bins to midnight. The resample is anchored to the timeline
  now, but it is why `aggregate` is tested across `Dt` from 15 minutes to
  30 days.
- **A timestep where no gauge reports at all** is refused rather than guessed.
  Trim the timeline to the period the network covers, or fill from a gridded
  product. (This is why the 4.1 collector zero-fills dry hours rather than
  leaving them as gaps: a report-by-exception feed would otherwise make almost
  every dry timestep all-gap.)
- **Check `coverage()` before building the field.** A gauge at 3% is a clock or
  unit problem, not a broken instrument, and it is far cheaper to catch here
  than in calibration.

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
| M3 | forcing builder (rainfall) | `forcing.*` (library only) | done |
| M4 | config + run (`--check`) | `config`, `run` | to do |
| M5 | calibration | `calibrate` | to do |
