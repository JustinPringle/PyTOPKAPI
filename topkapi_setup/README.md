# topkapi_setup

Config-driven setup toolkit for PyTOPKAPI (Phase 1 of the catchment/estuary
project). Turns raw GIS + met data into a runnable simulation.

## Status

| Stage | Module | State |
|---|---|---|
| M0 env + housekeeping | — | **done** — conda env builds on Py3; h5py modes fixed; Liebenbergsvlei runs green |
| 4.1 Terrain + network | `terrain.py` | **done (M1)** — mask, flowdir, network, slope, accumulation |
| Terrain quick-look | `viz.py` | **done** — multi-panel figure of the terrain outputs |
| 4.2 Parameter rasters | `params.py` | **next (M2)** |
| 4.3 Forcing builder | `forcing.py` | to do (M3) |
| 4.4 Config + run | `config.py`, `run.py` | to do (M4) |
| 4.5 Calibration | `calibrate.py` | to do (M5) |
| §3 CWQM adapter | `adapters/cwqm.py` | to do |

## `terrain.py` (M1)

```bash
python -m topkapi_setup.terrain \
    --dem catchment_utm36s.tif \
    --outlet 512345 6890123 \
    --a-thres 1000000 \
    --out projects/umhlanga/terrain
```

Produces `mask.tif`, `flowdir.tif`, `network.tif`, `slope.tif`,
`accumulation.tif` and a `terrain_manifest.json`. `build_terrain()` is the
library entry point; the CLI is a thin wrapper. The final step runs
`check_terrain()`, which drives the real `create_file.cell_connectivity` to
assert a single outlet before anything is written to the solver.

`accumulation.tif` is not consumed by `create_file`; it is emitted for QC and
for the viewer, and is the direct way to judge whether `A_thres` puts the
channel head in the right place.

## `viz.py` — terrain quick-look

```bash
python -m topkapi_setup.viz projects/umhlanga/terrain \
    --dem projects/umhlanga/dem_utm36s.tif \
    --out projects/umhlanga/terrain/quicklook.png
```

Renders a single figure with up to five panels: a catchment overview (hillshade
+ boundary + channel network + outlet), elevation (needs `--dem`), slope, D8
flow direction (categorical, with a compass legend), and the accumulation
drainage tree. `plot_terrain()` is the library entry point. `--dem` and
`accumulation.tif` are optional; missing-optional panels are dropped, not faked.

### Three contracts it must honour (all enforced in code)

1. **Flow-direction codes.** create_file's `cell_connectivity` decodes ArcGIS D8
   codes (`E=1, SE=2, S=4, SW=8, W=16, NW=32, N=64, NE=128`). This is identical
   to pysheds' default `dirmap`, so we emit that and set
   `flowdir_source = ArcGIS` (exact casing — the code compares to that string).
2. **Channel encoding is inverted.** create_file runs
   `network[network < 255] = 1; network[network == 255] = 0`. Non-channel cells
   must be exactly `255`; a `0` background would make everything a channel. We
   write channel=1, background=255.
3. **Flow direction is masked to the catchment.** `cell_connectivity` visits
   every raster cell and links any with a valid code, so cells outside the mask
   must be `0`. The single in-mask cell draining outside the mask is the outlet.

Plus: **slope is in degrees**, floored above zero (`DEFAULT_MIN_SLOPE_DEG`),
because create_file applies `tan(pi/180 * slope)` and a zero slope zeros the
routing gradient. This is the replacement for the old `zero_slope_management`.

### A_thres and the outlet confluence

`A_thres` (channel-initiation area, m^2) must be large enough that the outlet
reach is **single-thread**. create_file's Strahler ordering seeds from only the
first channel arc entering the outlet, so a confluence at the outlet raises a
cryptic `KeyError`. `check_terrain()` guards against this with an actionable
message. When a catchment is cropped at a gauge the outlet is single-thread
anyway; the trap is only sprung by an unrealistically small `A_thres`.

## Housekeeping applied to `pytopkapi/parameter_utils/create_file.py`

The parameter path predated modern NumPy/networkx and could not run as shipped.
Minimal, behaviour-preserving fixes were made so M1's rasters feed it cleanly:

- `from osgeo import gdal` made **lazy** (moved into the two functions that use
  it), so the pure-NumPy graph routines import and test without a conda GDAL.
- `np.int -> int`, `np.float -> float` (aliases removed in NumPy >= 1.24).
- `np.fromstring -> np.frombuffer` in the legacy ArcGIS binary reader.
- `channel_properties`: index downstream cell positionally (`X[indx]`) instead
  of via a length-1 boolean mask, which NumPy 2 refuses to assign to a scalar.
- `strahler_to_channel_manning`: `list(nx.topological_sort(G))[-1]` — modern
  networkx returns a generator.
- Docstring `ARCGIS -> ArcGIS` to match the string the code actually compares.

## Environment findings (for the pinned conda env)

- **pysheds 0.5 is not NumPy-2 clean**: it calls the removed `np.in1d`.
  `topkapi_setup/_compat.py` restores the alias (`np.isin`) as a temporary
  bridge; the durable fix is to pin `pysheds > 0.5` or `numpy < 2` in the conda
  env. Revisit at M0's env refresh.
- Prototyped here with pip wheels: `rasterio` (bundled GDAL), `pysheds`, `h5py`,
  `tqdm`, `pytest`. The osgeo Python bindings need system `libgdal-dev`; the
  project's plan to source GDAL from conda-forge stands.
