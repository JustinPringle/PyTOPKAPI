"""Quick-look viewer for the terrain stage (Phase 1, section 4.1).

Renders the rasters :func:`topkapi_setup.terrain.build_terrain` writes into a
single multi-panel figure, so a catchment can be eyeballed before it is fed to
the solver. The panels answer the questions a modeller actually asks of a fresh
delineation:

    * **Overview** -- did the catchment close on the right outlet, and does the
      channel network look like the real river? (hillshade + boundary + network)
    * **Elevation** -- is the DEM sane inside the mask? (needs ``--dem``)
    * **Slope**     -- where are the steep reaches, and did the floor bite?
    * **Flow direction** -- D8 codes, a fast check for striping or flat artefacts.
    * **Accumulation** -- the drainage tree; the direct way to judge ``A_thres``.

Only ``mask``/``flowdir``/``network``/``slope`` are required (they are always
emitted). ``elevation`` needs the original DEM passed with ``--dem``;
``accumulation`` is drawn only if ``accumulation.tif`` sits in the terrain dir
(terrain writes it by default). Missing-optional panels are dropped, not faked.

Library entry point is :func:`plot_terrain`; the CLI is a thin wrapper::

    python -m topkapi_setup.viz projects/umhlanga/terrain \\
        --dem projects/umhlanga/dem_utm36s.tif \\
        --out projects/umhlanga/terrain/quicklook.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from topkapi_setup.terrain import read_raster

# D8 (ArcGIS) codes -> compass label and unit (drow, dcol) step, for the
# flow-direction legend and the optional quiver overlay.
_D8 = {
    1:   ("E",  (0,  1)),
    2:   ("SE", (1,  1)),
    4:   ("S",  (1,  0)),
    8:   ("SW", (1, -1)),
    16:  ("W",  (0, -1)),
    32:  ("NW", (-1, -1)),
    64:  ("N",  (-1,  0)),
    128: ("NE", (-1,  1)),
}


def _extent(transform, shape):
    """(left, right, bottom, top) in map units for ``imshow(extent=...)``."""
    nrows, ncols = shape
    left, top = transform * (0, 0)
    right, bottom = transform * (ncols, nrows)
    return (left, right, bottom, top)


def _hillshade(dem, cell_size, azimuth=315.0, altitude=45.0, z=2.0):
    """Grey hillshade in [0, 1] from an elevation array.

    A local light model (no external dependency): shade from the surface normal
    against a light vector set by ``azimuth``/``altitude``. ``z`` exaggerates
    relief so gentle coastal catchments still read.
    """
    dem = np.asarray(dem, dtype=float)
    dy, dx = np.gradient(dem * z, cell_size)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shaded = (np.sin(alt) * np.sin(slope)
              + np.cos(alt) * np.cos(slope) * np.cos(az - aspect))
    return np.clip((shaded + 1) / 2.0, 0, 1)


def _load(terrain_dir, dem_path=None):
    """Read the panel arrays from a terrain output directory.

    Returns a dict of what is present: always ``mask/flowdir/network/slope`` and
    the shared ``transform``; ``elevation`` if ``dem_path`` is given;
    ``accumulation`` if ``accumulation.tif`` is in the directory; and the snapped
    ``outlet`` and cell counts from the manifest when available.
    """
    d = Path(terrain_dir)
    layers, transform = {}, None
    for name in ("mask", "flowdir", "network", "slope"):
        arr, tr, _, _ = read_raster(d / f"{name}.tif")
        layers[name] = arr
        transform = transform or tr

    acc_tif = d / "accumulation.tif"
    if acc_tif.exists():
        layers["accumulation"], _, _, _ = read_raster(acc_tif)

    if dem_path is not None:
        layers["elevation"], _, _, _ = read_raster(dem_path)

    manifest = d / "terrain_manifest.json"
    meta = json.loads(manifest.read_text()) if manifest.exists() else {}
    return layers, transform, meta


# --- Individual panels ------------------------------------------------------

def _panel_overview(ax, layers, extent, cell_size, meta):
    """Hillshade (or slope relief) + catchment boundary + channel network."""
    from matplotlib import colors

    mask = layers["mask"] == 1
    if "elevation" in layers:
        base = _hillshade(layers["elevation"], cell_size)
        title = "Catchment overview (hillshade + network)"
    else:  # fall back to a slope-driven relief when no DEM was supplied
        s = np.asarray(layers["slope"], float)
        base = 1.0 - np.clip(s / (np.nanpercentile(s, 98) or 1), 0, 1)
        title = "Catchment overview (slope relief + network)"

    ax.imshow(base, cmap="gray", extent=extent, origin="upper",
              interpolation="nearest", vmin=0, vmax=1)
    # dim everything outside the catchment
    outside = np.ma.masked_where(mask, np.ones_like(base))
    ax.imshow(outside, cmap=colors.ListedColormap(["#0b1e3a"]), alpha=0.55,
              extent=extent, origin="upper", interpolation="nearest")
    # catchment boundary
    ax.contour(mask.astype(float), levels=[0.5], colors="#ffd166",
               linewidths=1.6, extent=extent, origin="upper")
    # channel network (network==1 is channel)
    chan = np.ma.masked_where(layers["network"] != 1, np.ones_like(base))
    ax.imshow(chan, cmap=colors.ListedColormap(["#2a9df4"]),
              extent=extent, origin="upper", interpolation="nearest")
    if meta.get("outlet_xy"):
        ox, oy = meta["outlet_xy"]
        ax.plot(ox, oy, marker="v", ms=11, mfc="#ef476f", mec="white",
                mew=1.2, zorder=5, label="outlet")
        ax.legend(loc="upper right", framealpha=0.85, fontsize=8)
    ax.set_title(title, fontsize=10)


def _panel_elevation(ax, layers, extent):
    mask = layers["mask"] == 1
    z = np.ma.masked_where(~mask, np.asarray(layers["elevation"], float))
    im = ax.imshow(z, cmap="terrain", extent=extent, origin="upper",
                   interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="elevation (m)")
    ax.set_title("Elevation (masked)", fontsize=10)


def _panel_slope(ax, layers, extent):
    mask = layers["mask"] == 1
    s = np.ma.masked_where(~mask, np.asarray(layers["slope"], float))
    im = ax.imshow(s, cmap="magma", extent=extent, origin="upper",
                   interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="slope (deg)")
    ax.set_title("Surface slope (floored > 0)", fontsize=10)


def _panel_flowdir(ax, layers, extent):
    from matplotlib import colors

    mask = layers["mask"] == 1
    fdir = np.asarray(layers["flowdir"])
    codes = [64, 128, 1, 2, 4, 8, 16, 32]           # N, NE, E, SE, S, SW, W, NW
    idx = np.full(fdir.shape, -1, dtype=int)
    for i, code in enumerate(codes):
        idx[fdir == code] = i
    idx = np.ma.masked_where(~mask | (idx < 0), idx)
    cmap = colors.ListedColormap(
        ["#4062BB", "#59A5D8", "#5DD39E", "#BCE784",
         "#F4D35E", "#F49E4C", "#EF476F", "#B5179E"])
    im = ax.imshow(idx, cmap=cmap, vmin=-0.5, vmax=7.5,
                   extent=extent, origin="upper", interpolation="nearest")
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8, ticks=range(8))
    cbar.ax.set_yticklabels([_D8[c][0] for c in codes])
    ax.set_title("D8 flow direction (ArcGIS)", fontsize=10)


def _panel_accumulation(ax, layers, extent):
    from matplotlib import colors

    mask = layers["mask"] == 1
    acc = np.asarray(layers["accumulation"], float)
    acc = np.where(acc < 1, 1, acc)                 # keep log finite
    a = np.ma.masked_where(~mask, np.log10(acc))
    im = ax.imshow(a, cmap="cubehelix_r", extent=extent, origin="upper",
                   interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, shrink=0.8, label="log10 cells upstream")
    ax.set_title("Flow accumulation (drainage tree)", fontsize=10)


def plot_terrain(terrain_dir, dem_path=None, out_png=None, dpi=140, show=False):
    """Render a multi-panel quick-look of a terrain output directory.

    Parameters
    ----------
    terrain_dir : path
        Directory written by :func:`topkapi_setup.terrain.build_terrain`.
    dem_path : path, optional
        Original DEM. Enables the hillshade backdrop and the elevation panel.
    out_png : path, optional
        Where to save the figure. Defaults to ``<terrain_dir>/quicklook.png``.
    dpi, show :
        Figure resolution and whether to call ``plt.show()``.

    Returns the path of the written PNG.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, transform, meta = _load(terrain_dir, dem_path)
    extent = _extent(transform, layers["mask"].shape)
    cell_size = abs(transform.a)

    # Panels present, in reading order.
    panels = [("overview", _panel_overview)]
    if "elevation" in layers:
        panels.append(("elevation", _panel_elevation))
    panels.append(("slope", _panel_slope))
    panels.append(("flowdir", _panel_flowdir))
    if "accumulation" in layers:
        panels.append(("accumulation", _panel_accumulation))

    ncols = 3 if len(panels) > 4 else 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, fn) in zip(axes, panels):
        if name == "overview":
            fn(ax, layers, extent, cell_size, meta)
        else:
            fn(ax, layers, extent)
        ax.set_xlabel("easting (m)", fontsize=8)
        ax.set_ylabel("northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(style="plain")
    for ax in axes[len(panels):]:
        ax.axis("off")

    n_cells = meta.get("n_cells", int((layers["mask"] == 1).sum()))
    n_chan = meta.get("n_channel_cells", int((layers["network"] == 1).sum()))
    crs = meta.get("crs", "")
    fig.suptitle(
        f"Terrain quick-look  |  {n_cells} cells, {n_chan} channel cells  "
        f"|  cell {cell_size:g} m  |  {crs}",
        fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_png = str(out_png or (Path(terrain_dir) / "quicklook.png"))
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# --- Thin CLI ---------------------------------------------------------------

def _build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Render a quick-look figure of a terrain output directory.")
    p.add_argument("terrain_dir", help="Directory build_terrain wrote")
    p.add_argument("--dem", default=None,
                   help="Original DEM GeoTIFF (adds hillshade + elevation panel)")
    p.add_argument("--out", default=None, help="Output PNG (default: <dir>/quicklook.png)")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--show", action="store_true", help="Also open an interactive window")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    path = plot_terrain(args.terrain_dir, dem_path=args.dem, out_png=args.out,
                        dpi=args.dpi, show=args.show)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
