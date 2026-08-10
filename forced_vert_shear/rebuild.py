#!/usr/bin/env python3
"""Rebuild a locally-downloaded Zarr v3 array into a single 3D .npy (Numpy) array

Use this script when you've already downloaded a simulation/var/level array folder onto local disk
yourself (e.g. via Globus), e.g. `R4P50.zarr/r/3`. This folder will contain a subfolder `c`
containing multiple chunks (covering the region you specified) and a zarr.json file. 
This script will rebuild those zarr chunks into the original 3D array.

Options:
  -o, --output    output .npy file path (default if not specified: auto-named from zarr.json
                   attributes, written to current working directory). 
  -d, --outdir    directory to write into, created if needed 
                   (default if not specified: current working directory);
                   combines with the auto-generated name or a relative -o.
  --subbox        specify downloaded ranges of subvolume [X0 X1), [Y0 Y1), [Z0 Z1). 
                   If reassembling a full variable, omit this option. 
                   If you downloaded only a selection of shards, must specify the subbox to crop to that selection.
  --mem-gb        GB of Zarr data held in memory at once (default: 2).
  --progress      'auto' | 'bar' | 'lines' | 'none' (default: auto).
  -q, --quiet     suppress all progress and summary output.


Examples:
a) To rebuild download of full variable, use
   python rebuild.py downloaded_folder
   Ex. python rebuild.py 3
   The output file will be automatically saved as `sim_var_level.npy`
   based on attributes in zarr.json file, in the current working directory

b) To rebuild only a subvolume, use Option 1 in the README,
    which will automatically run this function and specify the subbox argument
    based on the chunks you have downloaded
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import zarr


DEFAULT_MEM_GB = 2                          # Zarr data held at once; see _tile_y
DEFAULT_MEM = int(DEFAULT_MEM_GB * (1 << 30))   # the same budget, in bytes

PROGRESS_CHOICES = ("auto", "bar", "lines", "none")


def _blocks(a, b, step):
    """Sub-ranges of [a, b) snapped to the chunk grid (each within one chunk)."""
    lo = a
    while lo < b:
        hi = min((lo // step + 1) * step, b)
        yield lo, hi
        lo = hi


def _tile_y(nz_out, dx, chunk_y, span_y, itemsize, mem_bytes):
    """How many y rows to hold at once: as many whole chunks as `mem_bytes` allows.

    The read tile is always the full output z extent, so each write is one run of
    `dy * nz_out` elements per x-plane rather than a 128-long dribble. When the
    budget covers the whole y span the tile becomes the entire x-slab, which is
    a single contiguous write -- the fastest case, and the usual one for all but
    the largest cases.

    Tiles stay chunk-aligned so no Zarr chunk is decoded twice.
    """
    per_row = dx * nz_out * itemsize
    if per_row <= 0:
        return span_y
    fit = max(1, (mem_bytes // per_row) // chunk_y) * chunk_y
    return min(span_y, fit)


def _progress_mode(progress):
    """Resolve 'auto' to the form that suits where output is going.

    A `\\r` bar is fine on a terminal but not desired in a Slurm log. Batch runs get periodic
    newline-terminated lines instead.
    """
    if progress not in PROGRESS_CHOICES:
        raise ValueError(f"progress must be one of {PROGRESS_CHOICES}, got {progress!r}")
    if progress != "auto":
        return progress
    try:
        return "bar" if sys.stdout.isatty() else "lines"
    except (AttributeError, ValueError):     # detached / closed stdout
        return "lines"


def rebuild(array_path, out_path=None, subbox=None, out_dir=None, verbose=True,
            mem_bytes=DEFAULT_MEM, progress="auto"):
    """Reassemble a local Zarr v3 array at `array_path` into `out_path`.

    `array_path` may live anywhere -- it's the array dir itself (the one holding
    zarr.json and c/), e.g. /some/scratch/R4P50.zarr/r/3.

    `out_path=None` (default) auto-names the file from zarr.json's attributes
    (see `_attrs_out`), falling back to a name derived from `array_path` if
    attributes are missing (see `_default_out`).

    `out_dir` is where the output lands. It's created if needed, and applies to
    auto-named and relative `out_path`s; an absolute `out_path` wins. Passing a
    directory as `out_path` is the same as passing it as `out_dir`. With both
    None the file is written to the current working directory.

    The output is a standard C-order .npy (load with np.load, memmap-friendly).
    That matches the layout of the Zarr chunks it's built from, so the rebuild
    is a straight copy.

    `mem_bytes` caps how much Zarr data is held at once, which sets the y tiling
    (see `_tile_y`). Raising it lengthens each write; it does not change the
    result.

    `progress` controls the running tile counter (`verbose=False` silences it
    either way): 'bar' redraws one line with `\\r`, 'lines' prints a new line
    every ~10% with elapsed time and rate, 'none' prints nothing, and 'auto'
    (the default) picks 'bar' on a terminal and 'lines' otherwise -- so logs
    stay clean without having to remember a flag.

    `subbox=(x0,x1,y0,y1,z0,z1)` crops to that half-open range instead of the
    whole array -- used after a Globus Transfer that pulled only the shards a
    selection touched, so the output is just the selection (not a mostly-zero
    full-size cube). Shards absent on disk read back as the store's fill value.
    """
    array_path = os.path.expanduser(array_path)
    out_path = _resolve_out(array_path, out_path, out_dir)

    za = zarr.open_array(array_path, mode="r")
    full, dtype = tuple(za.shape), za.dtype
    inner = tuple(za.chunks)                 # inner (read) chunk shape

    if subbox is None:
        x0, y0, z0 = 0, 0, 0
        x1, y1, z1 = full
    else:
        x0, x1, y0, y1, z0, z1 = subbox
        for a, b, n, nm in ((x0, x1, full[0], "x"), (y0, y1, full[1], "y"),
                            (z0, z1, full[2], "z")):
            if not (0 <= a < b <= n):
                raise ValueError(f"subbox {nm} range {a}:{b} out of bounds for "
                                 f"array size {n}")
    shape = (x1 - x0, y1 - y0, z1 - z0)

    if verbose:
        print(f"source : {array_path}")
        print(f"shape  : {full}  dtype: {dtype}  chunk: {inner}")
        if subbox is not None:
            print(f"subbox : x[{x0}:{x1}] y[{y0}:{y1}] z[{z0}:{z1}]  -> {shape}")
        print(f"size   : {np.prod(shape) * dtype.itemsize / 1e9:.2f} GB uncompressed")
        print(f"output : {out_path}")

    sx, sy, sz = shape
    itemsize = dtype.itemsize
    row = sy * sz * itemsize                 # bytes per x-plane of the output
    dy = _tile_y(sz, inner[0], inner[1], sy, itemsize, mem_bytes)

    xb = list(_blocks(x0, x1, inner[0]))
    yb = list(_blocks(y0, y1, dy))
    total = len(xb) * len(yb)
    mode = _progress_mode(progress) if verbose else "none"
    tick = max(1, total // (200 if mode == "bar" else 10))
    nbytes = int(np.prod(shape)) * itemsize

    if verbose:
        whole = len(yb) == 1
        print(f"tile   : {inner[0]} x {dy} x {sz}  "
              f"({inner[0] * dy * sz * itemsize / 1e9:.2f} GB per read)"
              + ("  [one contiguous write per x-slab]" if whole else ""))

    # The output is written with ordinary seek/write, not a memmap. 
    # Each read tile spans the full output z extent,
    # so a tile is one contiguous run per x-plane, and when it also spans the
    # full y extent, the whole slab is a single write.
    with open(out_path, "wb+") as f:
        np.lib.format.write_array_header_1_0(
            f, {"descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": False, "shape": shape})
        data_off = f.tell()
        f.truncate(data_off + sx * row)

        done = 0
        t0 = time.time()
        for (a0, a1), (b0, b1) in ((a, b) for a in xb for b in yb):
            buf = np.ascontiguousarray(za[a0:a1, b0:b1, z0:z1])
            if b1 - b0 == sy:                # whole y extent -> one run
                f.seek(data_off + (a0 - x0) * row)
                f.write(buf)
            else:
                for i in range(a1 - a0):
                    f.seek(data_off + (a0 - x0 + i) * row + (b0 - y0) * sz * itemsize)
                    f.write(buf[i])
            done += 1
            if mode != "none" and (done % tick == 0 or done == total):
                pct = 100 * done / total
                if mode == "bar":
                    print(f"\r  {done}/{total} tiles ({pct:5.1f}%)", end="", flush=True)
                else:
                    el = max(time.time() - t0, 1e-6)
                    print(f"  {done}/{total} tiles ({pct:5.1f}%)  {el:.0f} s  "
                          f"{nbytes * done / total / 1e6 / el:.0f} MB/s", flush=True)
        if mode == "bar":
            print()

        # Push the data out before the stat that follows
        f.flush()
        os.fsync(f.fileno())

    if verbose:
        print(f"done   : {os.path.getsize(out_path) / 1e9:.2f} GB written")
    return out_path


def _resolve_out(array_path, out_path, out_dir):
    """Settle on the output file path and make sure its directory exists.

    Auto-names the file when `out_path` is None, drops it in `out_dir` (or the
    cwd), and lets an absolute `out_path` override `out_dir`. An `out_path` that
    is itself an existing directory is treated as `out_dir`.
    """
    out_path = os.path.expanduser(out_path) if out_path else None
    out_dir = os.path.expanduser(out_dir) if out_dir else None

    if out_path and os.path.isdir(out_path):
        out_dir, out_path = out_path, None
    if out_path is None:
        out_path = _attrs_out(array_path) or _default_out(array_path)
    if out_dir:
        out_path = os.path.join(out_dir, out_path)   # absolute out_path wins

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    return out_path


def _attrs_out(array_path):
    """Build <simulation>_<variable>_L<level>.npy from zarr.json's attributes,
    e.g. {"simulation": "R1P1", "variable": "r", "level": 0} -> R1P1_r_L0.npy.
    Returns None if zarr.json is missing/unreadable or attributes are incomplete.
    """
    meta_path = os.path.join(array_path, "zarr.json")
    try:
        with open(meta_path) as f:
            attrs = json.load(f)["attributes"]
        sim, var, level = attrs["simulation"], attrs["variable"], attrs["level"]
    except OSError as e:
        print(f"note: could not read {meta_path} ({e}); "
              f"falling back to path-based output name", file=sys.stderr)
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f"note: {meta_path} missing expected attributes ({e}); "
              f"falling back to path-based output name", file=sys.stderr)
        return None
    return f"{sim}_{var}_L{level}.npy"


def _default_out(array_path):
    """R1P7.zarr/chi/0 -> chi_L0.npy  (var + level, in the cwd)."""
    p = os.path.normpath(array_path).split(os.sep)
    level = p[-1] if p else "0"
    var = p[-2] if len(p) >= 2 else "array"
    return f"{var}_L{level}.npy"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rebuild a local Zarr v3 array (one var/level) into a dense .npy file.")
    ap.add_argument("array_path",
                    help="path to the array dir, in any folder, "
                         "e.g. /scratch/dl/R1P7.zarr/chi/0")
    ap.add_argument("-o", "--output",
                    help="output .npy file, in any folder (default: "
                         "<simulation>_<var>_L<level>.npy from zarr.json attributes, "
                         "falling back to <var>_L<level>.npy if attributes are "
                         "missing). A directory here acts as --outdir")
    ap.add_argument("-d", "--outdir",
                    help="directory to write into, created if needed (default: cwd); "
                         "combines with the auto-generated name, or with a relative "
                         "-o/--output")
    ap.add_argument("--subbox", type=int, nargs=6,
                    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="crop to this half-open range (e.g. after a Globus batch "
                         "transfer of only the selection's shards)")
    ap.add_argument("--mem-gb", type=float, default=DEFAULT_MEM_GB,
                    help="GB of Zarr data held at once; larger means longer writes "
                         "(default: %(default)s)")
    ap.add_argument("--progress", choices=PROGRESS_CHOICES, default="auto",
                    help="running tile counter: 'bar' redraws one line, 'lines' "
                         "prints one every ~10%% (clean in a log file), 'none' is "
                         "silent, 'auto' picks bar on a terminal and lines "
                         "otherwise (default: %(default)s)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="no progress output at all, and none of the summary lines")
    args = ap.parse_args(argv)

    if not os.path.isdir(os.path.expanduser(args.array_path)):
        ap.error(f"not a directory: {args.array_path}")

    try:
        rebuild(args.array_path, args.output,
                subbox=tuple(args.subbox) if args.subbox else None,
                out_dir=args.outdir, verbose=not args.quiet,
                mem_bytes=int(args.mem_gb * (1 << 30)), progress=args.progress)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
