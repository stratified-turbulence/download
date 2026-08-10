"""zarr_app.py — machinery for zarr_download.ipynb.

Keep this file next to the notebook. The notebook is only four short calls:
    import zarr_app
    zarr_app.authenticate(CONFIG)   # step 2
    zarr_app.download_panel()       # step 3
    zarr_app.visualize_panel()      # step 4
Everything else lives here so the notebook stays free of code clutter.
"""

import os, math, itertools, time, threading, re, struct
import numpy as np
import requests
import zarr
import globus_sdk
import ipywidgets as widgets
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d  # noqa: F401  (registers the 3d projection)
from concurrent.futures import ThreadPoolExecutor, as_completed
from IPython.display import display, HTML

# module-level state, populated by configure()/authenticate() below
CONFIG = STORE = CATALOG = TOKEN = None

W = widgets.Layout       # shorthand used by both panel builders

# Baked-in defaults. The notebook's CONFIG cell only overrides the four
# user-facing knobs (CACHE_DIR, SAVE_DIR, MAX_FETCH_GB, DL_WORKERS); everything
# below — the collection coordinates, the OAuth app, the
# Range-fetch tuning, and the inventory to probe — lives here so users don't have
# to touch it. configure() merges the user's dict over these (see configure()).
DEFAULTS = {
    # --- Globus HTTPS endpoint of the mapped collection holding the stores ---
    ## THE FOLLOWING WILL BE UPDATED ONCE THE CONSTELLATION REPOSITORY IS PUBLISHED ##
    "COLLECTION_ID": "",   # collection UUID (Globus > collection > Overview)
    "HTTPS_BASE":    "",  # HTTPS server URL
    "STORE_ROOT":    "",   # dir that CONTAINS the <CASE>.zarr stores

    # --- Globus Native app used only for the OAuth login --------------------
    "CLIENT_ID":     "d47db6dc-0428-4076-9a6e-31927d7c7704",

    # --- local working dirs / limits (user-overridable in the notebook) -----
    "CACHE_DIR":     "./strata_cache",   # fetched shard files land here
    "SAVE_DIR":      "./strata_data",    # reconstructed .npy cubes land here
    "MAX_FETCH_GB":  20,                 # refuse selections larger than this
    "DL_WORKERS":    8,                  # concurrent shard downloads (1 = serial)

    # --- sub-shard HTTP Range fetching --------------------------------------
    "RANGE_FETCH":   "auto",             # "auto" | True | False
    "RANGE_MAX_FILL": 0.5,               # "auto" threshold (fraction of a shard's chunks)

    # --- Globus Transfer command generation (large sub-cubes; user submits) --
    "DEST_ENDPOINT": None,               # user's Globus Connect Personal endpoint UUID
    #  (the transfer lands under CACHE_DIR, which the endpoint must have access to)

    # --- what to probe for (the documented inventory) -----------------------
    "CANDIDATE_CASES": ["R1P1", "R1P7", "R1P50", "R4P1", "R4P7", "R4P50",
                        "R6P1", "R6P7", "R6P50", "R8P1", "R8P7", "R10P1", "R10P7"],
    "CANDIDATE_VARS":  ["u", "v", "w", "r", "ee", "chi"],
}
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
CMAP = "RdBu_r"
VARIABLE_INFO = {                       # descriptive labels for the var dropdown
    "u":   "u  -  velocity, x-component",
    "v":   "v  -  velocity, y-component",
    "w":   "w  -  velocity, z-component",
    "r":   "r  -  buoyancy / density perturbation",
    "ee":  "ee  -  kinetic-energy dissipation rate",
    "chi": "chi  -  scalar dissipation rate",
}

# Dissipation rates span many decades and are strictly positive, so the
# visualize plots them (and their color limits) on a log10 scale.
LOG_VARS = {"ee", "chi"}


def _case_key(case):
    """Natural sort key for 'R<n>P<m>' case names: order by R numerically, then
    by P numerically (so R1P1, R1P7, R1P50, ..., R10P1, R10P7 - not lexical)."""
    m = re.match(r"R(\d+)P(\d+)", case)
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9, case)


# ---------------------------------------------------------------- auth ------
def globus_https_token(cfg):
    scope = f"https://auth.globus.org/scopes/{cfg['COLLECTION_ID']}/https"
    client = globus_sdk.NativeAppAuthClient(cfg["CLIENT_ID"])
    client.oauth2_start_flow(requested_scopes=scope)
    qp = {"prompt": "login"}
    print("Open this URL, log in, and paste the auth code below:\n")
    print(client.oauth2_get_authorize_url(query_params=qp), "\n")
    code = input("Authorisation code: ").strip()
    tok = client.oauth2_exchange_code_for_tokens(code)
    for rs, data in tok.by_resource_server.items():
        if rs != "auth.globus.org":
            return data["access_token"]
    raise RuntimeError("No HTTPS token returned - check COLLECTION_ID.")


# ------------------------------------------------ HTTPS store (GET only) ----
class Store:
    """Read a Zarr v3 archive over Globus HTTPS. GET by explicit path only - the
    collection cannot be directory-listed, so every path is constructed."""

    def __init__(self, cfg, token):
        self.base = cfg["HTTPS_BASE"].rstrip("/") + "/" + cfg["STORE_ROOT"].strip("/")
        self.h = {"Authorization": f"Bearer {token}"}
        self.s = requests.Session()
        pool = cfg.get("DL_WORKERS", 8) + 4     # headroom so concurrent GETs don't
        ad = requests.adapters.HTTPAdapter(     # queue behind a too-small conn pool
            pool_connections=pool, pool_maxsize=pool)
        self.s.mount("https://", ad); self.s.mount("http://", ad)

    def _url(self, rel):
        return f"{self.base}/{rel}?download=1"

    def get_json(self, rel):
        r = self.s.get(self._url(rel), headers=self.h)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def download(self, rel, cache_dir, on_bytes=None):
        r = self.s.get(self._url(rel), headers=self.h, stream=True)
        if r.status_code == 404:
            return False                       # absent shard = all fill_value
        r.raise_for_status()
        dst = os.path.join(cache_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"                    # write to a temp name and rename on
        try:                                   # completion, so an interrupted transfer
            with open(tmp, "wb") as f:         # (e.g. Kernel > Interrupt) can never
                for chunk in r.iter_content(1 << 20):  # masquerade as a cached shard
                    f.write(chunk)
                    if on_bytes:               # feed the live rate/ETA meter
                        on_bytes(len(chunk))
        except BaseException:                  # interrupt/error: drop the partial so
            r.close()                          # nothing stale is left behind
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, dst)
        return True

    def file_size(self, rel):
        """Total size of a remote file in bytes (or None on 404). Tries HEAD, then a
        1-byte ranged GET's Content-Range total, so it works even where HEAD is
        unsupported. Used to turn a suffix range into an explicit start-end range:
        the Globus HTTPS server rejects the bytes=-N suffix form with HTTP 416."""
        r = self.s.head(self._url(rel), headers=self.h, allow_redirects=True)
        if r.status_code == 404:
            return None
        cl = r.headers.get("Content-Length")
        if r.ok and cl is not None:
            return int(cl)
        r = self.s.get(self._url(rel), headers={**self.h, "Range": "bytes=0-0"},
                       stream=True)
        try:
            if r.status_code == 404:
                return None
            r.raise_for_status()
            cr = r.headers.get("Content-Range", "")      # e.g. "bytes 0-0/1943011460"
            tail = cr.rsplit("/", 1)[-1] if "/" in cr else ""
            if tail.isdigit():
                return int(tail)
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None else None
        finally:
            r.close()

    def get_range(self, rel, suffix=None, start=None, length=None, on_bytes=None):
        """HTTP Range GET of [start, start+length); suffix=N asks for the last N
        bytes. Returns bytes (or None on 404). A suffix is resolved to an explicit
        start-end range via file_size() because the Globus HTTPS server rejects the
        bytes=-N suffix form with HTTP 416. Raises if the server ignores Range (HTTP
        200) so a mis-ranged request can never silently pull a whole 1.81 GB shard."""
        if suffix is not None:
            size = self.file_size(rel)
            if size is None:
                return None                              # 404 -> absent shard
            start = max(0, size - suffix); length = size - start
        h = dict(self.h)
        h["Range"] = f"bytes={start}-{start + length - 1}"
        r = self.s.get(self._url(rel), headers=h, stream=True)
        try:
            if r.status_code == 404:
                return None
            r.raise_for_status()
            if r.status_code != 206:
                raise RuntimeError(
                    f"Range not honoured for {rel} (HTTP {r.status_code}); set "
                    "CONFIG['RANGE_FETCH']=False to force whole-shard downloads.")
            buf = bytearray()
            for chunk in r.iter_content(1 << 20):
                buf += chunk
                if on_bytes:
                    on_bytes(len(chunk))
            return bytes(buf)
        finally:
            r.close()


# ------------------------------------------------- metadata parsing ---------
def _shard(meta):                              # outer chunk = one shard file
    return tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])

def _inner(meta):                              # inner (read-granularity) chunk
    for c in meta.get("codecs", []):
        if c.get("name") == "sharding_indexed":
            return tuple(c["configuration"]["chunk_shape"])
    return _shard(meta)                         # unsharded level: file == chunk

def _level_paths(group_meta):
    ms = group_meta.get("attributes", {}).get("multiscales")
    return [d["path"] for d in ms[0]["datasets"]] if ms else None


# -------------------------------------------------------- discovery ---------
def discover(store, cases, vars_):
    """Probe candidate cases/vars, read their zarr.json, and return
    CATALOG[case] = {tstamp, vars:{var:{levels:{lp:{shape,shard,inner,dtype}}}}}."""
    catalog = {}
    for case in cases:
        root = store.get_json(f"{case}.zarr/zarr.json")
        if root is None:
            continue
        vinfo = {}
        for var in vars_:
            grp = store.get_json(f"{case}.zarr/{var}/zarr.json")
            if grp is None:
                continue
            lps = _level_paths(grp)
            if lps is None:                     # fallback: probe 0, 1, 2, ...
                lps, L = [], 0
                while store.get_json(f"{case}.zarr/{var}/{L}/zarr.json"):
                    lps.append(str(L)); L += 1
            linfo = {}
            for lp in lps:
                am = store.get_json(f"{case}.zarr/{var}/{lp}/zarr.json")
                if am is None:
                    continue
                linfo[lp] = dict(shape=tuple(am["shape"]), shard=_shard(am),
                                 inner=_inner(am), dtype=am["data_type"])
            if linfo:
                vinfo[var] = dict(levels=linfo)
        if vinfo:
            catalog[case] = dict(
                tstamp=root.get("attributes", {}).get("tstamp"), vars=vinfo)
    return catalog


# --------------------------------------------- fetch / estimate / save ------
def estimate(shard, sel):
    """(n_shards, transfer_MB, reconstructed_MB) for a selection. No compression
    -> transfer = full shards touched, reconstructed = exactly the box."""
    x0, x1, y0, y1, z0, z1 = sel
    sx, sy, sz = shard
    n = ((math.ceil(x1 / sx) - x0 // sx) *
         (math.ceil(y1 / sy) - y0 // sy) *
         (math.ceil(z1 / sz) - z0 // sz))
    return n, n * sx * sy * sz * 4 / 1e6, (x1 - x0) * (y1 - y0) * (z1 - z0) * 4 / 1e6


def fetch_subbox(store, cache_dir, case, var, level, meta, sel,
                 on_start=None, on_file=None, on_bytes=None, workers=None):
    """Fetch the metadata + the shard files this selection touches into
    cache_dir (skip cached shards, always refresh metadata). Shard GETs run
    concurrently across `workers` threads: a single long-haul HTTPS stream is
    BDP-limited to a few MB/s, so N parallel streams multiply aggregate
    throughput. `on_bytes(n)` (if given) is called from the worker threads as
    bytes land, for a live rate/ETA meter. Returns the local array path."""
    ax = f"{case}.zarr/{var}/{level}"
    metas = [f"{case}.zarr/zarr.json", f"{case}.zarr/{var}/zarr.json", f"{ax}/zarr.json"]
    sx, sy, sz = meta["shard"]
    x0, x1, y0, y1, z0, z1 = sel
    data = [f"{ax}/c/{i}/{j}/{k}"
            for i in range(x0 // sx, math.ceil(x1 / sx))
            for j in range(y0 // sy, math.ceil(y1 / sy))
            for k in range(z0 // sz, math.ceil(z1 / sz))]
    need = [r for r in data if not os.path.exists(os.path.join(cache_dir, r))]
    if on_start:
        on_start(len(need))
    for rel in metas:                          # tiny JSON; fetch serially
        store.download(rel, cache_dir)
    if workers is None:
        workers = max(1, int(CONFIG.get("DL_WORKERS", 8)))
    if need:
        with ThreadPoolExecutor(max_workers=min(workers, len(need))) as ex:
            futs = [ex.submit(store.download, rel, cache_dir, on_bytes) for rel in need]
            for fut in as_completed(futs):
                fut.result()                   # surface any HTTP error, don't swallow
                if on_file:                    # per-shard tick (calling thread)
                    on_file()
    return os.path.join(cache_dir, ax)


# ------------------------------------ sub-shard HTTP Range fetching ---------
# A 768^3 shard is one 1.81 GB file, but a thin/small selection may touch only a
# few of its 6x6x6=216 inner 128^3 chunks. Each shard carries an index at its END:
# a (offset,length) uint64 pair per inner chunk in C-order, then a crc32c. We Range-
# GET just that index (~KB), parse it, then Range-GET only the inner chunks the
# selection needs and drop them straight into the output .npy. The inner codec is an
# uncompressed "bytes" codec, so a chunk is raw little-endian float32 (np.frombuffer,
# no decode). Absent shards (404) and MAX_UINT64 index entries are all-fill and are
# left as zeros (fill_value is 0.0).
_IDX_SENTINEL = 0xFFFFFFFFFFFFFFFF

def _grid(shard, inner):                        # inner chunks per shard, per dim
    return tuple(shard[d] // inner[d] for d in range(3))

def _index_size(shard, inner):                  # trailing index length, bytes
    gx, gy, gz = _grid(shard, inner)
    return gx * gy * gz * 16 + 4                 # 16 B/entry + 4 B crc32c

def parse_shard_index(idx_bytes, grid):
    """Decode a shard's end-of-file index -> {(ix,iy,iz): (offset,length)} for the
    present inner chunks; all-fill chunks (MAX_UINT64 sentinel) are omitted."""
    gx, gy, gz = grid
    ents = {}
    for c in range(gx * gy * gz):
        off, ln = struct.unpack_from("<QQ", idx_bytes, c * 16)
        if off == _IDX_SENTINEL:
            continue
        iz = c % gz; iy = (c // gz) % gy; ix = c // (gy * gz)   # C-order over the grid
        ents[(ix, iy, iz)] = (off, ln)
    return ents

def _touched_shards(shard, sel):
    x0, x1, y0, y1, z0, z1 = sel; sx, sy, sz = shard
    return [(i, j, k)
            for i in range(x0 // sx, math.ceil(x1 / sx))
            for j in range(y0 // sy, math.ceil(y1 / sy))
            for k in range(z0 // sz, math.ceil(z1 / sz))]

def _needed_chunks(inner, sel):
    """Inner chunks the selection overlaps (upper bound; treats all as present)."""
    x0, x1, y0, y1, z0, z1 = sel
    return ((math.ceil(x1 / inner[0]) - x0 // inner[0]) *
            (math.ceil(y1 / inner[1]) - y0 // inner[1]) *
            (math.ceil(z1 / inner[2]) - z0 // inner[2]))

def estimate_ranged(shard, inner, sel):
    """(n_chunks, transfer_MB) for a sub-shard Range fetch: needed inner chunks plus
    one index read per touched shard. Upper bound (sparse shards transfer less)."""
    need = _needed_chunks(inner, sel)
    cb = inner[0] * inner[1] * inner[2] * 4
    idx = _index_size(shard, inner) * len(_touched_shards(shard, sel))
    return need, (need * cb + idx) / 1e6

def use_ranged(cfg, shard, inner, sel):
    """Sub-shard Range vs whole-shard GET. "auto" (default) picks Range when the
    selection fills less than RANGE_MAX_FILL of the shards it touches; sub-shard has
    a small per-chunk round-trip cost, so a near-full box is cheaper whole."""
    mode = cfg.get("RANGE_FETCH", "auto")
    if mode is True or mode is False:
        return mode
    gx, gy, gz = _grid(shard, inner)
    cps = gx * gy * gz                          # inner chunks in a full shard
    nsh = len(_touched_shards(shard, sel))
    return _needed_chunks(inner, sel) < cfg.get("RANGE_MAX_FILL", 0.5) * nsh * cps

def globus_transfer_commands(cfg, case, var, level, meta, sel):
    """Build the copy-paste terminal commands for pulling **this selection** via the
    **Globus Transfer service** instead of the in-notebook HTTPS path. Nothing is
    submitted here -- the user runs it.

    Transfer gives byte-level restart/retry, integrity checks, and parallel GridFTP
    streams (can beat our single-stream WAN ceiling) and runs server-side, so it's
    the robust path for *large* sub-cubes. It moves only the shard files `sel`
    touches (+ the level's zarr.json) via `--batch`, mirroring the store; the
    matching `rebuild.py --subbox` crops back to exactly the selection. To pull a
    whole array, tick "Download entire box" so `sel` spans the full domain -- then
    the batch covers every shard and the rebuild needs no crop. Transfer is
    whole-file granularity, so the pull is shard-aligned (min unit = one shard).

    Returns a dict:
      download    -- the `globus transfer` command (paste-only, no comments)
      reconstruct -- the `python rebuild.py` command (paste-only, no comments)
      folder, sel, shards, full, dest_set -- info for the surrounding UI text
    """
    src = cfg["COLLECTION_ID"]
    root = cfg["STORE_ROOT"].strip("/")
    ax = f"{case}.zarr/{var}/{level}"
    src_dir = f"/{root}/{ax}/"                       # source array folder on the collection
    dst_ep = cfg.get("DEST_ENDPOINT") or "<YOUR_GCP_ENDPOINT_UUID>"
    # The transfer lands under CACHE_DIR -- the user's Globus Connect Personal
    # endpoint must have access to that path (set CACHE_DIR to an absolute path).
    dst_root = cfg["CACHE_DIR"].rstrip("/")
    folder = f"{case}_{var}_L{level}"
    dst_dir = f"{dst_root}/{folder}/"

    shards = _touched_shards(meta["shard"], sel)
    x0, x1, y0, y1, z0, z1 = sel
    nx, ny, nz = meta["shape"]
    full = (x0 == 0 and y0 == 0 and z0 == 0 and x1 == nx and y1 == ny and z1 == nz)
    batch = "\n".join(["zarr.json zarr.json"] +
                      [f"c/{i}/{j}/{k} c/{i}/{j}/{k}" for i, j, k in shards])

    # Step 1 -- pure `globus transfer` command, exactly what to paste.
    download = "\n".join([
        "globus transfer \\",
        f"  {src}:{src_dir} \\",
        f"  {dst_ep}:{dst_dir} \\",
        f"  --label {folder} \\",
        "  --batch - <<'EOF'",
        batch,
        "EOF",
    ])

    # Step 2 -- pure `rebuild.py` command, exactly what to paste.
    if full:
        reconstruct = f"python rebuild.py {dst_root}/{folder}"
    else:
        reconstruct = (f"python rebuild.py {dst_root}/{folder} \\\n"
                       f"  --subbox {x0} {x1} {y0} {y1} {z0} {z1}")

    return {
        "download": download,
        "reconstruct": reconstruct,
        "folder": folder,
        "sel": (x0, x1, y0, y1, z0, z1),
        "shards": len(shards),
        "shard_shape": meta["shard"],
        "full": full,
        # Destination path is always CACHE_DIR, so only the endpoint can be missing.
        "dest_set": bool(cfg.get("DEST_ENDPOINT")),
    }


def fetch_subbox_ranged(store, cache_dir, save_dir, case, var, level, meta, sel,
                        on_start=None, on_block=None, on_bytes=None, workers=None):
    """Fetch only the inner chunks `sel` touches (via each shard's end-of-file index)
    and write them straight into the output .npy memmap, so peak RAM is ~a handful of
    chunks. Shards already whole-cached under cache_dir are read locally instead of
    over HTTP. Returns (path, fname, size_MB) like save_subbox."""
    ax = f"{case}.zarr/{var}/{level}"
    shard, inner = meta["shard"], meta["inner"]
    grid = _grid(shard, inner)
    idx_size = _index_size(shard, inner)
    if workers is None:
        workers = max(1, int(CONFIG.get("DL_WORKERS", 8)))
    x0, x1, y0, y1, z0, z1 = sel
    for rel in (f"{case}.zarr/zarr.json", f"{case}.zarr/{var}/zarr.json",
                f"{ax}/zarr.json"):             # keep metadata cached (tiny, cheap)
        store.download(rel, cache_dir)

    fname = f"{case}_{var}_L{level}_x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}.npy"
    path = os.path.join(save_dir, fname)
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(meta["dtype"]),
                                    shape=(x1 - x0, y1 - y0, z1 - z0))
    # sparse zero-filled file; fill_value is 0.0, so any absent chunk stays correct.

    def _place(buf, gx0, gy0, gz0):             # drop one inner chunk into the output
        blk = np.frombuffer(buf, "<f4").reshape(inner)
        bx0 = max(gx0, x0); bx1 = min(gx0 + inner[0], x1)
        by0 = max(gy0, y0); by1 = min(gy0 + inner[1], y1)
        bz0 = max(gz0, z0); bz1 = min(gz0 + inner[2], z1)
        if bx0 >= bx1 or by0 >= by1 or bz0 >= bz1:
            return
        out[bx0 - x0:bx1 - x0, by0 - y0:by1 - y0, bz0 - z0:bz1 - z0] = \
            blk[bx0 - gx0:bx1 - gx0, by0 - gy0:by1 - gy0, bz0 - gz0:bz1 - gz0]

    # -- phase 1: fetch each touched shard's index (local cache or Range GET) --
    shards = _touched_shards(shard, sel)
    def _idx(shsk):
        si, sj, sk = shsk
        rel = f"{ax}/c/{si}/{sj}/{sk}"
        fp = os.path.join(cache_dir, rel)
        if os.path.exists(fp):                  # reuse a prior whole-shard download
            with open(fp, "rb") as fh:
                fh.seek(os.path.getsize(fp) - idx_size)
                return shsk, "local", fp, fh.read(idx_size)
        return shsk, "http", rel, store.get_range(rel, suffix=idx_size)
    with ThreadPoolExecutor(max_workers=min(workers, len(shards))) as ex:
        idx_res = list(ex.map(_idx, shards))

    gi0, gj0, gk0 = x0 // inner[0], y0 // inner[1], z0 // inner[2]
    gi1 = math.ceil(x1 / inner[0]); gj1 = math.ceil(y1 / inner[1]); gk1 = math.ceil(z1 / inner[2])
    http, local, n_idx = [], [], 0
    for (si, sj, sk), src, ref, idx in idx_res:
        if idx is None:                         # 404 -> absent shard, all fill
            continue
        if src == "http":
            n_idx += idx_size
        ents = parse_shard_index(idx, grid)
        bx, by, bz = si * grid[0], sj * grid[1], sk * grid[2]
        for gi in range(max(gi0, bx), min(gi1, bx + grid[0])):
            for gj in range(max(gj0, by), min(gj1, by + grid[1])):
                for gk in range(max(gk0, bz), min(gk1, bz + grid[2])):
                    ent = ents.get((gi - bx, gj - by, gk - bz))
                    if ent is None:             # sentinel -> all-fill, leave as 0
                        continue
                    off, ln = ent
                    coords = (gi * inner[0], gj * inner[1], gk * inner[2])
                    (local if src == "local" else http).append((ref, off, ln, coords))

    if on_start:
        on_start(len(http) + len(local))
    if on_bytes and n_idx:
        on_bytes(n_idx)                         # account the index reads

    # -- phase 1b: local reads from already-cached shards (main thread) --
    for fp, off, ln, (gx0, gy0, gz0) in local:
        with open(fp, "rb") as fh:
            fh.seek(off); _place(fh.read(ln), gx0, gy0, gz0)
        if on_block:
            on_block()

    # -- phase 2: HTTP Range GETs in parallel; decode/place on the main thread --
    def _get(task):
        rel, off, ln, coords = task
        return store.get_range(rel, start=off, length=ln, on_bytes=on_bytes), coords
    if http:
        with ThreadPoolExecutor(max_workers=min(workers, len(http))) as ex:
            for fut in as_completed([ex.submit(_get, t) for t in http]):
                buf, (gx0, gy0, gz0) = fut.result()
                _place(buf, gx0, gy0, gz0)
                if on_block:
                    on_block()
    out.flush(); del out
    return path, fname, os.path.getsize(path) / 1e6


def _blocks(a, b, step):
    """Sub-ranges of [a, b) snapped to the chunk grid (each within one chunk)."""
    lo = a
    while lo < b:
        hi = min((lo // step + 1) * step, b)
        yield lo, hi
        lo = hi


def save_subbox(array_path, save_dir, case, var, level, inner, sel,
                on_start=None, on_block=None):
    """Reassemble the selection to a .npy, block by block via a disk memmap, so
    peak RAM is ~one inner chunk rather than the whole cube."""
    za = zarr.open_array(array_path, mode="r")
    x0, x1, y0, y1, z0, z1 = sel
    fname = f"{case}_{var}_L{level}_x{x0}-{x1}_y{y0}-{y1}_z{z0}-{z1}.npy"
    path = os.path.join(save_dir, fname)
    xb = list(_blocks(x0, x1, inner[0]))
    yb = list(_blocks(y0, y1, inner[1]))
    zb = list(_blocks(z0, z1, inner[2]))
    if on_start:
        on_start(len(xb) * len(yb) * len(zb))
    out = np.lib.format.open_memmap(path, mode="w+", dtype=za.dtype,
                                    shape=(x1 - x0, y1 - y0, z1 - z0))
    try:
        for a0, a1 in xb:
            for b0, b1 in yb:
                for c0, c1 in zb:
                    out[a0 - x0:a1 - x0, b0 - y0:b1 - y0, c0 - z0:c1 - z0] = \
                        za[a0:a1, b0:b1, c0:c1]
                    if on_block:
                        on_block()
    finally:
        out.flush()
        del out
    return path, fname, os.path.getsize(path) / 1e6


# small shared 3d-box helper (used by the download & visualize wireframes)
_EDGES = [(0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7),
          (0, 1), (2, 3), (4, 5), (6, 7)]

def _draw_box(ax, x0, x1, y0, y1, z0, z1, **kw):
    pts = np.array(list(itertools.product([x0, x1], [y0, y1], [z0, z1])))
    for a, b in _EDGES:
        ax.plot(*zip(pts[a], pts[b]), **kw)


# --------------------------------------------------------- catalog ----------
def load_catalog():
    """Load the FROZEN, hardcoded catalog captured by build_catalog.py -- no
    network. The archive is being frozen for storage, so the notebook no longer
    live-probes the collection; a maintainer runs `python build_catalog.py` to
    refresh store_catalog.py. Returns (catalog, pending, generated) or
    (None, [], None) if the frozen file hasn't been generated yet."""
    try:
        import importlib, store_catalog
        importlib.reload(store_catalog)      # pick up a freshly regenerated file
        return (store_catalog.CATALOG,
                getattr(store_catalog, "PENDING", []),
                getattr(store_catalog, "GENERATED", None))
    except Exception:
        return None, [], None


# --------------------------------------------------------- run it -----------
def _authenticate():
    global TOKEN, STORE, CATALOG
    TOKEN = globus_https_token(CONFIG)
    STORE = Store(CONFIG, TOKEN)
    CATALOG, pending, generated = load_catalog()
    if CATALOG is None:
        print("\nAuthenticated, but no frozen catalog was found "
              "(app/store_catalog.py).\nA maintainer must generate it once:  "
              "python build_catalog.py")
        return
    print("Authenticated")


def _open_download_panel():
    if CATALOG is None:
        print("Run Step 2 (Authenticate & discover) and click its button first.")
        return

    def _meta():
        return CATALOG[dl_case.value]["vars"][dl_var.value]["levels"][dl_level.value]

    def _sel():
        m = _meta(); out = []
        for w, n in zip((dl_x, dl_y, dl_z), m["shape"]):
            a, b = w.value
            if b <= a:                                   # zero-width -> single plane
                a = min(a, n - 1); b = a + 1
            out += [a, b]
        return out

    def _ordinal(n):                                     # 2 -> "2nd", 4 -> "4th", ...
        if 11 <= n % 100 <= 13:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    dl_case  = widgets.Dropdown(description="Case:",  options=sorted(CATALOG, key=_case_key),
                                layout=W(width="200px"))
    dl_var   = widgets.Dropdown(description="Variable:", layout=W(width="300px"))
    dl_level = widgets.Dropdown(description="Level:", layout=W(width="300px"))
    dl_x = widgets.IntRangeSlider(description="x", continuous_update=False, layout=W(width="640px"))
    dl_y = widgets.IntRangeSlider(description="y", continuous_update=False, layout=W(width="640px"))
    dl_z = widgets.IntRangeSlider(description="z", continuous_update=False, layout=W(width="640px"))
    dl_full = widgets.Checkbox(value=False, description="Download entire box (full range)",
                               indent=False, layout=W(width="320px"))
    dl_est  = widgets.HTML()
    dl_wire = widgets.Output()
    _BTN_W = "300px"                        # both action buttons share this width
    dl_go   = widgets.Button(description="Download live (smaller jobs)", button_style="primary",
                             icon="download", layout=W(width=_BTN_W, height="auto"))
    dl_cli  = widgets.Button(description="Terminal command to download via Globus CLI (bigger jobs)",
                             icon="terminal", layout=W(width=_BTN_W, height="auto"),
                             tooltip="Print a copy-paste Globus Transfer command for this selection "
                                     "(robust path for large sub-cubes; you submit it)")
    for _b in (dl_go, dl_cli):              # allow the label to wrap instead of clipping
        _b.add_class("zarr-wrap-btn")
    dl_out  = widgets.Output()
    dl_cli_out = widgets.Output()

    def _fill_vars(*_):
        avail = CATALOG[dl_case.value]["vars"]
        order = list(VARIABLE_INFO)                  # u,v,w,r,ee,chi
        vs = [v for v in order if v in avail] + [v for v in sorted(avail) if v not in order]
        dl_var.options = [(VARIABLE_INFO.get(v, v), v) for v in vs]
        dl_var.value = vs[0] if vs else None         # ipywidgets>=8 won't auto-select
                                                     # when options are set post-construction
    def _fill_levels(*_):
        if not dl_var.value:
            return
        levels = CATALOG[dl_case.value]["vars"][dl_var.value]["levels"]
        lv = sorted(levels, key=int)
        base = levels[lv[0]]["shape"][0]              # finest level -> downsample factor 1
        opts = []                                     # label each level by its stride
        for l in lv:
            f = max(1, round(base / levels[l]["shape"][0]))
            opts.append((f"{l} (full resolution)" if f == 1
                         else f"{l} (every {_ordinal(f)} point)", l))
        dl_level.options = opts                       # (label, value): .value stays the
        dl_level.value = lv[0] if lv else None        # raw level string used as the path

    def _fill_ranges(*_):
        if not dl_level.value:
            return
        m = _meta()
        for w, n, step in zip((dl_x, dl_y, dl_z), m["shape"], m["inner"]):
            w.min = 0; w.max = n; w.step = step
            w.disabled = dl_full.value
            w.value = [0, n] if dl_full.value else [0, min(n, step * 4)]
        _update_preview()

    def _draw_wire(shape, sel):
        nx, ny, nz = shape
        x0, x1, y0, y1, z0, z1 = sel
        with dl_wire:
            dl_wire.clear_output(wait=True)
            fig = plt.figure(figsize=(5, 4)); ax = fig.add_subplot(111, projection="3d")
            _draw_box(ax, 0, nx, 0, ny, 0, nz, color="0.65", lw=0.8)
            _draw_box(ax, x0, x1, y0, y1, z0, z1, color="tab:red", lw=2.5)
            ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
            ax.set_box_aspect((nx, ny, nz))
            ax.set(xlabel="x", ylabel="y", zlabel="z")
            ax.set_title("Selection within full domain", fontsize=10)
            ax.view_init(elev=22, azim=-58); plt.show(); plt.close(fig)

    def _fmt_size(mb):                                    # MB, switching to GB past 1000 MB
        return f"{mb / 1024:.2f} GB" if mb > 1000 else f"{mb:.0f} MB"

    def _update_preview(*_):
        if not dl_level.value:
            return
        m = _meta(); sel = _sel()
        if use_ranged(CONFIG, m["shard"], m["inner"], sel):
            _, xfer = estimate_ranged(m["shard"], m["inner"], sel)   # MB actually fetched
        else:
            _, xfer, _ = estimate(m["shard"], sel)
        dl_est.value = f"<b>Transfer size of subcube:</b> {_fmt_size(xfer)}"
        _draw_wire(m["shape"], sel)


    def _on_full(ch):
        if not dl_level.value:
            return
        full = ch["new"]; m = _meta()
        for w, n in zip((dl_x, dl_y, dl_z), m["shape"]):
            w.disabled = full
            if full:
                w.value = [0, n]
        _update_preview()

    def _on_case(*_):                                # every case exposes the same vars &
        _fill_vars()                                 # level names, so switching case leaves
        _fill_levels()                               # dl_var/dl_level .value unchanged and
        _fill_ranges()                               # the observer cascade below never fires
                                                     # -> reset the sliders (min/max/step) and
                                                     # redraw the wireframe explicitly here.

    dl_case.observe(_on_case, "value")
    dl_var.observe(_fill_levels, "value")
    dl_level.observe(_fill_ranges, "value")
    dl_full.observe(_on_full, "value")
    for w in (dl_x, dl_y, dl_z):
        w.observe(_update_preview, "value")

    def _on_go(_):
        # Synchronous on the kernel main thread. Abort a long run with Kernel > Interrupt:
        # whole-shard GETs write .part -> rename so completed shards stay cached; a ranged
        # run's partial .npy is simply discarded and rebuilt on the next attempt.
        dl_go.disabled = True
        try:
            m, sel = _meta(), _sel()
            ranged = use_ranged(CONFIG, m["shard"], m["inner"], sel)
            x0, x1, y0, y1, z0, z1 = sel
            data = (x1 - x0) * (y1 - y0) * (z1 - z0) * 4 / 1e6      # reconstructed cube MB
            if data / 1024 > CONFIG["MAX_FETCH_GB"]:
                with dl_out:
                    dl_out.clear_output(wait=True)
                    display(HTML(f"<span style='color:#b00'><b>Selection too large:</b> "
                                 f"{data/1024:.1f} GB exceeds MAX_FETCH_GB="
                                 f"{CONFIG['MAX_FETCH_GB']}. Narrow it or pick a coarser "
                                 f"level.</span>"))
                return
            prog = widgets.FloatProgress(min=0, max=1, layout=W(width="420px"))
            lab  = widgets.HTML()
            with dl_out:
                dl_out.clear_output(wait=True); display(widgets.VBox([lab, prog]))

            def _fmt_eta(s):
                s = int(max(s, 0)); return f"{s // 60}:{s % 60:02d}"

            if ranged:
                nchunk, xfer_mb = estimate_ranged(m["shard"], m["inner"], sel)
                total = max(xfer_mb * 1e6, 1.0)         # upper-bound bytes for the bar/ETA
                prog.max = total
                dl = {"done": 0, "t0": 0.0, "last": 0.0, "lock": threading.Lock()}
                lab.value = (f"Reading shard index(es), then fetching up to {nchunk} inner "
                             f"chunk(s) (~{xfer_mb/1024:.2f} GB) via HTTP Range ...")

                def r_start(k):                          # index phase done; k chunks to fetch
                    with dl["lock"]:
                        dl["t0"] = time.time(); dl["last"] = 0.0
                    lab.value = (f"Fetching {k} inner chunk(s) (~{xfer_mb/1024:.2f} GB) via "
                                 f"HTTP Range ..." if k else "Nothing to fetch (all fill).")

                def on_bytes(nb):                        # from worker threads; throttled ~0.4s
                    now = time.time()
                    with dl["lock"]:
                        dl["done"] += nb
                        if dl["t0"] == 0.0:              # index bytes counted before fetch
                            return
                        if now - dl["last"] < 0.4 and dl["done"] < total:
                            return
                        dl["last"] = now; done = dl["done"]; el = now - dl["t0"]
                    prog.value = min(done, prog.max)
                    base = f"Fetching ... {done/1e6:,.0f}/{total/1e6:,.0f} MB"
                    if el < 0.5:
                        lab.value = base + "&nbsp;&middot;&nbsp; measuring rate ..."; return
                    rate = done / el; eta = (total - done) / rate if rate > 0 else 0.0
                    lab.value = (f"{base}&nbsp;&middot;&nbsp; {rate/1e6:.1f} MB/s"
                                 f"&nbsp;&middot;&nbsp; ETA {_fmt_eta(eta)}")

                path, fname, size = fetch_subbox_ranged(
                    STORE, CONFIG["CACHE_DIR"], CONFIG["SAVE_DIR"], dl_case.value,
                    dl_var.value, dl_level.value, m, sel, on_start=r_start, on_bytes=on_bytes)
                prog.value = prog.max
            else:
                sx, sy, sz = m["shard"]
                shard_bytes = sx * sy * sz * 4              # full-density shard (upper bound)
                st = {"done": 0.0, "total": 0.0}            # reused by the reconstruction phase
                dl = {"done": 0, "total": 0, "t0": 0.0, "last": 0.0, "lock": threading.Lock()}

                def s1(k):                                  # download starting: k shards to fetch
                    with dl["lock"]:
                        dl["done"] = 0; dl["total"] = k * shard_bytes
                        dl["t0"] = time.time(); dl["last"] = 0.0
                    prog.max = max(dl["total"], 1); prog.value = 0
                    lab.value = (f"Downloading {k} shard(s) (~{k*shard_bytes/1e9:.2f} GB) over "
                                 f"Globus HTTPS ..." if k else "All shards already cached.")

                def on_bytes(nb):                           # from worker threads; throttled ~0.4s
                    now = time.time()
                    with dl["lock"]:
                        dl["done"] += nb
                        if now - dl["last"] < 0.4 and dl["done"] < dl["total"]:
                            return
                        dl["last"] = now
                        done, total, el = dl["done"], dl["total"], now - dl["t0"]
                    prog.value = min(done, prog.max)
                    base = f"Downloading ... {done/1e6:,.0f}/{total/1e6:,.0f} MB"
                    if el < 0.5:                             # too early for a meaningful rate
                        lab.value = base + "&nbsp;&middot;&nbsp; measuring rate ..."
                        return
                    rate = done / el                         # average rate -> stable ETA
                    eta = (total - done) / rate if rate > 0 else 0.0
                    lab.value = (f"{base}&nbsp;&middot;&nbsp; {rate/1e6:.1f} MB/s"
                                 f"&nbsp;&middot;&nbsp; ETA {_fmt_eta(eta)}")

                ap = fetch_subbox(STORE, CONFIG["CACHE_DIR"], dl_case.value, dl_var.value,
                                  dl_level.value, m, sel, on_start=s1, on_bytes=on_bytes)

                def s2(k):
                    st["done"] = 0; st["total"] = max(k, 1); prog.max = st["total"]; prog.value = 0
                    lab.value = f"Reconstructing & saving ... 0/{k} blocks"

                def b2():
                    st["done"] += 1; prog.value = st["done"]
                    lab.value = (f"Reconstructing & saving ... "
                                 f"{int(st['done'])}/{int(st['total'])} blocks")

                path, fname, size = save_subbox(ap, CONFIG["SAVE_DIR"], dl_case.value,
                                                dl_var.value, dl_level.value, m["inner"],
                                                sel, on_start=s2, on_block=b2)
            with dl_out:
                dl_out.clear_output(wait=True)
                display(HTML(f"<b>Saved</b> &nbsp;<code>{fname}</code>&nbsp; ({size:.1f} MB)"))
        finally:
            dl_go.disabled = False


    def _on_cli(_):
        # Pure string generation (no network) -- turn the current wirebox selection
        # into copy-paste Globus Transfer commands for large sub-cubes.
        m, sel = _meta(), _sel()
        r = globus_transfer_commands(CONFIG, dl_case.value, dl_var.value,
                                     dl_level.value, m, sel)
        x0, x1, y0, y1, z0, z1 = r["sel"]

        def _box(text):                              # a paste-only terminal code box
            n = text.count("\n") + 1
            h = f"{min(max(n, 3), 16) * 20 + 16}px"
            return widgets.Textarea(value=text, layout=W(width="100%", height=h))

        note = ("" if r["dest_set"] else
                "<div style='color:#b00'><b>Set <code>DEST_ENDPOINT</code> in the "
                "cell&nbsp;1 CONFIG</b> (your Globus Connect Personal endpoint UUID) "
                "to fill the <code>&lt;...&gt;</code> placeholder below. The transfer "
                "lands under <code>CACHE_DIR</code>, which that endpoint must be able "
                "to access.</div>")

        with dl_cli_out:
            dl_cli_out.clear_output(wait=True)
            display(widgets.VBox([
                widgets.HTML(
                    "<b>Terminal commands for Globus transfer outside of notebook.</b><br>"
                    f"<code>{r['folder']}</code> &nbsp; "
                    f"x[{x0}:{x1}] y[{y0}:{y1}] z[{z0}:{z1}]"
                    + ("  (entire box)" if r["full"] else "") + "<br>"
                    + note),
                widgets.HTML("<b>Step 1 &mdash; Download.</b> Paste this into a "
                             "terminal to submit the transfer. It prints a Task ID; "
                             "track it with <code>globus task show &lt;TASK_ID&gt;</code> "
                             "or by using the web interface: "
                             "<a href='https://app.globus.org/activity/' target='_blank'>"
                             "https://app.globus.org/activity/</a>"),
                _box(r["download"]),
                widgets.HTML("<b>Step 2 &mdash; Reconstruct.</b> Once the transfer "
                             "finishes, navigate to SAVE_DIR and paste this into the terminal to reassemble the downloaded chunks into "
                             f"<code>{r['folder']}.npy</code>."),
                _box(r["reconstruct"]),
            ]))

    dl_go.on_click(_on_go)
    dl_cli.on_click(_on_cli)
    _fill_vars(); _fill_levels(); _fill_ranges()

    display(widgets.VBox([
        widgets.HTML("<style>.zarr-wrap-btn{white-space:normal !important;"
                     "height:auto !important;min-height:32px;line-height:1.25;"
                     "padding-top:5px;padding-bottom:5px;}</style>"),
        widgets.HTML(f"<b>Download a subdomain &rarr; {CONFIG['SAVE_DIR']}/*.npy</b>"),
        widgets.HBox([dl_case, dl_var, dl_level]),
        dl_x, dl_y, dl_z, dl_full,
        widgets.HTML("<hr style='margin:6px 0'>"),
        widgets.HBox([
            widgets.VBox([dl_est, widgets.HTML("<br>"),
                          widgets.VBox([dl_go, dl_cli]), dl_out],
                         layout=W(width="440px")),
            dl_wire,
        ]),
        dl_cli_out,
    ], layout=W(padding="10px", max_width="1120px")))


def _open_visualize_panel():
    import glob, re
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    PERP = {"xy": ("z", 2), "yz": ("x", 0), "xz": ("y", 1)}

    def parse_name(fname):
        m = re.match(r"(?P<case>[A-Za-z0-9]+)_(?P<var>u|v|w|r|ee|chi)_L(?P<lvl>\d+)"
                     r"_x(\d+)-(\d+)_y(\d+)-(\d+)_z(\d+)-(\d+)", os.path.basename(fname))
        if not m:
            return None
        sel = tuple(int(x) for x in m.groups()[3:])
        return m["case"], m["var"], m["lvl"], sel

    VS = {"data": None, "sel": (0, 1, 0, 1, 0, 1), "domain": (1, 1, 1),
          "vlim": (0, 1), "var": "", "busy": False}

    vf   = widgets.Dropdown(description="File:", layout=W(width="560px"),
                            style={"description_width": "40px"})
    vref = widgets.Button(description="Refresh", icon="refresh", layout=W(width="110px"))
    vld  = widgets.Button(description="Load", button_style="primary", layout=W(width="90px"))
    vor  = widgets.ToggleButtons(options=[("xy", "xy"), ("yz", "yz"), ("xz", "xz")],
                                 value="xy", description="Plane:")
    vpos = widgets.IntSlider(description="z", continuous_update=False, layout=W(width="520px"))
    vst  = widgets.Output(); vbox = widgets.Output(); vwire = widgets.Output(); vimg = widgets.Output()

    def _saved():
        return sorted(glob.glob(os.path.join(CONFIG["SAVE_DIR"], "*.npy")))

    def _refresh(_=None):
        fs = _saved(); vf.options = [(os.path.basename(f), f) for f in fs]
        vf.value = fs[0] if fs else None              # ipywidgets>=8 won't auto-select
        with vst:
            vst.clear_output(wait=True)
            print(f"{len(fs)} cube(s) in {CONFIG['SAVE_DIR']}" if fs
                  else "No .npy yet - download one above.")

    def _slice_wire(sel, o, pos):
        x0, x1, y0, y1, z0, z1 = sel
        if o == "xy":
            z = z0 + pos; corners = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        elif o == "yz":
            x = x0 + pos; corners = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
        else:
            y = y0 + pos; corners = [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]
        with vwire:
            vwire.clear_output(wait=True)
            fig = plt.figure(figsize=(4, 3.4)); ax = fig.add_subplot(111, projection="3d")
            _draw_box(ax, x0, x1, y0, y1, z0, z1, color="0.6", lw=0.9)
            ax.add_collection3d(Poly3DCollection([corners], alpha=0.35,
                                facecolor="tab:red", edgecolor="tab:red"))
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(z0, z1)
            ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
            ax.set(xlabel="x", ylabel="y", zlabel="z")
            ax.set_title("Slice position in sub-cube", fontsize=9)
            ax.view_init(elev=22, azim=-58); plt.show(); plt.close(fig)

    def _plot(*_):
        if VS["busy"] or VS["data"] is None:
            return
        d, sel = VS["data"], VS["sel"]
        x0, x1, y0, y1, z0, z1 = sel
        lo, hi = VS["vlim"]; p = vpos.value; o = vor.value
        logv = VS["var"] in LOG_VARS
        base = VARIABLE_INFO.get(VS["var"], VS["var"])
        label = f"log10({base})" if logv else base
        kw = dict(cmap=CMAP, vmin=lo, vmax=hi, origin="lower", aspect="equal")

        def sl(a):                                   # slice -> plotted array
            if not logv:
                return a
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(a > 0, np.log10(a), np.nan)   # non-positive -> blank
        with vimg:
            vimg.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            if o == "xy":
                im = ax.imshow(sl(d[:, :, p].T), extent=[x0, x1, y0, y1], **kw)
                ax.set(xlabel="x", ylabel="y", title=f"{label}  -  xy @ z = {z0 + p}")
            elif o == "yz":
                im = ax.imshow(sl(d[p, :, :].T), extent=[y0, y1, z0, z1], **kw)
                ax.set(xlabel="y", ylabel="z", title=f"{label}  -  yz @ x = {x0 + p}")
            else:
                im = ax.imshow(sl(d[:, p, :].T), extent=[x0, x1, z0, z1], **kw)
                ax.set(xlabel="x", ylabel="z", title=f"{label}  -  xz @ y = {y0 + p}")
            fig.colorbar(im, ax=ax, label=label); plt.tight_layout(); plt.show()
        _slice_wire(sel, o, p)

    def _domain():
        with vbox:
            vbox.clear_output(wait=True)
            nx, ny, nz = VS["domain"]; x0, x1, y0, y1, z0, z1 = VS["sel"]
            fig = plt.figure(figsize=(4, 3.4)); ax = fig.add_subplot(111, projection="3d")
            _draw_box(ax, 0, nx, 0, ny, 0, nz, color="0.7", lw=0.7)
            _draw_box(ax, x0, x1, y0, y1, z0, z1, color="tab:red", lw=2)
            ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
            ax.set_box_aspect((nx, ny, nz))
            ax.set(xlabel="x", ylabel="y", zlabel="z")
            ax.set_title("Sub-box in full domain", fontsize=9)
            ax.view_init(elev=22, azim=-58); plt.show(); plt.close(fig)

    def _set_axis(o, default="mid"):
        letter, axis = PERP[o]; n = VS["data"].shape[axis]
        vpos.description = letter; vpos.max = max(n - 1, 0)
        vpos.value = n // 2 if default == "mid" else min(default, vpos.max)

    def _on_orient(ch):
        if VS["busy"] or VS["data"] is None:
            return
        VS["busy"] = True; _set_axis(ch["new"]); VS["busy"] = False; _plot()

    def _on_load(_):
        if not vf.value:
            with vst:
                vst.clear_output(wait=True); print("Pick a file (Refresh if empty).")
            return
        d = np.load(vf.value, mmap_mode="r")            # memmap: touched pages only
        parsed = parse_name(vf.value)
        if parsed:
            case, var, lvl, sel = parsed
            domain = (CATALOG.get(case, {}).get("vars", {}).get(var, {})
                      .get("levels", {}).get(lvl, {}).get("shape", d.shape))
        else:
            var = os.path.basename(vf.value)
            sel = (0, d.shape[0], 0, d.shape[1], 0, d.shape[2]); domain = d.shape
        sub = np.asarray(d[::4, ::4, ::4])
        if var in LOG_VARS:                          # color limits on the log10 scale
            pos = sub[sub > 0]
            vals = np.log10(pos) if pos.size else np.array([0.0, 1.0])
        else:
            vals = sub
        VS.update(data=d, sel=sel, domain=tuple(domain), var=var, busy=True,
                  vlim=tuple(np.percentile(vals, [2, 98])))
        with vst:
            vst.clear_output(wait=True)
            print(f"Loaded {os.path.basename(vf.value)}   shape={d.shape}")
        if 1 in d.shape:                                 # a single saved plane
            o = {0: "yz", 1: "xz", 2: "xy"}[list(d.shape).index(1)]
            vor.value = o; _set_axis(o, default=0)
        else:
            vor.value = "xy"; _set_axis("xy")
        VS["busy"] = False; _domain(); _plot()

    vref.on_click(_refresh); vld.on_click(_on_load)
    vor.observe(_on_orient, "value"); vpos.observe(_plot, "value")
    _refresh()

    display(widgets.VBox([
        widgets.HTML("<b>Visualize a saved cube</b>"),
        widgets.HBox([vf, vref, vld]), vst, vor, vpos,
        widgets.HBox([widgets.VBox([vbox, vwire]), vimg]),
    ], layout=W(padding="10px", max_width="1100px")))


# ---------------------------------------------------------------------------
# Public API — the notebook calls just these four.
# ---------------------------------------------------------------------------
def configure(config=None):
    """Merge the notebook's CONFIG over the baked-in DEFAULTS and make working
    dirs. The notebook only needs to supply the four user knobs (CACHE_DIR,
    SAVE_DIR, MAX_FETCH_GB, DL_WORKERS); the collection coordinates and the rest
    come from DEFAULTS. Passing None uses the defaults verbatim; any keys given
    override them, so power users can still tweak anything."""
    global CONFIG
    CONFIG = {**DEFAULTS, **(config or {})}
    os.makedirs(CONFIG["CACHE_DIR"], exist_ok=True)
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)


def authenticate(config=None):
    """Step 2: log into Globus and load the catalog (runs on cell execution)."""
    if config is not None:
        configure(config)
    _authenticate()


def download_panel():
    """Step 3: open the interactive download panel (runs on cell execution)."""
    _open_download_panel()


def visualize_panel():
    """Step 4: open the interactive visualize panel (runs on cell execution)."""
    _open_visualize_panel()
