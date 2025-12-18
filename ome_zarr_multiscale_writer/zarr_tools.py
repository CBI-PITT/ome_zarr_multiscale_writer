import os, concurrent.futures
from pathlib import Path
import math
import threading, queue
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle, ShardingCodec
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Any, Dict, List
from enum import Enum


import multiprocessing as mp
from multiprocessing import shared_memory

# Local imports
from .helpers import (
    ceil_div,
    compute_xy_only_levels,
    ds2_mean_uint16,
    dsZ2_mean_uint16,
    level_factors,
)

VERBOSE = True
STORE_PATH = "volume.ome.zarr"

# at top-level (after imports)
blosc_threads = max(1, (os.cpu_count() or 1) // 2)
try:
    import blosc2  # zarr v3 uses python-blosc2

    blosc2.set_nthreads(blosc_threads)  # e.g., half your cores
except Exception:
    pass


@dataclass
class ChunkSpec:
    z: int = 8
    y: int = 512
    x: int = 512


@dataclass
@dataclass
class PyramidSpec:
    z_size_estimate: int  # upper bound; we trim on close()
    y: int
    x: int
    levels: int
    t_size: int = 1  # time dimension size (default 1 for backward compatibility)
    c_size: int = 1  # channel dimension size (default 1 for backward compatibility)


class FlushPad(Enum):
    DUPLICATE_LAST = "duplicate_last"  # repeat last plane to fill the chunk
    ZEROS = "zeros"  # pad with zeros
    DROP = "drop"  # do not flush the tail (you'll lose last planes)


@dataclass
class ChunkScheme:
    """
    Per-level chunking policy that evolves from base -> target as levels increase.
    - base applies at level 0 (e.g., z=8, y=512, x=512)
    - at each level l, we:
        zc = min(target.z, base.z * 2**l)         # z chunk grows toward target
        yc = max(target.y, max(1, base.y // 2**l))# y/x chunks shrink toward target
        xc = max(target.x, max(1, base.x // 2**l))
    Then we clamp to the level's array shape.
    """

    base: Tuple[int, int, int] = (1, 1024, 1024)  # (z, y, x) at level 0
    target: Tuple[int, int, int] = (64, 64, 64)  # desired asymptote

    def chunks_for_level(
        self, level: int, zyx_level_shape: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        z_l, y_l, x_l = zyx_level_shape
        bz, by, bx = self.base
        tz, ty, tx = self.target

        # Determine if chunks should grow or shrink based on base vs target
        if bz <= tz:
            # z chunks should grow from base to target
            zc = min(tz, max(1, bz * (2**level)))
        else:
            # z chunks should shrink from base to target
            zc = max(tz, max(1, bz // (2**level)))

        if by >= ty:
            # y chunks should shrink from base to target
            yc = max(ty, max(1, by // (2**level)))
        else:
            # y chunks should grow from base to target
            yc = min(ty, max(1, by * (2**level)))

        if bx >= tx:
            # x chunks should shrink from base to target
            xc = max(tx, max(1, bx // (2**level)))
        else:
            # x chunks should grow from base to target
            xc = min(tx, max(1, bx * (2**level)))

        # Clamp to the level's actual dimensions
        return (min(zc, z_l), min(yc, y_l), min(xc, x_l))

    def chunks_for_level_5d(
        self, level: int, tczycx_level_shape: Tuple[int, int, int, int, int]
    ) -> Tuple[int, int, int, int, int]:
        """
        Generate 5D chunks (t, c, z, y, x) where t and c are always 1 for chunking.
        The z, y, x dimensions follow the same logic as chunks_for_level().
        """
        t_l, c_l, z_l, y_l, x_l = tczycx_level_shape

        # For 5D chunking, we always chunk t and c as 1
        tc_chunk = 1
        cc_chunk = 1

        # Use existing logic for z, y, x dimensions
        zycx_chunks = self.chunks_for_level(level, (z_l, y_l, x_l))
        zc_chunk, yc_chunk, xc_chunk = zycx_chunks

        return (tc_chunk, cc_chunk, zc_chunk, yc_chunk, xc_chunk)


def _validate_divisible(
    chunks: Tuple[int, int, int], shards: Tuple[int, int, int]
) -> bool:
    return all(c % s == 0 for c, s in zip(chunks, shards))


def _ensure_v2_compressor(compressor):
    """
    If a zarr v3 BloscCodec is passed (e.g., OME-Zarr 0.5 style), convert it to a
    numcodecs.Blosc compatible with Zarr v2 (required for OME-Zarr 0.4).
    Otherwise return compressor unchanged.
    """

    compressor_default = "zstd"
    clevel_default = 5
    shuffle_default = 2

    import numcodecs
    from numcodecs import Blosc as BloscV2

    numcodecs.blosc.set_nthreads(blosc_threads)
    if BloscV2 is not None and isinstance(compressor, BloscCodec):
        cname_attr = getattr(compressor, "cname", compressor_default)
        # handle enum -> string
        cname = getattr(cname_attr, "value", cname_attr)
        if isinstance(cname, str):
            cname = cname.lower()
        clevel = int(getattr(compressor, "clevel", clevel_default))
        shuffle_attr = getattr(compressor, "shuffle", shuffle_default)
        # normalize shuffle to string key if enum
        shuffle_str = getattr(shuffle_attr, "value", shuffle_attr)
        if isinstance(shuffle_str, str):
            shuffle_str = shuffle_str.lower()
        shuffle_map = {
            "noshuffle": 0,
            "shuffle": 1,
            "bitshuffle": 2,
        }
        shuffle_int = shuffle_map.get(
            str(shuffle_str),
            1 if str(shuffle_str) not in ("0", "1", "2") else str(shuffle_str),
        )
        return BloscV2(cname=cname, clevel=clevel, shuffle=shuffle_int)


def _coerce_shards(
    chunks: Tuple[int, int, int], desired: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Guaranteed-valid shards for Zarr v3: choose a divisor of each chunk dim,
    <= desired, never 0.
    """
    out = []
    for d, c in zip(desired, chunks):
        d = int(d)
        c = int(c)
        # clamp to chunk dim first
        s = min(max(1, d), c)
        # step down by gcd until it divides
        g = math.gcd(s, c)
        if g == 0:  # extremely defensive; shouldn't happen
            g = 1
        # If gcd(s, c) < s, try gcd(down, c) until it divides
        while c % g != 0 and g > 1:
            s = max(1, g)
            g = math.gcd(s, c)
        if c % g != 0:
            # worst-case fallback: 1
            g = 1
        out.append(g)
    return tuple(out)


def pick_shards_for_level(
    desired: Tuple[int, int, int] | None,
    chunks: Tuple[int, int, int],
    lvl_shape: Tuple[int, int, int],
) -> Tuple[int, int, int] | None:
    """
    Zarr v3: choose a shard (super-chunk) shape so that
      - shards[i] is a multiple of chunks[i] (>= 1 * chunks[i])
      - shards[i] <= min(desired[i], lvl_shape[i])
      - if desired is None -> no sharding (return None)
    """
    if desired is None:
        return None
    out = []
    for d, c, n in zip(desired, chunks, lvl_shape):
        c = int(c)
        n = int(n)
        d = int(d)
        # upper bound can't exceed the level's extent
        cap = min(d, n)
        # at least one chunk per shard
        k = max(1, cap // c)  # number of chunks per shard along this axis
        s = k * c  # snap to multiple of chunk
        # (s <= cap <= n) and s % c == 0
        out.append(s)
    return tuple(out)


# ---------- Zarr init (multiscales ome-zarr 0.4 and 0.5) ----------
def init_ome_zarr(
    spec: PyramidSpec,
    path=STORE_PATH,
    chunk_scheme: ChunkScheme = ChunkScheme(),
    compressor=None,
    voxel_size=(1.0, 1.0, 1.0),
    unit="micrometer",
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),  # in units
    xy_levels: int = 0,
    shard_shape: Tuple[int, int, int] | None = None,
    ome_version: str = "0.5",
):
    # Map OME-NGFF version to Zarr store version
    zarr_version = 2 if ome_version == "0.4" else 3
    root = zarr.open_group(path, mode="a", zarr_version=zarr_version)
    arrs = []

    # Determine if we're dealing with 5D data (t,c,z,y,x) or legacy 3D (z,y,x)
    is_5d = spec.t_size > 1 or spec.c_size > 1

    for l in range(spec.levels):
        zf, yf, xf = level_factors(l, xy_levels)
        z_l = ceil_div(spec.z_size_estimate, zf)
        y_l = ceil_div(spec.y, yf)
        x_l = ceil_div(spec.x, xf)

        if is_5d:
            # 5D shape: (t, c, z, y, x)
            lvl_shape = (spec.t_size, spec.c_size, z_l, y_l, x_l)
            chunks = chunk_scheme.chunks_for_level_5d(l, lvl_shape)
            shards_l = (
                pick_shards_for_level(shard_shape, chunks[2:], lvl_shape[2:])
                if zarr_version == 3
                else None
            )
        else:
            # Legacy 3D shape: (z, y, x)
            lvl_shape = (z_l, y_l, x_l)
            chunks = chunk_scheme.chunks_for_level(l, lvl_shape)
            shards_l = (
                pick_shards_for_level(shard_shape, chunks, lvl_shape)
                if zarr_version == 3
                else None
            )

        name = f"{l}"
        if name in root:
            a = root[name]
            if a.shape != lvl_shape or a.dtype != np.uint16:
                raise ValueError(
                    f"Existing {name}: {a.shape}/{a.dtype} != {lvl_shape}/uint16"
                )
            # Handle resizing for 3D vs 5D arrays
            if is_5d:
                if a.shape[2] < z_l:
                    a.resize((spec.t_size, spec.c_size, z_l, y_l, x_l))
            else:
                if a.shape[0] < z_l:
                    a.resize((z_l, y_l, x_l))
        elif zarr_version == 3:
            create_kwargs: Dict[str, Any] = {
                "name": name,
                "shape": lvl_shape,
                "chunks": chunks,
                "dtype": "uint16",
            }
            if compressor is not None:
                # v3: list of codecs
                create_kwargs["compressors"] = [compressor]
            if shards_l is not None:
                # v3: inner shard (must divide chunks)
                create_kwargs["shards"] = shards_l
            create_kwargs["dimension_names"] = (
                ["t", "c", "z", "y", "x"] if is_5d else ["z", "y", "x"]
            )
            if VERBOSE:
                print(
                    f"[init] creating {name}: shape={lvl_shape} chunks={chunks} shards={shards_l}"
                )
            a = root.create_array(**create_kwargs)
        else:
            # Zarr v2 path
            if VERBOSE:
                print(
                    f"[init] creating {name} (Zarr v2): shape={lvl_shape} chunks={chunks}"
                )
            v2_comp = _ensure_v2_compressor(compressor)

            # optional: dimension hint for some tools
            try:
                dim_names = ["t", "c", "z", "y", "x"] if is_5d else ["z", "y", "x"]
                a.attrs["_ARRAY_DIMENSIONS"] = dim_names
            except Exception:
                pass

            # Work around AsyncGroup.create_array() not accepting `dimension_separator`
            from zarr import create as zcreate

            a = zcreate(
                shape=lvl_shape,
                chunks=chunks,
                dtype="uint16",
                compressor=v2_comp,  # numcodecs codec
                overwrite=False,
                store=root.store,  # same store as the group
                path=name,  # create under this group
                zarr_format=2,  # v2 array
                dimension_separator="/",  # nested directories in .zarray
            )

        arrs.append(a)

    # OME attributes: multiscales with per-axis physical scales
    dz, dy, dx = voxel_size
    datasets = []
    for l in range(spec.levels):
        zf, yf, xf = level_factors(l, xy_levels)

        if is_5d:
            # 5D scale factors: (t, c, z, y, x)
            # t and c have no spatial scaling (1.0), z,y,x have spatial scaling
            s = [1.0, 1.0, dz * zf, dy * yf, dx * xf]
        else:
            # Legacy 3D scale factors: (z, y, x)
            s = [dz * zf, dy * yf, dx * xf]

        datasets.append(
            {
                "path": f"{l}",
                "coordinateTransformations": [
                    {"type": "scale", "scale": s},
                    {
                        "type": "translation",
                        "translation": list(translation)
                        if not is_5d
                        else [0.0, 0.0] + list(translation),
                    },
                ],
            }
        )

    if is_5d:
        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": unit},
            {"name": "y", "type": "space", "unit": unit},
            {"name": "x", "type": "space", "unit": unit},
        ]
    else:
        axes = [
            {"name": "z", "type": "space", "unit": unit},
            {"name": "y", "type": "space", "unit": unit},
            {"name": "x", "type": "space", "unit": unit},
        ]
    if ome_version == "0.5":
        root.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "axes": axes,
                    "datasets": datasets,
                    "name": "image",
                    "type": "image",
                }
            ],
        }
    else:
        # OME-Zarr 0.4 stores multiscales at top level
        root.attrs["multiscales"] = [
            {
                "version": "0.4",
                "axes": axes,
                "datasets": datasets,
                "name": "image",
            }
        ]
    return root, arrs


# ---------- Live writer: true 3D decimation pipeline ----------
class Live3DPyramidWriter:
    """
    Streams true-3D (2x in z,y,x) pyramid while you acquire slices.
    Buffers complete Z-chunks per level and flushes only when chunks fill -> no read-modify-write.
    """

    def __init__(
        self,
        spec: PyramidSpec,
        voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        path: Union[str, Path] = STORE_PATH,
        max_workers: Optional[int] = None,
        chunk_scheme: ChunkScheme = ChunkScheme(),
        compressor: Optional[Any] = None,
        flush_pad: FlushPad = FlushPad.DUPLICATE_LAST,
        ingest_queue_size: int = 8,
        max_inflight_chunks: Optional[int] = None,
        async_close: bool = True,
        shard_shape: Optional[Tuple[int, int, int]] = None,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ome_version: str = "0.5",
        write_level0: bool = True,
    ):
        self.spec = spec
        self.chunk_scheme = chunk_scheme
        self.flush_pad = flush_pad
        self.xy_levels = compute_xy_only_levels(voxel_size)
        self.max_workers = max_workers or min(8, os.cpu_count() or 4)
        self.async_close = async_close
        self.finalize_future = None
        self.write_level0 = write_level0

        # Determine if we're dealing with 5D data (t,c,z,y,x) or legacy 3D (z,y,x)
        self.is_5d = spec.t_size > 1 or spec.c_size > 1

        self.root, self.arrs = init_ome_zarr(
            spec,
            str(path),  # Convert Path to string
            chunk_scheme=chunk_scheme,
            compressor=compressor,
            voxel_size=voxel_size,
            xy_levels=self.xy_levels,
            shard_shape=shard_shape,
            translation=translation,
            ome_version=ome_version,
        )

        self.levels = spec.levels
        self.z_counts = [0] * self.levels
        self.buffers = [None] * self.levels
        self.buf_fill = [0] * self.levels
        self.buf_start = [0] * self.levels
        self.zc = []
        self.yx_shapes = []

        for l in range(self.levels):
            zf, yf, xf = level_factors(l, self.xy_levels)
            z_l = ceil_div(self.spec.z_size_estimate, zf)
            y_l = ceil_div(self.spec.y, yf)
            x_l = ceil_div(self.spec.x, xf)

            if self.is_5d:
                # 5D arrays: (t, c, z, y, x), get chunks for z,y,x
                tc_zyx_chunks = self.chunk_scheme.chunks_for_level_5d(
                    l, (self.spec.t_size, self.spec.c_size, z_l, y_l, x_l)
                )
                tc, cc, zc, yc, xc = tc_zyx_chunks
            else:
                # Legacy 3D arrays: (z, y, x)
                zc, yc, xc = self.chunk_scheme.chunks_for_level(l, (z_l, y_l, x_l))

            self.zc.append(zc)
            self.yx_shapes.append((y_l, x_l))

        # Preserve existing level 0 when multiscales-only mode is requested
        if not self.write_level0 and self.arrs:
            if self.is_5d:
                # For 5D, initialize z_counts_5d with existing array dimensions
                if not hasattr(self, "_z_counts_5d"):
                    self._z_counts_5d = {}
                # Initialize all (t,c) combinations with the existing z size
                existing_z_size = self.arrs[0].shape[2]  # z dimension is index 2 in 5D
                for t_idx in range(self.spec.t_size):
                    for c_idx in range(self.spec.c_size):
                        key = (t_idx, c_idx)
                        self._z_counts_5d[key] = [0] * self.levels
                        self._z_counts_5d[key][0] = existing_z_size
                # Also update the legacy z_counts for compatibility
                self.z_counts[0] = existing_z_size
            else:
                self.z_counts[0] = self.arrs[0].shape[0]  # z dimension is index 0 in 3D

        # Concurrency primitives
        self.q = queue.Queue(maxsize=ingest_queue_size)
        self.stop = threading.Event()
        self.lock = (
            threading.Lock()
        )  # sequences z-indices & buffers (single writer to RAM)
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        # Allow at most ~2 chunks per worker in flight (tweak as you like)
        self.max_inflight_chunks = max_inflight_chunks or (self.max_workers * 40)
        self._inflight_sem = threading.Semaphore(self.max_inflight_chunks)

        self.worker = threading.Thread(target=self._consume, daemon=True)
        self.worker.start()

    # ---------- Public API ----------
    def push_slice(self, slice_u16: np.ndarray, t_index: int = 0, c_index: int = 0):
        """
        Push a slice to the writer. Supports multiple input formats:

        Legacy 3D mode (t_size=1, c_size=1):
        - slice_u16: 2D array with shape (y, x)
        - t_index and c_index are ignored (defaults to 0)

        5D mode (t_size>1 or c_size>1):
        - slice_u16: 2D array with shape (y, x) for specific (t,c)
        - OR 3D array with shape (z, y, x) for specific (t,c)
        - t_index and c_index specify the time and channel indices

        Args:
            slice_u16: Input data (2D or 3D numpy array with dtype uint16)
            t_index: Time index (default 0)
            c_index: Channel index (default 0)
        """
        assert slice_u16.dtype == np.uint16, "slice must be uint16"

        if self.is_5d:
            # Validate t,c indices
            if not (0 <= t_index < self.spec.t_size):
                raise ValueError(
                    f"t_index {t_index} out of range [0, {self.spec.t_size})"
                )
            if not (0 <= c_index < self.spec.c_size):
                raise ValueError(
                    f"c_index {c_index} out of range [0, {self.spec.c_size})"
                )

            # Handle different input formats
            if slice_u16.ndim == 2:
                # Single 2D slice (y, x) - wrap with metadata for processing
                data = (slice_u16, t_index, c_index)
            elif slice_u16.ndim == 3:
                # 3D data (z, y, x) - will be processed as multiple z-slices
                if slice_u16.shape[1:] != (self.spec.y, self.spec.x):
                    raise ValueError(
                        f"got y,x shape {slice_u16.shape[1:]}, expected {(self.spec.y, self.spec.x)}"
                    )
                data = (
                    slice_u16,
                    t_index,
                    c_index,
                    True,
                )  # True flag for 3D processing
            else:
                raise ValueError(f"slice_u16 must be 2D or 3D, got {slice_u16.ndim}D")
        else:
            # Legacy 3D mode
            if slice_u16.shape != (self.spec.y, self.spec.x):
                raise ValueError(
                    f"got {slice_u16.shape}, expected {(self.spec.y, self.spec.x)}"
                )
            data = slice_u16

        self.q.put(data, block=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        if self.async_close:
            self.close_async()  # Background finalize
        else:
            self.close_sync()  # synchronous finalize
        print("Live3DPyramidWriter: finalized.")

    def close_sync(self):
        self.q.put(None)
        self.stop.set()
        self.worker.join()

        with self.lock:
            # flush odd Z-pair tails at levels >= 1
            self._flush_pair_tails_all_the_way()

            # Flush any partially filled chunks w/o RMW by padding to full chunk size
            for l in range(self.levels):
                if self.buffers[l] is not None and self.buf_fill[l] > 0:
                    self._pad_and_flush_partial_chunk(l)

        self.pool.shutdown(wait=True)

        for l, a in enumerate(self.arrs):
            if l == 0 and not self.write_level0:
                continue
            if self.is_5d:
                # 5D arrays: find maximum z count across all (t,c) combinations
                max_z_count = 0
                if hasattr(self, "_z_counts_5d"):
                    for (t_idx, c_idx), counts in self._z_counts_5d.items():
                        max_z_count = max(max_z_count, counts[l])
                else:
                    # Fallback to regular z_counts if 5D not initialized
                    max_z_count = self.z_counts[l]
                a.resize((a.shape[0], a.shape[1], max_z_count, a.shape[3], a.shape[4]))
            else:
                # 3D arrays: (z, y, x) -> traditional resize
                a.resize((self.z_counts[l], a.shape[1], a.shape[2]))

    # inside class Live3DPyramidWriter

    def close_async(self):
        """
        Start flushing/closing in a background thread and return a Future.
        You can call future.result() later to wait for completion.
        """
        if getattr(self, "_finalize_future", None) is not None:
            raise RuntimeError("close_async already called")
        self._finalize_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.finalize_future = self._finalize_executor.submit(self._finalize)
        return self.finalize_future

    def _finalize(self):
        self.q.put(None)
        self.stop.set()
        self.worker.join()

        with self.lock:
            self._flush_pair_tails_all_the_way()
            for l in range(self.levels):
                if self.buffers[l] is not None and self.buf_fill[l] > 0:
                    self._pad_and_flush_partial_chunk(l)

        # for fut in self.pending_futs:
        #     fut.result()
        self.pool.shutdown(wait=True)

        for l, a in enumerate(self.arrs):
            if l == 0 and not self.write_level0:
                continue
            if self.is_5d:
                # 5D arrays: find maximum z count across all (t,c) combinations
                max_z_count = 0
                if hasattr(self, "_z_counts_5d"):
                    for (t_idx, c_idx), counts in self._z_counts_5d.items():
                        max_z_count = max(max_z_count, counts[l])
                else:
                    # Fallback to regular z_counts if 5D not initialized
                    max_z_count = self.z_counts[l]
                a.resize((a.shape[0], a.shape[1], max_z_count, a.shape[3], a.shape[4]))
            else:
                # 3D arrays: (z, y, x) -> traditional resize
                a.resize((self.z_counts[l], a.shape[1], a.shape[2]))

        # optional: mark completion for external watchers
        try:
            import pathlib

            store_path = getattr(self.root.store, "path", None)
            if store_path:
                pathlib.Path(store_path, ".READY").write_text("ok")
        except Exception:
            pass

    # ---------- Internals ----------

    def _flush_pair_tails_all_the_way(self):
        if self.is_5d:
            # Handle 5D pair buffers for each (t,c) combination
            if not hasattr(self, "_pair_buf_5d"):
                return
            changed = True
            while changed:
                changed = False
                for (t_index, c_index), pair_buf_list in self._pair_buf_5d.items():
                    for lvl in range(
                        max(1, self.xy_levels + 1), self.levels
                    ):  # start at first 3D level
                        buf = pair_buf_list[lvl]
                        if buf is None:
                            continue
                        if self.flush_pad == FlushPad.DUPLICATE_LAST:
                            tail = dsZ2_mean_uint16(buf, buf)
                        elif self.flush_pad == FlushPad.ZEROS:
                            tail = dsZ2_mean_uint16(
                                buf, np.zeros_like(buf, dtype=np.uint16)
                            )
                        else:  # DROP
                            pair_buf_list[lvl] = None
                            continue
                        pair_buf_list[lvl] = None
                        zL = self._reserve_z(lvl, t_index, c_index)
                        self._append_to_active_buffer(lvl, zL, tail, t_index, c_index)
                        self._emit_next(
                            lvl + 1, ds2_mean_uint16(tail), t_index, c_index
                        )
                        changed = True
        else:
            # Legacy 3D mode
            if not hasattr(self, "_pair_buf"):
                return
            changed = True
            while changed:
                changed = False
                for lvl in range(
                    max(1, self.xy_levels + 1), self.levels
                ):  # start at first 3D level
                    buf = self._pair_buf[lvl]
                    if buf is None:
                        continue
                    if self.flush_pad == FlushPad.DUPLICATE_LAST:
                        tail = dsZ2_mean_uint16(buf, buf)
                    elif self.flush_pad == FlushPad.ZEROS:
                        tail = dsZ2_mean_uint16(
                            buf, np.zeros_like(buf, dtype=np.uint16)
                        )
                    else:  # DROP
                        self._pair_buf[lvl] = None
                        continue
                    self._pair_buf[lvl] = None
                    zL = self._reserve_z(lvl)
                    self._append_to_active_buffer(lvl, zL, tail)
                    self._emit_next(lvl + 1, ds2_mean_uint16(tail))
                    changed = True

    def _consume(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            self._ingest_raw(item)

    def _reserve_z(self, level: int, t_index: int = 0, c_index: int = 0) -> int:
        if self.is_5d:
            # For 5D data, track z counts per (t,c) combination
            if not hasattr(self, "_z_counts_5d"):
                self._z_counts_5d = {}  # Dict[(t,c) -> List[int]]

            key = (t_index, c_index)
            if key not in self._z_counts_5d:
                self._z_counts_5d[key] = [0] * self.levels

            z = self._z_counts_5d[key][level]
            self._z_counts_5d[key][level] += 1
            return z
        else:
            # Legacy 3D mode: single z count per level
            z = self.z_counts[level]
            self.z_counts[level] += 1
            return z

    def _submit_write_chunk(
        self, level: int, z0: int, buf3d: np.ndarray, t_index: int = 0, c_index: int = 0
    ):
        # acquire *before* grabbing the lock (it's called from inside-lock code now)

        if (
            self.max_inflight_chunks == 1 and self.max_inflight_chunks == 1
        ):  # Helps with single threaded debugging
            if buf3d.ndim == 5:  # 5D data
                self.arrs[level][t_index, c_index, z0 : z0 + buf3d.shape[2], :, :] = (
                    buf3d[0, 0, :, :, :]
                )
            else:  # 3D data
                self.arrs[level][z0 : z0 + buf3d.shape[0], :, :] = buf3d
        else:
            self._inflight_sem.acquire()
            fut = self.pool.submit(
                self._write_chunk_slice,
                self.arrs[level],
                z0,
                buf3d,
                t_index,
                c_index,
                self.is_5d,
            )
            # Release the slot when done (and drop ref to the future immediately)
            fut.add_done_callback(lambda _f: self._inflight_sem.release())

    @staticmethod
    def _write_chunk_slice(arr, z0, buf3d, t_index, c_index, is_5d):
        if is_5d:
            # 5D: (t, c, z, y, x) -> write to specific t,c slice
            arr[t_index, c_index, z0 : z0 + buf3d.shape[2], :, :] = buf3d[0, 0, :, :, :]
        else:
            # 3D: (z, y, x) -> traditional write
            arr[z0 : z0 + buf3d.shape[0], :, :] = buf3d  # contiguous, aligned write

    def _ensure_active_buffer(
        self, level: int, start_z: int, t_index: int = 0, c_index: int = 0
    ):
        """Allocate active chunk buffer for a level if absent, starting at start_z."""
        if self.buffers[level] is None:
            zc = self.zc[level]
            y_l, x_l = self.yx_shapes[level]

            if self.is_5d:
                # 5D buffer: (t, c, z, y, x) - but we allocate per t,c so (1, 1, z, y, x)
                self.buffers[level] = np.empty((1, 1, zc, y_l, x_l), dtype=np.uint16)
                # Store current t,c indices for this buffer
                self._current_t = t_index
                self._current_c = c_index
            else:
                # Legacy 3D buffer: (z, y, x)
                self.buffers[level] = np.empty((zc, y_l, x_l), dtype=np.uint16)

            self.buf_fill[level] = 0
            self.buf_start[level] = start_z

    def _append_to_active_buffer(
        self,
        level: int,
        z_index: int,
        plane: np.ndarray,
        t_index: int = 0,
        c_index: int = 0,
    ):
        """Append plane into the active buffer; flush when full."""
        zc = self.zc[level]

        if self.buffers[level] is None:
            # Align start to chunk boundary; with strictly increasing z, z_index should already align when new chunk begins
            start_z = (z_index // zc) * zc
            self._ensure_active_buffer(level, start_z, t_index, c_index)

        # Check if t or c index changed - if so, flush current buffer and start new one
        if self.is_5d and (
            getattr(self, "_current_t", 0) != t_index
            or getattr(self, "_current_c", 0) != c_index
        ):
            if self.buf_fill[level] > 0:
                self._pad_and_flush_partial_chunk(level)
            start_z = (z_index // zc) * zc
            self._ensure_active_buffer(level, start_z, t_index, c_index)

        offset = z_index - self.buf_start[level]

        if self.is_5d:
            self.buffers[level][0, 0, offset, :, :] = plane
        else:
            self.buffers[level][offset, :, :] = plane

        self.buf_fill[level] += 1

        if self.buf_fill[level] == zc:
            # Full chunk -> flush and reset
            buf = self.buffers[level]
            z0 = self.buf_start[level]
            # hand a copy to the pool to avoid mutation races
            self.buffers[level] = None
            self.buf_fill[level] = 0
            self.buf_start[level] = z0 + zc
            self._submit_write_chunk(level, z0, buf, t_index, c_index)

    def _pad_and_flush_partial_chunk(self, level: int):
        """Pad the active buffer to full chunk size (duplicate last or zeros) and flush."""
        if level == 0 and not self.write_level0:
            return
        zc = self.zc[level]
        fill = self.buf_fill[level]
        if fill == 0:
            return
        buf = self.buffers[level]
        if self.flush_pad == FlushPad.DUPLICATE_LAST:
            last = buf[fill - 1 : fill, :, :]
            repeat = np.repeat(last, zc - fill, axis=0)
            padded = np.concatenate([buf[:fill], repeat], axis=0)
        elif self.flush_pad == FlushPad.ZEROS:
            pad = np.zeros((zc - fill, buf.shape[1], buf.shape[2]), dtype=np.uint16)
            padded = np.concatenate([buf[:fill], pad], axis=0)
        else:  # DROP
            # simply discard and roll back z_counts to the start of the partial chunk
            self.z_counts[level] = self.buf_start[level]
            self.buffers[level] = None
            self.buf_fill[level] = 0
            return

        self._submit_write_chunk(level, self.buf_start[level], padded)
        self.buffers[level] = None
        self.buf_fill[level] = 0
        self.buf_start[level] += zc

    def _ingest_raw(self, data):
        with self.lock:
            if self.is_5d:
                # Unpack 5D data: (slice_data, t_index, c_index, is_3d_flag)
                if isinstance(data, tuple) and len(data) >= 3:
                    img0, t_index, c_index = data[:3]
                    is_3d = (
                        len(data) == 4 and data[3]
                    )  # fourth element indicates 3D processing

                    if is_3d:
                        # Process 3D data (z, y, x) slice by slice
                        for z_idx, zplane in enumerate(img0):
                            # Level 0: optionally write into active chunk
                            if self.write_level0:
                                z0 = self._reserve_z(0, t_index, c_index)
                                self._append_to_active_buffer(
                                    0, z0, zplane, t_index, c_index
                                )

                            # Build and cascade upper levels (true 3D, factor 2^L)
                            if self.levels > 1:
                                self._emit_next(
                                    level=1,
                                    candidate_xy=ds2_mean_uint16(zplane),
                                    t_index=t_index,
                                    c_index=c_index,
                                )
                    else:
                        # Process 2D data (y, x)
                        # Level 0: optionally write into active chunk
                        if self.write_level0:
                            z0 = self._reserve_z(0, t_index, c_index)
                            self._append_to_active_buffer(0, z0, img0, t_index, c_index)

                        # Build and cascade upper levels (true 3D, factor 2^L)
                        if self.levels > 1:
                            self._emit_next(
                                level=1,
                                candidate_xy=ds2_mean_uint16(img0),
                                t_index=t_index,
                                c_index=c_index,
                            )
                else:
                    raise ValueError("Invalid 5D data format")
            else:
                # Legacy 3D mode: data is just the numpy array
                img0 = data
                # Level 0: optionally write into active chunk
                if self.write_level0:
                    z0 = self._reserve_z(0)
                    self._append_to_active_buffer(0, z0, img0)

                # Build and cascade upper levels (true 3D, factor 2^L)
                if self.levels > 1:
                    self._emit_next(level=1, candidate_xy=ds2_mean_uint16(img0))

    def _emit_next(
        self, level: int, candidate_xy: np.ndarray, t_index: int = 0, c_index: int = 0
    ):
        if level >= self.levels:
            return

        if level <= self.xy_levels:
            # XY-only stage: append every incoming slice (no Z pairing)
            zL = self._reserve_z(level, t_index, c_index)
            self._append_to_active_buffer(level, zL, candidate_xy, t_index, c_index)
            # continue XY decimation upward
            self._emit_next(level + 1, ds2_mean_uint16(candidate_xy), t_index, c_index)
            return

        # 3D stage: pair consecutive planes along Z
        if not hasattr(self, "_pair_buf"):
            self._pair_buf = [None] * self.levels

        # For 5D, we need separate pair buffers for each (t,c) combination
        if self.is_5d:
            if not hasattr(self, "_pair_buf_5d"):
                self._pair_buf_5d = {}  # Dict[(t,c) -> List[None|ndarray]]
            key = (t_index, c_index)
            if key not in self._pair_buf_5d:
                self._pair_buf_5d[key] = [None] * self.levels
            buf = self._pair_buf_5d[key][level]
            if buf is None:
                self._pair_buf_5d[key][level] = candidate_xy
                return
            out_3d = dsZ2_mean_uint16(buf, candidate_xy)
            self._pair_buf_5d[key][level] = None
        else:
            # Legacy 3D mode
            buf = self._pair_buf[level]
            if buf is None:
                self._pair_buf[level] = candidate_xy
                return
            out_3d = dsZ2_mean_uint16(buf, candidate_xy)
            self._pair_buf[level] = None

        zL = self._reserve_z(level, t_index, c_index)
        self._append_to_active_buffer(level, zL, out_3d, t_index, c_index)

        # propagate upward with further XY decimation
        self._emit_next(level + 1, ds2_mean_uint16(out_3d), t_index, c_index)


# ---------- Multiprocessing writer worker ----------
def omezarr_writer_worker(
    shm_name: str,
    frame_shape: tuple[int, int],
    ring_size: int,
    writer_kwargs: dict,
    work_q: mp.Queue,
    free_q: mp.Queue,
):
    """
    Child process:
    - Attaches to shared memory
    - Creates Live3DPyramidWriter
    - Loops reading slot indices from work_q
    - For each slot, takes the frame from shared memory and pushes it
    - Returns slot to free_q when done
    """

    import numpy as np

    # Attach to shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    Y, X = frame_shape
    ring = np.ndarray((ring_size, Y, X), dtype=np.uint16, buffer=shm.buf)

    writer = Live3DPyramidWriter(**writer_kwargs)

    try:
        while True:
            slot = work_q.get()
            if slot is None:
                break

            frame = ring[slot]  # view into shared memory
            writer.push_slice(frame)

            # Slot now reusable
            free_q.put(slot)
    finally:
        try:
            writer.close()
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "Error closing Live3DPyramidWriter in worker"
            )
        shm.close()
