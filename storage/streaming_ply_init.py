"""Streaming PLY -> SSD base-file initializer for pure paper offload.

The generated ``base_file.bin`` uses the TieredCache/LogStorage layout:
xyz(3) | scaling(3) | rotation(4) | opacity(1) | features_dc(3) |
features_rest(45), all float32.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

import utils.general_utils as utils
from utils.sh_utils import C0

RECORD_DTYPE = np.dtype([("morton", "<u8"), ("params", "<f4", (59,))])
MORTON_BITS = 30

# Scale initialization modes.
#   knn3: per-point log(sqrt(mean squared distance to the 3 nearest neighbours)),
#         the 3DGS / CLM `distCUDA2` rule, evaluated one Morton bucket at a time so
#         it stays out-of-core.  GPU `distCUDA2` when CUDA is available, otherwise
#         a scipy KD-tree with the identical formula.
#   morton_bucket_density_clamped: legacy bucket-volume density estimate (one
#         value per bucket).  Overestimates scale on surface point clouds.
SCALE_MODE_KNN3 = "knn3"
SCALE_MODE_LEGACY = "morton_bucket_density_clamped"
SCALE_MODES = (SCALE_MODE_KNN3, SCALE_MODE_LEGACY)
KNN_K = 3
KNN_MIN_DIST2 = 1e-7  # same floor as create_from_pcd / distCUDA2 callers
_KNN_BACKEND: Optional[str] = None


def _stream_log(message: str) -> None:
    try:
        log_file = utils.get_log_file()
    except Exception:
        log_file = None
    utils.log_and_print(message, log_file)


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / float(1024**3)


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


@dataclass(frozen=True)
class PlyHeader:
    path: str
    fmt: str
    vertex_count: int
    vertex_properties: Tuple[Tuple[str, str], ...]
    data_offset: int

    @property
    def vertex_dtype(self) -> np.dtype:
        fields = []
        for name, ply_type in self.vertex_properties:
            if ply_type not in PLY_DTYPE_MAP:
                raise ValueError(f"Unsupported PLY property type {ply_type!r} for {name!r}")
            fields.append((name, PLY_DTYPE_MAP[ply_type]))
        return np.dtype(fields)


def read_binary_ply_header(path: str | os.PathLike[str]) -> PlyHeader:
    """Read only the PLY header and return vertex layout metadata."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        first = f.readline()
        if first.strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")

        fmt = None
        vertex_count = None
        vertex_properties: List[Tuple[str, str]] = []
        current_element = None

        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError(f"{path} ended before end_header")
            line = line_bytes.decode("ascii", errors="replace").strip()
            if line == "end_header":
                data_offset = f.tell()
                break
            if not line or line.startswith("comment"):
                continue

            tokens = line.split()
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                current_element = tokens[1]
                if current_element == "vertex":
                    vertex_count = int(tokens[2])
            elif tokens[0] == "property" and current_element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("Streaming PLY init supports only scalar vertex properties")
                vertex_properties.append((tokens[2], tokens[1]))

    if fmt != "binary_little_endian":
        raise ValueError(
            f"Streaming PLY init supports binary_little_endian only, got {fmt!r}"
        )
    if vertex_count is None:
        raise ValueError(f"{path} has no vertex element")

    names = {name for name, _ in vertex_properties}
    required = {"x", "y", "z"}
    missing = required - names
    if missing:
        raise ValueError(f"{path} missing required vertex properties: {sorted(missing)}")

    return PlyHeader(
        path=path,
        fmt=fmt,
        vertex_count=int(vertex_count),
        vertex_properties=tuple(vertex_properties),
        data_offset=int(data_offset),
    )


def _iter_vertex_chunks(header: PlyHeader, chunk_vertices: int) -> Iterable[np.ndarray]:
    dtype = header.vertex_dtype
    remaining = header.vertex_count
    with open(header.path, "rb") as f:
        f.seek(header.data_offset)
        while remaining > 0:
            count = min(int(chunk_vertices), remaining)
            chunk = np.fromfile(f, dtype=dtype, count=count)
            if chunk.size != count:
                raise IOError(
                    f"Expected {count} PLY vertices, read {chunk.size}; file may be truncated"
                )
            remaining -= int(chunk.size)
            yield chunk


def _morton_codes_np(xyz: np.ndarray, scene_min: np.ndarray, scene_max: np.ndarray) -> np.ndarray:
    normalized = (xyz - scene_min[None, :]) / (scene_max[None, :] - scene_min[None, :] + 1e-8)
    quantized = np.clip((normalized * 1023.0).astype(np.uint32), 0, 1023)

    def spread_bits(v: np.ndarray) -> np.ndarray:
        v = (v | (v << np.uint32(16))) & np.uint32(0x030000FF)
        v = (v | (v << np.uint32(8))) & np.uint32(0x0300F00F)
        v = (v | (v << np.uint32(4))) & np.uint32(0x030C30C3)
        v = (v | (v << np.uint32(2))) & np.uint32(0x09249249)
        return v

    xx = spread_bits(quantized[:, 0])
    yy = spread_bits(quantized[:, 1])
    zz = spread_bits(quantized[:, 2])
    return (xx | (yy << np.uint32(1)) | (zz << np.uint32(2))).astype(np.uint64)


def _estimate_log_scale(scene_min: np.ndarray, scene_max: np.ndarray, n_points: int) -> float:
    extent = np.maximum(scene_max - scene_min, 1e-6)
    volume = float(np.prod(extent))
    density_side = (volume / max(int(n_points), 1)) ** (1.0 / 3.0)
    return float(math.log(max(density_side, 1e-4)))


def _estimate_bucket_log_scale(
    xyz: np.ndarray,
    n_points: int,
    global_log_scale: float,
    clamp_delta: float = 2.0,
) -> float:
    if int(n_points) <= 1 or xyz.size == 0:
        return float(global_log_scale)

    local_log_scale = _estimate_log_scale(
        xyz.min(axis=0),
        xyz.max(axis=0),
        int(n_points),
    )
    return float(np.clip(local_log_scale, global_log_scale - clamp_delta, global_log_scale + clamp_delta))


def _resolve_knn_backend() -> str:
    """Pick distCUDA2 on a CUDA device, else scipy.  Resolved once per process."""
    global _KNN_BACKEND
    if _KNN_BACKEND is not None:
        return _KNN_BACKEND
    backend = "scipy"
    try:
        import torch  # torch must be imported before simple_knn._C (libc10)

        if torch.cuda.is_available():
            from simple_knn._C import distCUDA2  # noqa: F401

            backend = "distCUDA2"
    except Exception:
        backend = "scipy"
    if backend == "scipy":
        from scipy.spatial import cKDTree  # noqa: F401  raises ImportError if unavailable
    _KNN_BACKEND = backend
    _stream_log(f"[STREAMING PLY INIT] knn3 scale backend: {backend}")
    return backend


def _knn3_log_scale(xyz: np.ndarray) -> np.ndarray:
    """Per-point log-scale from the 3 nearest neighbours inside ``xyz``.

    Identical to ``log(sqrt(clamp_min(distCUDA2(points), 1e-7)))`` used by
    ``create_from_pcd``; here ``xyz`` is one Morton bucket (or one streamed chunk
    of an equal-key bucket), so peak memory stays bounded by the sort budget.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    n = int(xyz.shape[0])
    k = min(KNN_K, n - 1)
    if k <= 0:
        return np.full((n,), 0.5 * math.log(KNN_MIN_DIST2), dtype=np.float32)

    backend = _resolve_knn_backend()
    if backend == "distCUDA2" and k == KNN_K:
        import torch
        from simple_knn._C import distCUDA2

        with torch.no_grad():
            points = torch.from_numpy(xyz).cuda()
            dist2 = torch.clamp_min(distCUDA2(points), KNN_MIN_DIST2)
            out = torch.log(torch.sqrt(dist2)).cpu().numpy().astype(np.float32)
            del points, dist2
        return out

    from scipy.spatial import cKDTree

    distances, _ = cKDTree(xyz).query(xyz, k=k + 1, workers=-1)
    dist2 = np.maximum((distances[:, 1:] ** 2).mean(axis=1), KNN_MIN_DIST2)
    return (0.5 * np.log(dist2)).astype(np.float32)


def _assign_scales(
    params: np.ndarray,
    scale_mode: str,
    n_points_hint: int,
    global_log_scale: float,
    scale_clamp_delta: float,
) -> None:
    """Fill columns 3:6 of ``params`` in place according to ``scale_mode``."""
    if scale_mode == SCALE_MODE_KNN3:
        params[:, 3:6] = _knn3_log_scale(params[:, 0:3])[:, None]
        return
    bucket_log_scale = _estimate_bucket_log_scale(
        params[:, 0:3],
        int(n_points_hint),
        global_log_scale=global_log_scale,
        clamp_delta=scale_clamp_delta,
    )
    params[:, 3:6] = np.float32(bucket_log_scale)


def _get_first_existing_name(names: set[str], candidates: Tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _rgb_to_sh(vertex_chunk: np.ndarray, rows: int) -> np.ndarray:
    names = set(vertex_chunk.dtype.names or ())

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        return np.stack(
            [
                vertex_chunk["f_dc_0"].astype(np.float32),
                vertex_chunk["f_dc_1"].astype(np.float32),
                vertex_chunk["f_dc_2"].astype(np.float32),
            ],
            axis=1,
        )

    red_name = _get_first_existing_name(names, ("red", "r", "diffuse_red"))
    green_name = _get_first_existing_name(names, ("green", "g", "diffuse_green"))
    blue_name = _get_first_existing_name(names, ("blue", "b", "diffuse_blue"))
    if red_name is not None and green_name is not None and blue_name is not None:
        rgb = np.stack(
            [
                vertex_chunk[red_name].astype(np.float32),
                vertex_chunk[green_name].astype(np.float32),
                vertex_chunk[blue_name].astype(np.float32),
            ],
            axis=1,
        )
        if rgb.max(initial=0.0) > 1.0:
            rgb *= 1.0 / 255.0
    else:
        rgb = np.full((rows, 3), 0.5, dtype=np.float32)
    return (rgb - 0.5) / np.float32(C0)


def _pack_gaussian_rows(
    vertex_chunk: np.ndarray,
    log_scale: float,
    opacity_logit: float,
) -> np.ndarray:
    rows = int(vertex_chunk.shape[0])
    out = np.zeros((rows, 59), dtype=np.float32)
    out[:, 0] = vertex_chunk["x"].astype(np.float32, copy=False)
    out[:, 1] = vertex_chunk["y"].astype(np.float32, copy=False)
    out[:, 2] = vertex_chunk["z"].astype(np.float32, copy=False)
    out[:, 3:6] = np.float32(log_scale)
    out[:, 6] = 1.0
    out[:, 10] = np.float32(opacity_logit)
    out[:, 11:14] = _rgb_to_sh(vertex_chunk, rows)
    return out


def _bucket_for_codes(codes: np.ndarray, bucket_bits: int, level: int) -> np.ndarray:
    shift = max(0, 30 - bucket_bits * (level + 1))
    mask = (1 << bucket_bits) - 1
    return ((codes >> np.uint64(shift)) & np.uint64(mask)).astype(np.int32)


def _write_records_by_bucket(
    header: PlyHeader,
    temp_dir: Path,
    scene_min: np.ndarray,
    scene_max: np.ndarray,
    log_scale: float,
    chunk_vertices: int,
    bucket_bits: int,
) -> List[Path]:
    num_buckets = 1 << bucket_bits
    bucket_paths = [temp_dir / f"bucket_{i:04d}.bin" for i in range(num_buckets)]
    opacity_logit = float(math.log(0.1 / 0.9))

    processed = 0
    for chunk in _iter_vertex_chunks(header, chunk_vertices):
        xyz = np.stack(
            [
                chunk["x"].astype(np.float32, copy=False),
                chunk["y"].astype(np.float32, copy=False),
                chunk["z"].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        codes = _morton_codes_np(xyz, scene_min, scene_max)
        params = _pack_gaussian_rows(chunk, log_scale, opacity_logit)
        bucket_ids = _bucket_for_codes(codes, bucket_bits=bucket_bits, level=0)

        for bucket_id in np.unique(bucket_ids):
            mask = bucket_ids == bucket_id
            records = np.empty(int(mask.sum()), dtype=RECORD_DTYPE)
            records["morton"] = codes[mask]
            records["params"] = params[mask]
            with open(bucket_paths[int(bucket_id)], "ab") as handle:
                records.tofile(handle)

        processed += int(chunk.shape[0])
        if processed == int(chunk.shape[0]) or processed % max(chunk_vertices * 10, 1) == 0:
            print(f"[STREAMING PLY INIT] pass2 bucketed {processed:,}/{header.vertex_count:,} vertices")

    return [path for path in bucket_paths if path.exists() and path.stat().st_size > 0]


def _update_block_bounds(block_bounds: np.ndarray, start_row: int, xyz: np.ndarray, block_size: int) -> None:
    local = 0
    rows = int(xyz.shape[0])
    while local < rows:
        global_row = start_row + local
        block_id = global_row // block_size
        within = global_row % block_size
        take = min(rows - local, block_size - within)
        chunk_xyz = xyz[local : local + take]
        bmin = chunk_xyz.min(axis=0)
        bmax = chunk_xyz.max(axis=0)
        if within == 0:
            block_bounds[block_id, :3] = bmin
            block_bounds[block_id, 3:] = bmax
        else:
            block_bounds[block_id, :3] = np.minimum(block_bounds[block_id, :3], bmin)
            block_bounds[block_id, 3:] = np.maximum(block_bounds[block_id, 3:], bmax)
        local += take


def _sort_bucket_to_base(
    bucket_path: Path,
    base_handle,
    block_bounds: np.ndarray,
    start_row: int,
    block_size: int,
    global_log_scale: float,
    scale_clamp_delta: float,
    temp_dir: Path,
    bucket_bits: int,
    level: int,
    max_sort_memory_mb: float,
    sort_stats: dict,
    delete_temp: bool = True,
    scale_mode: str = SCALE_MODE_KNN3,
) -> int:
    bucket_bytes = bucket_path.stat().st_size
    sort_stats["max_bucket_bytes"] = max(int(sort_stats.get("max_bucket_bytes", 0)), int(bucket_bytes))
    max_sort_bytes = max(int(float(max_sort_memory_mb) * 1024 * 1024), RECORD_DTYPE.itemsize)
    if bucket_bytes > max_sort_bytes:
        split_row = _sort_large_bucket_to_base(
            bucket_path=bucket_path,
            base_handle=base_handle,
            block_bounds=block_bounds,
            start_row=start_row,
            block_size=block_size,
            global_log_scale=global_log_scale,
            scale_clamp_delta=scale_clamp_delta,
            temp_dir=temp_dir,
            bucket_bits=bucket_bits,
            level=level,
            max_sort_memory_mb=max_sort_memory_mb,
            sort_stats=sort_stats,
            delete_temp=delete_temp,
            scale_mode=scale_mode,
        )
        return split_row

    records = np.fromfile(bucket_path, dtype=RECORD_DTYPE)
    if records.size == 0:
        return start_row
    sort_stats["in_memory_buckets"] = int(sort_stats.get("in_memory_buckets", 0)) + 1
    order = np.argsort(records["morton"], kind="stable")
    params = np.ascontiguousarray(records["params"][order])
    _assign_scales(
        params,
        scale_mode=scale_mode,
        n_points_hint=int(params.shape[0]),
        global_log_scale=global_log_scale,
        scale_clamp_delta=scale_clamp_delta,
    )
    base_handle.write(params.tobytes())
    _update_block_bounds(block_bounds, start_row, params[:, 0:3], block_size)
    if delete_temp:
        bucket_path.unlink(missing_ok=True)
    return start_row + int(params.shape[0])


def _max_morton_level(bucket_bits: int) -> int:
    return max(0, int(math.ceil(MORTON_BITS / int(bucket_bits))) - 1)


def _split_bucket_file(
    bucket_path: Path,
    split_dir: Path,
    bucket_bits: int,
    child_level: int,
) -> List[Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    child_paths = {}
    records_per_chunk = max(1, int((64 * 1024 * 1024) // RECORD_DTYPE.itemsize))

    with open(bucket_path, "rb") as handle:
        while True:
            records = np.fromfile(handle, dtype=RECORD_DTYPE, count=records_per_chunk)
            if records.size == 0:
                break
            child_ids = _bucket_for_codes(records["morton"], bucket_bits=bucket_bits, level=child_level)
            for child_id in np.unique(child_ids):
                child_id_int = int(child_id)
                child_path = child_paths.get(child_id_int)
                if child_path is None:
                    child_path = split_dir / f"bucket_{child_id_int:04d}.bin"
                    child_paths[child_id_int] = child_path
                mask = child_ids == child_id_int
                with open(child_path, "ab") as child_handle:
                    records[mask].tofile(child_handle)

    return [child_paths[key] for key in sorted(child_paths)]


def _scan_bucket_xyz_stats(bucket_path: Path) -> Tuple[int, np.ndarray, np.ndarray]:
    count = 0
    xyz_min = np.full((3,), np.inf, dtype=np.float32)
    xyz_max = np.full((3,), -np.inf, dtype=np.float32)
    records_per_chunk = max(1, int((64 * 1024 * 1024) // RECORD_DTYPE.itemsize))

    with open(bucket_path, "rb") as handle:
        while True:
            records = np.fromfile(handle, dtype=RECORD_DTYPE, count=records_per_chunk)
            if records.size == 0:
                break
            xyz = records["params"][:, 0:3]
            xyz_min = np.minimum(xyz_min, xyz.min(axis=0))
            xyz_max = np.maximum(xyz_max, xyz.max(axis=0))
            count += int(records.shape[0])

    return count, xyz_min, xyz_max


def _stream_equal_key_bucket_to_base(
    bucket_path: Path,
    base_handle,
    block_bounds: np.ndarray,
    start_row: int,
    block_size: int,
    global_log_scale: float,
    scale_clamp_delta: float,
    delete_temp: bool,
    scale_mode: str = SCALE_MODE_KNN3,
) -> int:
    count, xyz_min, xyz_max = _scan_bucket_xyz_stats(bucket_path)
    if count == 0:
        return start_row
    _stream_log(
        f"[STREAMING PLY INIT] pass3 stream equal-key {bucket_path.name}: "
        f"rows={count:,} start_row={start_row:,}"
    )
    bucket_log_scale = None
    if scale_mode != SCALE_MODE_KNN3:
        bucket_log_scale = _estimate_bucket_log_scale(
            np.stack([xyz_min, xyz_max], axis=0),
            count,
            global_log_scale=global_log_scale,
            clamp_delta=scale_clamp_delta,
        )

    row_cursor = start_row
    records_per_chunk = max(1, int((64 * 1024 * 1024) // RECORD_DTYPE.itemsize))
    with open(bucket_path, "rb") as handle:
        while True:
            records = np.fromfile(handle, dtype=RECORD_DTYPE, count=records_per_chunk)
            if records.size == 0:
                break
            params = np.ascontiguousarray(records["params"])
            if scale_mode == SCALE_MODE_KNN3:
                # Equal-key buckets share one Morton cell; kNN per 64 MiB chunk.
                params[:, 3:6] = _knn3_log_scale(params[:, 0:3])[:, None]
            else:
                params[:, 3:6] = np.float32(bucket_log_scale)
            base_handle.write(params.tobytes())
            _update_block_bounds(block_bounds, row_cursor, params[:, 0:3], block_size)
            row_cursor += int(params.shape[0])

    if delete_temp:
        bucket_path.unlink(missing_ok=True)
    return row_cursor


def _sort_large_bucket_to_base(
    bucket_path: Path,
    base_handle,
    block_bounds: np.ndarray,
    start_row: int,
    block_size: int,
    global_log_scale: float,
    scale_clamp_delta: float,
    temp_dir: Path,
    bucket_bits: int,
    level: int,
    max_sort_memory_mb: float,
    sort_stats: dict,
    delete_temp: bool,
    scale_mode: str = SCALE_MODE_KNN3,
) -> int:
    max_level = _max_morton_level(bucket_bits)
    if level >= max_level:
        sort_stats["streamed_equal_key_buckets"] = int(sort_stats.get("streamed_equal_key_buckets", 0)) + 1
        return _stream_equal_key_bucket_to_base(
            bucket_path=bucket_path,
            base_handle=base_handle,
            block_bounds=block_bounds,
            start_row=start_row,
            block_size=block_size,
            global_log_scale=global_log_scale,
            scale_clamp_delta=scale_clamp_delta,
            delete_temp=delete_temp,
            scale_mode=scale_mode,
        )

    child_level = level + 1
    bucket_bytes = bucket_path.stat().st_size
    split_dir = temp_dir / f"{bucket_path.stem}_split_l{child_level}"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    child_paths = _split_bucket_file(
        bucket_path=bucket_path,
        split_dir=split_dir,
        bucket_bits=bucket_bits,
        child_level=child_level,
    )
    sort_stats["recursive_splits"] = int(sort_stats.get("recursive_splits", 0)) + 1
    sort_stats["split_child_buckets"] = int(sort_stats.get("split_child_buckets", 0)) + len(child_paths)
    _stream_log(
        f"[STREAMING PLY INIT] pass3 split {bucket_path.name}: "
        f"level={level}->{child_level} size={_bytes_to_gib(bucket_bytes):.2f}GiB "
        f"children={len(child_paths)}"
    )
    if delete_temp:
        bucket_path.unlink(missing_ok=True)

    row_cursor = start_row
    for child_path in child_paths:
        row_cursor = _sort_bucket_to_base(
            bucket_path=child_path,
            base_handle=base_handle,
            block_bounds=block_bounds,
            start_row=row_cursor,
            block_size=block_size,
            global_log_scale=global_log_scale,
            scale_clamp_delta=scale_clamp_delta,
            temp_dir=temp_dir,
            bucket_bits=bucket_bits,
            level=child_level,
            max_sort_memory_mb=max_sort_memory_mb,
            sort_stats=sort_stats,
            delete_temp=delete_temp,
            scale_mode=scale_mode,
        )
    if delete_temp:
        shutil.rmtree(split_dir, ignore_errors=True)
    return row_cursor


def streaming_ply_to_ssd_base(
    ply_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    block_size: int,
    debug_fast_init_scales: bool,
    chunk_vertices: int = 1_000_000,
    bucket_bits: int = 10,
    scale_clamp_delta: float = 2.0,
    max_sort_memory_mb: float = 512.0,
    keep_temp: bool = False,
    scale_mode: str = SCALE_MODE_KNN3,
) -> dict:
    """Stream a binary PLY into a Morton-sorted SSD base file and metadata.

    ``scale_mode="knn3"`` (default) initializes every Gaussian from its 3 nearest
    neighbours inside its Morton bucket (distCUDA2 on GPU, scipy otherwise).
    ``scale_mode="morton_bucket_density_clamped"`` keeps the legacy per-bucket
    density estimate and still requires ``debug_fast_init_scales``.
    """
    scale_mode = str(scale_mode).lower()
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}; expected one of {SCALE_MODES}")
    if scale_mode == SCALE_MODE_LEGACY and not debug_fast_init_scales:
        raise RuntimeError(
            "streaming PLY init requires --debug_fast_init_scales; full distCUDA2 is not out-of-core"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    temp_dir = output / "streaming_init_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    header = read_binary_ply_header(ply_path)
    _stream_log(
        f"[STREAMING PLY INIT] pass1 header: vertices={header.vertex_count:,}, "
        f"properties={[name for name, _ in header.vertex_properties]}"
    )

    scene_min = np.full((3,), np.inf, dtype=np.float32)
    scene_max = np.full((3,), -np.inf, dtype=np.float32)
    processed = 0
    for chunk in _iter_vertex_chunks(header, chunk_vertices):
        xyz = np.stack(
            [
                chunk["x"].astype(np.float32, copy=False),
                chunk["y"].astype(np.float32, copy=False),
                chunk["z"].astype(np.float32, copy=False),
            ],
            axis=1,
        )
        scene_min = np.minimum(scene_min, xyz.min(axis=0))
        scene_max = np.maximum(scene_max, xyz.max(axis=0))
        processed += int(chunk.shape[0])
    _stream_log(
        f"[STREAMING PLY INIT] pass1 bbox: min={scene_min.tolist()} max={scene_max.tolist()} "
        f"vertices={processed:,}"
    )

    log_scale = _estimate_log_scale(scene_min, scene_max, header.vertex_count)
    _stream_log(f"[STREAMING PLY INIT] scale_mode={scale_mode}")
    if scale_mode == SCALE_MODE_KNN3:
        _resolve_knn_backend()
    bucket_paths = _write_records_by_bucket(
        header=header,
        temp_dir=temp_dir,
        scene_min=scene_min,
        scene_max=scene_max,
        log_scale=log_scale,
        chunk_vertices=chunk_vertices,
        bucket_bits=bucket_bits,
    )

    num_blocks = (header.vertex_count + int(block_size) - 1) // int(block_size)
    block_bounds = np.zeros((num_blocks, 6), dtype=np.float32)
    base_file = output / "base_file.bin"
    row_cursor = 0
    sort_stats = {
        "max_sort_memory_mb": float(max_sort_memory_mb),
        "recursive_splits": 0,
        "split_child_buckets": 0,
        "in_memory_buckets": 0,
        "streamed_equal_key_buckets": 0,
        "max_bucket_bytes": 0,
    }
    with open(base_file, "wb") as base_handle:
        sorted_bucket_paths = sorted(bucket_paths, key=lambda p: p.name)
        total_buckets = len(sorted_bucket_paths)
        for i, bucket_path in enumerate(sorted_bucket_paths):
            bucket_bytes = bucket_path.stat().st_size
            before_row = row_cursor
            _stream_log(
                f"[STREAMING PLY INIT] pass3 bucket {i + 1}/{total_buckets}: "
                f"{bucket_path.name} size={_bytes_to_gib(bucket_bytes):.2f}GiB "
                f"rows_before={before_row:,}/{header.vertex_count:,}"
            )
            row_cursor = _sort_bucket_to_base(
                bucket_path=bucket_path,
                base_handle=base_handle,
                block_bounds=block_bounds,
                start_row=row_cursor,
                block_size=int(block_size),
                global_log_scale=log_scale,
                scale_clamp_delta=float(scale_clamp_delta),
                temp_dir=temp_dir,
                bucket_bits=int(bucket_bits),
                level=0,
                max_sort_memory_mb=float(max_sort_memory_mb),
                sort_stats=sort_stats,
                delete_temp=not keep_temp,
                scale_mode=scale_mode,
            )
            _stream_log(
                f"[STREAMING PLY INIT] pass3 sorted {i + 1}/{total_buckets} buckets, "
                f"+rows={row_cursor - before_row:,}, rows={row_cursor:,}/{header.vertex_count:,}"
            )

    expected_size = int(header.vertex_count) * 59 * 4
    actual_size = base_file.stat().st_size
    if row_cursor != header.vertex_count:
        raise RuntimeError(
            f"Streaming init wrote {row_cursor:,} rows, expected {header.vertex_count:,}"
        )
    if actual_size != expected_size:
        raise RuntimeError(
            f"base_file.bin size mismatch: actual={actual_size}, expected={expected_size}"
        )

    bounds_path = output / "block_bounds.npy"
    np.save(bounds_path, block_bounds)

    manifest = {
        "version": 1,
        "source": "streaming_ply_to_ssd_base",
        "ply_path": os.fspath(ply_path),
        "base_file": str(base_file),
        "block_bounds": str(bounds_path),
        "total_points": int(header.vertex_count),
        "num_blocks": int(num_blocks),
        "block_size": int(block_size),
        "param_dim": 59,
        "dtype": "float32",
        "layout": "cache: xyz|scaling|rotation|opacity|features_dc|features_rest",
        "scene_min": scene_min.astype(float).tolist(),
        "scene_max": scene_max.astype(float).tolist(),
        "scale_mode": scale_mode,
        "global_log_scale": float(log_scale),
        "scale_clamp_delta": float(scale_clamp_delta),
        "knn": (
            {
                "k": KNN_K,
                "min_dist2": KNN_MIN_DIST2,
                "backend": _KNN_BACKEND,
                "neighborhood": "morton bucket (in-memory sort unit); 64 MiB chunk for equal-key buckets",
            }
            if scale_mode == SCALE_MODE_KNN3
            else None
        ),
        "external_sort": sort_stats,
        "morton": {
            "type": "standard_10bit",
            "bucket_bits": int(bucket_bits),
        },
    }
    manifest_path = output / "streaming_init_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    manifest["manifest_path"] = str(manifest_path)

    if keep_temp:
        manifest["temp_dir"] = str(temp_dir)
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)

    _stream_log(
        f"[STREAMING PLY INIT] wrote base_file={base_file} "
        f"size={actual_size / (1024 ** 3):.2f}GB blocks={num_blocks:,}"
    )
    return manifest


def _main(argv=None) -> int:
    """Standalone base builder: python -m storage.streaming_ply_init --ply ... --output ..."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream a binary PLY into a Morton-sorted TideGS SSD base (base_file.bin, "
        "block_bounds.npy, streaming_init_manifest.json) without starting training."
    )
    parser.add_argument("--ply", required=True, help="binary_little_endian PLY with x,y,z (+rgb)")
    parser.add_argument("--output", required=True, help="output directory for the base")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--scale-mode", choices=SCALE_MODES, default=SCALE_MODE_KNN3)
    parser.add_argument("--bucket-bits", type=int, default=10)
    parser.add_argument("--sort-memory-mb", type=float, default=512.0,
                        help="max in-RAM bucket size; also bounds the per-call kNN working set")
    parser.add_argument("--chunk-vertices", type=int, default=1_000_000)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args(argv)

    manifest = streaming_ply_to_ssd_base(
        ply_path=args.ply,
        output_dir=args.output,
        block_size=args.block_size,
        debug_fast_init_scales=(args.scale_mode == SCALE_MODE_LEGACY),
        chunk_vertices=args.chunk_vertices,
        bucket_bits=args.bucket_bits,
        max_sort_memory_mb=args.sort_memory_mb,
        keep_temp=args.keep_temp,
        scale_mode=args.scale_mode,
    )
    summary = {
        key: manifest.get(key)
        for key in ("manifest_path", "base_file", "block_bounds", "total_points", "num_blocks", "scale_mode", "knn")
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
