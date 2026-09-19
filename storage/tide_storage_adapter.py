"""TideGS SSD storage adapter.

This adapter owns the release storage path: streaming PLY init or checkpoint
resume, tiered RAM cache, frustum culling, async prefetch, and camera
scheduling.  It does not materialize or sort a full CPU Gaussian parameter
table.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.cluster import KMeans

from .async_pipeline import AsyncPipeline, TSPScheduler
from .config import StorageConfig, get_config_for_scene_size
from .gaussian_block import FrustumCuller
from .log_storage_manager import LogStorageManager
from .streaming_ply_init import read_binary_ply_header, streaming_ply_to_ssd_base
from .tiered_cache_manager import TieredCacheManager


def _link_resume_base_file(source_base_file: Path, active_storage_dir: Path) -> Path:
    """Expose checkpoint base as the active storage base without copying it."""
    active_storage_dir.mkdir(parents=True, exist_ok=True)
    active_base_file = active_storage_dir / "base_file.bin"
    if active_base_file.exists() or active_base_file.is_symlink():
        active_base_file.unlink()
    active_base_file.symlink_to(source_base_file.resolve())
    return active_base_file


class _CacheCommitJob:
    def __init__(
        self,
        payload,
        block_ids: List[int],
        bounds_managed_externally: bool = False,
    ):
        self.payload = payload
        self.block_ids = tuple(sorted(set(int(block_id) for block_id in block_ids)))
        self.bounds_managed_externally = bool(bounds_managed_externally)
        self.done = threading.Event()
        self.result = 0
        self.error = None

    def wait_gpu(self) -> None:
        wait_gpu = getattr(self.payload, "wait_gpu", None)
        if callable(wait_gpu):
            wait_gpu()

    def wait(self) -> int:
        self.done.wait()
        if self.error is not None:
            raise RuntimeError("background cache commit failed") from self.error
        return int(self.result)


class _BoundsRefreshJob:
    def __init__(self, payload):
        self.payload = payload
        self.done = threading.Event()
        self.result = 0
        self.error = None

    def wait(self) -> int:
        self.done.wait()
        if self.error is not None:
            raise RuntimeError("background bounds refresh failed") from self.error
        return int(self.result)


class TideStorageAdapter:
    """Adapter for the out-of-core SSD training path."""

    def __init__(
        self,
        gaussians,
        cameras: List,
        config: StorageConfig,
    ):
        self.config = config
        self.gaussians = gaussians
        self.cameras = cameras
        self.skip_camera_clustering = config.skip_camera_clustering
        self.use_6plane = config.use_6plane
        self.max_patch_files = int(config.max_patch_files)
        self.max_stale_patch_gb = float(config.max_stale_patch_gb)
        self.max_patch_total_gb = float(config.max_patch_total_gb)
        self.min_free_gb = float(config.min_free_gb)
        self.paper_debug_logging = bool(config.debug_logging)
        self.schedule_cache_enabled = bool(config.schedule_cache_enabled)
        self.schedule_cache_dir = (
            Path(config.schedule_cache_dir)
            if config.schedule_cache_dir
            else Path(config.ssd_cache_dir) / "camera_schedule_cache"
        )

        self.execution_metrics = {
            "paper_cache_sync_calls": 0,
            "paper_cache_sync_blocks": 0,
            "paper_cache_sync_staged_blocks": 0,
            "paper_bounds_refresh_blocks": 0,
            "background_cache_commit_jobs": 0,
            "background_cache_commit_blocks": 0,
            "background_cache_commit_waits": 0,
            "background_bounds_jobs": 0,
            "background_bounds_blocks": 0,
            "background_bounds_waits": 0,
            "visibility_full_culls": 0,
            "visibility_incremental_repairs": 0,
            "visibility_repaired_blocks": 0,
        }
        self._cache_commit_queue: Queue = Queue()
        self._cache_commit_lock = threading.RLock()
        self._pending_cache_commits: Dict[int, _CacheCommitJob] = {}
        self._cache_commit_error = None
        self._cache_commit_thread = None
        self._bounds_refresh_queue: Queue = Queue()
        self._bounds_refresh_error = None
        self._bounds_refresh_thread = None
        self._resident_state = None
        self._resident_working_set = None
        self._bounds_state_lock = threading.RLock()
        self._bounds_generation = 0
        self._visibility_cache = {}
        self._bounds_change_log = OrderedDict()
        self._bounds_change_history_limit = 64
        self.streaming_init_manifest = None
        self._closed = False

        resume_manifest = getattr(gaussians, "_pure_ssd_resume_manifest", None)
        use_pure_ssd_resume = bool(getattr(gaussians, "_pure_ssd_resume_pending", False))
        streaming_ply_path = getattr(gaussians, "_streaming_init_ply_path", "")
        use_streaming_init = bool(getattr(gaussians, "_streaming_ply_init_pending", False))

        if use_pure_ssd_resume:
            self._init_from_resume_manifest(
                resume_manifest=resume_manifest,
                storage_dir=config.ssd_cache_dir,
                block_size=config.block_size,
                max_ram_gb=config.max_ram_gb,
                num_clusters=config.num_camera_clusters,
            )
        elif use_streaming_init:
            self._init_from_streaming_ply(
                ply_path=streaming_ply_path,
                storage_dir=config.ssd_cache_dir,
                block_size=config.block_size,
                max_ram_gb=config.max_ram_gb,
                num_clusters=config.num_camera_clusters,
            )
        else:
            raise RuntimeError(
                "TideStorageAdapter requires a prepared streaming PLY init or "
                "pure SSD checkpoint resume shell. Full in-memory materialization "
                "is intentionally unsupported."
            )

        self._log(f"[TideStorageAdapter] Initializing for {self.num_points:,} points")
        self._log(f"[TideStorageAdapter] Block size: {self.block_size:,}")
        self._log(f"[TideStorageAdapter] RAM cache: {self.max_ram_gb} GB")
        self._log(f"[TideStorageAdapter] Camera clusters: {self.num_clusters}")

        self._initialize_storage()
        self._start_cache_commit_worker()
        self._start_bounds_refresh_worker()
        if self.skip_camera_clustering:
            self._log("[TideStorageAdapter] Skipping camera scheduling")
            self.camera_clusters = None
            self.scheduler = None
        else:
            self._initialize_cameras()
        self._initialize_pipeline()
        print("[TideStorageAdapter] Initialization complete")

    def _log(self, message: str) -> None:
        if bool(getattr(self, "paper_debug_logging", False)):
            print(message)

    def _warn(self, message: str) -> None:
        print(message)

    def _start_cache_commit_worker(self) -> None:
        self._cache_commit_thread = threading.Thread(
            target=self._cache_commit_worker,
            name="tide-cache-commit",
            daemon=True,
        )
        self._cache_commit_thread.start()

    def _cache_commit_worker(self) -> None:
        while True:
            job = self._cache_commit_queue.get()
            if job is None:
                self._cache_commit_queue.task_done()
                return
            try:
                updated_blocks = job.payload.wait()
                job.result = self.sync_cache_from_cpu_views(
                    updated_blocks,
                    refresh_bounds=not job.bounds_managed_externally,
                )
                if job.result != len(job.block_ids):
                    raise RuntimeError(
                        f"Incomplete cache commit: expected={len(job.block_ids)} "
                        f"committed={job.result}"
                    )
            except Exception as exc:
                job.error = exc
                self._cache_commit_error = exc
            finally:
                release = getattr(job.payload, "release", None)
                if callable(release):
                    release()
                with self._cache_commit_lock:
                    for block_id in job.block_ids:
                        if self._pending_cache_commits.get(block_id) is job:
                            self._pending_cache_commits.pop(block_id, None)
                job.done.set()
                self._cache_commit_queue.task_done()

    def _start_bounds_refresh_worker(self) -> None:
        self._bounds_refresh_thread = threading.Thread(
            target=self._bounds_refresh_worker,
            name="tide-bounds-refresh",
            daemon=True,
        )
        self._bounds_refresh_thread.start()

    def _bounds_refresh_worker(self) -> None:
        while True:
            job = self._bounds_refresh_queue.get()
            if job is None:
                self._bounds_refresh_queue.task_done()
                return
            try:
                with torch.cuda.nvtx.range("Tide bounds: wait D2H"):
                    block_ids, bounds = job.payload.wait()
                with torch.cuda.nvtx.range("Tide bounds: publish generation"):
                    job.result = self.update_block_bounds(block_ids, bounds)
            except Exception as exc:
                job.error = exc
                self._bounds_refresh_error = exc
            finally:
                job.done.set()
                self._bounds_refresh_queue.task_done()

    def submit_bounds_refresh(self, payload) -> int:
        if payload is None:
            return 0
        block_ids = list(getattr(payload, "block_ids", []))
        if not block_ids:
            return 0
        self.execution_metrics["background_bounds_jobs"] += 1
        self.execution_metrics["background_bounds_blocks"] += len(block_ids)
        self._bounds_refresh_queue.put(_BoundsRefreshJob(payload))
        return len(block_ids)

    def wait_for_bounds_refresh(self) -> None:
        if self._bounds_refresh_queue.unfinished_tasks:
            self.execution_metrics["background_bounds_waits"] += 1
        self._bounds_refresh_queue.join()
        if self._bounds_refresh_error is not None:
            raise RuntimeError("background bounds refresh worker failed") from self._bounds_refresh_error

    def submit_cache_writeback(
        self,
        payload,
        *,
        bounds_managed_externally: bool = False,
    ) -> int:
        self.check_writeback_error()
        block_ids = list(getattr(payload, "block_ids", []))
        if not block_ids:
            release = getattr(payload, "release", None)
            if callable(release):
                release()
            return 0

        job = _CacheCommitJob(
            payload,
            block_ids,
            bounds_managed_externally=bounds_managed_externally,
        )
        with self._cache_commit_lock:
            for block_id in job.block_ids:
                self._pending_cache_commits[block_id] = job
        self.execution_metrics["background_cache_commit_jobs"] += 1
        self.execution_metrics["background_cache_commit_blocks"] += len(job.block_ids)
        self._cache_commit_queue.put(job)
        return len(job.block_ids)

    def bind_resident_writeback(self, resident_state, working_set) -> None:
        self._resident_state = resident_state
        self._resident_working_set = working_set

    def flush_resident_dirty(self) -> int:
        resident_state = self._resident_state
        if resident_state is None:
            self.drain_cache_writebacks()
            return 0

        resident_state.synchronize_prefetch()
        return self.writeback_resident_blocks(
            resident_state.dirty_blocks(), wait=True,
            bounds_managed_externally=False,
        )

    def writeback_resident_blocks(
        self, block_ids: List[int], *, wait: bool = False,
        bounds_managed_externally: bool = True,
    ) -> int:
        self.check_writeback_error()
        block_ids = sorted(set(int(block_id) for block_id in block_ids))
        if not block_ids:
            if wait:
                self.drain_cache_writebacks()
            return 0

        resident_state = self._resident_state
        payload = self._resident_working_set.stage_updated_blocks(
            block_ids, source_buffer=resident_state.active_buffer,
        )
        try:
            staged_ids = [] if payload is None else payload.block_ids
            if set(staged_ids) != set(block_ids):
                raise RuntimeError(
                    "Incomplete GPU resident writeback: "
                    f"requested={block_ids[:8]} staged={staged_ids[:8]}"
                )
            staged = self.submit_cache_writeback(
                payload, bounds_managed_externally=bounds_managed_externally,
            )
        except Exception:
            if payload is not None:
                payload.wait_gpu()
                payload.release()
            raise
        if wait:
            self.drain_cache_writebacks()
        resident_state.mark_blocks_written_back(block_ids)
        return staged

    def check_writeback_error(self) -> None:
        if self._cache_commit_error is not None:
            raise RuntimeError("background cache commit worker failed") from self._cache_commit_error

    def wait_for_cache_blocks(self, block_ids: List[int]) -> None:
        self.check_writeback_error()
        requested = set(int(block_id) for block_id in block_ids)
        while requested:
            self.check_writeback_error()
            with self._cache_commit_lock:
                jobs = {
                    self._pending_cache_commits[block_id]
                    for block_id in requested
                    if block_id in self._pending_cache_commits
                }
            if not jobs:
                return
            self.execution_metrics["background_cache_commit_waits"] += 1
            for job in jobs:
                job.wait()

    def filter_cache_prefetch_candidates(self, block_ids: List[int]) -> List[int]:
        with self._cache_commit_lock:
            pending = set(self._pending_cache_commits)
        return [int(block_id) for block_id in block_ids if int(block_id) not in pending]

    def wait_for_pending_gpu_copies(self) -> None:
        self.check_writeback_error()
        with self._cache_commit_lock:
            jobs = set(self._pending_cache_commits.values())
        for job in jobs:
            job.wait_gpu()

    def drain_cache_writebacks(self) -> None:
        self._cache_commit_queue.join()
        self.check_writeback_error()

    def _init_from_resume_manifest(
        self,
        *,
        resume_manifest: Optional[dict],
        storage_dir: str,
        block_size: Optional[int],
        max_ram_gb: Optional[float],
        num_clusters: Optional[int],
    ) -> None:
        if not resume_manifest:
            raise ValueError("Pure SSD resume was requested but no checkpoint manifest is attached")

        is_prebuilt = bool(resume_manifest.get("_prebuilt_base_reuse", False))
        self.streaming_init_manifest = dict(resume_manifest)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.storage_dir = Path(storage_dir) / timestamp
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.num_points = int(resume_manifest["total_points"])
        config = get_config_for_scene_size(self.num_points)
        self.block_size = int(resume_manifest.get("block_size", block_size or config.block_size))
        self.max_ram_gb = max_ram_gb if max_ram_gb is not None else config.max_ram_gb
        self.num_clusters = num_clusters if num_clusters is not None else config.num_camera_clusters
        self.num_blocks = int(resume_manifest["num_blocks"])
        self.block_bounds = np.load(resume_manifest["block_bounds"])
        self.scene_min = np.asarray(resume_manifest["scene_min"], dtype=np.float32)
        self.scene_max = np.asarray(resume_manifest["scene_max"], dtype=np.float32)
        self.scene_center = (self.scene_min + self.scene_max) / 2.0
        self.scene_radius = np.linalg.norm(self.scene_max - self.scene_min) / 2.0

        base_file = Path(resume_manifest["base_file"])
        expected_size = (
            self.num_points
            * int(resume_manifest.get("param_dim", 59))
            * np.dtype(np.float32).itemsize
        )
        actual_size = base_file.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Pure SSD checkpoint base size mismatch: got {actual_size}, expected {expected_size}"
            )

        active_base_file = _link_resume_base_file(base_file, self.storage_dir)
        checkpoint_type = str(resume_manifest.get("checkpoint_type", "pure_ssd_snapshot"))
        if is_prebuilt:
            print(f"[PURE SSD PREBUILT] reusing SSD base: {base_file}")
            print(f"[PURE SSD PREBUILT] active patch/cache dir: {self.storage_dir}")
            print(f"[PURE SSD PREBUILT] active base link: {active_base_file}")
            self.gaussians.initialize_from_streaming_ssd_manifest(self.streaming_init_manifest)
        else:
            print(f"[PURE SSD RESUME] using SSD {checkpoint_type} base: {base_file}")
            print(f"[PURE SSD RESUME] active patch/cache dir: {self.storage_dir}")
            print(f"[PURE SSD RESUME] active base link: {active_base_file}")
            self.gaussians.initialize_from_pure_ssd_checkpoint_manifest(self.streaming_init_manifest)

    def _init_from_streaming_ply(
        self,
        *,
        ply_path: str,
        storage_dir: str,
        block_size: Optional[int],
        max_ram_gb: Optional[float],
        num_clusters: Optional[int],
    ) -> None:
        if not ply_path:
            raise ValueError("Pure SSD streaming init requested but no dense PLY path is attached")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.storage_dir = Path(storage_dir) / timestamp
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"[TideStorageAdapter] Storage directory with timestamp: {self.storage_dir}")

        header = read_binary_ply_header(ply_path)
        self.num_points = int(header.vertex_count)
        config = get_config_for_scene_size(self.num_points)
        self.block_size = int(block_size if block_size is not None else config.block_size)
        self.max_ram_gb = max_ram_gb if max_ram_gb is not None else config.max_ram_gb
        self.num_clusters = num_clusters if num_clusters is not None else config.num_camera_clusters
        self.num_blocks = (self.num_points + self.block_size - 1) // self.block_size

        print("[TideStorageAdapter] Streaming PLY directly to SSD base file")
        self.streaming_init_manifest = streaming_ply_to_ssd_base(
            ply_path=ply_path,
            output_dir=self.storage_dir,
            block_size=self.block_size,
            debug_fast_init_scales=bool(self.config.fast_init_scales),
            bucket_bits=int(self.config.bucket_bits),
            max_sort_memory_mb=float(self.config.sort_memory_mb),
            scale_mode=str(self.config.init_scale_mode),
        )
        self.num_points = int(self.streaming_init_manifest["total_points"])
        self.num_blocks = int(self.streaming_init_manifest["num_blocks"])
        self.block_size = int(self.streaming_init_manifest["block_size"])
        self.block_bounds = np.load(self.streaming_init_manifest["block_bounds"])
        self.scene_min = np.asarray(self.streaming_init_manifest["scene_min"], dtype=np.float32)
        self.scene_max = np.asarray(self.streaming_init_manifest["scene_max"], dtype=np.float32)
        self.scene_center = (self.scene_min + self.scene_max) / 2.0
        self.scene_radius = np.linalg.norm(self.scene_max - self.scene_min) / 2.0

        self.gaussians.initialize_from_streaming_ssd_manifest(self.streaming_init_manifest)

    def _initialize_storage(self) -> None:
        self._log("[TideStorageAdapter] Binding prebuilt SSD base file")
        self.storage = LogStorageManager(
            storage_dir=str(self.storage_dir),
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            point_dim=59,
            verbose=self.paper_debug_logging,
            max_patch_files=self.max_patch_files,
            max_stale_patch_gb=self.max_stale_patch_gb,
            max_patch_total_gb=self.max_patch_total_gb,
            min_free_gb=self.min_free_gb,
        )
        storage_index = None
        if self.streaming_init_manifest:
            storage_index = self.streaming_init_manifest.get("storage_index")
        if storage_index:
            self.storage.load_index_manifest(storage_index)
            print(f"[PURE SSD RESUME] loaded log-structured storage index: {storage_index}")
        self.cache = TieredCacheManager(
            storage_manager=self.storage,
            max_ram_gb=self.max_ram_gb,
            block_size=self.block_size,
            point_dim=59,
            verbose=self.paper_debug_logging,
        )

    def _initialize_cameras(self) -> None:
        self._log("[TideStorageAdapter] Building camera schedule")
        camera_positions = []
        camera_directions = []
        for cam_info in self.cameras:
            R = np.array(cam_info.R).reshape(3, 3)
            T = np.array(cam_info.T).reshape(3, 1)
            camera_center = -R @ T
            camera_positions.append(camera_center.flatten())
            view_dir = R[:, 2]
            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-12)
            camera_directions.append(view_dir)

        camera_positions = np.asarray(camera_positions)
        camera_directions = np.asarray(camera_directions)
        self.camera_positions = camera_positions
        self.camera_directions = camera_directions

        cached = self._try_load_camera_schedule_cache(camera_positions, camera_directions)
        if cached is not None:
            self.camera_clusters = cached["camera_clusters"]
            self.block_hotspots = self._compute_block_hotspots()
            self._cached_training_schedule = cached["training_schedule"]
            self.scheduler = None
            self._log(
                f"[TideStorageAdapter] Loaded camera schedule cache: "
                f"{len(self._cached_training_schedule)} cameras"
            )
            return

        kmeans = KMeans(
            n_clusters=min(int(self.num_clusters), len(self.cameras)),
            random_state=42,
        )
        self.camera_clusters = kmeans.fit_predict(camera_positions)
        self.block_hotspots = self._compute_block_hotspots()
        self.scheduler = TSPScheduler(
            camera_positions=camera_positions,
            camera_directions=camera_directions,
            camera_clusters=self.camera_clusters,
            block_hotspots=self.block_hotspots,
        )
        self._cached_training_schedule = None
        self._log(f"[TideStorageAdapter] {len(self.cameras)} cameras in {self.num_clusters} clusters")

    def _camera_schedule_cache_key(
        self,
        camera_positions: np.ndarray,
        camera_directions: np.ndarray,
    ) -> str:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(camera_positions, dtype=np.float32).tobytes())
        h.update(np.ascontiguousarray(camera_directions, dtype=np.float32).tobytes())
        h.update(np.asarray(self.scene_min, dtype=np.float32).tobytes())
        h.update(np.asarray(self.scene_max, dtype=np.float32).tobytes())
        key_data = {
            "version": 1,
            "num_cameras": int(len(self.cameras)),
            "num_clusters": int(min(int(self.num_clusters), len(self.cameras))),
            "num_points": int(self.num_points),
            "num_blocks": int(self.num_blocks),
            "block_size": int(self.block_size),
            "use_6plane": bool(self.use_6plane),
        }
        h.update(json.dumps(key_data, sort_keys=True).encode("utf-8"))
        return h.hexdigest()[:24]

    def _camera_schedule_cache_path(
        self,
        camera_positions: np.ndarray,
        camera_directions: np.ndarray,
    ) -> Path:
        key = self._camera_schedule_cache_key(camera_positions, camera_directions)
        return self.schedule_cache_dir / f"pure_ssd_camera_schedule_{key}.npz"

    def _try_load_camera_schedule_cache(
        self,
        camera_positions: np.ndarray,
        camera_directions: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not self.schedule_cache_enabled:
            return None
        cache_path = self._camera_schedule_cache_path(camera_positions, camera_directions)
        if not cache_path.is_file():
            self._log(f"[TideStorageAdapter] Camera schedule cache miss: {cache_path}")
            return None
        try:
            data = np.load(cache_path)
            camera_clusters = data["camera_clusters"].astype(np.int64, copy=False)
            training_schedule = data["training_schedule"].astype(np.int64, copy=False).tolist()
        except Exception as exc:
            self._warn(f"[TideStorageAdapter] WARNING: Camera schedule cache ignored: {exc}")
            return None
        if len(camera_clusters) != len(self.cameras) or len(training_schedule) != len(self.cameras):
            self._warn("[TideStorageAdapter] WARNING: Camera schedule cache ignored: size mismatch")
            return None
        return {
            "camera_clusters": camera_clusters,
            "training_schedule": [int(i) for i in training_schedule],
        }

    def _save_camera_schedule_cache(
        self,
        schedule: List[int],
    ) -> None:
        if not self.schedule_cache_enabled:
            return
        cache_path = self._camera_schedule_cache_path(self.camera_positions, self.camera_directions)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            camera_clusters=np.asarray(self.camera_clusters, dtype=np.int64),
            training_schedule=np.asarray(schedule, dtype=np.int64),
        )
        self._log(f"[TideStorageAdapter] Saved camera schedule cache: {cache_path}")

    def _compute_block_hotspots(self) -> Dict[int, float]:
        hotspots = {}
        for block_id in range(self.num_blocks):
            bounds = self.block_bounds[block_id]
            block_center = (bounds[:3] + bounds[3:]) / 2.0
            distance = np.linalg.norm(block_center - self.scene_center)
            hotspots[block_id] = 1.0 / (1.0 + distance)
        return hotspots

    def _initialize_pipeline(self) -> None:
        self._log("[Pipeline] Using CPU block-wise frustum culling")
        self.culler = FrustumCuller(
            block_bounds=self.block_bounds,
            scene_radius=self.scene_radius,
            verbose=self.paper_debug_logging,
        )

        self.pipeline = AsyncPipeline(
            cache_manager=self.cache,
            frustum_culler=self.culler,
            block_size=self.block_size,
            point_dim=59,
            prefetch_ahead=3,
            verbose=self.paper_debug_logging,
        )
        self.pipeline.start()

    def get_training_schedule(self, shuffle: bool = False, repeat_epochs: int = 1) -> List[int]:
        if getattr(self, "_cached_training_schedule", None) is not None:
            schedule = list(self._cached_training_schedule)
        else:
            schedule = self.scheduler.get_tsp_camera_order()
            self._save_camera_schedule_cache(schedule)
        if shuffle:
            import random

            print("[WARNING] Shuffling enabled in SSD schedule - this will reduce cache hit rates")
            random.shuffle(schedule)
        if repeat_epochs > 1:
            schedule = schedule * repeat_epochs

        self._log(
            f"[SSD Schedule] Generated {len(schedule)} camera indices "
            f"(TSP order, {self.num_clusters} clusters, shuffle={shuffle})"
        )
        return schedule

    def get_visible_blocks(
        self,
        camera_idx: int,
        *,
        wait_for_refresh: bool = True,
    ) -> List[int]:
        if wait_for_refresh:
            self.wait_for_bounds_refresh()
        camera_idx = int(camera_idx)
        with self._bounds_state_lock:
            cached = self._visibility_cache.get(camera_idx)
            if cached is not None and cached[0] == self._bounds_generation:
                return list(cached[1])

        cam_info = self.cameras[camera_idx]
        R = np.array(cam_info.R).reshape(3, 3)
        T = np.array(cam_info.T).reshape(3, 1)
        camera_pos = (-R @ T).flatten()

        view_mat = np.eye(4, dtype=np.float32)
        view_mat[:3, :3] = R.T
        view_mat[:3, 3] = T.flatten()
        cv_to_gl = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        view_mat = cv_to_gl @ view_mat

        fov_y = cam_info.FovY if hasattr(cam_info, "FovY") else np.deg2rad(45)
        if hasattr(cam_info, "width") and hasattr(cam_info, "height"):
            aspect = cam_info.width / cam_info.height
        else:
            aspect = 16.0 / 9.0

        near = 0.01
        far = max(float(self.scene_radius) * 2.0, 1000.0)
        f = 1.0 / np.tan(fov_y / 2.0)
        proj_mat = np.array(
            [
                [f / aspect, 0, 0, 0],
                [0, f, 0, 0],
                [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

        if self.paper_debug_logging and (
            not hasattr(self, "_coord_check_done") or not self._coord_check_done
        ):
            self._coord_check_done = True
            gl_forward = -view_mat[2, :3]
            bb_min = self.block_bounds[:, :3].min(axis=0)
            bb_max = self.block_bounds[:, 3:].max(axis=0)
            n_sample = min(50, len(self.cameras))
            cam_positions = []
            for ci in range(n_sample):
                Ri = np.array(self.cameras[ci].R).reshape(3, 3)
                Ti = np.array(self.cameras[ci].T).reshape(3, 1)
                cam_positions.append((-Ri @ Ti).flatten())
            cam_positions = np.asarray(cam_positions)
            overlap = np.all(cam_positions.max(axis=0) >= bb_min) and np.all(
                cam_positions.min(axis=0) <= bb_max
            )
            print(f"[COORD_CHECK] Camera 0 position : {camera_pos}")
            print(f"[COORD_CHECK] Camera 0 GL-forward: {gl_forward}")
            print(f"[COORD_CHECK] View matrix det(R3x3): {np.linalg.det(view_mat[:3, :3]):.6f}")
            print(f"[COORD_CHECK] Block bounds min: {bb_min}")
            print(f"[COORD_CHECK] Block bounds max: {bb_max}")
            print(f"[COORD_CHECK] Camera pos min  : {cam_positions.min(axis=0)}")
            print(f"[COORD_CHECK] Camera pos max  : {cam_positions.max(axis=0)}")
            print(f"[COORD_CHECK] Camera-Block AABB overlap: {overlap}")
            print(f"[COORD_CHECK] Proj[2,2]={proj_mat[2, 2]:.6f} (OpenGL: should be < 0)")
            print(f"[COORD_CHECK] near={near}, far={far}, scene_radius={self.scene_radius}")

        with self._bounds_state_lock:
            bounds_generation = self._bounds_generation
            changed_blocks = self._changed_blocks_since_locked(
                cached[0] if cached is not None else None,
                bounds_generation,
            )
            can_repair = (
                cached is not None
                and changed_blocks is not None
                and len(changed_blocks) < int(self.culler.num_blocks)
                and hasattr(self.culler, "cull_subset")
            )
            if can_repair:
                changed_visible = self.culler.cull_subset(
                    camera_position=camera_pos,
                    view_matrix=view_mat,
                    projection_matrix=proj_mat,
                    block_ids=changed_blocks,
                    use_6plane=self.use_6plane,
                )
                visible_set = set(cached[1])
                visible_set.difference_update(changed_blocks)
                visible_set.update(changed_visible)
                visible_blocks = sorted(visible_set)
                self.execution_metrics.setdefault("visibility_incremental_repairs", 0)
                self.execution_metrics.setdefault("visibility_repaired_blocks", 0)
                self.execution_metrics["visibility_incremental_repairs"] += 1
                self.execution_metrics["visibility_repaired_blocks"] += len(changed_blocks)
            else:
                visible_blocks = self.culler.cull(
                    camera_position=camera_pos,
                    view_matrix=view_mat,
                    projection_matrix=proj_mat,
                    use_6plane=self.use_6plane,
                )
                self.execution_metrics.setdefault("visibility_full_culls", 0)
                self.execution_metrics["visibility_full_culls"] += 1

        if self.paper_debug_logging:
            if not hasattr(self, "_logged_cameras"):
                self._logged_cameras = set()
            if len(self._logged_cameras) < 20 or camera_idx % 100 == 0:
                if camera_idx not in self._logged_cameras:
                    print(
                        f"[FrustumCuller] Camera {camera_idx}: "
                        f"{len(visible_blocks)}/{self.culler.num_blocks} blocks visible "
                        f"({100 * len(visible_blocks) / self.culler.num_blocks:.1f}%)"
                    )
                    self._logged_cameras.add(camera_idx)

        with self._bounds_state_lock:
            if bounds_generation == self._bounds_generation:
                self._visibility_cache[camera_idx] = (
                    bounds_generation,
                    tuple(visible_blocks),
                )
                if len(self._visibility_cache) > 500:
                    oldest_key = next(iter(self._visibility_cache))
                    del self._visibility_cache[oldest_key]

        return visible_blocks

    def _changed_blocks_since_locked(self, cached_generation, current_generation):
        if cached_generation is None or cached_generation > current_generation:
            return None
        if cached_generation == current_generation:
            return set()

        change_log = getattr(self, "_bounds_change_log", None)
        if change_log is None:
            return None
        changed = set()
        for generation in range(int(cached_generation) + 1, int(current_generation) + 1):
            generation_blocks = change_log.get(generation)
            if generation_blocks is None:
                return None
            changed.update(generation_blocks)
        return changed

    def get_visible_blocks_batch(self, camera_ids: List[int]):
        self.wait_for_bounds_refresh()
        with self._bounds_state_lock:
            generation = int(self._bounds_generation)
            camera_blocks = {
                int(camera_id): self.get_visible_blocks(
                    int(camera_id),
                    wait_for_refresh=False,
                )
                for camera_id in camera_ids
            }
        return generation, camera_blocks

    def refresh_block_bounds_from_blocks(self, updated_blocks_dict: Dict[int, torch.Tensor]) -> int:
        if not updated_blocks_dict:
            return 0

        block_ids = []
        bounds = []
        for block_id, block_tensor in updated_blocks_dict.items():
            if block_tensor is None:
                continue
            valid_rows = None
            if hasattr(self, "num_points") and hasattr(self, "block_size"):
                block_start = int(block_id) * int(self.block_size)
                valid_rows = max(0, min(int(self.block_size), int(self.num_points) - block_start))
            if torch.is_tensor(block_tensor):
                if block_tensor.ndim != 2 or block_tensor.shape[1] < 3 or block_tensor.shape[0] == 0:
                    continue
                row_count = int(block_tensor.shape[0])
                rows_to_use = row_count if valid_rows is None else min(row_count, valid_rows)
                if rows_to_use <= 0:
                    continue
                xyz = block_tensor.detach()[:rows_to_use, :3]
                xyz_cpu = xyz.cpu() if xyz.is_cuda else xyz
                xyz_np = xyz_cpu.numpy()
            else:
                block_array = np.asarray(block_tensor)
                if block_array.ndim != 2 or block_array.shape[1] < 3 or block_array.shape[0] == 0:
                    continue
                row_count = int(block_array.shape[0])
                rows_to_use = row_count if valid_rows is None else min(row_count, valid_rows)
                if rows_to_use <= 0:
                    continue
                xyz_np = block_array[:rows_to_use, :3]

            xyz_np = np.asarray(xyz_np, dtype=np.float32)
            xyz_min = xyz_np.min(axis=0)
            xyz_max = xyz_np.max(axis=0)
            block_ids.append(int(block_id))
            bounds.append(np.concatenate([xyz_min, xyz_max]).astype(np.float32, copy=False))

        if not block_ids:
            return 0

        return self.update_block_bounds(block_ids, np.stack(bounds, axis=0))

    def update_block_bounds(self, block_ids, bounds) -> int:
        block_ids_np = np.asarray(block_ids, dtype=np.int64)
        if torch.is_tensor(bounds):
            bounds = bounds.detach().cpu().numpy()
        bounds_np = np.asarray(bounds, dtype=np.float32)
        if bounds_np.shape != (block_ids_np.size, 6):
            raise ValueError(
                f"Expected bounds shape {(block_ids_np.size, 6)}, got {bounds_np.shape}"
            )
        with self._bounds_state_lock:
            if not self.block_bounds.flags.writeable:
                self.block_bounds = np.array(self.block_bounds, dtype=np.float32, copy=True)
            self.block_bounds[block_ids_np] = bounds_np

            culler = getattr(self, "culler", None)
            if culler is not None and hasattr(culler, "update_block_bounds"):
                culler.update_block_bounds(block_ids_np, bounds_np)
            self._bounds_generation += 1
            if not hasattr(self, "_bounds_change_log"):
                self._bounds_change_log = OrderedDict()
            generation = int(self._bounds_generation)
            self._bounds_change_log[generation] = tuple(
                sorted(set(int(block_id) for block_id in block_ids_np.tolist()))
            )
            history_limit = int(getattr(self, "_bounds_change_history_limit", 64))
            while len(self._bounds_change_log) > history_limit:
                self._bounds_change_log.popitem(last=False)

        refreshed = int(block_ids_np.size)
        self.execution_metrics["paper_bounds_refresh_blocks"] += refreshed
        return refreshed

    def get_bounds_generation(self) -> int:
        with self._bounds_state_lock:
            return int(self._bounds_generation)

    def sync_cache_from_cpu_views(
        self,
        updated_blocks_dict: Dict[int, torch.Tensor],
        *,
        refresh_bounds: bool = True,
    ):
        if not updated_blocks_dict:
            return 0

        staged = len(updated_blocks_dict)
        self.execution_metrics["paper_cache_sync_calls"] += 1
        self.execution_metrics["paper_cache_sync_blocks"] += staged
        staged_valid = self.cache.upsert_dirty_block_batch(updated_blocks_dict)
        self.execution_metrics["paper_cache_sync_staged_blocks"] += staged_valid
        if refresh_bounds:
            self.refresh_block_bounds_from_blocks(updated_blocks_dict)
        return staged_valid

    def get_stats(self) -> Dict:
        return {
            "execution": dict(self.execution_metrics),
            "pipeline": self.pipeline.get_stats(),
            "cache": self.cache.get_stats(),
            "storage": self.storage.get_stats(),
        }

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._log("[TideStorageAdapter] Shutting down...")
        errors = []

        def run_cleanup(name, cleanup):
            try:
                cleanup()
            except Exception as exc:
                errors.append((name, exc))

        run_cleanup("flush resident dirty blocks", self.flush_resident_dirty)
        run_cleanup("drain cache writebacks", self.drain_cache_writebacks)
        run_cleanup("stop cache commit worker", lambda: self._cache_commit_queue.put(None))
        run_cleanup("join cache commit queue", self._cache_commit_queue.join)
        if self._cache_commit_thread is not None:
            run_cleanup(
                "join cache commit thread",
                lambda: self._cache_commit_thread.join(timeout=5.0),
            )
        run_cleanup("wait for bounds refresh", self.wait_for_bounds_refresh)
        run_cleanup("stop bounds refresh worker", lambda: self._bounds_refresh_queue.put(None))
        run_cleanup("join bounds refresh queue", self._bounds_refresh_queue.join)
        if self._bounds_refresh_thread is not None:
            run_cleanup(
                "join bounds refresh thread",
                lambda: self._bounds_refresh_thread.join(timeout=5.0),
            )
        run_cleanup("shutdown async pipeline", self.pipeline.shutdown)
        run_cleanup("shutdown cache", self.cache.shutdown)
        run_cleanup(
            "compact storage",
            lambda: self.storage.maybe_compact(min_patches=2, force=True),
        )
        run_cleanup("close storage", self.storage.close)

        if errors:
            details = "; ".join(f"{name}: {exc}" for name, exc in errors)
            self._log(f"[TideStorageAdapter] Shutdown completed with errors: {details}")
            raise RuntimeError(f"TideStorageAdapter shutdown failed: {details}") from errors[0][1]

        self._log("[TideStorageAdapter] Shutdown complete")
