"""
Log-Structured Storage Manager for Billion-scale Gaussian Splatting.

Implements append-only patch-based storage to optimize for SSD characteristics:
- Sequential writes for performance and longevity
- Metadata index for efficient block lookup
- Patch file management with compaction support
"""

import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch


@dataclass
class BlockLocation:
    """Metadata for locating a block in storage."""
    file_id: int  # 0 = base_file, 1+ = patch files
    offset: int   # Byte offset in the file
    size: int     # Size in bytes
    version: int  # Version number for tracking updates


class StorageCapacityError(RuntimeError):
    """Raised before a patch write would consume the configured free-space reserve."""


class LogStorageManager:
    """
    Log-Structured Storage Manager with append-only semantics.

    Architecture:
    - base_file.bin: Initial complete Gaussian dataset
    - patch_XXX.bin: Update logs (append-only)
    - Metadata index maps block_id -> (file, offset, size)
    """

    def __init__(
        self,
        storage_dir: str,
        block_size: int,
        num_blocks: int,
        point_dim: int = 59,  # 3(xyz) + 3(scale) + 4(rot) + 3(opacity) + 48(SH)
        dtype: torch.dtype = torch.float32,
        verbose: bool = True,
        max_patch_files: int = 16,
        max_stale_patch_gb: float = 64.0,
        max_patch_total_gb: float = 128.0,
        min_free_gb: float = 64.0,
    ):
        """
        Initialize Log-Structured Storage Manager.

        Args:
            storage_dir: Directory for base and patch files
            block_size: Number of Gaussian points per block
            num_blocks: Total number of blocks
            point_dim: Dimension per Gaussian point (default 59)
            dtype: Data type for storage
            verbose: Print routine storage lifecycle messages
            max_patch_files: Compact when the active patch count reaches this value
            max_stale_patch_gb: Compact when reclaimable stale patch data reaches this size
            max_patch_total_gb: Compact when active patch data reaches this size
            min_free_gb: Free-space reserve enforced before writes and compaction
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.block_size = block_size
        self.num_blocks = num_blocks
        self.point_dim = point_dim
        self.dtype = dtype
        self.verbose = bool(verbose)
        self.bytes_per_point = point_dim * torch.finfo(dtype).bits // 8
        self.bytes_per_block = block_size * self.bytes_per_point
        self.max_patch_files = max(2, int(max_patch_files))
        self.max_stale_patch_bytes = max(0, int(float(max_stale_patch_gb) * (1024 ** 3)))
        self.max_patch_total_bytes = max(0, int(float(max_patch_total_gb) * (1024 ** 3)))
        self.min_free_bytes = max(0, int(float(min_free_gb) * (1024 ** 3)))

        # Metadata index: block_id -> BlockLocation
        # 用于快速查找任意 Block 当前存储在文件的哪个位置
        self.index: Dict[int, BlockLocation] = {}
        # 线程锁: 防止多线程同时修改索引导致竞争条件
        self.index_lock = threading.Lock()

        # File handles (lazy opened)
        # file_handles: 缓存已打开的文件对象，避免重复打开关闭
        self.file_handles: Dict[int, object] = {}  # file_id -> open file handle
        # 维护 file_id 到磁盘物理路径的映射
        self.file_paths: Dict[int, Path] = {0: self.storage_dir / "base_file.bin"}

        # Patch file counter
        self.next_patch_id = 1
        # 计数器锁：确保多线程并发场景 patch 时ID不会冲突
        self.patch_counter_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._operation_condition = threading.Condition()
        self._maintenance_active = False
        self._active_operations = 0

        # Statistics
        self.stats = {
            'reads': 0,
            'writes': 0,
            'patches_created': 0,
            'patch_write_jobs': 0,
            'patch_write_bytes': 0,
            'patch_write_time': 0.0,
            'compactions': 0,
            'compaction_time': 0.0,
            'compaction_input_bytes': 0,
            'compaction_output_bytes': 0,
            'compaction_reclaimed_bytes': 0,
            'compactions_deferred': 0,
        }

        # 执行初始化逻辑， 建立初始的文件索引映射
        self._initialize_index()
        self._remove_abandoned_temp_files()

    @contextmanager
    def _storage_operation(self):
        with self._operation_condition:
            while self._maintenance_active:
                self._operation_condition.wait()
            self._active_operations += 1
        try:
            yield
        finally:
            with self._operation_condition:
                self._active_operations -= 1
                if self._active_operations == 0:
                    self._operation_condition.notify_all()

    @contextmanager
    def _exclusive_maintenance(self):
        with self._operation_condition:
            while self._maintenance_active:
                self._operation_condition.wait()
            self._maintenance_active = True
            while self._active_operations:
                self._operation_condition.wait()
        try:
            yield
        finally:
            with self._operation_condition:
                self._maintenance_active = False
                self._operation_condition.notify_all()

    def _remove_abandoned_temp_files(self) -> None:
        for path in self.storage_dir.glob(".tide_compact_*.tmp"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _check_free_space(self, additional_bytes: int, operation: str) -> None:
        free_bytes = shutil.disk_usage(self.storage_dir).free
        required = max(0, int(additional_bytes)) + self.min_free_bytes
        if free_bytes < required:
            raise StorageCapacityError(
                f"Insufficient space for {operation}: free={free_bytes / (1024 ** 3):.2f} GiB, "
                f"required={required / (1024 ** 3):.2f} GiB including "
                f"reserve={self.min_free_bytes / (1024 ** 3):.2f} GiB"
            )

    def _initialize_index(self):
        """Initialize metadata index pointing all blocks to base file."""
        base_file = self.file_paths[0]

        if base_file.exists():
            # Load existing base file
            for block_id in range(self.num_blocks):
                self.index[block_id] = BlockLocation(
                    file_id=0,
                    offset=block_id * self.bytes_per_block,
                    size=self.bytes_per_block,
                    version=0
                )
            if self.verbose:
                print(f"[LogStorage] Loaded base file with {self.num_blocks} blocks")
        else:
            # Create empty base file
            if self.verbose:
                print(f"[LogStorage] Creating new base file at {base_file}")
            with open(base_file, 'wb') as f:
                # Pre-allocate space
                # 将文件指针移动到文件最后一个字节的位置
                f.seek(self.num_blocks * self.bytes_per_block - 1)
                # 写入一个空字节，操作系统会立刻为文件分配这个大的逻辑大小，不需要将中间都填满0
                f.write(b'\0')

            # 初始化索引，将所有 block 指向这个新创建的空文件
            for block_id in range(self.num_blocks):
                self.index[block_id] = BlockLocation(
                    file_id=0,
                    offset=block_id * self.bytes_per_block,
                    size=self.bytes_per_block,
                    version=0
                )

    def read_blocks(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        with self._storage_operation():
            return self._read_blocks_uncoordinated(block_ids)

    def _read_blocks_uncoordinated(self, block_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Read multiple blocks from storage.
        从存储中批量读取多个 blocks
        Groups reads by file to minimize seeking and maximize sequential access.
        核心优化：按文件分组，按offset排序，最小化seeking时间，最大化顺序访问
        Args:
            block_ids: List of block IDs to read

        Returns:
            Dictionary mapping block_id -> tensor data
        """
        if not block_ids:
            return {}

        result = {}

        # Group blocks by file for efficient batched reads
        # 建立分组字典： k是文件id，v是(block_id, Blocklocation)列表
        file_groups: Dict[int, List[Tuple[int, BlockLocation]]] = {}

        # 读取索引时加锁，防止读取过程中索引被后台写线程修改
        with self.index_lock:
            for block_id in block_ids:
                if block_id not in self.index:
                    raise ValueError(f"Block {block_id} not in index")

                location = self.index[block_id]
                if location.file_id not in file_groups:
                    file_groups[location.file_id] = []
                file_groups[location.file_id].append((block_id, location))

        # Read from each file
        for file_id, blocks in file_groups.items():
            file_path = self.file_paths[file_id]

            # 核心优化: Sort by offset for sequential reading
            # 这样 f.seek() 总是向后移动，不会来回跳跃，极大提升了SSD 的吞吐量
            blocks.sort(key=lambda x: x[1].offset)

            with open(file_path, 'rb') as f:
                for block_id, location in blocks:
                    if location.size <= 0:
                        print(
                            f"[LogStorage] WARNING: block {block_id} points to an empty patch record; "
                            "falling back to base file"
                        )
                        result[block_id] = self._read_base_block(block_id)
                        continue

                    # 移动指针到block的起始位置
                    f.seek(location.offset)
                    # 读取指定大小的二进制数据
                    data_bytes = f.read(location.size)

                    # Convert to tensor (copy to make it writable)
                    np_array = np.frombuffer(data_bytes, dtype=np.float32).copy() 
                    curr_num_points = np_array.size // self.point_dim
                    if curr_num_points <= 0:
                        print(
                            f"[LogStorage] WARNING: block {block_id} read as empty from {file_path}; "
                            "falling back to base file"
                        )
                        tensor = self._read_base_block(block_id)
                    else:
                        tensor = torch.from_numpy(np_array).reshape(curr_num_points, self.point_dim)
                    result[block_id] = tensor

            self.stats['reads'] += len(blocks)

        return result

    def _read_base_block(self, block_id: int) -> torch.Tensor:
        """Read a block directly from the immutable base segment."""
        base_path = self.file_paths[0]
        offset = int(block_id) * self.bytes_per_block
        with open(base_path, 'rb') as f:
            f.seek(offset)
            data_bytes = f.read(self.bytes_per_block)
        np_array = np.frombuffer(data_bytes, dtype=np.float32).copy()
        curr_num_points = np_array.size // self.point_dim
        if curr_num_points <= 0:
            raise RuntimeError(f"Base segment returned empty data for block {block_id}")
        return torch.from_numpy(np_array).reshape(curr_num_points, self.point_dim)

    def write_patch(
        self,
        block_dict: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
    ) -> int:
        if not block_dict:
            return -1
        estimated_bytes = sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in block_dict.values()
            if tensor is not None and torch.is_tensor(tensor)
        )
        with self._write_lock:
            if self._compaction_needed():
                self.maybe_compact(min_patches=2)
            self._check_free_space(estimated_bytes, "patch write")
            with self._storage_operation():
                patch_id = self._write_patch_uncoordinated(
                    block_dict,
                    block_versions=block_versions,
                )
            if self._compaction_needed():
                self.maybe_compact(min_patches=2)
            return patch_id

    def _write_patch_uncoordinated(
        self,
        block_dict: Dict[int, torch.Tensor],
        block_versions: Optional[Dict[int, int]] = None,
    ) -> int:
        """
        Write updated blocks to a new patch file (append-only).

        Args:
            block_dict: Dictionary mapping block_id -> updated tensor data

        Returns:
            patch_id: ID of the created patch file
        """
        if not block_dict:
            return -1

        valid_blocks = {}
        for block_id, tensor in block_dict.items():
            if tensor is None or tensor.numel() == 0:
                print(f"[LogStorage] WARNING: skip empty dirty block {block_id}")
                continue
            if tensor.dim() != 2 or tensor.shape[1] != self.point_dim:
                raise ValueError(
                    f"Invalid dirty block {block_id} shape {tuple(tensor.shape)}; "
                    f"expected (*, {self.point_dim})"
                )
            if tensor.shape[0] > self.block_size:
                raise ValueError(
                    f"Invalid dirty block {block_id} rows {tensor.shape[0]}; "
                    f"expected at most {self.block_size}"
                )
            valid_blocks[int(block_id)] = tensor

        if not valid_blocks:
            print("[LogStorage] Skipped patch creation: no non-empty dirty blocks")
            return -1

        incoming_versions = {
            int(block_id): int(version)
            for block_id, version in (block_versions or {}).items()
        }
        with self.index_lock:
            accepted_blocks = {}
            for block_id, tensor in valid_blocks.items():
                current_version = (
                    self.index[block_id].version if block_id in self.index else 0
                )
                incoming_version = incoming_versions.get(block_id)
                if incoming_version is None or incoming_version > current_version:
                    accepted_blocks[block_id] = tensor
        valid_blocks = accepted_blocks

        if not valid_blocks:
            return -1

        # Allocate new patch file ID
        with self.patch_counter_lock:
            patch_id = self.next_patch_id
            self.next_patch_id += 1

        # Create patch file
        timestamp = int(time.time() * 1000000)
        patch_file = self.storage_dir / f"patch_{patch_id:06d}_{timestamp}.bin"
        # Write blocks sequentially
        current_offset = 0
        updated_locations = {}

        write_started = time.perf_counter()
        try:
            with open(patch_file, 'wb') as f:
                for block_id in sorted(valid_blocks.keys()):  # Sort for determinism
                    tensor = valid_blocks[block_id]

                    if tensor.is_cuda:
                        tensor = tensor.cpu()

                    data_bytes = tensor.numpy().astype(np.float32, copy=False).tobytes()
                    f.write(data_bytes)

                    with self.index_lock:
                        old_version = self.index[block_id].version if block_id in self.index else 0
                    incoming_version = incoming_versions.get(block_id)
                    updated_locations[block_id] = BlockLocation(
                        file_id=patch_id,
                        offset=current_offset,
                        size=len(data_bytes),
                        version=(
                            incoming_version
                            if incoming_version is not None
                            else old_version + 1
                        ),
                    )
                    current_offset += len(data_bytes)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            patch_file.unlink(missing_ok=True)
            raise
        finally:
            self.stats['patch_write_time'] += time.perf_counter() - write_started

        # Update index atomically
        # 只有文件全部写完落盘后，才会更新内存索引，确保不会读到写了一半的文件
        with self.index_lock:
            self.file_paths[patch_id] = patch_file
            for block_id, location in updated_locations.items():
                self.index[block_id] = location

        self.stats['writes'] += len(valid_blocks)
        self.stats['patches_created'] += 1
        self.stats['patch_write_jobs'] += 1
        self.stats['patch_write_bytes'] += current_offset

        print(f"[LogStorage] Created patch {patch_id} with {len(valid_blocks)} blocks ({current_offset / 1024 / 1024:.2f} MB)")

        return patch_id

    def get_block_versions(self, block_ids: List[int]) -> Dict[int, int]:
        with self.index_lock:
            return {
                int(block_id): int(self.index[int(block_id)].version)
                for block_id in block_ids
                if int(block_id) in self.index
            }

    def export_index_manifest(
        self,
        manifest_path: str | Path,
        patches_dir: str | Path,
        copy_patch_files: Optional[bool] = None,
        patch_file_mode: str = "hardlink",
    ) -> Dict:
        with self._write_lock:
            with self._storage_operation():
                return self._export_index_manifest_uncoordinated(
                    manifest_path=manifest_path,
                    patches_dir=patches_dir,
                    copy_patch_files=copy_patch_files,
                    patch_file_mode=patch_file_mode,
                )

    def _export_index_manifest_uncoordinated(
        self,
        manifest_path: str | Path,
        patches_dir: str | Path,
        copy_patch_files: Optional[bool] = None,
        patch_file_mode: str = "hardlink",
    ) -> Dict:
        """Persist the current log-structured index for incremental checkpoints.

        The immutable base file is referenced by absolute path. Patch files are
        hard-linked into the checkpoint by default, avoiding duplicate physical
        bytes while keeping the checkpoint valid after cache-side garbage collection.
        """
        if copy_patch_files is not None:
            if not copy_patch_files:
                raise ValueError("Reference-only checkpoints are unsafe with patch garbage collection")
            patch_file_mode = "copy"
        patch_file_mode = str(patch_file_mode).lower()
        if patch_file_mode not in {"hardlink", "copy"}:
            raise ValueError(
                f"Invalid patch_file_mode={patch_file_mode!r}; expected hardlink or copy"
            )
        manifest_path = Path(manifest_path)
        patches_dir = Path(patches_dir)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        with self.index_lock:
            index_snapshot = dict(self.index)
            file_paths_snapshot = dict(self.file_paths)
        with self.patch_counter_lock:
            next_patch_id = int(self.next_patch_id)

        referenced_file_ids = {0}
        referenced_file_ids.update(int(location.file_id) for location in index_snapshot.values())
        files = {}
        patch_files = 0
        patch_bytes = 0
        copied_patch_files = 0
        copied_patch_bytes = 0
        linked_patch_files = 0
        linked_patch_bytes = 0
        for file_id, source_path in sorted(file_paths_snapshot.items()):
            file_id = int(file_id)
            if file_id not in referenced_file_ids:
                continue
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"Log storage file missing for file_id={file_id}: {source_path}")

            linked = False
            copied = False
            if file_id == 0:
                checkpoint_path = source_path.resolve()
            else:
                checkpoint_path = patches_dir / source_path.name
                if source_path.resolve() != checkpoint_path.resolve():
                    if checkpoint_path.exists():
                        same_file = os.path.samefile(source_path, checkpoint_path)
                        if same_file and patch_file_mode == "hardlink":
                            linked = True
                        else:
                            checkpoint_path.unlink()
                    if not checkpoint_path.exists():
                        if patch_file_mode == "hardlink":
                            try:
                                os.link(source_path, checkpoint_path)
                                linked = True
                            except OSError:
                                shutil.copy2(source_path, checkpoint_path)
                                copied = True
                        else:
                            shutil.copy2(source_path, checkpoint_path)
                            copied = True
                checkpoint_path = checkpoint_path.resolve()

            size = int(checkpoint_path.stat().st_size)
            if file_id != 0:
                patch_files += 1
                patch_bytes += size
                if copied:
                    copied_patch_files += 1
                    copied_patch_bytes += size
                if linked:
                    linked_patch_files += 1
                    linked_patch_bytes += size

            files[str(file_id)] = {
                "path": str(checkpoint_path),
                "role": "base" if file_id == 0 else "patch",
                "copied": bool(copied),
                "linked": bool(linked),
                "size": size,
            }

        index_payload = {
            str(block_id): {
                "file_id": int(location.file_id),
                "offset": int(location.offset),
                "size": int(location.size),
                "version": int(location.version),
            }
            for block_id, location in sorted(index_snapshot.items())
        }
        manifest = {
            "version": 1,
            "storage_type": "log_structured_blocks",
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
            "point_dim": int(self.point_dim),
            "dtype": "float32",
            "bytes_per_block": int(self.bytes_per_block),
            "next_patch_id": int(next_patch_id),
            "files": files,
            "index": index_payload,
            "patch_file_mode": patch_file_mode,
            "patch_files": int(patch_files),
            "patch_bytes": int(patch_bytes),
            "copied_patch_files": int(copied_patch_files),
            "copied_patch_bytes": int(copied_patch_bytes),
            "linked_patch_files": int(linked_patch_files),
            "linked_patch_bytes": int(linked_patch_bytes),
        }
        manifest_temp = manifest_path.with_name(f".{manifest_path.name}.tmp")
        with open(manifest_temp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_temp, manifest_path)
        return manifest

    def load_index_manifest(self, manifest_path: str | Path) -> Dict:
        """Load a previously persisted log-structured index."""
        manifest_path = Path(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        if int(manifest.get("block_size", -1)) != int(self.block_size):
            raise ValueError(
                f"Index block_size mismatch: manifest={manifest.get('block_size')} runtime={self.block_size}"
            )
        if int(manifest.get("num_blocks", -1)) != int(self.num_blocks):
            raise ValueError(
                f"Index num_blocks mismatch: manifest={manifest.get('num_blocks')} runtime={self.num_blocks}"
            )
        if int(manifest.get("point_dim", -1)) != int(self.point_dim):
            raise ValueError(
                f"Index point_dim mismatch: manifest={manifest.get('point_dim')} runtime={self.point_dim}"
            )

        loaded_file_paths: Dict[int, Path] = {}
        for file_id_raw, file_info in manifest.get("files", {}).items():
            file_id = int(file_id_raw)
            path = Path(file_info["path"])
            if not path.exists():
                raise FileNotFoundError(f"Index file_id={file_id} path does not exist: {path}")
            loaded_file_paths[file_id] = path

        loaded_index: Dict[int, BlockLocation] = {}
        for block_id_raw, location in manifest.get("index", {}).items():
            block_id = int(block_id_raw)
            file_id = int(location["file_id"])
            if file_id not in loaded_file_paths:
                raise KeyError(f"Block {block_id} references missing file_id={file_id}")
            loaded_index[block_id] = BlockLocation(
                file_id=file_id,
                offset=int(location["offset"]),
                size=int(location["size"]),
                version=int(location["version"]),
            )

        if len(loaded_index) != int(self.num_blocks):
            raise ValueError(
                f"Index block count mismatch: manifest={len(loaded_index)} runtime={self.num_blocks}"
            )

        self.close()
        with self.index_lock:
            self.file_paths = loaded_file_paths
            self.index = loaded_index
        with self.patch_counter_lock:
            max_file_id = max(loaded_file_paths.keys()) if loaded_file_paths else 0
            self.next_patch_id = max(int(manifest.get("next_patch_id", 1)), max_file_id + 1)
        print(
            f"[LogStorage] Loaded index manifest {manifest_path} "
            f"files={len(self.file_paths)} next_patch_id={self.next_patch_id}"
        )
        return manifest

    def _patch_state(self):
        with self.index_lock:
            index_snapshot = dict(self.index)
            file_paths_snapshot = dict(self.file_paths)
        patch_paths = {
            int(file_id): Path(path)
            for file_id, path in file_paths_snapshot.items()
            if int(file_id) != 0
        }
        patch_bytes = sum(
            path.stat().st_size for path in patch_paths.values() if path.exists()
        )
        live_block_ids = sorted(
            block_id
            for block_id, location in index_snapshot.items()
            if int(location.file_id) != 0
        )
        live_bytes = sum(int(index_snapshot[block_id].size) for block_id in live_block_ids)
        return index_snapshot, file_paths_snapshot, patch_paths, patch_bytes, live_block_ids, live_bytes

    def _compaction_needed(self) -> bool:
        _, _, patch_paths, patch_bytes, _, live_bytes = self._patch_state()
        stale_bytes = max(0, patch_bytes - live_bytes)
        return len(patch_paths) >= self.max_patch_files or (
            self.max_patch_total_bytes > 0
            and patch_bytes >= self.max_patch_total_bytes
        ) or (
            self.max_stale_patch_bytes > 0
            and stale_bytes >= self.max_stale_patch_bytes
        )

    def _is_managed_patch(self, path: Path) -> bool:
        try:
            return (
                path.parent.resolve() == self.storage_dir.resolve()
                and path.name.startswith("patch_")
                and path.suffix == ".bin"
            )
        except OSError:
            return False

    def compact_patches(self, min_patches: int = 2, force: bool = False) -> bool:
        """Merge the latest live patch records without rewriting or deleting the base.

        Readers and writers pause only while a compacted delta is built and the in-memory
        index is switched. Patch paths outside this run directory are never deleted; this
        protects resume checkpoints and shared immutable data.
        """
        with self._write_lock:
            with self._exclusive_maintenance():
                (
                    index_snapshot,
                    file_paths_snapshot,
                    patch_paths,
                    patch_bytes,
                    live_block_ids,
                    live_bytes,
                ) = self._patch_state()
                if len(patch_paths) < max(1, int(min_patches)):
                    return False
                if not force and not self._compaction_needed():
                    return False

                start_time = time.perf_counter()
                self._check_free_space(live_bytes, "patch compaction")
                with self.patch_counter_lock:
                    compacted_file_id = self.next_patch_id
                    self.next_patch_id += 1

                timestamp = int(time.time() * 1_000_000)
                temp_path = self.storage_dir / f".tide_compact_{compacted_file_id:06d}_{timestamp}.tmp"
                final_path = self.storage_dir / f"patch_{compacted_file_id:06d}_{timestamp}_compact.bin"
                new_locations: Dict[int, BlockLocation] = {}
                output_bytes = 0
                switched = False

                try:
                    with open(temp_path, "wb") as output:
                        for chunk_start in range(0, len(live_block_ids), 256):
                            block_ids = live_block_ids[chunk_start:chunk_start + 256]
                            blocks = self._read_blocks_uncoordinated(block_ids)
                            for block_id in block_ids:
                                tensor = blocks[block_id]
                                data = tensor.numpy().astype(np.float32, copy=False).tobytes()
                                output.write(data)
                                previous = index_snapshot[block_id]
                                new_locations[block_id] = BlockLocation(
                                    file_id=compacted_file_id,
                                    offset=output_bytes,
                                    size=len(data),
                                    version=int(previous.version),
                                )
                                output_bytes += len(data)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temp_path, final_path)

                    new_index = dict(index_snapshot)
                    new_index.update(new_locations)
                    new_file_paths = {0: Path(file_paths_snapshot[0])}
                    if live_block_ids:
                        new_file_paths[compacted_file_id] = final_path
                    with self.index_lock:
                        self.index = new_index
                        self.file_paths = new_file_paths
                        self.file_handles.clear()
                    switched = True
                finally:
                    temp_path.unlink(missing_ok=True)
                    if not switched:
                        final_path.unlink(missing_ok=True)

                managed_input_bytes = 0
                for path in patch_paths.values():
                    if path == final_path or not self._is_managed_patch(path):
                        continue
                    try:
                        managed_input_bytes += path.stat().st_size
                        path.unlink()
                    except FileNotFoundError:
                        pass

                compaction_elapsed = time.perf_counter() - start_time
                self.stats["compactions"] += 1
                self.stats["compaction_time"] += compaction_elapsed
                self.stats["compaction_input_bytes"] += int(patch_bytes)
                self.stats["compaction_output_bytes"] += int(output_bytes)
                self.stats["compaction_reclaimed_bytes"] += max(
                    0, int(managed_input_bytes) - int(output_bytes)
                )
                print(
                    f"[LogStorage] Compacted {len(patch_paths)} patches into "
                    f"{len(live_block_ids)} live blocks ({output_bytes / (1024 ** 3):.2f} GiB) "
                    f"in {compaction_elapsed:.2f}s"
                )
                return True

    def maybe_compact(self, min_patches: int = 2, force: bool = False) -> bool:
        """Run maintenance when space permits, without invalidating a completed write."""
        try:
            return self.compact_patches(min_patches=min_patches, force=force)
        except StorageCapacityError as exc:
            self.stats["compactions_deferred"] += 1
            print(f"[LogStorage] Compaction deferred: {exc}")
            return False

    def compact(self, min_patches: int = 10) -> bool:
        """Compatibility wrapper for explicit maintenance callers."""
        return self.compact_patches(min_patches=min_patches, force=True)

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        _, file_paths, patch_paths, patch_bytes, live_block_ids, live_bytes = self._patch_state()
        total_files = len(file_paths)
        total_size = sum(p.stat().st_size for p in file_paths.values() if p.exists())
        free_bytes = shutil.disk_usage(self.storage_dir).free

        return {
            **self.stats,
            'total_files': total_files,
            'total_size_mb': total_size / 1024 / 1024,
            'num_patches': len(patch_paths),
            'patch_size_mb': patch_bytes / 1024 / 1024,
            'live_patch_blocks': len(live_block_ids),
            'live_patch_size_mb': live_bytes / 1024 / 1024,
            'stale_patch_size_mb': max(0, patch_bytes - live_bytes) / 1024 / 1024,
            'free_space_gb': free_bytes / (1024 ** 3),
        }

    def get_timing_stats(self) -> Dict:
        """Return write and compaction counters without scanning storage files."""
        return {
            'patch_write_jobs': self.stats['patch_write_jobs'],
            'patch_write_bytes': self.stats['patch_write_bytes'],
            'patch_write_time': self.stats['patch_write_time'],
            'compaction_jobs': self.stats['compactions'],
            'compaction_time': self.stats['compaction_time'],
        }

    def close(self):
        """Close all file handles."""
        for fh in self.file_handles.values():
            if hasattr(fh, 'close'):
                fh.close()
        self.file_handles.clear()
        try:
            stats = self.get_stats()
        except Exception as exc:
            stats = {"error": str(exc)}
        if "error" in stats:
            print(f"[LogStorage] Closed with stats error: {stats['error']}")
        else:
            print(
                "[LogStorage] Closed summary: "
                f"reads={stats.get('reads', 0)} "
                f"writes={stats.get('writes', 0)} "
                f"patches={stats.get('patches_created', 0)} "
                f"files={stats.get('total_files', 0)} "
                f"size_mb={stats.get('total_size_mb', 0.0):.1f}"
            )
