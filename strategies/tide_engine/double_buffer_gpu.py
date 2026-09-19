"""
Double-Buffered GPU Working Set with N+1 Prefetch

This module implements pipeline parallelism for the Pure SSD/Tide path:
- Double buffer: Two GPU working sets that alternate each iteration
- N+1 prefetch: While GPU computes iteration N, load data for N+1 in background

Architecture:
    Iteration N:
        GPU Compute Stream:  [Forward/Backward on Buffer A]
        GPU Prefetch Stream: [Load N+1 data into Buffer B]
    
    Iteration N+1:
        GPU Compute Stream:  [Forward/Backward on Buffer B]  
        GPU Prefetch Stream: [Load N+2 data into Buffer A]

This overlaps N+1 preparation with current-batch compute when the compute
window is long enough to hide that work.
"""

import torch
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the tested PyTorch wheel includes Triton.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _copy_resident_rows_kernel(
        source_xyz,
        source_scaling,
        source_rotation,
        source_opacity,
        source_features_dc,
        source_features_rest,
        source_global_ids,
        source_starts,
        target_starts,
        target_xyz,
        target_scaling,
        target_rotation,
        target_opacity,
        target_features_dc,
        target_features_rest,
        target_global_ids,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
        COPY_BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * COPY_BLOCK_SIZE + tl.arange(0, COPY_BLOCK_SIZE)
        total_values = num_rows * 59
        mask = offsets < total_values
        row = offsets // 59
        column = offsets - row * 59
        block_position = row // BLOCK_ROWS
        row_in_block = row - block_position * BLOCK_ROWS
        source_row = tl.load(source_starts + block_position, mask=mask, other=0) + row_in_block
        target_row = tl.load(target_starts + block_position, mask=mask, other=0) + row_in_block

        value = tl.zeros((COPY_BLOCK_SIZE,), dtype=tl.float32)
        xyz_mask = mask & (column < 3)
        scaling_mask = mask & (column >= 3) & (column < 6)
        rotation_mask = mask & (column >= 6) & (column < 10)
        opacity_mask = mask & (column == 10)
        dc_mask = mask & (column >= 11) & (column < 14)
        rest_mask = mask & (column >= 14)
        value += tl.load(
            source_xyz + source_row * 3 + column,
            mask=xyz_mask,
            other=0.0,
        )
        value += tl.load(
            source_scaling + source_row * 3 + (column - 3),
            mask=scaling_mask,
            other=0.0,
        )
        value += tl.load(
            source_rotation + source_row * 4 + (column - 6),
            mask=rotation_mask,
            other=0.0,
        )
        value += tl.load(
            source_opacity + source_row,
            mask=opacity_mask,
            other=0.0,
        )
        value += tl.load(
            source_features_dc + source_row * 3 + (column - 11),
            mask=dc_mask,
            other=0.0,
        )
        value += tl.load(
            source_features_rest + source_row * 45 + (column - 14),
            mask=rest_mask,
            other=0.0,
        )

        tl.store(target_xyz + target_row * 3 + column, value, mask=xyz_mask)
        tl.store(
            target_scaling + target_row * 3 + (column - 3), value, mask=scaling_mask
        )
        tl.store(
            target_rotation + target_row * 4 + (column - 6), value, mask=rotation_mask
        )
        tl.store(target_opacity + target_row, value, mask=opacity_mask)
        tl.store(
            target_features_dc + target_row * 3 + (column - 11), value, mask=dc_mask
        )
        tl.store(
            target_features_rest + target_row * 45 + (column - 14),
            value,
            mask=rest_mask,
        )
        global_mask = mask & (column == 0)
        global_id = tl.load(source_global_ids + source_row, mask=global_mask, other=0)
        tl.store(target_global_ids + target_row, global_id, mask=global_mask)

    @triton.jit
    def _fill_cold_global_ids_kernel(
        block_ids,
        target_global_ids,
        target_offset,
        num_rows,
        BLOCK_ROWS: tl.constexpr,
        COPY_BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0) * COPY_BLOCK_SIZE + tl.arange(0, COPY_BLOCK_SIZE)
        mask = row < num_rows
        block_position = row // BLOCK_ROWS
        row_in_block = row - block_position * BLOCK_ROWS
        block_id = tl.load(block_ids + block_position, mask=mask, other=0)
        tl.store(
            target_global_ids + target_offset + row,
            block_id * BLOCK_ROWS + row_in_block,
            mask=mask,
        )

    @triton.jit
    def _unpack_streamed_rows_kernel(
        source,
        target_xyz,
        target_scaling,
        target_rotation,
        target_opacity,
        target_features_dc,
        target_features_rest,
        target_offset,
        num_rows,
        SOURCE_UNIFIED: tl.constexpr,
        COPY_BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * COPY_BLOCK_SIZE + tl.arange(0, COPY_BLOCK_SIZE)
        mask = offsets < num_rows * 59
        row = offsets // 59
        column = offsets - row * 59

        source_column = column
        if SOURCE_UNIFIED:
            source_column = tl.where(
                (column >= 3) & (column < 10), column + 1, source_column
            )
            source_column = tl.where(column == 10, 3, source_column)
        value = tl.load(
            source + row * 59 + source_column,
            mask=mask,
            other=0.0,
        )
        target_row = target_offset + row

        xyz_mask = mask & (column < 3)
        scaling_mask = mask & (column >= 3) & (column < 6)
        rotation_mask = mask & (column >= 6) & (column < 10)
        opacity_mask = mask & (column == 10)
        dc_mask = mask & (column >= 11) & (column < 14)
        rest_mask = mask & (column >= 14)
        tl.store(target_xyz + target_row * 3 + column, value, mask=xyz_mask)
        tl.store(
            target_scaling + target_row * 3 + (column - 3),
            value,
            mask=scaling_mask,
        )
        tl.store(
            target_rotation + target_row * 4 + (column - 6),
            value,
            mask=rotation_mask,
        )
        tl.store(target_opacity + target_row, value, mask=opacity_mask)
        tl.store(
            target_features_dc + target_row * 3 + (column - 11),
            value,
            mask=dc_mask,
        )
        tl.store(
            target_features_rest + target_row * 45 + (column - 14),
            value,
            mask=rest_mask,
        )


def _build_block_starts(
    block_to_local_slice: Dict[int, slice],
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a compact block-start lookup table.

    Returns a tensor of shape (num_blocks,) where entry *k* holds the
    local-index offset of block *k* inside the working set, or -1 if
    the block is not loaded.  Size: O(num_blocks) ≈ O(N/B), typically
    a few hundred KB even for billion-scale scenes — versus the O(N)
    tensor it replaces (~8 GB at 1 B Gaussians).
    """
    block_starts = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
    valid_items = [
        (int(block_id), int(block_slice.start))
        for block_id, block_slice in block_to_local_slice.items()
        if 0 <= int(block_id) < num_blocks
    ]
    if valid_items:
        mapping = torch.tensor(valid_items, dtype=torch.long, device=device)
        block_starts.index_copy_(0, mapping[:, 0], mapping[:, 1])
    return block_starts


def _global_ids_to_local(
    global_ids: torch.Tensor,
    block_starts: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map global Gaussian IDs to working-set-local indices.

    Uses the compact *block_starts* table (size num_blocks) instead of
    an N-sized lookup tensor.  Returns -1 for IDs whose block is not
    loaded in the current working set.
    """
    num_blocks = block_starts.shape[0]
    block_ids = torch.div(global_ids, block_size, rounding_mode='floor')
    within_block = global_ids - block_ids * block_size
    safe_block_ids = block_ids.clamp(0, num_blocks - 1)
    starts = block_starts[safe_block_ids]
    invalid = (block_ids != safe_block_ids) | (starts < 0)
    local_ids = starts + within_block
    local_ids[invalid] = -1
    return local_ids


@dataclass
class GPUBuffer:
    """A single GPU buffer holding all Gaussian parameters"""
    xyz: Optional[torch.Tensor] = None           # (M, 3)
    scaling: Optional[torch.Tensor] = None       # (M, 3)
    rotation: Optional[torch.Tensor] = None      # (M, 4)
    opacity: Optional[torch.Tensor] = None       # (M, 1)
    features_dc: Optional[torch.Tensor] = None   # (M, 3)
    features_rest: Optional[torch.Tensor] = None # (M, 45)
    
    # Mapping info
    local_to_global_idx: Optional[torch.Tensor] = None
    _block_starts: Optional[torch.Tensor] = None
    loaded_blocks: List[int] = None
    num_gaussians: int = 0
    
    # Filters for each camera in the batch (local indices)
    filters_local: List[torch.Tensor] = None
    block_to_local_slice: Dict[int, slice] = None
    resident_blocks: List[int] = None
    streamed_blocks: List[int] = None
    evicted_blocks: List[int] = None
    
    def __post_init__(self):
        if self.loaded_blocks is None:
            self.loaded_blocks = []
        if self.filters_local is None:
            self.filters_local = []
        if self.block_to_local_slice is None:
            self.block_to_local_slice = {}
        if self.resident_blocks is None:
            self.resident_blocks = []
        if self.streamed_blocks is None:
            self.streamed_blocks = []
        if self.evicted_blocks is None:
            self.evicted_blocks = []
    
    def is_empty(self) -> bool:
        return self.xyz is None
    
    def clear(self):
        """Release GPU memory"""
        self.xyz = None
        self.scaling = None
        self.rotation = None
        self.opacity = None
        self.features_dc = None
        self.features_rest = None
        self.local_to_global_idx = None
        self._block_starts = None
        self.loaded_blocks = []
        self.filters_local = []
        self.block_to_local_slice = {}
        self.resident_blocks = []
        self.streamed_blocks = []
        self.evicted_blocks = []
        self.num_gaussians = 0


@dataclass(frozen=True)
class ResidentHandoffResult:
    """Outcome of making retained blocks ready for the next activation."""

    ready_blocks: int
    copied_blocks: int


class DoubleBufferGPUWorkingSet:
    """
    Double-buffered GPU working set manager with N+1 prefetch.
    
    Key features:
    1. Two GPU buffers that alternate each iteration
    2. Background prefetch thread loads next iteration's data
    3. Event-coordinated swap between buffers
    4. Seamless integration with existing training loop
    """
    
    def __init__(
        self,
        num_total_gaussians: int,
        block_size: int,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        Initialize double-buffered GPU working set.
        
        Args:
            num_total_gaussians: Total number of Gaussians in scene
            block_size: Number of Gaussians per block
            device: GPU device
            verbose: Print routine buffer lifecycle messages
        """
        self.num_total = num_total_gaussians
        self.block_size = block_size
        self.device = torch.device(device)
        self.num_blocks = (num_total_gaussians + block_size - 1) // block_size
        self.verbose = bool(verbose)
        
        # ====================================================================
        # Double buffer: Two GPU working sets
        # ====================================================================
        self.buffer_a = GPUBuffer()
        self.buffer_b = GPUBuffer()
        
        # Which buffer is currently active (being used for compute)
        self.active_buffer_idx = 0  # 0 = A, 1 = B
        
        # ====================================================================
        # Prefetch management
        # ====================================================================
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.resident_refresh_stream = torch.cuda.Stream(device=self.device)
        self.prefetch_complete_event = torch.cuda.Event()
        self.resident_refresh_complete_event: Optional[torch.cuda.Event] = None
        self.prefetch_in_progress = False
        self.prefetch_iteration = -1
        self._prefetch_future: Optional[Future] = None
        self._prefetch_finalized = True
        self._deferred_resident_blocks: List[int] = []
        self._target_layout_ready = threading.Event()
        self._persistent_plan = None
        self._before_persistent_apply = None
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tide-nplus1",
        )
        self._resident_plan_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="tide-resident-plan",
        )
        self._resident_plan_future: Optional[Future] = None
        self._executor_shutdown = False
        self._host_staging_slots: List[Optional[torch.Tensor]] = [None, None]
        self._gpu_staging_slots: List[Optional[torch.Tensor]] = [None, None]
        self._host_staging_capacities = [0, 0]
        self._host_staging_events: List[Optional[torch.cuda.Event]] = [None, None]
        self._next_host_staging_slot = 0
        self._dirty_bitmap = torch.zeros(
            self.num_blocks,
            dtype=torch.bool,
            device=self.device,
        )
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'swaps': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'total_prefetch_time_ms': 0.0,
            'batched_materializations': 0,
            'batched_h2d_calls': 0,
            'batched_streamed_blocks': 0,
            'batched_resident_blocks': 0,
            'host_staging_bytes': 0,
            'host_staging_pinned': False,
            'dirty_blocks_marked': 0,
            'dirty_blocks_written_back': 0,
            'persistent_slot_updates': 0,
            'persistent_streamed_blocks': 0,
            'persistent_ab_fallbacks': 0,
            'persistent_omega_reuses': 0,
            'resident_refresh_blocks': 0,
            'resident_plan_jobs': 0,
            'resident_plan_time_ms': 0.0,
        }
        
        self._log("[DoubleBufferGPU] Initialized with 2 GPU buffers")
        self._log(f"[DoubleBufferGPU] Total Gaussians: {num_total_gaussians:,}")
        self._log(f"[DoubleBufferGPU] Block size: {block_size}, Blocks: {self.num_blocks}")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)
    
    @property
    def active_buffer(self) -> GPUBuffer:
        """Get currently active buffer (for compute)"""
        return self.buffer_a if self.active_buffer_idx == 0 else self.buffer_b
    
    @property
    def loading_buffer(self) -> GPUBuffer:
        """Get buffer being loaded (for prefetch)"""
        return self.buffer_b if self.active_buffer_idx == 0 else self.buffer_a
    
    # ========================================================================
    # Compatibility properties - expose active buffer's tensors directly
    # ========================================================================
    @property
    def gpu_xyz(self) -> Optional[torch.Tensor]:
        return self.active_buffer.xyz
    
    @property
    def gpu_scaling(self) -> Optional[torch.Tensor]:
        return self.active_buffer.scaling
    
    @property
    def gpu_rotation(self) -> Optional[torch.Tensor]:
        return self.active_buffer.rotation
    
    @property
    def gpu_opacity(self) -> Optional[torch.Tensor]:
        return self.active_buffer.opacity
    
    @property
    def gpu_features_dc(self) -> Optional[torch.Tensor]:
        return self.active_buffer.features_dc
    
    @property
    def gpu_features_rest(self) -> Optional[torch.Tensor]:
        return self.active_buffer.features_rest
    
    @property
    def local_to_global_idx(self) -> Optional[torch.Tensor]:
        return self.active_buffer.local_to_global_idx
    
    def global_to_local(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Map global Gaussian IDs to working-set-local indices.

        Uses a compact O(num_blocks) lookup instead of an O(N) tensor,
        saving ~8 GB VRAM at 1 B Gaussians.  Returns -1 for IDs whose
        block is not in the current working set.
        """
        buf = self.active_buffer
        if buf._block_starts is None:
            raise RuntimeError(
                '[DoubleBufferGPU] global_to_local called but _block_starts '
                'has not been built yet (buffer is empty or not materialized)'
            )
        return _global_ids_to_local(global_ids, buf._block_starts, self.block_size)

    def mark_dirty_blocks(self, block_ids: List[int]) -> int:
        block_ids = sorted(
            set(
                int(block_id)
                for block_id in block_ids
                if 0 <= int(block_id) < self.num_blocks
            )
        )
        if not block_ids:
            return 0
        ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        with self.lock:
            self._dirty_bitmap.index_fill_(0, ids, True)
            self.stats['dirty_blocks_marked'] += len(block_ids)
        return len(block_ids)

    def dirty_blocks_for_eviction(self, evicted_block_ids: List[int]) -> List[int]:
        evicted = sorted(
            set(
                int(block_id)
                for block_id in evicted_block_ids
                if 0 <= int(block_id) < self.num_blocks
            )
        )
        if not evicted:
            return []
        evicted_gpu = torch.tensor(evicted, dtype=torch.long, device=self.device)
        with self.lock:
            selected = evicted_gpu[self._dirty_bitmap.index_select(0, evicted_gpu)]
        return selected.cpu().tolist()

    def dirty_blocks(self) -> List[int]:
        with self.lock:
            dirty = torch.nonzero(self._dirty_bitmap, as_tuple=False).squeeze(1)
        return dirty.cpu().tolist()

    def mark_blocks_written_back(self, block_ids: List[int]) -> None:
        block_ids = sorted(
            set(
                int(block_id)
                for block_id in block_ids
                if 0 <= int(block_id) < self.num_blocks
            )
        )
        if not block_ids:
            return
        ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        with self.lock:
            self._dirty_bitmap.index_fill_(0, ids, False)
            self.stats['dirty_blocks_written_back'] += len(block_ids)

    def submit_resident_plan(
        self,
        plan_iteration: int,
        planner: Callable,
        *args,
        **kwargs,
    ) -> Future:
        previous = self._resident_plan_future
        if previous is not None and not previous.done():
            raise RuntimeError("previous resident plan is still running")

        def _run():
            started = time.perf_counter()
            try:
                return planner(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self.lock:
                    self.stats['resident_plan_time_ms'] += elapsed_ms

        with self.lock:
            self.stats['resident_plan_jobs'] += 1
        future = self._resident_plan_executor.submit(_run)
        future.iteration = int(plan_iteration)
        self._resident_plan_future = future
        return future

    def _build_persistent_plan(
        self,
        next_blocks: List[int],
        resident_set: set,
        evicted_set: set,
        *,
        allow_resident_copy: bool,
        defer_resident_copy: bool,
    ):
        source = self.active_buffer
        current_blocks = list(source.loaded_blocks)
        if (
            not allow_resident_copy
            or not defer_resident_copy
            or source.is_empty()
            or not current_blocks
            or len(next_blocks) != len(current_blocks)
        ):
            return None

        current_set = set(int(block_id) for block_id in current_blocks)
        next_set = set(int(block_id) for block_id in next_blocks)
        keep_set = current_set & next_set
        incoming = sorted(next_set - current_set)
        evicted = current_set - next_set
        if (
            keep_set != set(int(block_id) for block_id in resident_set)
            or evicted != set(int(block_id) for block_id in evicted_set)
            or len(incoming) != len(evicted)
        ):
            return None

        for block_id in current_blocks:
            block_slice = source.block_to_local_slice.get(int(block_id))
            if (
                block_slice is None
                or int(block_slice.stop - block_slice.start) != self.block_size
            ):
                return None
        for block_id in incoming:
            if min(
                self.block_size,
                self.num_total - int(block_id) * self.block_size,
            ) != self.block_size:
                return None
        if int(source.xyz.shape[0]) != len(current_blocks) * self.block_size:
            return None

        free_slices = sorted(
            (
                source.block_to_local_slice[int(block_id)]
                for block_id in evicted
            ),
            key=lambda block_slice: int(block_slice.start),
        )
        target_slices = {
            block_id: block_slice
            for block_id, block_slice in zip(incoming, free_slices)
        }
        next_slices = {
            int(block_id): source.block_to_local_slice[int(block_id)]
            for block_id in keep_set
        }
        next_slices.update(target_slices)
        next_order = [
            block_id
            for block_id, _ in sorted(
                next_slices.items(), key=lambda item: int(item[1].start)
            )
        ]
        return {
            'keep_blocks': sorted(keep_set),
            'incoming_blocks': incoming,
            'evicted_blocks': sorted(evicted),
            'target_slices': target_slices,
            'next_slices': next_slices,
            'next_order': next_order,
        }

    @property
    def loaded_blocks(self) -> List[int]:
        return self.active_buffer.loaded_blocks
    
    @property
    def filters_local(self) -> List[torch.Tensor]:
        return self.active_buffer.filters_local
    
    def swap_buffers(self):
        """
        Swap active and loading buffers.
        
        Called at the start of each iteration AFTER prefetch is complete.
        """
        persistent_plan = self._persistent_plan
        next_blocks = (
            persistent_plan['next_order'] if persistent_plan is not None
            else self.loading_buffer.loaded_blocks
        )
        missing_dirty = set(self.dirty_blocks()) - set(next_blocks)
        if missing_dirty:
            raise RuntimeError(
                f"Cannot evict GPU-dirty blocks before writeback: {sorted(missing_dirty)[:8]}"
            )
        if persistent_plan is not None:
            if self._before_persistent_apply is not None:
                self._before_persistent_apply()
            source = self.loading_buffer
            target = self.active_buffer
            incoming_blocks = persistent_plan['incoming_blocks']
            target_slices = persistent_plan['target_slices']
            with torch.no_grad():
                self._copy_resident_blocks(
                    source,
                    target,
                    incoming_blocks,
                    target_slices,
                )
            target.block_to_local_slice = dict(persistent_plan['next_slices'])
            target.loaded_blocks = list(persistent_plan['next_order'])
            target.resident_blocks = list(persistent_plan['keep_blocks'])
            target.streamed_blocks = list(incoming_blocks)
            target.evicted_blocks = list(persistent_plan['evicted_blocks'])
            target.num_gaussians = int(target.xyz.shape[0])
            target._block_starts = _build_block_starts(
                target.block_to_local_slice,
                self.num_blocks,
                self.device,
            )
            source.clear()
            with self.lock:
                self._persistent_plan = None
                self._before_persistent_apply = None
                self.prefetch_in_progress = False
                self.stats['swaps'] += 1
                self.stats['persistent_slot_updates'] += 1
                self.stats['persistent_streamed_blocks'] += len(incoming_blocks)
            return

        with self.lock:
            # Wait for any ongoing prefetch to complete
            if self.prefetch_in_progress:
                self.prefetch_complete_event.synchronize()
                if self.resident_refresh_complete_event is not None:
                    self.resident_refresh_complete_event.synchronize()
                self.prefetch_in_progress = False
            
            # Swap
            self.active_buffer_idx = 1 - self.active_buffer_idx
            self.stats['swaps'] += 1

    @staticmethod
    def _component_columns(layout) -> Dict[str, slice]:
        from storage.block_reader import BlockLayout

        if layout == BlockLayout.UNIFIED:
            return {
                'xyz': slice(0, 3),
                'opacity': slice(3, 4),
                'scaling': slice(4, 7),
                'rotation': slice(7, 11),
                'features_dc': slice(11, 14),
                'features_rest': slice(14, 59),
            }
        if layout == BlockLayout.CACHE:
            return {
                'xyz': slice(0, 3),
                'scaling': slice(3, 6),
                'rotation': slice(6, 10),
                'opacity': slice(10, 11),
                'features_dc': slice(11, 14),
                'features_rest': slice(14, 59),
            }
        raise ValueError(f'Unsupported block layout: {layout!r}')

    def _allocate_target_tensors(self, target: GPUBuffer, num_rows: int) -> None:
        target.xyz = torch.empty(num_rows, 3, device=self.device, dtype=torch.float32)
        target.scaling = torch.empty(num_rows, 3, device=self.device, dtype=torch.float32)
        target.rotation = torch.empty(num_rows, 4, device=self.device, dtype=torch.float32)
        target.opacity = torch.empty(num_rows, 1, device=self.device, dtype=torch.float32)
        target.features_dc = torch.empty(num_rows, 3, device=self.device, dtype=torch.float32)
        target.features_rest = torch.empty(num_rows, 45, device=self.device, dtype=torch.float32)
        target.local_to_global_idx = torch.empty(
            num_rows, device=self.device, dtype=torch.long
        )

    def _acquire_host_staging(
        self,
        required_rows: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        slot_idx = self._next_host_staging_slot
        self._next_host_staging_slot = (slot_idx + 1) % len(self._host_staging_slots)
        previous_copy = self._host_staging_events[slot_idx]
        if previous_copy is not None:
            previous_copy.synchronize()

        staging = self._host_staging_slots[slot_idx]
        gpu_staging = self._gpu_staging_slots[slot_idx]
        if (
            required_rows <= self._host_staging_capacities[slot_idx]
            and staging is not None
            and gpu_staging is not None
        ):
            return staging[:required_rows], gpu_staging[:required_rows], slot_idx

        try:
            staging = torch.empty(
                (required_rows, 59), dtype=torch.float32, pin_memory=True
            )
        except RuntimeError:
            staging = torch.empty((required_rows, 59), dtype=torch.float32)
        gpu_staging = torch.empty(
            (required_rows, 59), dtype=torch.float32, device=self.device
        )
        self._host_staging_slots[slot_idx] = staging
        self._gpu_staging_slots[slot_idx] = gpu_staging
        self._host_staging_capacities[slot_idx] = required_rows
        self.stats['host_staging_bytes'] = sum(
            tensor.numel() * tensor.element_size()
            for slot in self._host_staging_slots if slot is not None
            for tensor in (slot,)
        )
        self.stats['host_staging_pinned'] = all(
            tensor.is_pinned()
            for slot in self._host_staging_slots if slot is not None
            for tensor in (slot,)
        )
        return staging[:required_rows], gpu_staging[:required_rows], slot_idx

    def _pack_streamed_blocks(
        self,
        block_ids: List[int],
        block_lengths: Dict[int, int],
        reader_blocks: Dict[int, torch.Tensor],
        reader_layout,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        total_rows = sum(block_lengths[block_id] for block_id in block_ids)
        staging, gpu_staging, slot_idx = self._acquire_host_staging(total_rows)
        output = staging[:total_rows]
        pieces = [
            reader_blocks[block_id][:block_lengths[block_id]]
            for block_id in block_ids
        ]
        if len(pieces) == 1:
            output.copy_(pieces[0])
        else:
            torch.cat(pieces, dim=0, out=output)
        return output, gpu_staging[:total_rows], total_rows, slot_idx

    def _copy_resident_blocks(
        self,
        source: GPUBuffer,
        target: GPUBuffer,
        block_ids: List[int],
        target_slices: Dict[int, slice],
    ) -> int:
        if not block_ids:
            return 0
        num_rows = sum(
            target_slices[block_id].stop - target_slices[block_id].start
            for block_id in block_ids
        )
        if triton is None:
            for block_id in block_ids:
                source_slice = source.block_to_local_slice[block_id]
                target_slice = target_slices[block_id]
                target.xyz[target_slice].copy_(source.xyz[source_slice])
                target.scaling[target_slice].copy_(source.scaling[source_slice])
                target.rotation[target_slice].copy_(source.rotation[source_slice])
                target.opacity[target_slice].copy_(source.opacity[source_slice])
                target.features_dc[target_slice].copy_(source.features_dc[source_slice])
                target.features_rest[target_slice].copy_(source.features_rest[source_slice])
                target.local_to_global_idx[target_slice].copy_(
                    source.local_to_global_idx[source_slice]
                )
            return num_rows

        source_starts = torch.tensor(
            [source.block_to_local_slice[block_id].start for block_id in block_ids],
            dtype=torch.long,
            device=self.device,
        )
        target_starts = torch.tensor(
            [target_slices[block_id].start for block_id in block_ids],
            dtype=torch.long,
            device=self.device,
        )
        grid = (triton.cdiv(num_rows * 59, 256),)
        _copy_resident_rows_kernel[grid](
            source.xyz,
            source.scaling,
            source.rotation,
            source.opacity,
            source.features_dc,
            source.features_rest,
            source.local_to_global_idx,
            source_starts,
            target_starts,
            target.xyz,
            target.scaling,
            target.rotation,
            target.opacity,
            target.features_dc,
            target.features_rest,
            target.local_to_global_idx,
            num_rows,
            BLOCK_ROWS=self.block_size,
            COPY_BLOCK_SIZE=256,
            num_warps=4,
        )
        return num_rows

    def finalize_retained_blocks(
        self,
        block_ids: List[int],
        target: str = 'loading',
    ) -> ResidentHandoffResult:
        persistent_plan = self._persistent_plan
        if target == 'loading' and persistent_plan is not None:
            requested_ids = set(int(block_id) for block_id in block_ids)
            kept = set(persistent_plan['keep_blocks'])
            reused = len(requested_ids & kept)
            with self.lock:
                self._prefetch_finalized = True
                self._deferred_resident_blocks = []
                self.stats['persistent_omega_reuses'] += reused
            return ResidentHandoffResult(ready_blocks=reused, copied_blocks=0)

        if target == 'loading':
            self._wait_for_target_layout()
        source_buffer = self.active_buffer
        target_buffer = self.loading_buffer if target == 'loading' else self.active_buffer
        if source_buffer.is_empty() or target_buffer.is_empty() or not block_ids:
            return ResidentHandoffResult(ready_blocks=0, copied_blocks=0)
        refresh_ids = sorted(
            int(block_id)
            for block_id in set(block_ids)
            if int(block_id) in source_buffer.block_to_local_slice
            and int(block_id) in target_buffer.block_to_local_slice
        )
        if not refresh_ids:
            return ResidentHandoffResult(ready_blocks=0, copied_blocks=0)
        with torch.cuda.stream(self.resident_refresh_stream):
            self.resident_refresh_stream.wait_stream(torch.cuda.current_stream(self.device))
            self._copy_resident_blocks(
                source_buffer,
                target_buffer,
                refresh_ids,
                target_buffer.block_to_local_slice,
            )
            resident_event = torch.cuda.Event()
            resident_event.record(self.resident_refresh_stream)
            self.resident_refresh_complete_event = resident_event
        with self.lock:
            self._prefetch_finalized = True
            self._deferred_resident_blocks = []
            self.stats['resident_refresh_blocks'] += len(refresh_ids)
        return ResidentHandoffResult(
            ready_blocks=len(refresh_ids),
            copied_blocks=len(refresh_ids),
        )

    def _wait_for_target_layout(self) -> None:
        while not self._target_layout_ready.wait(timeout=0.01):
            prefetch_future = self._prefetch_future
            if prefetch_future is not None and prefetch_future.done():
                prefetch_future.result()

        prefetch_future = self._prefetch_future
        if prefetch_future is not None and prefetch_future.done():
            prefetch_future.result()

    def _fill_streamed_global_ids(
        self,
        target: GPUBuffer,
        block_ids: List[int],
        target_offset: int,
        num_rows: int,
    ) -> None:
        if not block_ids:
            return
        block_ids_gpu = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        if triton is None:
            rows = torch.arange(num_rows, dtype=torch.long, device=self.device)
            block_positions = torch.div(rows, self.block_size, rounding_mode='floor')
            within_block = rows - block_positions * self.block_size
            target.local_to_global_idx[target_offset:target_offset + num_rows] = (
                block_ids_gpu[block_positions] * self.block_size + within_block
            )
            return
        grid = (triton.cdiv(num_rows, 256),)
        _fill_cold_global_ids_kernel[grid](
            block_ids_gpu,
            target.local_to_global_idx,
            target_offset,
            num_rows,
            BLOCK_ROWS=self.block_size,
            COPY_BLOCK_SIZE=256,
            num_warps=4,
        )

    def _materialize_from_block_reader(
        self,
        target: GPUBuffer,
        source: GPUBuffer,
        sorted_visible_blocks: List[int],
        resident_set: set,
        evicted_set: set,
        allow_resident_copy: bool,
        block_reader,
        defer_resident_copy: bool = False,
    ) -> None:
        visible_set = set(sorted_visible_blocks)
        resident_blocks = sorted(
            block_id
            for block_id in visible_set & resident_set
            if allow_resident_copy
            and source is not None
            and not source.is_empty()
            and block_id in source.block_to_local_slice
        )
        resident_block_set = set(resident_blocks)
        streamed_blocks = sorted(visible_set - resident_block_set)
        target_order = resident_blocks + streamed_blocks

        block_lengths = {
            block_id: min(
                self.block_size,
                self.num_total - block_id * self.block_size,
            )
            for block_id in target_order
        }
        invalid = [
            block_id for block_id, block_len in block_lengths.items() if block_len <= 0
        ]
        if invalid:
            raise ValueError(f'Invalid block IDs during N+1 materialization: {invalid}')

        offset = 0
        for block_id in target_order:
            block_len = block_lengths[block_id]
            target.block_to_local_slice[block_id] = slice(offset, offset + block_len)
            offset += block_len
        self._allocate_target_tensors(target, offset)
        self._target_layout_ready.set()

        resident_rows = sum(block_lengths[block_id] for block_id in resident_blocks)
        if not defer_resident_copy:
            self._copy_resident_blocks(
                source,
                target,
                resident_blocks,
                target.block_to_local_slice,
            )

        if streamed_blocks:
            streamed_rows = sum(block_lengths[block_id] for block_id in streamed_blocks)
            read_batch = getattr(block_reader, 'read_batch', None)
            if callable(read_batch):
                staging, packed_gpu, staging_slot = self._acquire_host_staging(streamed_rows)
                block_batch = read_batch(
                    streamed_blocks,
                    out=staging[:streamed_rows],
                )
                packed_cpu = block_batch.tensor
                if tuple(packed_cpu.shape) != (streamed_rows, 59):
                    raise ValueError(
                        f'Invalid N+1 block batch shape {tuple(packed_cpu.shape)}; '
                        f'expected {(streamed_rows, 59)}'
                    )
                for block_id in streamed_blocks:
                    block_slice = block_batch.block_slices.get(block_id)
                    expected_rows = block_lengths[block_id]
                    if block_slice is None or block_slice.stop - block_slice.start != expected_rows:
                        raise ValueError(
                            f'Invalid N+1 batch slice for block {block_id}: '
                            f'got={block_slice}, expected_rows={expected_rows}'
                        )
            else:
                reader_blocks = block_reader.read_blocks(streamed_blocks)
                missing = [block_id for block_id in streamed_blocks if block_id not in reader_blocks]
                if missing:
                    raise KeyError(
                        f'BlockReader omitted {len(missing)} N+1 blocks; first missing={missing[:8]}'
                    )
                packed_cpu, packed_gpu, streamed_rows, staging_slot = self._pack_streamed_blocks(
                    streamed_blocks,
                    block_lengths,
                    reader_blocks,
                    block_reader.layout,
                )
            packed_gpu.copy_(packed_cpu, non_blocking=packed_cpu.is_pinned())
            if triton is None:  # pragma: no cover
                columns = self._component_columns(block_reader.layout)
                target_slice = slice(resident_rows, resident_rows + streamed_rows)
                target.xyz[target_slice].copy_(packed_gpu[:, columns['xyz']])
                target.scaling[target_slice].copy_(packed_gpu[:, columns['scaling']])
                target.rotation[target_slice].copy_(packed_gpu[:, columns['rotation']])
                target.opacity[target_slice].copy_(packed_gpu[:, columns['opacity']])
                target.features_dc[target_slice].copy_(packed_gpu[:, columns['features_dc']])
                target.features_rest[target_slice].copy_(packed_gpu[:, columns['features_rest']])
            else:
                from storage.block_reader import BlockLayout

                grid = (triton.cdiv(streamed_rows * 59, 256),)
                _unpack_streamed_rows_kernel[grid](
                    packed_gpu,
                    target.xyz,
                    target.scaling,
                    target.rotation,
                    target.opacity,
                    target.features_dc,
                    target.features_rest,
                    resident_rows,
                    streamed_rows,
                    SOURCE_UNIFIED=block_reader.layout == BlockLayout.UNIFIED,
                    COPY_BLOCK_SIZE=256,
                    num_warps=4,
                )
            self._fill_streamed_global_ids(
                target,
                streamed_blocks,
                resident_rows,
                streamed_rows,
            )
            staging_event = torch.cuda.Event()
            staging_event.record(self.prefetch_stream)
            self._host_staging_events[staging_slot] = staging_event
            self.stats['batched_h2d_calls'] += 1
        else:
            streamed_rows = 0

        target.loaded_blocks = target_order
        target.num_gaussians = resident_rows + streamed_rows
        target.resident_blocks = resident_blocks
        target.streamed_blocks = streamed_blocks
        target.evicted_blocks = sorted(evicted_set)
        target._block_starts = _build_block_starts(
            target.block_to_local_slice, self.num_blocks, self.device
        )
        self.stats['batched_materializations'] += 1
        self.stats['batched_resident_blocks'] += len(resident_blocks)
        self.stats['batched_streamed_blocks'] += len(streamed_blocks)

    def _run_block_reader_prefetch(
        self,
        *,
        target_buffer: GPUBuffer,
        source_buffer: GPUBuffer,
        sorted_visible_blocks: List[int],
        resident_set: set,
        evicted_set: set,
        filters_global: List[torch.Tensor],
        allow_resident_copy: bool,
        defer_resident_copy: bool,
        block_reader,
        before_target_reuse,
    ) -> None:
        start_time = time.time()
        if before_target_reuse is not None:
            before_target_reuse()

        with torch.cuda.device(self.device), torch.cuda.stream(self.prefetch_stream):
            target_buffer.clear()
            if not sorted_visible_blocks:
                target_buffer.evicted_blocks = sorted(evicted_set)
                self._target_layout_ready.set()
                self.prefetch_complete_event.record(self.prefetch_stream)
                with self.lock:
                    self._prefetch_finalized = True
                return

            with torch.cuda.nvtx.range("Tide N+1: block read and pack"):
                self._materialize_from_block_reader(
                    target=target_buffer,
                    source=source_buffer,
                    sorted_visible_blocks=sorted_visible_blocks,
                    resident_set=resident_set,
                    evicted_set=evicted_set,
                    allow_resident_copy=allow_resident_copy,
                    block_reader=block_reader,
                    defer_resident_copy=defer_resident_copy,
                )
            target_buffer.filters_local = []
            for filter_global in filters_global:
                if filter_global is not None and len(filter_global) > 0:
                    filter_local = _global_ids_to_local(
                        filter_global.to(self.device),
                        target_buffer._block_starts,
                        self.block_size,
                    )
                    target_buffer.filters_local.append(filter_local[filter_local >= 0])
                else:
                    target_buffer.filters_local.append(
                        torch.empty(0, dtype=torch.long, device=self.device)
                    )

            deferred_blocks = target_buffer.resident_blocks if defer_resident_copy else []
            with self.lock:
                self._deferred_resident_blocks = list(deferred_blocks)
            self.prefetch_complete_event.record(self.prefetch_stream)
            if not deferred_blocks:
                with self.lock:
                    self._prefetch_finalized = True

        self.stats['total_prefetch_time_ms'] += (time.time() - start_time) * 1000.0
    
    def start_prefetch(
        self,
        iteration: int,
        visible_block_ids: List[int],
        filters_global: Optional[List[torch.Tensor]] = None,
        resident_block_ids: Optional[List[int]] = None,
        evicted_block_ids: Optional[List[int]] = None,
        allow_resident_copy: bool = True,
        block_reader: 'Optional[object]' = None,
        defer_resident_copy: bool = False,
        before_target_reuse=None,
    ):
        """
        Start async prefetch of next iteration's data into loading buffer.

        This is non-blocking! Returns immediately while prefetch happens in background.

        Args:
            iteration: The iteration this data is for (N+1)
            visible_block_ids: Block IDs needed for iteration N+1
            filters_global: Optional global Gaussian indices for each camera
            resident_block_ids: Optional blocks to keep resident from the active buffer (Omega)
            evicted_block_ids: Optional blocks evicted from the current working set (Delta-)
            allow_resident_copy: Whether to reuse resident blocks from the active buffer
            block_reader: Source for block reads implementing the ``BlockReader`` protocol.
            defer_resident_copy: Delay retained-block copying until the current
                optimizer step has completed.
            before_target_reuse: Optional fence invoked by the worker immediately
                before reusing the loading GPU buffer.
        """
        if block_reader is None:
            raise ValueError("Tide prefetch requires a block reader.")
        previous_iteration = None
        with self.lock:
            if self.prefetch_in_progress:
                previous_iteration = self.prefetch_iteration

        if previous_iteration is not None:
            self.wait_for_prefetch(previous_iteration)

        with self.lock:
            self.prefetch_in_progress = True
            self.prefetch_iteration = iteration
            self.prefetch_complete_event = torch.cuda.Event()
            self.resident_refresh_complete_event = None
            self._prefetch_finalized = False
            self._deferred_resident_blocks = []
            self._target_layout_ready.clear()
            self._persistent_plan = None
            self._before_persistent_apply = None

        sorted_visible_blocks = sorted(
            set(int(block_id) for block_id in visible_block_ids)
        )
        resident_set = set(
            int(block_id) for block_id in (resident_block_ids or [])
        )
        evicted_set = set(
            int(block_id) for block_id in (evicted_block_ids or [])
        )
        persistent_plan = self._build_persistent_plan(
            sorted_visible_blocks,
            resident_set,
            evicted_set,
            allow_resident_copy=allow_resident_copy,
            defer_resident_copy=defer_resident_copy,
        )
        if persistent_plan is not None:
            self._persistent_plan = persistent_plan
            self._before_persistent_apply = before_target_reuse
            prefetch_blocks = persistent_plan['incoming_blocks']
            prefetch_resident_set = set()
            prefetch_evicted_set = evicted_set
            prefetch_deferred = False
            worker_before_target_reuse = None
        else:
            self.stats['persistent_ab_fallbacks'] += 1
            prefetch_blocks = sorted_visible_blocks
            prefetch_resident_set = resident_set
            prefetch_evicted_set = evicted_set
            prefetch_deferred = defer_resident_copy
            worker_before_target_reuse = before_target_reuse
        self._prefetch_future = self._prefetch_executor.submit(
            self._run_block_reader_prefetch,
            target_buffer=self.loading_buffer,
            source_buffer=self.active_buffer,
            sorted_visible_blocks=prefetch_blocks,
            resident_set=prefetch_resident_set,
            evicted_set=prefetch_evicted_set,
            filters_global=list(filters_global or []),
            allow_resident_copy=allow_resident_copy,
            defer_resident_copy=prefetch_deferred,
            block_reader=block_reader,
            before_target_reuse=worker_before_target_reuse,
        )
        self._prefetch_future.add_done_callback(
            lambda _future: self._target_layout_ready.set()
        )
        return

    def synchronize_prefetch(self) -> None:
        if self._resident_plan_future is not None:
            self._resident_plan_future.result()
        if self._prefetch_future is not None:
            self._prefetch_future.result()
        self.prefetch_stream.synchronize()
        self.resident_refresh_stream.synchronize()

    def discard_prefetch(self) -> None:
        self.synchronize_prefetch()
        with self.lock:
            self.prefetch_in_progress = False
            self.prefetch_iteration = -1
            self._prefetch_finalized = True
            self._deferred_resident_blocks = []
            self._persistent_plan = None
            self._before_persistent_apply = None

    def wait_for_prefetch(self, iteration: int) -> bool:
        """
        Wait for prefetch to complete (if needed).
        
        Args:
            iteration: Expected iteration number
            
        Returns:
            True if prefetch was ready, False if had to wait
        """
        with self.lock:
            if not self.prefetch_in_progress:
                self.stats['prefetch_misses'] += 1
                return False
            
            if self.prefetch_iteration != iteration:
                self.stats['prefetch_misses'] += 1
                return False

        prefetch_future = self._prefetch_future
        if prefetch_future is not None:
            prefetch_future.result()

        with self.lock:
            finalized = self._prefetch_finalized
            deferred_blocks = list(self._deferred_resident_blocks)
        if not finalized:
            handoff = self.finalize_retained_blocks(
                deferred_blocks,
                target='loading',
            )
            if handoff.ready_blocks != len(deferred_blocks):
                raise RuntimeError(
                    "N+1 retained-block handoff was incomplete: "
                    f"expected={len(deferred_blocks)} ready={handoff.ready_blocks}"
                )

        resident_event = self.resident_refresh_complete_event
        
        # Check if already complete (non-blocking query)
        if (
            self.prefetch_complete_event.query()
            and (resident_event is None or resident_event.query())
        ):
            self.stats['prefetch_hits'] += 1
            with self.lock:
                self.prefetch_in_progress = False
            return True
        
        # Have to wait
        self.prefetch_complete_event.synchronize()
        if resident_event is not None:
            resident_event.synchronize()
        self.stats['prefetch_hits'] += 1
        with self.lock:
            self.prefetch_in_progress = False
        return True
    
    
    
    def get_stats(self) -> Dict[str, any]:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        # Calculate hit rate
        total = stats['prefetch_hits'] + stats['prefetch_misses']
        if total > 0:
            stats['hit_rate'] = stats['prefetch_hits'] / total
        else:
            stats['hit_rate'] = 0.0
        
        # Calculate average prefetch time
        if stats['prefetch_hits'] > 0:
            stats['avg_prefetch_time_ms'] = stats['total_prefetch_time_ms'] / stats['prefetch_hits']
        else:
            stats['avg_prefetch_time_ms'] = 0.0
        
        # Current buffer stats
        stats['active_buffer'] = 'A' if self.active_buffer_idx == 0 else 'B'
        stats['active_gaussians'] = self.active_buffer.num_gaussians
        stats['active_blocks'] = len(self.active_buffer.loaded_blocks)
        
        return stats
    
    def clear(self):
        """Clear both buffers"""
        self.synchronize_prefetch()
        if not self._executor_shutdown:
            self._prefetch_executor.shutdown(wait=True)
            self._resident_plan_executor.shutdown(wait=True)
            self._executor_shutdown = True
        self.buffer_a.clear()
        self.buffer_b.clear()
        self._host_staging_slots = [None, None]
        self._gpu_staging_slots = [None, None]
        self._host_staging_capacities = [0, 0]
        self._host_staging_events = [None, None]
        self._target_layout_ready.clear()
        self.resident_refresh_complete_event = None
        self._dirty_bitmap.zero_()
        self.stats['host_staging_bytes'] = 0
        self.stats['host_staging_pinned'] = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        self.clear()
