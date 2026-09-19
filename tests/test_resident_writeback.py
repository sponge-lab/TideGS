import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from storage.block_reader import TieredCacheBlockReader
from storage.config import StorageConfig
from storage.pure_ssd_checkpoint import write_pure_ssd_snapshot_checkpoint
from storage.tide_storage_adapter import TideStorageAdapter
from strategies.tide_engine.double_buffer_gpu import DoubleBufferGPUWorkingSet
from strategies.tide_engine.gpu_working_set import GPUWorkingSet
from strategies.tide_engine.runtime import (
    activate_paper_prefetched_buffer,
    load_paper_stage1_working_set,
    seed_paper_active_buffer_from_manager,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidentWritebackTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.base = torch.zeros(10, 59)
        self.base.numpy().tofile(self.root / "base_file.bin")
        np.save(self.root / "block_bounds.npy", np.zeros((3, 6), dtype=np.float32))
        manifest = {
            "total_points": 10, "num_blocks": 3, "block_size": 4, "param_dim": 59,
            "base_file": str(self.root / "base_file.bin"),
            "block_bounds": str(self.root / "block_bounds.npy"),
            "scene_min": [0, 0, 0], "scene_max": [1, 1, 1],
        }
        self.model = SimpleNamespace(
            _pure_ssd_resume_manifest=manifest,
            _pure_ssd_resume_pending=True,
            initialize_from_pure_ssd_checkpoint_manifest=Mock(),
            active_sh_degree=0,
        )
        self.staging = GPUWorkingSet(10, 4)
        self.resident = DoubleBufferGPUWorkingSet(10, 4)
        self.addCleanup(self.staging.clear)
        self.addCleanup(self.resident.clear)
        self.adapter = TideStorageAdapter(
            self.model, [], StorageConfig(
                ssd_cache_dir=str(self.root / "cache"), block_size=4,
                max_ram_gb=0.01, min_free_gb=0, skip_camera_clustering=True,
            ),
        )
        self.addCleanup(self.adapter.shutdown)
        self.adapter.bind_resident_writeback(self.resident, self.staging)
        self.reader = TieredCacheBlockReader(
            self.adapter.cache, total_gaussians=10, block_size=4,
            before_read=self.adapter.wait_for_cache_blocks,
        )
        self.model.gpu_working_set_manager = self.staging
        self.model._block_reader = self.reader
        self.model._paper_ab_runtime_stats = {
            "prefetch_hits": 0, "cold_starts": 0, "prefetch_misses": 0, "sync_fallbacks": 0,
        }

    def load(self, block_ids):
        self.staging.load_visible_blocks_with_retention(block_ids, block_reader=self.reader)
        seed_paper_active_buffer_from_manager(
            self.model, self.resident,
            lambda model, *args, **kwargs: model.gpu_working_set_manager.local_to_global_idx,
        )

    def update(self, block_id, value):
        source = self.resident.active_buffer
        block_slice = source.block_to_local_slice[block_id]
        for name in ("xyz", "scaling", "rotation", "opacity", "features_dc", "features_rest"):
            getattr(source, name)[block_slice].fill_(value)
        self.resident.mark_dirty_blocks([block_id])

    def test_staging_reads_owner_buffer_instead_of_stale_manager_mapping(self):
        self.load([0, 2])
        self.update(0, 7)
        self.update(2, 9)
        self.staging.block_to_gpu_slice = {1: slice(0, 4)}
        self.staging.gpu_xyz = torch.zeros_like(self.staging.gpu_xyz)

        self.assertEqual(self.adapter.flush_resident_dirty(), 2)
        blocks = self.reader.read_blocks([0, 2])
        torch.testing.assert_close(blocks[0], torch.full((4, 59), 7.0))
        torch.testing.assert_close(blocks[2], torch.full((2, 59), 9.0))
        self.assertEqual(self.resident.dirty_blocks(), [])

    def test_stale_plan_fallback_checkpoint_and_shutdown_preserve_updates(self):
        self.load([0, 1])
        self.update(0, 7)
        self.update(1, 8)
        self.resident.start_prefetch(
            iteration=3, visible_block_ids=[0, 1], resident_block_ids=[0, 1],
            block_reader=self.reader, defer_resident_copy=True,
        )
        self.model._paper_plan_camera_ids = [2, 3]
        self.model._paper_plan_bounds_generation = 0

        result = load_paper_stage1_working_set(
            gaussians=self.model,
            args=SimpleNamespace(
                gaussian_block_size=4, paper_resident_selection_policy="topc_strict_active_first",
                paper_resident_capacity_blocks=3,
            ),
            iteration=3, total_n_gaussians=10, visible_block_ids=[1, 2],
            current_camera_blocks={2: [1], 3: [2]}, training_schedule=[0, 1, 2, 3],
            storage_adapter=self.adapter, should_log=False,
            get_double_buffer_gpu_fn=lambda **kwargs: self.resident,
            ensure_local_to_global_mapping_fn=Mock(),
            resolve_current_iteration_resident_blocks_fn=lambda **kwargs: ([1, 2], "test"),
            current_bounds_generation=1, current_camera_ids=[2, 3],
        )
        torch.testing.assert_close(result[0]["xyz"][:4], torch.full((4, 3), 8.0, device="cuda"))
        self.assertEqual(self.resident.dirty_blocks(), [])
        self.load([0, 1, 2])
        torch.testing.assert_close(self.resident.active_buffer.xyz[:4], torch.full((4, 3), 7.0, device="cuda"))
        self.update(2, 9)
        write_pure_ssd_snapshot_checkpoint(
            storage_adapter=self.adapter, gaussians=self.model,
            checkpoint_dir=self.root / "checkpoint", iteration=4, next_iteration=5,
            args=SimpleNamespace(), chunk_blocks=1,
        )
        self.adapter.shutdown()
        base_file = self.root / "checkpoint" / "ssd_snapshot" / "base_file.bin"
        restored = torch.from_numpy(np.fromfile(base_file, dtype=np.float32).reshape(10, 59))
        expected = torch.cat((torch.full((4, 59), 7.0), torch.full((4, 59), 8.0), torch.full((2, 59), 9.0)))
        torch.testing.assert_close(restored, expected)

    def test_swap_rejects_unwritten_dirty_eviction(self):
        self.load([0, 1])
        self.update(0, 7)
        self.update(1, 8)
        self.resident.start_prefetch(
            iteration=3, visible_block_ids=[1, 2], resident_block_ids=[1],
            evicted_block_ids=[0], block_reader=self.reader,
        )
        self.assertTrue(self.resident.wait_for_prefetch(3))
        with self.assertRaisesRegex(RuntimeError, "Cannot evict GPU-dirty"):
            self.resident.swap_buffers()
        self.adapter.writeback_resident_blocks([0])
        self.resident.swap_buffers()
        torch.testing.assert_close(self.reader.read_blocks([0])[0], torch.full((4, 59), 7.0))
        self.assertEqual(self.resident.dirty_blocks(), [1])
        self.adapter.flush_resident_dirty()
        torch.testing.assert_close(self.reader.read_blocks([1])[1], torch.full((4, 59), 8.0))

    def test_activation_refreshes_block_lookup_after_swap(self):
        self.load([0, 1])
        self.resident.start_prefetch(
            iteration=3, visible_block_ids=[1, 2], resident_block_ids=[1],
            evicted_block_ids=[0], block_reader=self.reader,
        )
        self.assertTrue(self.resident.wait_for_prefetch(3))
        self.resident.swap_buffers()
        activate_paper_prefetched_buffer(self.model, self.resident, Mock())
        self.assertEqual(
            self.staging.global_to_local(torch.tensor([0, 4, 8], device="cuda")).tolist(),
            [-1, 0, 4],
        )

    def test_missing_block_is_rejected_before_consuming_staging_slot(self):
        self.load([0])
        with self.assertRaisesRegex(RuntimeError, "absent from the active GPU buffer"):
            self.staging.stage_updated_blocks([0, 1], source_buffer=self.resident.active_buffer)
        self.assertTrue(all(event.is_set() for event in self.staging._writeback_slot_available))
