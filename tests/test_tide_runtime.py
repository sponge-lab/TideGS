import unittest
import io
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from storage.block_reader import TieredCacheBlockReader, resolve_block_reader_backend
from strategies.tide_engine import runtime
from strategies.tide_engine.double_buffer_gpu import DoubleBufferGPUWorkingSet
from strategies.tide_engine.gpu_working_set import GPUWorkingSet


class TidePrefetchTest(unittest.TestCase):
    def setUp(self):
        ranges = mock.patch.object(
            runtime.torch.cuda.nvtx,
            "range",
            side_effect=lambda *args, **kwargs: nullcontext(),
        )
        ranges.start()
        self.addCleanup(ranges.stop)

    def test_next_batch_prefetches_only_incoming_blocks_and_defers_resident_copy(self):
        cache = SimpleNamespace(prefetch_future=mock.Mock(return_value=1))
        reader = TieredCacheBlockReader(cache, total_gaussians=8, block_size=2)
        adapter = SimpleNamespace(
            get_visible_blocks_batch=mock.Mock(return_value=(7, {2: [1], 3: [2]})),
            wait_for_pending_gpu_copies=mock.Mock(),
        )
        double_buffer = SimpleNamespace(start_prefetch=mock.Mock())

        result = runtime.plan_and_start_resident_prefetch(
            storage_adapter=adapter,
            training_schedule=[0, 1, 2, 3],
            iteration=1,
            batch_size=2,
            current_block_ids=[0, 1],
            schedule_ordering="trajectory",
            current_resident_blocks=[0, 1],
            current_resident_recency_scores={},
            resident_selection_policy="topc_strict_active_first",
            resident_lambda=0.3,
            resident_recency_decay=0.95,
            resident_capacity_blocks=2,
            balanced_seed_fraction=0.25,
            active_block_reader=reader,
            double_buffer=double_buffer,
        )

        adapter.get_visible_blocks_batch.assert_called_once_with([2, 3])
        cache.prefetch_future.assert_called_once_with([2])
        double_buffer.start_prefetch.assert_called_once_with(
            iteration=3,
            visible_block_ids=[1, 2],
            filters_global=[],
            resident_block_ids=[1],
            evicted_block_ids=[0],
            allow_resident_copy=True,
            block_reader=reader,
            defer_resident_copy=True,
            before_target_reuse=adapter.wait_for_pending_gpu_copies,
        )
        self.assertEqual(result["bounds_generation"], 7)
        self.assertEqual(result["camera_ids"], [2, 3])
        self.assertEqual(result["future_submitted"], 1)
        self.assertTrue(result["prefetch_started"])

    def load_working_set(self, *, prefetched):
        tensors = {"xyz": object()}
        retention_stats = {"source": "test"}
        manager = SimpleNamespace(
            load_visible_blocks_with_retention=mock.Mock(
                return_value=(tensors, retention_stats),
            ),
        )
        reader = object()
        gaussians = SimpleNamespace(
            gpu_working_set_manager=manager,
            _block_reader=reader,
            _paper_plan_bounds_generation=7,
            _paper_plan_camera_ids=[2, 3],
            _paper_ab_runtime_stats={
                "prefetch_hits": 0,
                "cold_starts": 0,
                "prefetch_misses": 0,
                "sync_fallbacks": 0,
            },
        )
        double_buffer = SimpleNamespace(
            wait_for_prefetch=mock.Mock(return_value=prefetched),
            swap_buffers=mock.Mock(),
            discard_prefetch=mock.Mock(),
        )
        select_resident = mock.Mock(return_value=([1, 2, 3], "test"))
        with mock.patch.object(
            runtime,
            "activate_paper_prefetched_buffer",
            return_value=(tensors, retention_stats),
        ) as activate:
            result = runtime.load_paper_stage1_working_set(
                gaussians=gaussians,
                args=SimpleNamespace(
                    gaussian_block_size=2,
                    paper_resident_selection_policy="topc_strict_active_first",
                    paper_resident_capacity_blocks=2,
                ),
                iteration=3,
                total_n_gaussians=8,
                visible_block_ids=[1, 2],
                current_camera_blocks={2: [1], 3: [2]},
                training_schedule=[0, 1, 2, 3],
                storage_adapter=mock.Mock(),
                should_log=False,
                get_double_buffer_gpu_fn=mock.Mock(return_value=double_buffer),
                ensure_local_to_global_mapping_fn=mock.Mock(),
                resolve_current_iteration_resident_blocks_fn=select_resident,
                current_bounds_generation=7,
                current_camera_ids=[2, 3],
            )

        self.assertIs(result[0], tensors)
        self.assertIs(result[1], retention_stats)
        double_buffer.wait_for_prefetch.assert_called_once_with(3)
        return result, gaussians, double_buffer, select_resident, activate

    def test_ready_prefetch_activates_buffer_without_synchronous_reads(self):
        result, gaussians, double_buffer, select_resident, activate = self.load_working_set(
            prefetched=True,
        )

        self.assertTrue(result[2])
        self.assertEqual(result[3], "prefetched_ab_buffer")
        double_buffer.swap_buffers.assert_called_once_with()
        activate.assert_called_once()
        select_resident.assert_not_called()
        gaussians.gpu_working_set_manager.load_visible_blocks_with_retention.assert_not_called()
        self.assertEqual(gaussians._paper_ab_runtime_stats["prefetch_hits"], 1)
        self.assertEqual(gaussians._paper_ab_runtime_stats["sync_fallbacks"], 0)

    def test_missing_prefetch_uses_tiered_reader_and_restricts_resident_blocks(self):
        result, gaussians, double_buffer, select_resident, activate = self.load_working_set(
            prefetched=False,
        )

        self.assertFalse(result[2])
        self.assertEqual(result[3], "sync_resident_set(test)")
        double_buffer.swap_buffers.assert_not_called()
        activate.assert_not_called()
        select_resident.assert_called_once()
        gaussians.gpu_working_set_manager.load_visible_blocks_with_retention.assert_called_once_with(
            visible_block_ids=[1, 2],
            active_blocks_ram=None,
            enable_retention=False,
            block_reader=gaussians._block_reader,
        )
        self.assertEqual(gaussians._paper_ab_runtime_stats["prefetch_misses"], 1)
        self.assertEqual(gaussians._paper_ab_runtime_stats["sync_fallbacks"], 1)

    def test_stale_prefetch_is_not_activated_and_current_set_is_loaded(self):
        tensors = {"xyz": object()}
        manager = SimpleNamespace(
            load_visible_blocks_with_retention=mock.Mock(
                return_value=(tensors, {"source": "sync"}),
            ),
        )
        gaussians = SimpleNamespace(
            gpu_working_set_manager=manager,
            _block_reader=object(),
            _paper_plan_bounds_generation=7,
            _paper_plan_camera_ids=[2, 3],
            _paper_expected_resident_blocks=[0, 1],
            _paper_resident_recency_scores={},
            _paper_ab_runtime_stats={
                "prefetch_hits": 0,
                "cold_starts": 0,
                "prefetch_misses": 0,
                "sync_fallbacks": 0,
            },
        )
        double_buffer = SimpleNamespace(
            wait_for_prefetch=mock.Mock(return_value=True),
            swap_buffers=mock.Mock(),
            discard_prefetch=mock.Mock(),
        )
        args = SimpleNamespace(
            gaussian_block_size=2,
            paper_resident_selection_policy="topc_strict_active_first",
            paper_resident_capacity_blocks=2,
            paper_resident_lambda=0.3,
            paper_resident_recency_decay=0.95,
            paper_balanced_seed_fraction=0.25,
        )

        result = runtime.load_paper_stage1_working_set(
            gaussians=gaussians,
            args=args,
            iteration=3,
            total_n_gaussians=20,
            visible_block_ids=[0, 2],
            current_camera_blocks={2: [0], 3: [2]},
            training_schedule=[0, 1, 2, 3],
            storage_adapter=mock.Mock(),
            should_log=False,
            get_double_buffer_gpu_fn=mock.Mock(return_value=double_buffer),
            ensure_local_to_global_mapping_fn=mock.Mock(),
            resolve_current_iteration_resident_blocks_fn=runtime.resolve_current_iteration_resident_blocks,
            current_bounds_generation=8,
            current_camera_ids=[2, 3],
        )

        self.assertFalse(result[2])
        self.assertEqual(result[3], "sync_resident_set(bootstrap_topc_over_k1)")
        double_buffer.swap_buffers.assert_not_called()
        manager.load_visible_blocks_with_retention.assert_called_once_with(
            visible_block_ids=[0, 2],
            active_blocks_ram=None,
            enable_retention=False,
            block_reader=gaussians._block_reader,
        )

    def _load_after_bounds_change(self, *, expected_resident, visible, capacity):
        tensors = {"xyz": object()}
        manager = SimpleNamespace(
            load_visible_blocks_with_retention=mock.Mock(
                return_value=(tensors, {"source": "sync"}),
            ),
        )
        gaussians = SimpleNamespace(
            gpu_working_set_manager=manager,
            _block_reader=object(),
            _paper_plan_bounds_generation=7,
            _paper_plan_camera_ids=[2, 3],
            _paper_expected_resident_blocks=expected_resident,
            _paper_resident_recency_scores={},
            _paper_ab_runtime_stats={
                "prefetch_hits": 0,
                "cold_starts": 0,
                "prefetch_misses": 0,
                "sync_fallbacks": 0,
            },
        )
        double_buffer = SimpleNamespace(
            wait_for_prefetch=mock.Mock(return_value=True),
            swap_buffers=mock.Mock(),
            discard_prefetch=mock.Mock(),
        )
        args = SimpleNamespace(
            gaussian_block_size=2,
            paper_resident_selection_policy="topc_strict_active_first",
            paper_resident_capacity_blocks=capacity,
            paper_resident_lambda=0.3,
            paper_resident_recency_decay=0.95,
            paper_balanced_seed_fraction=0.25,
        )
        with mock.patch.object(
            runtime,
            "activate_paper_prefetched_buffer",
            return_value=(tensors, {"source": "prefetched"}),
        ) as activate:
            result = runtime.load_paper_stage1_working_set(
                gaussians=gaussians,
                args=args,
                iteration=3,
                total_n_gaussians=20,
                visible_block_ids=visible,
                current_camera_blocks={2: visible[:1], 3: visible[1:]},
                training_schedule=[0, 1, 2, 3],
                storage_adapter=mock.Mock(),
                should_log=False,
                get_double_buffer_gpu_fn=mock.Mock(return_value=double_buffer),
                ensure_local_to_global_mapping_fn=mock.Mock(),
                resolve_current_iteration_resident_blocks_fn=runtime.resolve_current_iteration_resident_blocks,
                current_bounds_generation=8,
                current_camera_ids=[2, 3],
            )
        return result, gaussians, double_buffer, activate

    def test_bounds_change_alone_keeps_prefetched_plan_when_it_covers_visible_blocks(self):
        result, gaussians, double_buffer, activate = self._load_after_bounds_change(
            expected_resident=[1, 2, 5],
            visible=[1, 2],
            capacity=4,
        )

        self.assertTrue(result[2])
        self.assertEqual(result[3], "prefetched_ab_buffer")
        double_buffer.swap_buffers.assert_called_once_with()
        double_buffer.discard_prefetch.assert_not_called()
        activate.assert_called_once()
        gaussians.gpu_working_set_manager.load_visible_blocks_with_retention.assert_not_called()
        self.assertEqual(gaussians._paper_ab_runtime_stats["prefetch_hits"], 1)
        self.assertEqual(gaussians._paper_ab_runtime_stats["sync_fallbacks"], 0)
        self.assertEqual(gaussians._paper_ab_runtime_stats["revalidated_hits"], 1)

    def test_bounds_change_keeps_full_capacity_plan_when_visible_set_exceeds_capacity(self):
        result, gaussians, double_buffer, activate = self._load_after_bounds_change(
            expected_resident=[1, 2],
            visible=[1, 2, 3],
            capacity=2,
        )

        self.assertTrue(result[2])
        double_buffer.swap_buffers.assert_called_once_with()
        activate.assert_called_once()
        self.assertEqual(gaussians._paper_ab_runtime_stats["sync_fallbacks"], 0)

    def test_stale_fallback_replans_when_old_set_has_no_overlap(self):
        tensors = {"xyz": object()}
        manager = SimpleNamespace(
            load_visible_blocks_with_retention=mock.Mock(
                return_value=(tensors, {"source": "sync"}),
            ),
        )
        gaussians = SimpleNamespace(
            gpu_working_set_manager=manager,
            _block_reader=object(),
            _paper_plan_bounds_generation=7,
            _paper_plan_camera_ids=[2, 3],
            _paper_expected_resident_blocks=[0, 1],
            _paper_resident_recency_scores={},
            _paper_ab_runtime_stats={
                "prefetch_hits": 0,
                "cold_starts": 0,
                "prefetch_misses": 0,
                "sync_fallbacks": 0,
            },
        )
        double_buffer = SimpleNamespace(
            wait_for_prefetch=mock.Mock(return_value=False),
            discard_prefetch=mock.Mock(),
        )
        args = SimpleNamespace(
            gaussian_block_size=2,
            paper_resident_selection_policy="topc_strict_active_first",
            paper_resident_capacity_blocks=2,
            paper_resident_lambda=0.3,
            paper_resident_recency_decay=0.95,
            paper_balanced_seed_fraction=0.25,
        )

        runtime.load_paper_stage1_working_set(
            gaussians=gaussians,
            args=args,
            iteration=3,
            total_n_gaussians=20,
            visible_block_ids=[2, 3, 4, 5, 6],
            current_camera_blocks={2: [2, 3], 3: [4, 5, 6]},
            training_schedule=[0, 1, 2, 3],
            storage_adapter=mock.Mock(),
            should_log=False,
            get_double_buffer_gpu_fn=mock.Mock(return_value=double_buffer),
            ensure_local_to_global_mapping_fn=mock.Mock(),
            resolve_current_iteration_resident_blocks_fn=runtime.resolve_current_iteration_resident_blocks,
            current_bounds_generation=8,
            current_camera_ids=[2, 3],
        )

        loaded = manager.load_visible_blocks_with_retention.call_args.kwargs["visible_block_ids"]
        self.assertEqual(loaded, [2, 3])
        self.assertLessEqual(len(loaded), args.paper_resident_capacity_blocks)

    def test_reader_resolution_rejects_full_ram_backend(self):
        self.assertEqual(resolve_block_reader_backend("auto"), "tiered_cache")
        self.assertEqual(resolve_block_reader_backend("tiered_cache"), "tiered_cache")
        with self.assertRaises(ValueError):
            resolve_block_reader_backend("unified_params")

    def test_stale_plan_preserves_updated_and_departing_blocks(self):
        cache = {block_id: torch.zeros(2, 3) for block_id in range(3)}
        manager = SimpleNamespace(
            loaded_blocks=[0, 1],
            gpu_xyz=torch.cat((torch.full((2, 3), 7.0), torch.full((2, 3), 8.0))),
        )
        events = []

        def flush():
            events.append("flush")
            for position, block_id in enumerate(manager.loaded_blocks):
                cache[block_id] = manager.gpu_xyz[position * 2:position * 2 + 2].clone()

        def load(**kwargs):
            events.append("load")
            manager.loaded_blocks = kwargs["visible_block_ids"]
            manager.gpu_xyz = torch.cat([cache[block_id] for block_id in manager.loaded_blocks])
            return {"xyz": manager.gpu_xyz}, {}

        manager.load_visible_blocks_with_retention = load
        adapter = SimpleNamespace(flush_resident_dirty=flush, check_writeback_error=mock.Mock())
        double_buffer = SimpleNamespace(
            wait_for_prefetch=mock.Mock(return_value=True),
            discard_prefetch=mock.Mock(),
        )
        gaussians = SimpleNamespace(
            gpu_working_set_manager=manager,
            _block_reader=object(),
            _paper_plan_bounds_generation=7,
            _paper_plan_camera_ids=[2, 3],
            _paper_ab_runtime_stats={
                "prefetch_hits": 0, "cold_starts": 0,
                "prefetch_misses": 0, "sync_fallbacks": 0,
            },
        )
        result = runtime.load_paper_stage1_working_set(
            gaussians=gaussians,
            args=SimpleNamespace(
                gaussian_block_size=2,
                paper_resident_selection_policy="topc_strict_active_first",
                paper_resident_capacity_blocks=3,
            ),
            iteration=3,
            total_n_gaussians=6,
            visible_block_ids=[1, 2],
            current_camera_blocks={2: [1], 3: [2]},
            training_schedule=[0, 1, 2, 3],
            storage_adapter=adapter,
            should_log=False,
            get_double_buffer_gpu_fn=lambda **kwargs: double_buffer,
            ensure_local_to_global_mapping_fn=mock.Mock(),
            resolve_current_iteration_resident_blocks_fn=lambda **kwargs: ([1, 2], "test"),
            current_bounds_generation=8,
            current_camera_ids=[2, 3],
        )

        self.assertEqual(events, ["flush", "load"])
        torch.testing.assert_close(cache[0], torch.full((2, 3), 7.0))
        torch.testing.assert_close(result[0]["xyz"][:2], torch.full((2, 3), 8.0))
        double_buffer.discard_prefetch.assert_called_once_with()


class TideWritebackTest(unittest.TestCase):
    def setUp(self):
        ranges = mock.patch.object(
            runtime.torch.cuda.nvtx,
            "range",
            side_effect=lambda *args, **kwargs: nullcontext(),
        )
        ranges.start()
        self.addCleanup(ranges.stop)
        self.adapter = SimpleNamespace(
            writeback_resident_blocks=mock.Mock(return_value=1),
        )
        self.double_buffer = SimpleNamespace(
            finalize_retained_blocks=mock.Mock(
                return_value=SimpleNamespace(ready_blocks=1, copied_blocks=1),
            ),
        )

    def writeback(self, updated_block_ids=(2,)):
        return runtime.apply_paper_writeback_payload(
            storage_adapter=self.adapter,
            double_buffer=self.double_buffer,
            updated_block_ids=list(updated_block_ids),
            omega_blocks=[1],
        )

    def test_ready_handoff_submits_dirty_evictions_without_blocking(self):
        self.assertEqual(self.writeback(), (1, 1, 1, 1))
        self.double_buffer.finalize_retained_blocks.assert_called_once_with(
            block_ids=[1], target="loading",
        )
        self.adapter.writeback_resident_blocks.assert_called_once_with([2])

    def test_incomplete_handoff_fails_before_replacing_dirty_gpu_data(self):
        self.double_buffer.finalize_retained_blocks.return_value.ready_blocks = 0

        with self.assertRaisesRegex(RuntimeError, "Incomplete resident handoff"):
            self.writeback()

        self.adapter.writeback_resident_blocks.assert_not_called()

    def test_failed_writeback_propagates_error(self):
        self.adapter.writeback_resident_blocks.side_effect = OSError("write failed")

        with self.assertRaisesRegex(OSError, "write failed"):
            self.writeback()


    def test_empty_evictions_still_finalize_retained_blocks(self):
        self.assertEqual(self.writeback(updated_block_ids=()), (0, 1, 1, 0))
        self.double_buffer.finalize_retained_blocks.assert_called_once()
        self.adapter.writeback_resident_blocks.assert_not_called()


class TideRuntimeContractTest(unittest.TestCase):
    def test_full_ram_model_is_rejected_before_batch_processing(self):
        from strategies.tide_engine import engine

        model = SimpleNamespace(
            gpu_working_set_manager=object(),
            optimizer=SimpleNamespace(is_ssd_offload_mode=True),
            _unified_params=torch.empty((1, 59)),
        )
        with self.assertRaisesRegex(RuntimeError, "full RAM parameter table"):
            engine.clm_offload_train_one_batch(
                gaussians=model, scene=None, batched_cameras=[],
                background=None, pipe_args=None, comm_stream=None,
                storage_adapter=object(), runtime_args=SimpleNamespace(),
            )

    def test_prefetch_requires_reader_before_mutating_buffer_state(self):
        with self.assertRaisesRegex(ValueError, "requires a block reader"):
            DoubleBufferGPUWorkingSet.start_prefetch(
                SimpleNamespace(), iteration=1, visible_block_ids=[0],
            )

    def test_initialization_without_mode_selector_keeps_resident_state(self):
        args = SimpleNamespace(
            paper_optimizer_backend="gpu_resident",
            paper_optimizer_state_mode="resident_blocks",
            paper_optimizer_deferred_mode="off",
            gaussian_block_size=4,
        )
        model = SimpleNamespace(use_gpu_features=True)
        result = runtime.initialize_paper_mode_runtime_state(
            gaussians=model, args=args, iteration=2, log_file=io.StringIO(),
        )
        self.assertEqual(result, ("off", "gpu_resident"))
        self.assertEqual(model._paper_optimizer_block_size, 4)
        self.assertEqual(model._paper_optimizer_state_mode, "resident_blocks")

        model.use_gpu_features = False
        with self.assertRaisesRegex(RuntimeError, "working-set features"):
            runtime.initialize_paper_mode_runtime_state(
                gaussians=model, args=args, iteration=2,
            )

    def test_facade_forwards_current_batch_inputs(self):
        from strategies.tide_engine import engine

        inputs = {
            name: object() for name in (
                "gaussians", "scene", "batched_cameras", "background", "pipe_args",
                "comm_stream", "storage_adapter", "training_schedule", "runtime_args",
            )
        }
        with mock.patch.object(engine, "clm_offload_train_one_batch", autospec=True) as train:
            self.assertIs(runtime.train_tide_batch(**inputs), train.return_value)
        train.assert_called_once_with(
            inputs["gaussians"], inputs["scene"], inputs["batched_cameras"],
            inputs["background"], inputs["pipe_args"], inputs["comm_stream"],
            storage_adapter=inputs["storage_adapter"],
            training_schedule=inputs["training_schedule"],
            runtime_args=inputs["runtime_args"],
        )


class TideSynchronousMaterializationTest(unittest.TestCase):
    def test_tiered_reader_preserves_layout_tail_rows_and_updated_blocks(self):
        source = torch.arange(5 * 59, dtype=torch.float32).reshape(5, 59)
        cache = SimpleNamespace(prefetch=mock.Mock(
            side_effect=lambda blocks: {
                block: source[block * 2:min(block * 2 + 2, 5)] for block in blocks
            },
        ))
        reader = TieredCacheBlockReader(cache, total_gaussians=5, block_size=2)
        with mock.patch("torch.cuda.Stream"):
            manager = GPUWorkingSet(5, block_size=2, device="cpu")
        self.addCleanup(manager.clear, release_cuda_cache=False)

        tensors, stats = manager.load_visible_blocks_with_retention(
            visible_block_ids=[2, 0], enable_retention=False, block_reader=reader,
        )
        expected = source[[0, 1, 4]]
        columns = {
            "xyz": slice(0, 3), "scaling": slice(3, 6), "rotation": slice(6, 10),
            "opacity": slice(10, 11), "features_dc": slice(11, 14),
            "features_rest": slice(14, 59),
        }
        for name, selection in columns.items():
            torch.testing.assert_close(tensors[name], expected[:, selection])
        self.assertEqual(manager.loaded_blocks, [0, 2])
        self.assertEqual(manager.block_to_gpu_slice[2], slice(2, 3))
        torch.testing.assert_close(manager.local_to_global_idx, torch.tensor([0, 1, 4]))
        self.assertEqual(stats["num_gaussians"], 3)

        source[:2].add_(1000)
        tensors, stats = manager.load_visible_blocks_with_retention(
            visible_block_ids=[0], enable_retention=True, block_reader=reader,
        )
        for name, selection in columns.items():
            torch.testing.assert_close(tensors[name], source[:2, selection])
        self.assertEqual(stats["hotspot_count"], 1)
        self.assertEqual(stats["data_reused_count"], 0)
        self.assertEqual(cache.prefetch.call_args_list, [mock.call([0, 2]), mock.call([0])])


if __name__ == "__main__":
    unittest.main()
