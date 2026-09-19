import threading
import time
import unittest
from queue import Queue
from unittest.mock import Mock

import torch

from storage.block_reader import TieredCacheBlockReader
from storage.tide_storage_adapter import TideStorageAdapter


class _Payload:
    def __init__(self, block_id, tensor):
        self.block_ids = [block_id]
        self.tensor = tensor
        self.ready = threading.Event()
        self.released = threading.Event()

    def wait_gpu(self):
        self.ready.wait(timeout=2.0)

    def wait(self):
        if not self.ready.wait(timeout=2.0):
            raise TimeoutError("payload was not released")
        return {self.block_ids[0]: self.tensor}

    def release(self):
        self.released.set()


class _Cache:
    def __init__(self, block_id, tensor):
        self.data = {block_id: tensor}
        self.future_hints = []

    def prefetch(self, block_ids):
        return {block_id: self.data[block_id] for block_id in block_ids}

    def prefetch_future(self, block_ids):
        self.future_hints.append(list(block_ids))
        return len(block_ids)


class _ResidentState:
    def __init__(self, block_ids):
        self._dirty = set(block_ids)
        self.active_buffer = object()
        self.synchronize_prefetch = Mock()

    def dirty_blocks(self):
        return sorted(self._dirty)

    def mark_blocks_written_back(self, block_ids):
        self._dirty.difference_update(block_ids)


class _WorkingSet:
    def __init__(self, payloads):
        self.payloads = payloads

    def stage_updated_blocks(self, block_ids, *, source_buffer):
        self.source_buffer = source_buffer
        payload = _Payload(block_ids[0], torch.full((4, 59), 9.0))
        payload.ready.set()
        self.payloads.append(payload)
        return payload


class AsyncCacheCommitTest(unittest.TestCase):
    def setUp(self):
        self.adapter = TideStorageAdapter.__new__(TideStorageAdapter)
        self.adapter.execution_metrics = {
            "paper_cache_sync_calls": 0,
            "paper_cache_sync_blocks": 0,
            "paper_cache_sync_staged_blocks": 0,
            "paper_bounds_refresh_blocks": 0,
            "background_cache_commit_jobs": 0,
            "background_cache_commit_blocks": 0,
            "background_cache_commit_waits": 0,
        }
        self.adapter._cache_commit_queue = Queue()
        self.adapter._cache_commit_lock = threading.RLock()
        self.adapter._pending_cache_commits = {}
        self.adapter._cache_commit_error = None
        self.adapter._cache_commit_thread = None
        self.payloads = []
        self.allow_commit_error = False
        self.refresh_bounds_flags = []
        self.cache = _Cache(3, torch.zeros(4, 59))

        def sync_cache(updated, *, refresh_bounds=True):
            self.refresh_bounds_flags.append(bool(refresh_bounds))
            self.cache.data.update(
                {block_id: tensor.clone() for block_id, tensor in updated.items()}
            )
            return len(updated)

        self.adapter.sync_cache_from_cpu_views = sync_cache
        self.adapter._start_cache_commit_worker()

    def tearDown(self):
        for payload in self.payloads:
            payload.ready.set()
        self.adapter._cache_commit_queue.join()
        self.adapter._cache_commit_queue.put(None)
        self.adapter._cache_commit_queue.join()
        self.adapter._cache_commit_thread.join(timeout=2.0)
        if not self.allow_commit_error:
            self.adapter.check_writeback_error()

    def test_reader_waits_for_new_version_and_hint_skips_pending_block(self):
        updated = torch.full((4, 59), 7.0)
        payload = _Payload(3, updated)
        self.payloads.append(payload)
        self.assertEqual(self.adapter.submit_cache_writeback(payload), 1)

        reader = TieredCacheBlockReader(
            cache_manager=self.cache,
            total_gaussians=16,
            block_size=4,
            before_read=self.adapter.wait_for_cache_blocks,
            filter_hint=self.adapter.filter_cache_prefetch_candidates,
        )
        self.assertEqual(reader.hint_future([3]), 0)
        self.assertEqual(self.cache.future_hints, [])

        result = {}
        thread = threading.Thread(
            target=lambda: result.update(reader.read_blocks([3])),
        )
        thread.start()
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        payload.ready.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(torch.equal(result[3], updated))
        self.assertTrue(payload.released.is_set())
        self.assertEqual(self.refresh_bounds_flags[-1], True)

    def test_training_commit_skips_duplicate_cpu_bounds_refresh(self):
        payload = _Payload(3, torch.full((4, 59), 5.0))
        self.payloads.append(payload)
        self.assertEqual(
            self.adapter.submit_cache_writeback(
                payload,
                bounds_managed_externally=True,
            ),
            1,
        )
        payload.ready.set()
        self.adapter.drain_cache_writebacks()
        self.assertEqual(self.refresh_bounds_flags[-1], False)

    def test_checkpoint_flushes_gpu_dirty_resident_blocks(self):
        resident = _ResidentState([3])
        working_set = _WorkingSet(self.payloads)
        self.adapter.bind_resident_writeback(resident, working_set)

        self.assertEqual(self.adapter.flush_resident_dirty(), 1)
        self.assertEqual(resident.dirty_blocks(), [])
        self.assertTrue(torch.equal(self.cache.data[3], torch.full((4, 59), 9.0)))
        self.assertTrue(self.payloads[-1].released.is_set())
        self.assertIs(working_set.source_buffer, resident.active_buffer)
        resident.synchronize_prefetch.assert_called_once_with()

    def test_incomplete_payload_keeps_dirty_state_and_releases_staging(self):
        resident = _ResidentState([2, 3])
        self.adapter.bind_resident_writeback(resident, _WorkingSet(self.payloads))

        with self.assertRaisesRegex(RuntimeError, "Incomplete GPU resident writeback"):
            self.adapter.flush_resident_dirty()

        self.assertEqual(resident.dirty_blocks(), [2, 3])
        self.assertTrue(self.payloads[-1].released.is_set())

    def test_failed_commit_keeps_dirty_state_and_propagates_error(self):
        self.allow_commit_error = True
        resident = _ResidentState([3])
        self.adapter.bind_resident_writeback(resident, _WorkingSet(self.payloads))
        self.adapter.sync_cache_from_cpu_views = Mock(side_effect=OSError("disk unavailable"))

        with self.assertRaisesRegex(RuntimeError, "background cache commit worker failed"):
            self.adapter.flush_resident_dirty()

        self.assertEqual(resident.dirty_blocks(), [3])
        self.assertTrue(self.payloads[-1].released.is_set())
        torch.testing.assert_close(self.cache.data[3], torch.zeros(4, 59))

    def test_partial_cache_commit_fails_instead_of_acknowledging_writeback(self):
        self.allow_commit_error = True
        resident = _ResidentState([3])
        self.adapter.bind_resident_writeback(resident, _WorkingSet(self.payloads))
        self.adapter.sync_cache_from_cpu_views = Mock(return_value=0)

        with self.assertRaisesRegex(RuntimeError, "background cache commit worker failed"):
            self.adapter.flush_resident_dirty()

        self.assertEqual(resident.dirty_blocks(), [3])
        self.assertIn("Incomplete cache commit", str(self.adapter._cache_commit_error))

    def test_pending_commit_is_drained_even_when_no_gpu_blocks_are_dirty(self):
        resident = _ResidentState([])
        self.adapter.bind_resident_writeback(resident, _WorkingSet(self.payloads))
        payload = _Payload(3, torch.full((4, 59), 11.0))
        self.payloads.append(payload)
        self.adapter.submit_cache_writeback(payload)
        finished = threading.Event()
        thread = threading.Thread(target=lambda: (self.adapter.flush_resident_dirty(), finished.set()))
        thread.start()
        self.assertFalse(finished.wait(timeout=0.05))
        payload.ready.set()
        thread.join(timeout=2.0)
        self.assertTrue(finished.is_set())
        torch.testing.assert_close(self.cache.data[3], torch.full((4, 59), 11.0))


if __name__ == "__main__":
    unittest.main()
