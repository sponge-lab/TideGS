import threading
import time
import unittest
import importlib.util
import sys
import types
from collections import Counter
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    class _FakeTensor:
        def __init__(self, shape, value):
            self.shape = tuple(shape)
            self.value = float(value)

        def clone(self):
            return _FakeTensor(self.shape, self.value)

        def numel(self):
            total = 1
            for dim in self.shape:
                total *= dim
            return total

        def dim(self):
            return len(self.shape)

    torch = types.SimpleNamespace(
        Tensor=_FakeTensor,
        full=lambda shape, value: _FakeTensor(shape, value),
        equal=lambda left, right: (
            isinstance(left, _FakeTensor)
            and isinstance(right, _FakeTensor)
            and left.shape == right.shape
            and left.value == right.value
        ),
        is_tensor=lambda value: isinstance(value, _FakeTensor),
    )
    sys.modules["torch"] = torch

sys.modules.setdefault("psutil", types.SimpleNamespace())

_MODULE_PATH = Path(__file__).resolve().parents[1] / "storage" / "tiered_cache_manager.py"
_SPEC = importlib.util.spec_from_file_location("tiered_cache_manager_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
TieredCacheManager = _MODULE.TieredCacheManager


class FakeStorage:
    def __init__(self, block_size=2, point_dim=3):
        self.block_size = block_size
        self.point_dim = point_dim
        self.data = {
            block_id: torch.full((block_size, point_dim), float(block_id))
            for block_id in range(32)
        }
        self.read_counts = Counter()
        self.call_count = 0
        self.lock = threading.Lock()
        self.block_first_read = False
        self.fail_first_read = False
        self.read_started = threading.Event()
        self.release_read = threading.Event()

    def read_blocks(self, block_ids):
        block_ids = [int(block_id) for block_id in block_ids]
        with self.lock:
            self.call_count += 1
            call_index = self.call_count
            for block_id in block_ids:
                self.read_counts[block_id] += 1

        if self.block_first_read and call_index == 1:
            self.read_started.set()
            if not self.release_read.wait(timeout=2.0):
                raise TimeoutError("test timed out waiting to release first read")

        if self.fail_first_read and call_index == 1:
            raise RuntimeError("injected read failure")

        return {block_id: self.data[block_id].clone() for block_id in block_ids}

    def write_patch(self, dirty_blocks):
        return None


class TieredCacheSingleFlightTest(unittest.TestCase):
    def make_cache(self, storage):
        cache = TieredCacheManager(
            storage,
            max_ram_gb=1.0,
            block_size=storage.block_size,
            point_dim=storage.point_dim,
            verbose=False,
        )
        self.addCleanup(cache.shutdown)
        return cache

    def run_prefetch_in_thread(self, cache, block_ids):
        result = {}
        errors = []

        def target():
            try:
                result.update(cache.prefetch(block_ids))
            except Exception as exc:  # pragma: no cover - asserted by tests
                errors.append(exc)

        thread = threading.Thread(target=target)
        thread.start()
        return thread, result, errors

    def test_future_read_serves_urgent_without_second_ssd_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        self.assertEqual(cache.prefetch_future([7]), 1)
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread, result, errors = self.run_prefetch_in_thread(cache, [7])
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[7], storage.data[7]))
        self.assertEqual(storage.read_counts[7], 1)

        stats = cache.get_stats()
        self.assertEqual(stats["urgent_prefetch_blocks"], 0)
        self.assertEqual(stats["future_prefetch_reserved"], 1)
        self.assertGreaterEqual(stats["inflight_wait_blocks"], 1)

    def test_future_hint_skips_block_already_claimed_by_urgent(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        thread, result, errors = self.run_prefetch_in_thread(cache, [3])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        self.assertEqual(cache.prefetch_future([3]), 0)

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[3], storage.data[3]))
        self.assertEqual(storage.read_counts[3], 1)
        self.assertGreaterEqual(cache.get_stats()["future_prefetch_skipped"], 1)

    def test_two_urgent_prefetches_share_one_ssd_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        cache = self.make_cache(storage)

        thread_a, result_a, errors_a = self.run_prefetch_in_thread(cache, [5])
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread_b, result_b, errors_b = self.run_prefetch_in_thread(cache, [5])
        time.sleep(0.05)
        self.assertTrue(thread_b.is_alive())

        storage.release_read.set()
        thread_a.join(timeout=2.0)
        thread_b.join(timeout=2.0)

        self.assertFalse(thread_a.is_alive())
        self.assertFalse(thread_b.is_alive())
        self.assertEqual(errors_a, [])
        self.assertEqual(errors_b, [])
        self.assertTrue(torch.equal(result_a[5], storage.data[5]))
        self.assertTrue(torch.equal(result_b[5], storage.data[5]))
        self.assertEqual(storage.read_counts[5], 1)
        self.assertEqual(cache.get_stats()["urgent_prefetch_blocks"], 1)

    def test_flushing_buffer_hit_does_not_touch_ssd_or_inflight(self):
        storage = FakeStorage()
        cache = self.make_cache(storage)
        tensor = torch.full((storage.block_size, storage.point_dim), 9.0)

        with cache.flushing_lock:
            cache.flushing_buffer[9] = (tensor, time.time(), 0)
        try:
            result = cache.prefetch([9])
            self.assertTrue(torch.equal(result[9], tensor))
            self.assertEqual(storage.read_counts[9], 0)
            self.assertEqual(cache.get_stats()["inflight_read_blocks"], 0)
        finally:
            with cache.flushing_lock:
                cache.flushing_buffer.pop(9, None)

    def test_future_failure_wakes_urgent_and_allows_fallback_read(self):
        storage = FakeStorage()
        storage.block_first_read = True
        storage.fail_first_read = True
        cache = self.make_cache(storage)

        self.assertEqual(cache.prefetch_future([11]), 1)
        self.assertTrue(storage.read_started.wait(timeout=2.0))

        thread, result, errors = self.run_prefetch_in_thread(cache, [11])
        time.sleep(0.05)
        self.assertTrue(thread.is_alive())

        storage.release_read.set()
        thread.join(timeout=2.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(torch.equal(result[11], storage.data[11]))
        self.assertEqual(storage.read_counts[11], 2)

        stats = cache.get_stats()
        self.assertEqual(stats["future_prefetch_errors"], 1)
        self.assertGreaterEqual(stats["inflight_fallback_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
