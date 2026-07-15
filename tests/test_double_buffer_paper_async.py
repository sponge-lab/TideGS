import threading
import time
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

_HAS_CUDA = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
if _HAS_CUDA:
    from storage.block_reader import BlockLayout
    from strategies.tide_engine.double_buffer_gpu import DoubleBufferGPUWorkingSet
else:
    BlockLayout = None
    DoubleBufferGPUWorkingSet = None


@unittest.skipUnless(_HAS_CUDA, "CUDA is required")
class PaperAsyncPrefetchTest(unittest.TestCase):
    class BlockingReader:
        layout = BlockLayout.CACHE if BlockLayout is not None else None

        def __init__(self, blocks, fail=False):
            self.blocks = blocks
            self.fail = fail
            self.started = threading.Event()
            self.release = threading.Event()
            self.calls = []

        def read_blocks(self, block_ids):
            self.calls.append(list(block_ids))
            self.started.set()
            if not self.release.wait(timeout=5.0):
                raise TimeoutError("test reader was not released")
            if self.fail:
                raise RuntimeError("injected Paper prefetch failure")
            return {block_id: self.blocks[block_id] for block_id in block_ids}

    def make_buffer(self):
        double_buffer = DoubleBufferGPUWorkingSet(
            num_total_gaussians=8,
            block_size=4,
            device="cuda",
            verbose=False,
        )
        active = double_buffer.active_buffer
        active.xyz = torch.full((4, 3), 1.0, device="cuda")
        active.scaling = torch.full((4, 3), 2.0, device="cuda")
        active.rotation = torch.full((4, 4), 3.0, device="cuda")
        active.opacity = torch.full((4, 1), 4.0, device="cuda")
        active.features_dc = torch.full((4, 3), 5.0, device="cuda")
        active.features_rest = torch.full((4, 45), 6.0, device="cuda")
        active.local_to_global_idx = torch.arange(4, device="cuda")
        active.loaded_blocks = [0]
        active.block_to_local_slice = {0: slice(0, 4)}
        active.num_gaussians = 4
        return double_buffer

    @staticmethod
    def cache_block(base):
        block = torch.empty((4, 59), dtype=torch.float32)
        block[:, 0:3] = base + 0
        block[:, 3:6] = base + 1
        block[:, 6:10] = base + 2
        block[:, 10:11] = base + 3
        block[:, 11:14] = base + 4
        block[:, 14:59] = base + 5
        return block

    def test_cold_read_does_not_block_submit_and_omega_is_refreshed(self):
        double_buffer = self.make_buffer()
        reader = self.BlockingReader({1: self.cache_block(20.0)})
        try:
            start = time.perf_counter()
            double_buffer.start_paper_prefetch(
                iteration=2,
                visible_block_ids=[0, 1],
                filters_global=[],
                resident_block_ids=[0],
                evicted_block_ids=[],
                block_reader=reader,
            )
            submit_elapsed = time.perf_counter() - start

            self.assertTrue(reader.started.wait(timeout=2.0))
            self.assertLess(submit_elapsed, 2.0)
            self.assertEqual(reader.calls, [[1]])

            self.assertTrue(double_buffer.wait_for_paper_omega_before_update(2))
            torch.cuda.current_stream().synchronize()
            double_buffer.active_buffer.xyz.add_(10.0)

            reader.release.set()
            self.assertTrue(double_buffer.wait_for_paper_host_enqueue(2))
            refreshed = double_buffer.refresh_blocks_from_block_cache(
                block_cache={0: self.cache_block(11.0)},
                block_ids=[0],
                target="loading",
            )
            self.assertEqual(refreshed, 1)
            self.assertTrue(double_buffer.wait_for_prefetch(2))
            double_buffer.swap_buffers()

            active = double_buffer.active_buffer
            self.assertTrue(torch.all(active.xyz[0:4] == 11.0).item())
            self.assertTrue(torch.all(active.xyz[4:8] == 20.0).item())
            self.assertEqual(active.resident_blocks, [0])
            self.assertEqual(active.streamed_blocks, [1])
        finally:
            reader.release.set()
            torch.cuda.synchronize()
            double_buffer.clear()

    def test_background_failure_falls_back_without_deadlock(self):
        double_buffer = self.make_buffer()
        reader = self.BlockingReader({1: self.cache_block(20.0)}, fail=True)
        try:
            double_buffer.start_paper_prefetch(
                iteration=2,
                visible_block_ids=[0, 1],
                filters_global=[],
                resident_block_ids=[0],
                evicted_block_ids=[],
                block_reader=reader,
            )
            self.assertTrue(reader.started.wait(timeout=2.0))
            reader.release.set()

            self.assertFalse(double_buffer.wait_for_paper_host_enqueue(2))
            self.assertFalse(double_buffer.wait_for_prefetch(2))
            self.assertEqual(double_buffer.get_stats()["prefetch_failures"], 1)
        finally:
            reader.release.set()
            torch.cuda.synchronize()
            double_buffer.clear()

    def test_wrong_iteration_does_not_consume_async_job(self):
        double_buffer = self.make_buffer()
        reader = self.BlockingReader({1: self.cache_block(20.0)})
        try:
            double_buffer.start_paper_prefetch(
                iteration=2,
                visible_block_ids=[1],
                filters_global=[],
                resident_block_ids=[],
                evicted_block_ids=[0],
                block_reader=reader,
            )
            self.assertTrue(reader.started.wait(timeout=2.0))
            self.assertFalse(double_buffer.wait_for_prefetch(99))

            reader.release.set()
            self.assertTrue(double_buffer.wait_for_prefetch(2))
            double_buffer.swap_buffers()
            self.assertEqual(double_buffer.active_buffer.loaded_blocks, [1])
        finally:
            reader.release.set()
            torch.cuda.synchronize()
            double_buffer.clear()


if __name__ == "__main__":
    unittest.main()
