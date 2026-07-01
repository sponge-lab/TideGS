import unittest

from tools.summarize_pure_ssd_pipeline import derive_metrics_window_rows


class MetricsWindowTest(unittest.TestCase):
    def test_cumulative_snapshot_diffs_into_window_metrics(self):
        snapshot_rows = [
            {
                "sample_idx": "0",
                "batch_idx": "0",
                "iteration": "1",
                "iter_end": "17",
                "bsz": "16",
                "cache_hits": "100",
                "cache_misses": "20",
                "prefetches": "40",
                "hit_rate_cumulative": "0.833333",
                "ssd_bytes_read_urgent": str(1 * 1024 * 1024),
                "ssd_bytes_read_future": str(2 * 1024 * 1024),
                "ssd_bytes_written_async": str(3 * 1024 * 1024),
                "ssd_bytes_written_sync": str(4 * 1024 * 1024),
                "urgent_blocks": "7",
                "urgent_misses": "5",
                "future_submitted": "11",
                "future_blocks": "13",
                "future_skipped": "2",
                "future_reserved": "17",
                "urgent_storage_read_calls": "1",
                "urgent_storage_read_blocks": "2",
                "urgent_storage_read_time_ms": "4.0",
                "future_storage_read_calls": "3",
                "future_storage_read_blocks": "13",
                "future_storage_read_time_ms": "30.0",
                "inflight_wait_blocks": "19",
                "inflight_fallback_blocks": "1",
                "inflight_wait_time_ms": "31.5",
                "cache_size": "10",
                "dirty_blocks": "3",
                "flushing_blocks": "1",
                "flush_q": "2",
                "future_pending": "4",
                "ram_usage_mb": "512.5",
                "setup_ms": "1.0",
                "ssd_cull_load_ms": "2.0",
                "gauss_cull_legacy_ms": "30.0",
                "n1_prefetch_ms": "12.0",
                "gauss_cull_excl_n1_ms": "18.0",
                "train_ms": "40.0",
                "optim_ms": "5.0",
                "writeback_ms": "6.0",
                "total_ms": "84.0",
            },
            {
                "sample_idx": "1",
                "batch_idx": "5",
                "iteration": "81",
                "iter_end": "97",
                "bsz": "16",
                "cache_hits": "160",
                "cache_misses": "40",
                "prefetches": "55",
                "hit_rate_cumulative": "0.800000",
                "ssd_bytes_read_urgent": str(3 * 1024 * 1024),
                "ssd_bytes_read_future": str(5 * 1024 * 1024),
                "ssd_bytes_written_async": str(8 * 1024 * 1024),
                "ssd_bytes_written_sync": str(4 * 1024 * 1024),
                "urgent_blocks": "9",
                "urgent_misses": "6",
                "future_submitted": "20",
                "future_blocks": "25",
                "future_skipped": "4",
                "future_reserved": "30",
                "urgent_storage_read_calls": "2",
                "urgent_storage_read_blocks": "3",
                "urgent_storage_read_time_ms": "6.5",
                "future_storage_read_calls": "5",
                "future_storage_read_blocks": "25",
                "future_storage_read_time_ms": "51.0",
                "inflight_wait_blocks": "22",
                "inflight_fallback_blocks": "3",
                "inflight_wait_time_ms": "44.0",
                "cache_size": "15",
                "dirty_blocks": "8",
                "flushing_blocks": "2",
                "flush_q": "5",
                "future_pending": "6",
                "ram_usage_mb": "700.0",
                "setup_ms": "1.5",
                "ssd_cull_load_ms": "2.5",
                "gauss_cull_legacy_ms": "36.0",
                "n1_prefetch_ms": "21.0",
                "gauss_cull_excl_n1_ms": "15.0",
                "train_ms": "41.0",
                "optim_ms": "5.5",
                "writeback_ms": "6.5",
                "total_ms": "92.0",
            },
        ]

        windows = derive_metrics_window_rows(snapshot_rows)

        self.assertEqual(windows[0]["delta_cache_hits"], 0.0)
        self.assertEqual(windows[0]["delta_ssd_read_urgent_mb"], 0.0)
        self.assertEqual(windows[1]["window_batches"], 5.0)
        self.assertEqual(windows[1]["delta_cache_hits"], 60.0)
        self.assertEqual(windows[1]["delta_cache_misses"], 20.0)
        self.assertAlmostEqual(windows[1]["window_hit_rate"], 0.75)
        self.assertEqual(windows[1]["delta_prefetches"], 15.0)
        self.assertEqual(windows[1]["delta_ssd_read_urgent_mb"], 2.0)
        self.assertEqual(windows[1]["delta_ssd_read_future_mb"], 3.0)
        self.assertEqual(windows[1]["delta_ssd_write_async_mb"], 5.0)
        self.assertEqual(windows[1]["delta_ssd_write_sync_mb"], 0.0)
        self.assertEqual(windows[1]["delta_future_reserved"], 13.0)
        self.assertEqual(windows[1]["delta_future_storage_read_time_ms"], 21.0)
        self.assertEqual(windows[1]["delta_urgent_storage_read_blocks"], 1.0)
        self.assertEqual(windows[1]["delta_inflight_wait_time_ms"], 12.5)
        self.assertEqual(windows[1]["cache_size"], 15.0)
        self.assertEqual(windows[1]["dirty_blocks"], 8.0)
        self.assertEqual(windows[1]["n1_prefetch_ms"], 21.0)
        self.assertEqual(windows[1]["gauss_cull_excl_n1_ms"], 15.0)

    def test_missing_new_fields_keep_old_snapshots_compatible(self):
        snapshot_rows = [
            {
                "sample_idx": "0",
                "batch_idx": "0",
                "iteration": "1",
                "iter_end": "17",
                "bsz": "16",
                "cache_hits": "10",
                "cache_misses": "0",
            },
            {
                "sample_idx": "1",
                "batch_idx": "1",
                "iteration": "17",
                "iter_end": "33",
                "bsz": "16",
                "cache_hits": "12",
                "cache_misses": "3",
            },
        ]

        windows = derive_metrics_window_rows(snapshot_rows)

        self.assertEqual(windows[1]["delta_cache_hits"], 2.0)
        self.assertEqual(windows[1]["delta_cache_misses"], 3.0)
        self.assertAlmostEqual(windows[1]["window_hit_rate"], 0.4)
        self.assertEqual(windows[1]["delta_future_reserved"], 0.0)
        self.assertEqual(windows[1]["delta_future_storage_read_time_ms"], 0.0)
        self.assertEqual(windows[1]["delta_inflight_wait_blocks"], 0.0)
        self.assertEqual(windows[1]["delta_ssd_read_future_mb"], 0.0)
        self.assertEqual(windows[1]["cache_size"], 0.0)
        self.assertEqual(windows[1]["n1_prefetch_ms"], 0.0)

    def test_restarted_snapshot_append_starts_new_window(self):
        snapshot_rows = [
            {
                "sample_idx": "0",
                "batch_idx": "0",
                "iteration": "1",
                "cache_hits": "10",
                "cache_misses": "2",
                "ssd_bytes_read_future": str(4 * 1024 * 1024),
            },
            {
                "sample_idx": "1",
                "batch_idx": "5",
                "iteration": "41",
                "cache_hits": "20",
                "cache_misses": "2",
                "ssd_bytes_read_future": str(7 * 1024 * 1024),
            },
            {
                "sample_idx": "2",
                "batch_idx": "0",
                "iteration": "1",
                "cache_hits": "9",
                "cache_misses": "1",
                "ssd_bytes_read_future": str(3 * 1024 * 1024),
            },
        ]

        windows = derive_metrics_window_rows(snapshot_rows)

        self.assertEqual(windows[1]["delta_cache_hits"], 10.0)
        self.assertEqual(windows[1]["delta_ssd_read_future_mb"], 3.0)
        self.assertEqual(windows[2]["window_batches"], 0.0)
        self.assertEqual(windows[2]["delta_cache_hits"], 0.0)
        self.assertEqual(windows[2]["delta_cache_misses"], 0.0)
        self.assertEqual(windows[2]["delta_ssd_read_future_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()
