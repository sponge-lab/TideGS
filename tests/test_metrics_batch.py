import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tools.summarize_pure_ssd_pipeline import update_metrics_batch_peak_summary


def load_runtime_module():
    root = Path(__file__).resolve().parents[1]
    if "torch" not in sys.modules:
        class _FakeTensor:
            def __init__(self, shape, value=0.0):
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

            def element_size(self):
                return 4

        sys.modules["torch"] = types.SimpleNamespace(
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
    storage_pkg = types.ModuleType("storage")
    storage_pkg.__path__ = []
    schedule_mod = types.ModuleType("storage.schedule_utils")
    schedule_mod.get_current_and_next_camera_batches = lambda *args, **kwargs: None
    sys.modules.setdefault("storage", storage_pkg)
    sys.modules.setdefault("storage.schedule_utils", schedule_mod)

    strategies_pkg = types.ModuleType("strategies")
    strategies_pkg.__path__ = [str(root / "strategies")]
    tide_pkg = types.ModuleType("strategies.tide_engine")
    tide_pkg.__path__ = [str(root / "strategies" / "tide_engine")]
    sys.modules.setdefault("strategies", strategies_pkg)
    sys.modules.setdefault("strategies.tide_engine", tide_pkg)

    module_name = "strategies.tide_engine.runtime"
    spec = importlib.util.spec_from_file_location(
        module_name,
        root / "strategies" / "tide_engine" / "runtime.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runtime = load_runtime_module()


class FakeCache:
    def __init__(self, stats):
        self.stats = dict(stats)

    def get_stats(self):
        return dict(self.stats)


class FakeStorageAdapter:
    def __init__(self, stats):
        self.cache = FakeCache(stats)


def make_stats(**overrides):
    stats = {
        "cache_size": 10,
        "dirty_blocks": 3,
        "ram_usage_mb": 128.0,
        "future_prefetch_blocks": 0,
        "urgent_prefetch_blocks": 0,
        "ssd_bytes_read_future": 0,
        "ssd_bytes_read_urgent": 0,
        "future_storage_read_calls": 0,
        "future_storage_read_blocks": 0,
        "future_storage_read_time": 0.0,
        "urgent_storage_read_calls": 0,
        "urgent_storage_read_blocks": 0,
        "urgent_storage_read_time": 0.0,
        "future_prefetch_reserved": 0,
        "inflight_wait_blocks": 0,
        "inflight_fallback_blocks": 0,
        "inflight_wait_time": 0.0,
        "bytes_per_block": 1024 * 1024,
    }
    stats.update(overrides)
    return stats


def make_perf_times():
    return {
        "iter_start": 0.0,
        "stage1_setup_done": 0.001,
        "stage1_5_ssd_done": 0.003,
        "n1_prefetch_start": 0.004,
        "n1_prefetch_done": 0.006,
        "stage2_3_culling_done": 0.010,
        "stage4_train_done": 0.020,
        "stage5_optim_start": 0.021,
        "stage5_optim_done": 0.025,
        "stage5_writeback_done": 0.030,
        "iter_end": 0.040,
    }


class FakeArgs:
    gaussian_block_size = 4


class FakePaperCache:
    def __init__(self):
        self.prefetch_calls = []

    def prefetch(self, block_ids):
        self.prefetch_calls.append(list(block_ids))
        return {int(block_id): f"prefetched-{int(block_id)}" for block_id in block_ids}


class FakePaperStorageAdapter:
    def __init__(self):
        self.cache = FakePaperCache()
        self.synced_blocks = None

    def sync_cache_from_cpu_views(self, updated_blocks_dict):
        self.synced_blocks = dict(updated_blocks_dict)
        return len(updated_blocks_dict)


class FakeDoubleBuffer:
    def __init__(self, gpu_refreshed):
        self.gpu_refreshed = set(int(block_id) for block_id in gpu_refreshed)
        self.gpu_refresh_calls = []
        self.cpu_refresh_calls = []

    def refresh_blocks_from_active_buffer(self, block_ids, target="loading"):
        self.gpu_refresh_calls.append((list(block_ids), target))
        return set(block_id for block_id in block_ids if int(block_id) in self.gpu_refreshed)

    def refresh_blocks_from_block_cache(self, block_cache, block_ids, target="loading"):
        self.cpu_refresh_calls.append((dict(block_cache), list(block_ids), target))
        return sum(1 for block_id in block_ids if int(block_id) in block_cache)


def apply_writeback_with_fake_double_buffer(double_buffer, updated_blocks, omega_blocks):
    storage = FakePaperStorageAdapter()

    def build_updated_blocks_dict_from_gpu_fn(**kwargs):
        return dict(updated_blocks)

    def unexpected_sync_from_gpu_views(**kwargs):
        raise AssertionError("CPU-view sync should not be used in direct GPU path")

    def unexpected_materialize_cpu_views(**kwargs):
        raise AssertionError("CPU-view materialization should not be used in direct GPU path")

    staged, refreshed, candidates = runtime.apply_paper_writeback_payload(
        storage_adapter=storage,
        args=FakeArgs(),
        payload_iteration=1,
        current_iteration=1,
        updated_block_ids=list(updated_blocks.keys()),
        omega_blocks=list(omega_blocks),
        total_n_gaussians=64,
        original_xyz=None,
        original_scaling=None,
        original_rotation=None,
        original_opacity=None,
        original_features_dc=None,
        original_features_rest=None,
        gpu_working_set_manager=None,
        build_updated_blocks_dict_from_gpu_fn=build_updated_blocks_dict_from_gpu_fn,
        sync_updated_blocks_from_gpu_views_to_cpu_fn=unexpected_sync_from_gpu_views,
        materialize_updated_blocks_from_cpu_views_fn=unexpected_materialize_cpu_views,
        get_double_buffer_gpu_fn=lambda **kwargs: double_buffer,
    )
    return storage, staged, refreshed, candidates


def read_batch_rows(model_path):
    with open(Path(model_path) / "metrics_batch.tsv", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class MetricsBatchTest(unittest.TestCase):
    def test_writeback_refreshes_updated_omega_from_gpu_first(self):
        double_buffer = FakeDoubleBuffer(gpu_refreshed={2, 3})
        storage, staged, refreshed, candidates = apply_writeback_with_fake_double_buffer(
            double_buffer=double_buffer,
            updated_blocks={1: "block-1", 2: "block-2", 3: "block-3"},
            omega_blocks=[2, 3, 4],
        )

        self.assertEqual(staged, 3)
        self.assertEqual(refreshed, 2)
        self.assertEqual(candidates, 3)
        self.assertEqual(storage.synced_blocks, {1: "block-1", 2: "block-2", 3: "block-3"})
        self.assertEqual(double_buffer.gpu_refresh_calls, [([2, 3], "loading")])
        self.assertEqual(double_buffer.cpu_refresh_calls, [])
        self.assertEqual(storage.cache.prefetch_calls, [])

    def test_writeback_refresh_falls_back_to_cpu_for_gpu_misses_only(self):
        double_buffer = FakeDoubleBuffer(gpu_refreshed={2})
        storage, staged, refreshed, candidates = apply_writeback_with_fake_double_buffer(
            double_buffer=double_buffer,
            updated_blocks={2: "block-2", 3: "block-3", 5: "block-5"},
            omega_blocks=[2, 3, 4],
        )

        self.assertEqual(staged, 3)
        self.assertEqual(refreshed, 2)
        self.assertEqual(candidates, 3)
        self.assertEqual(double_buffer.gpu_refresh_calls, [([2, 3], "loading")])
        self.assertEqual(len(double_buffer.cpu_refresh_calls), 1)
        block_cache, block_ids, target = double_buffer.cpu_refresh_calls[0]
        self.assertEqual(block_cache, {3: "block-3"})
        self.assertEqual(block_ids, [3])
        self.assertEqual(target, "loading")
        self.assertEqual(storage.cache.prefetch_calls, [])

    def test_writeback_does_not_refresh_unchanged_omega(self):
        double_buffer = FakeDoubleBuffer(gpu_refreshed=set())
        storage, staged, refreshed, candidates = apply_writeback_with_fake_double_buffer(
            double_buffer=double_buffer,
            updated_blocks={2: "block-2", 3: "block-3"},
            omega_blocks=[4, 5],
        )

        self.assertEqual(staged, 2)
        self.assertEqual(refreshed, 0)
        self.assertEqual(candidates, 2)
        self.assertEqual(double_buffer.gpu_refresh_calls, [])
        self.assertEqual(double_buffer.cpu_refresh_calls, [])
        self.assertEqual(storage.cache.prefetch_calls, [])

    def test_per_batch_deltas_and_rt_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime.write_paper_metrics_batch(
                iteration=1,
                perf_times=make_perf_times(),
                storage_adapter=FakeStorageAdapter(make_stats()),
                model_path=tmp,
                batch_size=16,
                paper_stage_metrics={
                    "resident_capacity": 8,
                    "r_t_size": 6,
                    "r_t_next_size": 7,
                    "k_t_size": 9,
                    "k_t_next_size": 10,
                    "delta_plus": 4,
                    "delta_minus": 3,
                    "hint_requested": 4,
                    "hint_submitted": 4,
                },
            )
            runtime.write_paper_metrics_batch(
                iteration=17,
                perf_times=make_perf_times(),
                storage_adapter=FakeStorageAdapter(make_stats(
                    future_prefetch_blocks=4,
                    urgent_prefetch_blocks=2,
                    ssd_bytes_read_future=8 * 1024 * 1024,
                    ssd_bytes_read_urgent=3 * 1024 * 1024,
                    future_storage_read_calls=1,
                    future_storage_read_blocks=4,
                    future_storage_read_time=0.004,
                    urgent_storage_read_calls=1,
                    urgent_storage_read_blocks=2,
                    urgent_storage_read_time=0.002,
                    future_prefetch_reserved=4,
                    inflight_wait_blocks=3,
                    inflight_fallback_blocks=1,
                    inflight_wait_time=0.012,
                    cache_size=14,
                    dirty_blocks=5,
                    ram_usage_mb=256.0,
                )),
                model_path=tmp,
                batch_size=16,
                paper_stage_metrics={
                    "resident_capacity": 8,
                    "r_t_size": 7,
                    "r_t_next_size": 8,
                    "k_t_size": 11,
                    "k_t_next_size": 12,
                    "delta_plus": 4,
                    "delta_minus": 2,
                    "hint_requested": 4,
                    "hint_submitted": 4,
                },
            )

            rows = read_batch_rows(tmp)
            self.assertEqual(rows[0]["reset"], "1")
            self.assertEqual(rows[0]["future_read_blocks_delta"], "0.000000")
            self.assertEqual(rows[1]["reset"], "0")
            self.assertEqual(rows[1]["batch_idx"], "1")
            self.assertEqual(rows[1]["future_read_blocks_delta"], "4.000000")
            self.assertEqual(rows[1]["future_read_mb_delta"], "8.000000")
            self.assertEqual(rows[1]["urgent_read_blocks_delta"], "2.000000")
            self.assertEqual(rows[1]["urgent_read_mb_delta"], "3.000000")
            self.assertEqual(rows[1]["future_storage_read_calls_delta"], "1.000000")
            self.assertEqual(rows[1]["future_storage_read_blocks_delta"], "4.000000")
            self.assertEqual(rows[1]["future_storage_read_time_ms_delta"], "4.000000")
            self.assertEqual(rows[1]["future_storage_read_bw_mb_s"], "2000.000000")
            self.assertEqual(rows[1]["urgent_storage_read_calls_delta"], "1.000000")
            self.assertEqual(rows[1]["urgent_storage_read_blocks_delta"], "2.000000")
            self.assertEqual(rows[1]["urgent_storage_read_time_ms_delta"], "2.000000")
            self.assertEqual(rows[1]["urgent_storage_read_bw_mb_s"], "1500.000000")
            self.assertEqual(rows[1]["future_reserved_delta"], "4.000000")
            self.assertEqual(rows[1]["inflight_wait_blocks_delta"], "3.000000")
            self.assertEqual(rows[1]["inflight_fallback_blocks_delta"], "1.000000")
            self.assertEqual(rows[1]["inflight_wait_time_ms_delta"], "12.000000")
            self.assertEqual(rows[1]["future_read_blocks_vs_rt"], "0.500000")
            self.assertEqual(rows[1]["future_read_blocks_vs_cap"], "0.500000")
            self.assertEqual(rows[1]["resident_capacity"], "8")
            self.assertEqual(rows[1]["bytes_per_block"], "1048576")
            self.assertEqual(rows[1]["cap_read_mb"], "8.000000")
            self.assertAlmostEqual(float(rows[1]["cap_transfer_ms_at_3p3gibs"]), 2.367424, places=6)
            self.assertAlmostEqual(float(rows[1]["cap_transfer_share_at_3p3gibs"]), 0.0591856, places=6)
            self.assertEqual(rows[1]["delta_plus"], "4")

    def test_counter_reset_never_writes_negative_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime.write_paper_metrics_batch(
                iteration=81,
                perf_times=make_perf_times(),
                storage_adapter=FakeStorageAdapter(make_stats(
                    future_prefetch_blocks=10,
                    ssd_bytes_read_future=20 * 1024 * 1024,
                )),
                model_path=tmp,
                batch_size=16,
                paper_stage_metrics={"resident_capacity": 8},
            )
            runtime.write_paper_metrics_batch(
                iteration=1,
                perf_times=make_perf_times(),
                storage_adapter=FakeStorageAdapter(make_stats(
                    future_prefetch_blocks=2,
                    ssd_bytes_read_future=4 * 1024 * 1024,
                )),
                model_path=tmp,
                batch_size=16,
                paper_stage_metrics={"resident_capacity": 8},
            )

            rows = read_batch_rows(tmp)
            self.assertEqual(rows[1]["reset"], "1")
            self.assertEqual(rows[1]["future_read_blocks_delta"], "0.000000")
            self.assertEqual(rows[1]["future_read_mb_delta"], "0.000000")

    def test_missing_stage_metrics_are_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime.write_paper_metrics_batch(
                iteration=1,
                perf_times=make_perf_times(),
                storage_adapter=FakeStorageAdapter(make_stats(future_prefetch_blocks=1)),
                model_path=tmp,
                batch_size=16,
                paper_stage_metrics={},
            )

            row = read_batch_rows(tmp)[0]
            self.assertEqual(row["resident_capacity"], "0")
            self.assertEqual(row["r_t_size"], "0")
            self.assertEqual(row["delta_plus"], "0")
            self.assertEqual(row["hint_requested"], "0")
            self.assertEqual(row["future_read_blocks_vs_rt"], "")

    def test_summary_extracts_batch_peak(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_batch.tsv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "batch_idx",
                        "reset",
                        "future_read_blocks_delta",
                        "future_read_mb_delta",
                        "future_read_blocks_vs_rt",
                        "future_read_blocks_vs_cap",
                        "delta_plus",
                        "future_storage_read_time_ms_delta",
                        "total_ms",
                        "cap_transfer_share_at_3p3gibs",
                    ],
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow({
                    "batch_idx": 0,
                    "reset": 1,
                    "future_read_blocks_delta": 0,
                    "future_read_mb_delta": 0,
                    "future_read_blocks_vs_rt": 0,
                    "future_read_blocks_vs_cap": 0,
                    "delta_plus": 4,
                    "future_storage_read_time_ms_delta": 0,
                    "total_ms": 10,
                    "cap_transfer_share_at_3p3gibs": 0,
                })
                writer.writerow({
                    "batch_idx": 3,
                    "reset": 0,
                    "future_read_blocks_delta": 7,
                    "future_read_mb_delta": 6.5,
                    "future_read_blocks_vs_rt": 0.875,
                    "future_read_blocks_vs_cap": 0.875,
                    "delta_plus": 9,
                    "future_storage_read_time_ms_delta": 2,
                    "total_ms": 20,
                    "cap_transfer_share_at_3p3gibs": 0.1,
                })
                writer.writerow({
                    "batch_idx": 4,
                    "reset": 0,
                    "future_read_blocks_delta": 5,
                    "future_read_mb_delta": 4.0,
                    "future_read_blocks_vs_rt": 0.625,
                    "future_read_blocks_vs_cap": 0.625,
                    "delta_plus": 6,
                    "future_storage_read_time_ms_delta": 1,
                    "total_ms": 20,
                    "cap_transfer_share_at_3p3gibs": 0.05,
                })

            summary = {}
            update_metrics_batch_peak_summary(Path(tmp), summary)

            self.assertEqual(summary["peak_future_read_blocks"], 7.0)
            self.assertEqual(summary["peak_future_read_mb"], 6.5)
            self.assertEqual(summary["peak_future_read_blocks_vs_rt"], 0.875)
            self.assertEqual(summary["peak_future_read_blocks_vs_cap"], 0.875)
            self.assertEqual(summary["peak_future_read_batch_idx"], 3)
            self.assertEqual(summary["peak_delta_plus"], 9.0)
            self.assertEqual(summary["peak_cap_transfer_share_at_3p3gibs"], 0.1)
            self.assertAlmostEqual(summary["avg_future_storage_read_time_ms"], 1.5)
            self.assertAlmostEqual(summary["avg_future_storage_read_share_pct"], 7.5)
            self.assertAlmostEqual(summary["avg_future_storage_read_bw_mb_s"], 3500.0)
            self.assertEqual(summary["peak_future_storage_read_time_ms"], 2.0)


if __name__ == "__main__":
    unittest.main()
