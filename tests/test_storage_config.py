import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from plyfile import PlyData, PlyElement

from storage.config import StorageConfig, get_config_for_scene_size
from storage.streaming_ply_init import streaming_ply_to_ssd_base
from storage.tide_storage_adapter import TideStorageAdapter


class _ModelWithoutArgs:
    def __init__(self, ply_path, resume_manifest=None):
        self._streaming_init_ply_path = str(ply_path)
        self._streaming_ply_init_pending = resume_manifest is None
        self._pure_ssd_resume_manifest = resume_manifest
        self._pure_ssd_resume_pending = resume_manifest is not None
        self.manifest = None

    @property
    def args(self):
        raise AssertionError("The storage adapter must not read model.args")

    def initialize_from_streaming_ssd_manifest(self, manifest):
        self.manifest = manifest

    def initialize_from_pure_ssd_checkpoint_manifest(self, manifest):
        self.manifest = manifest


class StorageConfigTest(unittest.TestCase):
    def setUp(self):
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.root = Path(tempdir.name)
        self.ply_path = self.root / "input.ply"
        vertices = np.zeros(48, dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        generator = np.random.default_rng(42)
        for coordinate in ("x", "y", "z"):
            vertices[coordinate] = generator.uniform(-2.0, 2.0, len(vertices))
        for color in ("red", "green", "blue"):
            vertices[color] = generator.integers(0, 256, len(vertices))
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            text=False,
            byte_order="<",
        ).write(str(self.ply_path))
        pipeline_patch = mock.patch("storage.tide_storage_adapter.AsyncPipeline")
        pipeline_patch.start()
        self.addCleanup(pipeline_patch.stop)

    def make_adapter(self, config, resume_manifest=None):
        model = _ModelWithoutArgs(self.ply_path, resume_manifest)
        adapter = TideStorageAdapter(gaussians=model, cameras=[], config=config)
        self.addCleanup(adapter.shutdown)
        return adapter

    def test_streaming_config_preserves_base_and_bounds_without_model_args(self):
        for bucket_bits, sort_memory_mb in ((10, 512.0), (4, 0.001)):
            with self.subTest(bucket_bits=bucket_bits, sort_memory_mb=sort_memory_mb):
                reference = streaming_ply_to_ssd_base(
                    ply_path=self.ply_path,
                    output_dir=self.root / f"reference_{bucket_bits}",
                    block_size=8,
                    debug_fast_init_scales=True,
                    bucket_bits=bucket_bits,
                    max_sort_memory_mb=sort_memory_mb,
                )
                config = StorageConfig(
                    ssd_cache_dir=str(self.root / f"cache_{bucket_bits}"),
                    block_size=8,
                    max_ram_gb=0.02,
                    num_camera_clusters=7,
                    skip_camera_clustering=True,
                    use_6plane=False,
                    max_patch_files=7,
                    max_stale_patch_gb=0.125,
                    max_patch_total_gb=0.25,
                    min_free_gb=0,
                    debug_logging=True,
                    schedule_cache_enabled=False,
                    schedule_cache_dir=str(self.root / "schedules"),
                    fast_init_scales=True,
                    bucket_bits=bucket_bits,
                    sort_memory_mb=sort_memory_mb,
                )
                with mock.patch(
                    "storage.tide_storage_adapter.streaming_ply_to_ssd_base",
                    wraps=streaming_ply_to_ssd_base,
                ) as initialize:
                    adapter = self.make_adapter(config)
                self.assertEqual(initialize.call_args.kwargs["bucket_bits"], bucket_bits)
                self.assertEqual(initialize.call_args.kwargs["max_sort_memory_mb"], sort_memory_mb)
                actual = adapter.streaming_init_manifest
                self.assertEqual(
                    Path(actual["base_file"]).read_bytes(),
                    Path(reference["base_file"]).read_bytes(),
                )
                np.testing.assert_array_equal(
                    np.load(actual["block_bounds"]),
                    np.load(reference["block_bounds"]),
                )
                self.assertEqual(adapter.gaussians.manifest, actual)
                self.assertEqual(adapter.cache.max_ram_bytes, int(0.02 * 1024**3))
                self.assertEqual(adapter.num_clusters, 7)
                self.assertFalse(adapter.use_6plane)
                self.assertTrue(adapter.paper_debug_logging)
                self.assertFalse(adapter.schedule_cache_enabled)
                self.assertEqual(adapter.schedule_cache_dir, self.root / "schedules")
                self.assertEqual(adapter.storage.max_patch_files, 7)
                self.assertEqual(adapter.storage.max_stale_patch_bytes, int(0.125 * 1024**3))
                self.assertEqual(adapter.storage.max_patch_total_bytes, int(0.25 * 1024**3))
                self.assertEqual(adapter.storage.min_free_bytes, 0)

    def test_unspecified_sizes_keep_scene_presets(self):
        config = StorageConfig(
            ssd_cache_dir=str(self.root / "auto_cache"),
            block_size=None,
            max_ram_gb=None,
            num_camera_clusters=None,
            skip_camera_clustering=True,
            fast_init_scales=True,
            min_free_gb=0,
        )
        adapter = self.make_adapter(config)
        defaults = get_config_for_scene_size(48)
        self.assertEqual(adapter.block_size, defaults.block_size)
        self.assertEqual(adapter.max_ram_gb, defaults.max_ram_gb)
        self.assertEqual(adapter.num_clusters, defaults.num_camera_clusters)
        self.assertTrue(adapter.schedule_cache_enabled)
        self.assertEqual(
            adapter.schedule_cache_dir,
            Path(config.ssd_cache_dir) / "camera_schedule_cache",
        )

    def test_resume_and_prebuilt_keep_manifest_block_size(self):
        manifest = streaming_ply_to_ssd_base(
            ply_path=self.ply_path,
            output_dir=self.root / "saved_base",
            block_size=8,
            debug_fast_init_scales=True,
        )
        for prebuilt in (False, True):
            with self.subTest(prebuilt=prebuilt):
                config = StorageConfig(
                    ssd_cache_dir=str(self.root / f"resume_{prebuilt}"),
                    block_size=32,
                    max_ram_gb=0.02,
                    num_camera_clusters=3,
                    skip_camera_clustering=True,
                    min_free_gb=0,
                )
                resume_manifest = dict(manifest, _prebuilt_base_reuse=prebuilt)
                adapter = self.make_adapter(config, resume_manifest)
                self.assertEqual(adapter.block_size, 8)
                self.assertEqual(adapter.storage.block_size, 8)
                self.assertEqual(adapter.cache.block_size, 8)
                self.assertEqual(config.block_size, 32)
                self.assertEqual(adapter.max_ram_gb, 0.02)
                self.assertEqual(adapter.num_clusters, 3)
                self.assertEqual(
                    (adapter.storage_dir / "base_file.bin").resolve(),
                    Path(manifest["base_file"]).resolve(),
                )
                self.assertEqual(adapter.gaussians.manifest["block_size"], 8)
                expected = np.fromfile(manifest["base_file"], dtype=np.float32).reshape(48, 59)
                np.testing.assert_array_equal(adapter.storage.read_blocks([0])[0].numpy(), expected[:8])

    def test_fast_scale_requirement_remains_explicit(self):
        config = StorageConfig(
            ssd_cache_dir=str(self.root / "invalid_cache"),
            skip_camera_clustering=True,
            init_scale_mode="morton_bucket_density_clamped",
        )
        with self.assertRaisesRegex(RuntimeError, "debug_fast_init_scales"):
            self.make_adapter(config)

    def test_knn3_scale_matches_three_nearest_neighbours(self):
        manifest = streaming_ply_to_ssd_base(
            ply_path=self.ply_path,
            output_dir=self.root / "knn3_base",
            block_size=8,
            debug_fast_init_scales=False,
            scale_mode="knn3",
        )
        self.assertEqual(manifest["scale_mode"], "knn3")
        self.assertEqual(manifest["knn"]["k"], 3)
        rows = np.fromfile(manifest["base_file"], dtype=np.float32).reshape(48, 59)
        xyz = rows[:, :3].astype(np.float32)
        # Brute-force reference: 3 nearest neighbours inside each point's level-0
        # Morton bucket (the unit the streaming initializer sorts in memory).
        from storage.streaming_ply_init import _bucket_for_codes, _morton_codes_np

        scene_min = np.asarray(manifest["scene_min"], dtype=np.float32)
        scene_max = np.asarray(manifest["scene_max"], dtype=np.float32)
        buckets = _bucket_for_codes(_morton_codes_np(xyz, scene_min, scene_max), bucket_bits=10, level=0)
        expected = np.empty(48, dtype=np.float64)
        for i in range(48):
            same = np.nonzero(buckets == buckets[i])[0]
            same = same[same != i]
            if same.size == 0:
                expected[i] = 0.5 * np.log(1e-7)
                continue
            dist2 = np.sort(((xyz[same].astype(np.float64) - xyz[i]) ** 2).sum(-1))[:3]
            expected[i] = 0.5 * np.log(max(dist2.mean(), 1e-7))
        np.testing.assert_allclose(rows[:, 3], expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(rows[:, 3], rows[:, 4])
        np.testing.assert_array_equal(rows[:, 3], rows[:, 5])


if __name__ == "__main__":
    unittest.main()
