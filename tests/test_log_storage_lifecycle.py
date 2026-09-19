import errno
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from storage.log_storage_manager import LogStorageManager, StorageCapacityError
from storage.pure_ssd_checkpoint import prune_checkpoint_history


class LogStorageLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.storage_dir = self.root / "cache"
        self.storage = LogStorageManager(
            storage_dir=str(self.storage_dir),
            block_size=2,
            num_blocks=4,
            point_dim=3,
            verbose=False,
            max_patch_files=32,
            max_stale_patch_gb=0,
            max_patch_total_gb=0,
            min_free_gb=0,
        )
        base = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        self.storage.file_paths[0].write_bytes(base.numpy().tobytes())
        self.base_bytes = self.storage.file_paths[0].read_bytes()

    def tearDown(self):
        self.storage.close()
        self.tempdir.cleanup()

    @staticmethod
    def block(value):
        return torch.full((2, 3), float(value), dtype=torch.float32)

    def test_sparse_compaction_preserves_latest_data_versions_and_base(self):
        first = self.storage.write_patch({0: self.block(10), 1: self.block(11)})
        second = self.storage.write_patch({0: self.block(20), 2: self.block(12)})
        old_paths = [self.storage.file_paths[first], self.storage.file_paths[second]]
        versions = {block_id: location.version for block_id, location in self.storage.index.items()}

        self.assertTrue(self.storage.compact_patches(force=True))

        self.assertEqual(self.storage.file_paths[0].read_bytes(), self.base_bytes)
        self.assertEqual(len(self.storage.file_paths), 2)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))
        self.assertTrue(torch.equal(self.storage.read_blocks([1])[1], self.block(11)))
        self.assertTrue(torch.equal(self.storage.read_blocks([2])[2], self.block(12)))
        self.assertEqual(
            {block_id: location.version for block_id, location in self.storage.index.items()},
            versions,
        )
        self.assertTrue(all(not path.exists() for path in old_paths))

    def test_checkpoint_hardlink_survives_runtime_patch_gc(self):
        self.storage.write_patch({0: self.block(10), 1: self.block(11)})
        self.storage.write_patch({0: self.block(20)})
        self.storage.compact_patches(force=True)
        runtime_patch = next(path for file_id, path in self.storage.file_paths.items() if file_id != 0)

        checkpoint_dir = self.root / "checkpoint"
        manifest_path = checkpoint_dir / "storage_index.json"
        manifest = self.storage.export_index_manifest(
            manifest_path=manifest_path,
            patches_dir=checkpoint_dir / "patches",
            patch_file_mode="hardlink",
        )
        checkpoint_patch = Path(next(
            info["path"] for file_id, info in manifest["files"].items() if file_id != "0"
        ))
        self.assertEqual(os.stat(runtime_patch).st_ino, os.stat(checkpoint_patch).st_ino)
        self.assertEqual(manifest["copied_patch_bytes"], 0)

        self.storage.write_patch({0: self.block(30)})
        self.storage.compact_patches(force=True)
        self.assertFalse(runtime_patch.exists())
        self.assertTrue(checkpoint_patch.exists())

        resume_dir = self.root / "resume"
        resume = LogStorageManager(
            storage_dir=str(resume_dir),
            block_size=2,
            num_blocks=4,
            point_dim=3,
            verbose=False,
            min_free_gb=0,
        )
        try:
            resume.load_index_manifest(manifest_path)
            self.assertTrue(torch.equal(resume.read_blocks([0])[0], self.block(20)))
            self.assertTrue(torch.equal(resume.read_blocks([1])[1], self.block(11)))
        finally:
            resume.close()

    def test_checkpoint_hardlink_falls_back_to_copy(self):
        self.storage.write_patch({0: self.block(10)})
        checkpoint_dir = self.root / "copy_fallback"

        with mock.patch(
            "storage.log_storage_manager.os.link",
            side_effect=OSError(errno.EXDEV, "cross-device link"),
        ):
            manifest = self.storage.export_index_manifest(
                manifest_path=checkpoint_dir / "storage_index.json",
                patches_dir=checkpoint_dir / "patches",
                patch_file_mode="hardlink",
            )

        self.assertEqual(manifest["copied_patch_files"], 1)
        self.assertEqual(manifest["linked_patch_files"], 0)
        self.assertGreater(manifest["copied_patch_bytes"], 0)

    def test_failed_atomic_switch_keeps_old_index_and_patches(self):
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        old_paths = dict(self.storage.file_paths)
        old_index = dict(self.storage.index)

        with mock.patch("storage.log_storage_manager.os.replace", side_effect=OSError("injected")):
            with self.assertRaisesRegex(OSError, "injected"):
                self.storage.compact_patches(force=True)

        self.assertEqual(self.storage.file_paths, old_paths)
        self.assertEqual(self.storage.index, old_index)
        self.assertTrue(all(path.exists() for path in old_paths.values()))
        self.assertEqual(list(self.storage_dir.glob(".tide_compact_*.tmp")), [])
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_patch_threshold_triggers_automatic_compaction(self):
        self.storage.max_patch_files = 2
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20), 1: self.block(11)})

        self.assertEqual(self.storage.get_stats()["num_patches"], 1)
        self.assertEqual(self.storage.get_stats()["compactions"], 1)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_storage_timing_breakdown_tracks_patch_writes_and_compactions(self):
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})

        stats = self.storage.get_stats()
        self.assertEqual(stats["patch_write_jobs"], 2)
        self.assertGreater(stats["patch_write_bytes"], 0)
        self.assertGreaterEqual(stats["patch_write_time"], 0.0)
        self.assertEqual(stats["compactions"], 0)
        self.assertEqual(stats["compaction_time"], 0.0)

        self.storage.compact_patches(force=True)
        stats = self.storage.get_stats()
        self.assertEqual(stats["compactions"], 1)
        self.assertGreater(stats["compaction_time"], 0.0)

    def test_total_patch_size_triggers_automatic_compaction(self):
        self.storage.max_patch_files = 32
        self.storage.max_patch_total_bytes = self.block(0).numel() * self.block(0).element_size()

        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({1: self.block(11)})

        stats = self.storage.get_stats()
        self.assertEqual(stats["compactions"], 1)
        self.assertEqual(stats["num_patches"], 1)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(10)))
        self.assertTrue(torch.equal(self.storage.read_blocks([1])[1], self.block(11)))

    def test_live_delta_size_does_not_retrigger_compaction(self):
        self.storage.max_stale_patch_bytes = 1
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        self.assertEqual(self.storage.get_stats()["compactions"], 1)

        self.storage.write_patch({1: self.block(11)})

        stats = self.storage.get_stats()
        self.assertEqual(stats["compactions"], 1)
        self.assertEqual(stats["num_patches"], 2)
        self.assertEqual(stats["stale_patch_size_mb"], 0)

    def test_low_space_defers_maintenance_without_changing_data(self):
        self.storage.write_patch({0: self.block(10)})
        self.storage.write_patch({0: self.block(20)})
        self.storage.min_free_bytes = 1024
        with mock.patch(
            "storage.log_storage_manager.shutil.disk_usage",
            return_value=mock.Mock(total=2048, used=2048, free=0),
        ):
            self.assertFalse(self.storage.maybe_compact(force=True))

        self.assertEqual(self.storage.stats["compactions_deferred"], 1)
        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))

    def test_free_space_reserve_rejects_write_before_creating_patch(self):
        self.storage.min_free_bytes = 1024
        before = set(self.storage_dir.iterdir())
        with mock.patch(
            "storage.log_storage_manager.shutil.disk_usage",
            return_value=mock.Mock(total=2048, used=2048, free=0),
        ):
            with self.assertRaises(StorageCapacityError):
                self.storage.write_patch({0: self.block(10)})
        self.assertEqual(set(self.storage_dir.iterdir()), before)

    def test_stale_versioned_write_does_not_overwrite_newer_block(self):
        self.storage.write_patch({0: self.block(20)}, block_versions={0: 2})

        self.storage.write_patch({0: self.block(10)}, block_versions={0: 1})

        self.assertTrue(torch.equal(self.storage.read_blocks([0])[0], self.block(20)))
        self.assertEqual(self.storage.index[0].version, 2)
        self.assertEqual(self.storage.get_stats()["num_patches"], 1)

    def test_checkpoint_retention_keeps_newest_numeric_directories(self):
        model_path = self.root / "model"
        for iteration in (100, 200, 300):
            path = model_path / "checkpoints" / str(iteration)
            path.mkdir(parents=True)
            (path / "manifest.json").write_text(json.dumps({"iteration": iteration}))

        removed = prune_checkpoint_history(model_path, keep_last=2)

        self.assertEqual([path.name for path in removed], ["100"])
        self.assertEqual(
            sorted(path.name for path in (model_path / "checkpoints").iterdir()),
            ["200", "300"],
        )


if __name__ == "__main__":
    unittest.main()
