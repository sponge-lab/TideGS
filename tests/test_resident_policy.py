import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from storage.pure_ssd_checkpoint import (
    build_resident_policy_config,
    load_pure_ssd_checkpoint_manifest,
    resident_policy_resume_message,
    write_pure_ssd_incremental_checkpoint,
)
from strategies.tide_engine.resident_policy import compute_topc_resident_transition


def _transition(*, balanced=False, active_first=True, **kwargs):
    return compute_topc_resident_transition(
        balanced_camera_seeds=balanced,
        enforce_next_active_coverage=active_first,
        **kwargs,
    )


class ActiveFirstResidentPolicyTest(unittest.TestCase):
    def test_fit_keeps_all_active_then_highest_recency_stale(self):
        transition = _transition(
            current_active_blocks=[1],
            next_active_blocks=[10, 11],
            current_resident_blocks=[1, 2, 3, 4],
            previous_recency_scores={2: 0.9, 3: 0.5, 4: 0.1},
            lambda_weight=0.0,
            resident_capacity_blocks=4,
        )

        self.assertEqual(transition.next_resident_blocks, [1, 2, 10, 11])
        self.assertEqual(transition.next_active_coverage, 2)
        self.assertEqual(transition.optional_selected_blocks, [1, 2])

    def test_equal_capacity_selects_exactly_active_set(self):
        transition = _transition(
            current_active_blocks=[1, 2, 3],
            next_active_blocks=[10, 11, 12],
            current_resident_blocks=[1, 2, 3],
            lambda_weight=0.3,
            resident_capacity_blocks=3,
        )

        self.assertEqual(transition.next_resident_blocks, [10, 11, 12])
        self.assertEqual(transition.optional_selected_blocks, [])

    def test_balanced_fit_keeps_all_active_before_stale_blocks(self):
        transition = _transition(
            balanced=True,
            current_active_blocks=[1, 2, 3, 4],
            next_active_blocks=[10, 11, 12],
            current_resident_blocks=[1, 2, 3, 4],
            next_camera_blocks={0: [10], 1: [11], 2: [12]},
            lambda_weight=0.3,
            resident_capacity_blocks=4,
            balanced_seed_fraction=0.25,
        )

        self.assertTrue({10, 11, 12}.issubset(transition.next_resident_blocks))
        self.assertEqual(transition.next_active_coverage, 3)
        self.assertEqual(len(transition.optional_selected_blocks), 1)
        self.assertEqual(
            transition.resident_selection_policy,
            "topc_balanced_active_first",
        )

    def test_overflow_strict_selects_only_active_blocks(self):
        transition = _transition(
            current_active_blocks=[1, 2, 3],
            next_active_blocks=[10, 11, 12, 13, 14],
            current_resident_blocks=[1, 2, 3],
            lambda_weight=0.3,
            resident_capacity_blocks=3,
        )

        self.assertEqual(len(transition.next_resident_blocks), 3)
        self.assertTrue(set(transition.next_resident_blocks).issubset({10, 11, 12, 13, 14}))
        self.assertEqual(transition.optional_selected_blocks, [])
        self.assertEqual(transition.resident_selection_policy, "topc_strict_active_first")

    def test_overflow_balanced_seeds_cover_cameras(self):
        camera_blocks = {
            0: [10, 11],
            1: [12, 13],
            2: [14, 15],
            3: [16, 17],
        }
        transition = _transition(
            balanced=True,
            current_active_blocks=[1, 2, 3, 4],
            next_active_blocks=list(range(10, 18)),
            current_resident_blocks=[1, 2, 3, 4],
            next_camera_blocks=camera_blocks,
            lambda_weight=0.3,
            resident_capacity_blocks=4,
            balanced_seed_fraction=0.25,
        )

        self.assertEqual(transition.next_camera_coverage, 4)
        self.assertEqual(len(transition.camera_seed_blocks), 4)
        self.assertEqual(transition.optional_selected_blocks, [])
        self.assertEqual(
            transition.resident_selection_policy,
            "topc_balanced_active_first",
        )

    def test_lambda_does_not_weaken_active_first_constraint(self):
        for lambda_weight in (0.0, 0.3, 1.0):
            with self.subTest(lambda_weight=lambda_weight):
                transition = _transition(
                    current_active_blocks=[1, 2, 3, 4],
                    next_active_blocks=[10, 11, 12],
                    current_resident_blocks=[1, 2, 3, 4],
                    lambda_weight=lambda_weight,
                    resident_capacity_blocks=4,
                )
                self.assertTrue({10, 11, 12}.issubset(transition.next_resident_blocks))

    def test_empty_duplicate_and_zero_capacity_inputs(self):
        duplicate_transition = _transition(
            current_active_blocks=[1, 1, 2],
            next_active_blocks=[10, 10, 11],
            current_resident_blocks=[1, 1, 2],
            lambda_weight=0.3,
            resident_capacity_blocks=2,
        )
        self.assertEqual(duplicate_transition.next_resident_blocks, [10, 11])

        empty_transition = _transition(
            current_active_blocks=[],
            next_active_blocks=[],
            current_resident_blocks=[],
            lambda_weight=0.3,
            resident_capacity_blocks=0,
        )
        self.assertEqual(empty_transition.next_resident_blocks, [])
        self.assertEqual(empty_transition.next_active_coverage, 0)

    def test_tie_break_is_deterministic(self):
        kwargs = dict(
            current_active_blocks=[4, 3, 2, 1],
            next_active_blocks=[14, 13, 12, 11, 10],
            current_resident_blocks=[4, 3, 2, 1],
            lambda_weight=1.0,
            resident_capacity_blocks=3,
        )
        first = _transition(**kwargs)
        second = _transition(**kwargs)
        self.assertEqual(first.next_resident_blocks, [10, 11, 12])
        self.assertEqual(first.next_resident_blocks, second.next_resident_blocks)

    def test_transition_sets_are_consistent(self):
        transition = _transition(
            current_active_blocks=[1, 2, 3],
            next_active_blocks=[3, 4, 5],
            current_resident_blocks=[1, 2, 3],
            lambda_weight=0.3,
            resident_capacity_blocks=3,
        )
        keep = set(transition.keep_resident_blocks)
        stream_in = set(transition.stream_in_blocks)
        evict = set(transition.evict_blocks)
        self.assertFalse(keep & stream_in)
        self.assertFalse(keep & evict)
        self.assertFalse(stream_in & evict)
        self.assertEqual(keep | stream_in, set(transition.next_resident_blocks))
        self.assertEqual(keep | evict, set(transition.current_resident_blocks))

    def test_legacy_policy_can_retain_stale_blocks(self):
        transition = _transition(
            active_first=False,
            balanced=True,
            current_active_blocks=[1, 2, 3, 4],
            next_active_blocks=[10, 11, 12],
            current_resident_blocks=[1, 2, 3, 4],
            lambda_weight=0.3,
            resident_capacity_blocks=4,
            balanced_seed_fraction=0.25,
        )

        self.assertLess(transition.next_active_coverage, 3)
        self.assertGreater(len(transition.optional_selected_blocks), 0)
        self.assertEqual(transition.resident_selection_policy, "topc_balanced")


class ResidentPolicyCheckpointMetadataTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            paper_resident_selection_policy="topc_balanced_active_first",
            paper_resident_capacity_blocks=2048,
            paper_resident_lambda=0.3,
            paper_resident_recency_decay=0.95,
            paper_balanced_seed_fraction=0.25,
        )

    def test_new_checkpoint_metadata_matches(self):
        config = build_resident_policy_config(self.args)
        message = resident_policy_resume_message({"resident_policy": config}, self.args)
        self.assertIn("checkpoint/current match", message)

    def test_legacy_checkpoint_without_metadata_is_allowed(self):
        message = resident_policy_resume_message({}, self.args)
        self.assertIn("has no resident-policy metadata", message)
        self.assertIn("topc_balanced_active_first", message)

    def test_policy_override_is_explicit(self):
        saved = build_resident_policy_config(self.args)
        saved["selection_policy"] = "topc_balanced"
        message = resident_policy_resume_message({"resident_policy": saved}, self.args)
        self.assertIn("checkpoint/current override", message)
        self.assertIn("changed=selection_policy", message)

    def test_incremental_checkpoint_persists_policy_metadata(self):
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            base_file = root / "base_file.bin"
            base_file.write_bytes(np.zeros((2, 59), dtype=np.float32).tobytes())

            class Storage:
                point_dim = 59

                def maybe_compact(self, min_patches, force):
                    return False

                def export_index_manifest(self, manifest_path, patches_dir, patch_file_mode):
                    patches_dir.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "block_size": 2,
                        "num_blocks": 1,
                        "point_dim": 59,
                        "files": {
                            "0": {
                                "path": str(base_file),
                                "role": "base",
                                "copied": False,
                                "linked": False,
                                "size": base_file.stat().st_size,
                            }
                        },
                        "index": {
                            "0": {"file_id": 0, "offset": 0, "size": 472, "version": 0}
                        },
                        "patch_file_mode": patch_file_mode,
                        "patch_files": 0,
                        "patch_bytes": 0,
                        "copied_patch_files": 0,
                        "copied_patch_bytes": 0,
                        "linked_patch_files": 0,
                        "linked_patch_bytes": 0,
                    }
                    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
                    return payload

            storage_adapter = SimpleNamespace(
                storage=Storage(),
                cache=None,
                num_points=2,
                block_size=2,
                num_blocks=1,
                block_bounds=np.zeros((1, 6), dtype=np.float32),
                scene_min=np.zeros(3, dtype=np.float32),
                scene_max=np.ones(3, dtype=np.float32),
                streaming_init_manifest={},
            )
            gaussians = SimpleNamespace(_unified_params=None, active_sh_degree=0)
            args = SimpleNamespace(
                **vars(self.args),
                pure_ssd_checkpoint_patch_mode="hardlink",
                pure_ssd_checkpoint_mode="incremental",
                pure_ssd_checkpoint_keep_last=1,
            )
            checkpoint_dir = root / "checkpoint"

            write_pure_ssd_incremental_checkpoint(
                storage_adapter=storage_adapter,
                gaussians=gaussians,
                checkpoint_dir=checkpoint_dir,
                iteration=16,
                next_iteration=17,
                args=args,
            )

            manifest = load_pure_ssd_checkpoint_manifest(checkpoint_dir)
            self.assertEqual(
                manifest["resident_policy"],
                build_resident_policy_config(self.args),
            )
            self.assertIn(
                "checkpoint/current match",
                resident_policy_resume_message(manifest, self.args),
            )


if __name__ == "__main__":
    unittest.main()
