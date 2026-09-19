"""CPU-only tests for the checkpoint evaluation protocol helpers."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import eval_protocol as proto


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _make_checkpoint(root: Path, iteration=100000, patch_bytes=4096, complete=True, base_size=8192):
    cdir = root / "checkpoints" / str(iteration)
    patches = cdir / "ssd_delta" / "patches"
    patches.mkdir(parents=True)
    base = root / "base_file.bin"
    if not base.exists():
        base.write_bytes(b"\0" * base_size)
    patch = patches / "patch_000001_1_compact.bin"
    patch.write_bytes(b"\0" * patch_bytes)
    (cdir / "training_state.pth").write_bytes(b"x")
    (cdir / "ssd_delta" / "block_bounds.npy").write_bytes(b"x")
    _write_json(cdir / "ssd_delta" / "storage_index.json", {
        "block_size": 4096, "num_blocks": 2,
        "files": {"0": {"path": str(base), "role": "base", "size": base_size},
                  "1": {"path": str(patch), "role": "patch", "size": patch_bytes}},
        "index": {"0": {"file_id": 1, "offset": 0, "size": 2048, "version": 1},
                  "1": {"file_id": 1, "offset": 2048, "size": 2048, "version": 1}},
    })
    manifest = {
        "checkpoint_type": "pure_ssd_incremental", "checkpoint_iter": iteration, "iteration": iteration,
        "next_iteration": iteration + 1, "total_points": 8, "num_blocks": 2, "block_size": 4096, "param_dim": 59,
        "base_file": str(base), "storage_index": "ssd_delta/storage_index.json",
        "block_bounds": "ssd_delta/block_bounds.npy", "training_state": "training_state.pth",
        "patches_dir": "ssd_delta/patches", "patch_files": 1, "patch_bytes": patch_bytes, "active_sh_degree": 3,
    }
    if complete:
        _write_json(cdir / proto.CHECKPOINT_MANIFEST, manifest)
    return cdir, patch


def _cameras(n):
    return [{"eval_index": i, "dataset_frame_index": i * 45, "image_id": f"test/block_{i % 6}/{i:04d}.png",
             "image_name": f"{i:04d}.png"} for i in range(n)]


def _make_eval_dir(root: Path, iteration, views, rows, status="complete", cams=None, summary_views=None):
    cams = cams if cams is not None else _cameras(views)
    h = proto.camera_protocol_hash(cams)
    d = root / "evaluations" / proto.eval_dir_name(iteration, views, "20260919_000000_job1")
    d.mkdir(parents=True)
    _write_json(d / proto.SELECTED_CAMERAS_FILE, {"camera_protocol_hash": h, "cameras": cams})
    with open(d / proto.PER_CAMERA_FILE, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(proto.PER_CAMERA_FIELDS))
        w.writeheader()
        for i in range(rows):
            w.writerow({"eval_index": i, "dataset_frame_index": i * 45, "image_id": cams[i]["image_id"],
                        "image_name": cams[i]["image_name"], "mse": 0.004, "psnr": 24.0, "ssim": 0.8,
                        "lpips": "", "visible_blocks": 100})
    _write_json(d / proto.STATUS_FILE, {"status": status, "completed_views": rows, "camera_protocol_hash": h})
    if status == "complete":
        _write_json(d / proto.SUMMARY_FILE, {"num_test_cameras": summary_views if summary_views is not None else rows,
                                             "mean_psnr": 24.0, "mean_ssim": 0.8, "camera_protocol_hash": h})
    return d, h


class CameraProtocolTest(unittest.TestCase):
    def test_linspace_200_of_9066_matches_reader_formula_and_is_deterministic(self):
        a = proto.sample_frame_indices(9066, 200)
        b = proto.sample_frame_indices(9066, 200)
        ref = np.unique(np.linspace(0, 9065, num=200, dtype=int)).tolist()
        self.assertEqual(a, b)
        self.assertEqual(a, ref)
        self.assertEqual(len(a), 200)
        self.assertEqual(len(set(a)), 200)
        self.assertEqual((a[0], a[1], a[-1]), (0, 45, 9065))

    def test_quick_32_view_subset_is_a_different_protocol(self):
        full = set(proto.sample_frame_indices(9066, 200))
        quick = proto.sample_frame_indices(9066, 32)
        self.assertNotEqual(proto.camera_protocol_hash(_cameras(200)), proto.camera_protocol_hash(_cameras(32)))
        self.assertFalse(set(quick) <= full)

    def test_select_test_frames_reads_transforms_in_order(self):
        with tempfile.TemporaryDirectory() as d:
            frames = [{"file_name": f"blk/{i:04d}.png", "transform_matrix": []} for i in range(50)]
            p = Path(d) / "transforms_test.json"
            p.write_text(json.dumps({"frames": frames}))
            sel = proto.select_test_frames(p, 5)
            self.assertEqual([c["dataset_frame_index"] for c in sel], [0, 12, 24, 36, 49])
            self.assertEqual(sel[1]["image_id"], "test/blk/0012.png")
            self.assertEqual(sel[1]["eval_index"], 1)

    def test_protocol_hash_depends_only_on_index_frame_and_id(self):
        cams = _cameras(200)
        h1 = proto.camera_protocol_hash(cams)
        cams2 = [dict(c, source_image_path="/elsewhere") for c in cams]
        self.assertEqual(h1, proto.camera_protocol_hash(cams2))
        cams3 = [dict(c) for c in cams]
        cams3[7]["dataset_frame_index"] += 1
        self.assertNotEqual(h1, proto.camera_protocol_hash(cams3))

    def test_diagnostic_indices_are_uniform_and_fixed(self):
        self.assertEqual(proto.diagnostic_eval_indices(200, 10), [0, 22, 44, 66, 88, 110, 132, 154, 176, 199])
        self.assertEqual(proto.diagnostic_eval_indices(200, 10), proto.diagnostic_eval_indices(200, 10))
        self.assertEqual(proto.diagnostic_eval_indices(32, 10), [0, 3, 6, 10, 13, 17, 20, 24, 27, 31])
        self.assertEqual(proto.diagnostic_eval_indices(5, 10), [0, 1, 2, 3, 4])
        self.assertEqual(proto.diagnostic_eval_indices(200, 0), [])


class CheckpointStatusTest(unittest.TestCase):
    def test_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            cdir, _ = _make_checkpoint(Path(d))
            st = proto.checkpoint_status(cdir)
            self.assertTrue(st["complete"], st["reasons"])
            self.assertEqual(st["iteration"], 100000)

    def test_directory_without_manifest_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            cdir, _ = _make_checkpoint(Path(d), complete=False)
            st = proto.checkpoint_status(cdir)
            self.assertTrue(st["exists"])
            self.assertFalse(st["complete"])
            self.assertIn("missing", st["reasons"][0])

    def test_missing_directory(self):
        st = proto.checkpoint_status("/nonexistent/checkpoints/100000")
        self.assertFalse(st["exists"])
        self.assertFalse(st["complete"])

    def test_missing_or_truncated_patch_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            cdir, patch = _make_checkpoint(Path(d))
            patch.write_bytes(b"\0" * 10)
            st = proto.checkpoint_status(cdir)
            self.assertFalse(st["complete"])
            patch.unlink()
            st = proto.checkpoint_status(cdir)
            self.assertFalse(st["complete"])
            self.assertTrue(any("missing" in r for r in st["reasons"]))

    def test_missing_training_state_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            cdir, _ = _make_checkpoint(Path(d))
            (cdir / "training_state.pth").unlink()
            self.assertFalse(proto.checkpoint_status(cdir)["complete"])


class EvaluationStatusTest(unittest.TestCase):
    def test_complete_200_view_evaluation_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=200)
            es = proto.evaluation_status(ed, 200, h)
            self.assertTrue(es["complete"], es["reasons"])
            self.assertEqual(proto.find_evaluation_dirs(d, 100000, 200), [ed])
            self.assertEqual(proto.find_evaluation_dirs(d, 100000, 32), [])

    def test_partial_rows_are_not_complete_but_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=150, status="running")
            es = proto.evaluation_status(ed, 200, h)
            self.assertFalse(es["complete"])
            self.assertEqual(es["completed_views"], 150)
            self.assertTrue(es["resumable"])

    def test_status_complete_but_csv_short_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=150, status="complete", summary_views=200)
            es = proto.evaluation_status(ed, 200, h)
            self.assertFalse(es["complete"])
            self.assertFalse(es["resumable"])  # claims complete: leave it alone, start fresh

    def test_quick_32_view_run_never_satisfies_200_view_skip(self):
        with tempfile.TemporaryDirectory() as d:
            _make_eval_dir(Path(d), 100000, 32, rows=32, cams=_cameras(32))
            self.assertEqual(proto.find_evaluation_dirs(d, 100000, 200), [])

    def test_wrong_camera_protocol_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            cams = _cameras(200)
            cams[3]["dataset_frame_index"] = 999
            ed, _ = _make_eval_dir(Path(d), 100000, 200, rows=200, cams=cams)
            expected = proto.camera_protocol_hash(_cameras(200))
            es = proto.evaluation_status(ed, 200, expected)
            self.assertFalse(es["complete"])
            self.assertFalse(es["resumable"])

    def test_rows_with_nan_metrics_are_not_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=200)
            p = ed / proto.PER_CAMERA_FILE
            lines = p.read_text().splitlines()
            lines[5] = lines[5].replace("0.8", "nan")
            p.write_text("\n".join(lines) + "\n")
            rows = proto.read_completed_rows(p)
            self.assertEqual(len(rows), 199)
            self.assertNotIn(4, rows)
            self.assertFalse(proto.evaluation_status(ed, 200, h)["complete"])

    def test_missing_summary_or_status_is_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=200)
            (ed / proto.SUMMARY_FILE).unlink()
            self.assertFalse(proto.evaluation_status(ed, 200, h)["complete"])
        with tempfile.TemporaryDirectory() as d:
            ed, h = _make_eval_dir(Path(d), 100000, 200, rows=200)
            (ed / proto.STATUS_FILE).unlink()
            self.assertFalse(proto.evaluation_status(ed, 200, h)["complete"])


class CompactionGuardTest(unittest.TestCase):
    def test_detects_compaction_tmp_in_live_cache(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            self.assertEqual(proto.active_compaction_files([run]), [])
            live = run / "cache" / "20260919_015752"
            live.mkdir(parents=True)
            (live / "patch_000182_1.bin").write_bytes(b"x")
            self.assertEqual(proto.active_compaction_files([run]), [])
            tmp = live / ".tide_compact_000183_1789760000000000.tmp"
            tmp.write_bytes(b"x")
            self.assertEqual(proto.active_compaction_files([run, run / "missing"]), [tmp])


if __name__ == "__main__":
    unittest.main()
