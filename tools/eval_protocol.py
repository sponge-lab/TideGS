"""Shared, torch-free helpers for the TideGS checkpoint evaluation workflow.

Used by ``tools/evaluate_pure_ssd_native.py`` (the evaluator) and
``scripts/evaluate_missing_checkpoints.py`` (the wrapper) so that checkpoint
discovery, completed-evaluation detection, the fixed camera protocol and the
diagnostic-view selection are defined in exactly one place.

Everything here is read-only with respect to checkpoints and training caches.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

CHECKPOINT_MANIFEST = "pure_ssd_checkpoint.json"
DEFAULT_ITERATIONS = (100000, 200000, 300000, 400000, 500000)
FULL_VIEWS = 200
DEFAULT_DIAGNOSTIC_VIEWS = 10
PROTOCOL_VERSION = 1

SUMMARY_FILE = "metrics_summary.json"
PER_CAMERA_FILE = "metrics_per_camera.csv"
SELECTED_CAMERAS_FILE = "selected_cameras.json"
STATUS_FILE = "evaluation_status.json"
DIAGNOSTICS_DIR = "diagnostics"
DIAGNOSTIC_VIEWS_FILE = "selected_diagnostic_views.json"
CONSOLE_LOG = "console.log"

REQUIRED_ROW_METRICS = ("mse", "psnr", "ssim")
PER_CAMERA_FIELDS = (
    "eval_index", "dataset_frame_index", "image_id", "image_name",
    "mse", "psnr", "ssim", "lpips", "visible_blocks",
)


# --------------------------------------------------------------------------- #
# Camera protocol
# --------------------------------------------------------------------------- #
def sample_frame_indices(n_frames: int, n_views: int, mode: str = "linspace", start: int = 0) -> List[int]:
    """Mirror ``scene.dataset_readers._sample_frames_for_debug`` exactly.

    Returns the dataset frame indices (in ``transforms_test.json`` order) that
    the Scene loads for ``n_views`` cameras.  Deterministic: depends only on the
    frame count, the view count and the mode.
    """
    if n_views is None or n_views <= 0 or n_frames <= n_views:
        return list(range(n_frames))
    mode = str(mode).lower()
    if mode == "linspace":
        idx = np.unique(np.linspace(0, n_frames - 1, num=n_views, dtype=int))
    elif mode == "contiguous":
        idx = np.arange(0, n_views, dtype=int)
    elif mode == "window":
        s = min(int(start), max(0, n_frames - n_views))
        idx = np.arange(s, s + n_views, dtype=int)
    else:
        raise ValueError(f"unknown camera_sample_mode {mode!r}")
    return [int(i) for i in idx.tolist()]


def frame_ref(frame: dict) -> str:
    if frame.get("file_name"):
        return str(frame["file_name"])
    if frame.get("file_path"):
        return str(frame["file_path"])
    raise KeyError("frame has neither file_name nor file_path")


def select_test_frames(transforms_test_path: str | Path, n_views: int,
                       mode: str = "linspace", start: int = 0) -> List[Dict]:
    """Return the audit record of the cameras the protocol selects.

    Each entry: eval_index, dataset_frame_index, file (as in transforms_test.json),
    image_id (``test/<file>``, the decoded-cache key used by the loader).
    """
    with open(transforms_test_path, "r", encoding="utf-8") as f:
        frames = json.load(f)["frames"]
    indices = sample_frame_indices(len(frames), n_views, mode, start)
    out = []
    for eval_index, fi in enumerate(indices):
        ref = frame_ref(frames[fi]).replace("\\", "/")
        out.append({
            "eval_index": eval_index,
            "dataset_frame_index": int(fi),
            "file": ref,
            "image_id": f"test/{ref.lstrip('/')}",
            "image_name": os.path.basename(ref),
        })
    return out


def camera_protocol_hash(cameras: Sequence[Dict]) -> str:
    """Stable signature of an ordered camera list (index, frame, image_id)."""
    payload = [[int(c["eval_index"]), int(c["dataset_frame_index"]), str(c["image_id"])] for c in cameras]
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def diagnostic_eval_indices(n_views: int, n_diag: int = DEFAULT_DIAGNOSTIC_VIEWS) -> List[int]:
    """Deterministic, uniformly spread eval indices used for saved images.

    For n_views=200, n_diag=10 -> [0, 22, 44, 66, 88, 110, 132, 154, 176, 199].
    """
    if n_diag <= 0 or n_views <= 0:
        return []
    if n_diag >= n_views:
        return list(range(n_views))
    return [int(i) for i in np.unique(np.linspace(0, n_views - 1, num=n_diag, dtype=int)).tolist()]


# --------------------------------------------------------------------------- #
# Checkpoint discovery / completeness
# --------------------------------------------------------------------------- #
def _resolve(path_value: str, base: Path) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else (base / p)


def checkpoint_status(checkpoint_dir: str | Path) -> Dict:
    """Read-only completeness check of a pure SSD checkpoint directory.

    The trainer writes the manifest last (atomic replace), so a manifest means
    the writer finished; we still verify every file the manifest and storage
    index reference, so a checkpoint whose patches were removed or truncated is
    reported as incomplete instead of being handed to the evaluator.
    """
    cdir = Path(checkpoint_dir)
    reasons: List[str] = []
    result = {"path": str(cdir), "exists": cdir.is_dir(), "complete": False,
              "reasons": reasons, "iteration": None, "manifest": None}
    if not cdir.is_dir():
        reasons.append("directory missing")
        return result
    mpath = cdir / CHECKPOINT_MANIFEST
    if not mpath.is_file():
        reasons.append(f"{CHECKPOINT_MANIFEST} missing (checkpoint still being written or never completed)")
        return result
    try:
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        reasons.append(f"manifest unreadable: {exc}")
        return result
    result["manifest"] = manifest
    for key in ("total_points", "num_blocks", "block_size", "param_dim", "next_iteration"):
        if key not in manifest:
            reasons.append(f"manifest missing key {key}")
    result["iteration"] = int(manifest.get("checkpoint_iter", manifest.get("iteration", -1)))

    training_state = _resolve(manifest.get("training_state", "training_state.pth"), cdir)
    if not training_state.is_file():
        reasons.append(f"training_state missing: {training_state}")
    bounds = _resolve(manifest.get("block_bounds", "ssd_delta/block_bounds.npy"), cdir)
    if not bounds.is_file():
        reasons.append(f"block_bounds missing: {bounds}")
    base_file = manifest.get("base_file")
    if not base_file or not _resolve(base_file, cdir).is_file():
        reasons.append(f"base_file missing: {base_file}")

    ctype = str(manifest.get("checkpoint_type", "pure_ssd_snapshot"))
    if ctype == "pure_ssd_incremental":
        index_path = _resolve(manifest.get("storage_index", "ssd_delta/storage_index.json"), cdir)
        if not index_path.is_file():
            reasons.append(f"storage_index missing: {index_path}")
        else:
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
                files = index.get("files", {})
                if not files:
                    reasons.append("storage_index lists no files")
                for fid, info in files.items():
                    p = Path(info["path"])
                    if not p.is_file():
                        reasons.append(f"indexed file {fid} missing: {p}")
                    elif "size" in info and p.stat().st_size != int(info["size"]):
                        reasons.append(f"indexed file {fid} size {p.stat().st_size} != {info['size']}")
                if int(manifest.get("num_blocks", -1)) != int(index.get("num_blocks", -2)):
                    reasons.append("storage_index num_blocks != manifest num_blocks")
            except (OSError, ValueError, KeyError) as exc:
                reasons.append(f"storage_index unreadable: {exc}")
        patches_dir = _resolve(manifest.get("patches_dir", "ssd_delta/patches"), cdir)
        expected_files = int(manifest.get("patch_files", 0))
        if expected_files > 0:
            if not patches_dir.is_dir():
                reasons.append(f"patches_dir missing: {patches_dir}")
            else:
                patches = sorted(p for p in patches_dir.iterdir() if p.name.startswith("patch_") and p.suffix == ".bin")
                if len(patches) != expected_files:
                    reasons.append(f"patch file count {len(patches)} != manifest patch_files {expected_files}")
                if "patch_bytes" in manifest:
                    total = sum(p.stat().st_size for p in patches)
                    if total != int(manifest["patch_bytes"]):
                        reasons.append(f"patch bytes {total} != manifest patch_bytes {manifest['patch_bytes']}")
    else:
        snapshot_dir = _resolve(manifest.get("snapshot_dir", "ssd_snapshot"), cdir)
        if not snapshot_dir.is_dir():
            reasons.append(f"snapshot_dir missing: {snapshot_dir}")
    result["complete"] = not reasons
    return result


def find_checkpoint_dir(run_dir: str | Path, iteration: int) -> Path:
    return Path(run_dir) / "checkpoints" / str(int(iteration))


# --------------------------------------------------------------------------- #
# Evaluation outputs
# --------------------------------------------------------------------------- #
def eval_dir_name(iteration: int, views: int, unique_id: str) -> str:
    return f"iter{int(iteration)}_views{int(views)}_{unique_id}"


def find_evaluation_dirs(run_dir: str | Path, iteration: int, views: int) -> List[Path]:
    root = Path(run_dir) / "evaluations"
    if not root.is_dir():
        return []
    prefix = f"iter{int(iteration)}_views{int(views)}_"
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))


def _finite(value) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def read_completed_rows(csv_path: str | Path) -> Dict[int, Dict[str, str]]:
    """Rows of a per-camera CSV whose required metrics are all finite.

    Used for completeness checks and for resuming a partial evaluation.  A row
    is only trusted when eval_index parses and mse/psnr/ssim are finite.
    """
    rows: Dict[int, Dict[str, str]] = {}
    p = Path(csv_path)
    if not p.is_file():
        return rows
    with open(p, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["eval_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(_finite(row.get(k)) for k in REQUIRED_ROW_METRICS):
                rows[idx] = row
    return rows


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def evaluation_status(eval_dir: str | Path, expected_views: int,
                      expected_hash: Optional[str] = None) -> Dict:
    """Decide whether ``eval_dir`` holds a COMPLETE evaluation of ``expected_views`` views.

    All of the following must hold, otherwise the directory is incomplete:
      * metrics_summary.json exists and num_test_cameras == expected_views
      * evaluation_status.json says status == "complete" with completed_views == expected_views
      * metrics_per_camera.csv has exactly expected_views trusted rows with eval_index 0..N-1
      * selected_cameras.json lists expected_views cameras and its hash matches the
        summary, the status file and (when given) the expected protocol hash
    """
    d = Path(eval_dir)
    reasons: List[str] = []
    out = {"path": str(d), "complete": False, "reasons": reasons, "summary": None,
           "status": None, "camera_hash": None, "completed_views": 0, "resumable": False}
    if not d.is_dir():
        reasons.append("directory missing")
        return out
    summary = _load_json(d / SUMMARY_FILE)
    status = _load_json(d / STATUS_FILE)
    cams = _load_json(d / SELECTED_CAMERAS_FILE)
    out["summary"], out["status"] = summary, status

    cam_hash = None
    if not cams or "cameras" not in cams:
        reasons.append(f"{SELECTED_CAMERAS_FILE} missing")
    else:
        if len(cams["cameras"]) != int(expected_views):
            reasons.append(f"selected cameras {len(cams['cameras'])} != {expected_views}")
        cam_hash = camera_protocol_hash(cams["cameras"])
        if cams.get("camera_protocol_hash") not in (None, cam_hash):
            reasons.append("selected_cameras.json hash does not match its camera list")
        if expected_hash and cam_hash != expected_hash:
            reasons.append("camera protocol hash differs from the expected protocol")
    out["camera_hash"] = cam_hash

    rows = read_completed_rows(d / PER_CAMERA_FILE)
    out["completed_views"] = len(rows)
    if len(rows) != int(expected_views) or set(rows) != set(range(int(expected_views))):
        reasons.append(f"per-camera rows {len(rows)} != {expected_views}")

    if not status:
        reasons.append(f"{STATUS_FILE} missing")
    else:
        if status.get("status") != "complete":
            reasons.append(f"status is {status.get('status')!r}, not complete")
        if int(status.get("completed_views", -1)) != int(expected_views):
            reasons.append("status completed_views != expected views")
        if cam_hash and status.get("camera_protocol_hash") not in (None, cam_hash):
            reasons.append("status camera hash mismatch")

    if not summary:
        reasons.append(f"{SUMMARY_FILE} missing")
    else:
        if int(summary.get("num_test_cameras", -1)) != int(expected_views):
            reasons.append("summary num_test_cameras != expected views")
        if cam_hash and summary.get("camera_protocol_hash") not in (None, cam_hash):
            reasons.append("summary camera hash mismatch")
        for k in ("mean_psnr", "mean_ssim"):
            if not _finite(summary.get(k)):
                reasons.append(f"summary {k} missing")

    out["complete"] = not reasons
    # A directory is resumable when its camera protocol matches and it was not completed.
    out["resumable"] = (
        not out["complete"] and cam_hash is not None
        and (expected_hash is None or cam_hash == expected_hash)
        and (status or {}).get("status") in ("running", "failed", "interrupted")
    )
    return out


# --------------------------------------------------------------------------- #
# Training-side storage activity (read-only)
# --------------------------------------------------------------------------- #
def active_compaction_files(watch_dirs: Iterable[str | Path]) -> List[Path]:
    """``.tide_compact_*.tmp`` files under ``<dir>/cache`` (any depth 0-1)."""
    found: List[Path] = []
    for d in watch_dirs:
        cache = Path(d) / "cache"
        if not cache.is_dir():
            continue
        found.extend(cache.glob(".tide_compact_*.tmp"))
        found.extend(cache.glob("*/.tide_compact_*.tmp"))
    return sorted(set(found))
