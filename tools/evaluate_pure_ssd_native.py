#!/usr/bin/env python3
"""Render Pure SSD checkpoints without materializing the full Gaussian table.

Read-only with respect to the checkpoint and the training cache: it restores
the checkpoint shell, loads only the blocks visible to each test camera from
the checkpoint's own patch files, and renders with the existing gsplat
pipeline.  Outputs (all inside --output_dir):

  metrics_summary.json        mean/median/min/max PSNR, mean SSIM (+LPIPS when available)
  metrics_per_camera.csv      one row per view, written and flushed after every view
  selected_cameras.json       the exact camera protocol (eval index -> dataset frame)
  evaluation_status.json      running / complete / failed / interrupted
  diagnostics/                10 fixed views: render, GT and side-by-side comparison
  console.log, native_eval.log, env.txt
  cache/                      evaluation-owned SSD cache, removed on success

Correctness notes (do not regress):
  * GT arrives as uint8 [0,255] from the raw decoded cache and is divided by
    255 before any metric, exactly like the training loss.
  * active_sh_degree is restored from the checkpoint manifest; the
    PREBUILT/base-reuse bind path does not do it.
  * mean_psnr is the mean of per-image PSNR (psnr_of_mean_mse is secondary).
"""
from __future__ import annotations

import argparse, csv, datetime as _dt, json, math, os, shutil, socket, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils.general_utils as utils
from arguments import AuxiliaryParams, BenchmarkParams, DebugParams, ModelParams, OptimizationParams, PipelineParams
from scene import OffloadSceneDataset, Scene
from storage.config import StorageConfig
from storage.pure_ssd_checkpoint import load_pure_ssd_checkpoint_manifest
from storage.tide_storage_adapter import TideStorageAdapter
from storage.block_reader import TieredCacheBlockReader
from strategies.tide_engine.gaussian_model import TideGaussianModel
from strategies.tide_engine.engine import calculate_filters
from strategies.base_engine import pipeline_forward_one_step
from utils.loss_utils import ssim as gaussian_window_ssim
from tools import eval_protocol as proto


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
class _Tee:
    """Duplicate a text stream into a file (console.log)."""

    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()

    def fileno(self):
        return self._stream.fileno()


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _move_camera(camera):
    camera.world_view_transform = camera.world_view_transform.cuda()
    camera.full_proj_transform = camera.full_proj_transform.cuda()
    camera.K = camera.create_k_on_gpu()
    camera.camtoworlds = torch.inverse(camera.world_view_transform.transpose(0, 1)).unsqueeze(0)
    camera.original_image = camera.original_image_backup.cuda(non_blocking=True)


def _render_one(camera, model, scene, background):
    xyz = model.get_xyz
    opacity = model.get_opacity
    scaling = model.get_scaling
    rotation = model.get_rotation
    filters, _, _ = calculate_filters([camera], xyz, opacity, scaling, rotation)
    filt = filters[0]
    if filt.numel() == 0:
        return torch.zeros_like(camera.original_image, dtype=torch.float32)
    xyz = xyz.index_select(0, filt)
    opacity = opacity.index_select(0, filt)
    scaling = scaling.index_select(0, filt)
    rotation = rotation.index_select(0, filt)
    features = model.get_features.index_select(0, filt)
    rendered, _, _ = pipeline_forward_one_step(
        filtered_opacity_gpu=opacity,
        filtered_scaling_gpu=scaling,
        filtered_rotation_gpu=rotation,
        filtered_xyz_gpu=xyz,
        filtered_shs=features,
        camera=camera,
        scene=scene,
        gaussians=model,
        background=background,
        pipe_args=None,
        eval=True,
    )
    return rendered.float().clamp(0.0, 1.0)


def _gt_unit_range(camera) -> torch.Tensor:
    """uint8 [0,255] -> float [0,1]; the training loss does exactly this."""
    img = camera.original_image
    assert img.dtype == torch.uint8, f"expected uint8 raw GT, got {img.dtype}"
    return img.float().div(255.0).clamp(0.0, 1.0)


def _to_uint8_hwc(img: torch.Tensor) -> np.ndarray:
    """[3,H,W] float in [0,1] -> HxWx3 uint8 (plain scaling, no other processing)."""
    return (img.detach().clamp(0, 1).mul(255.0).round().to(torch.uint8)
            .permute(1, 2, 0).contiguous().cpu().numpy())


def _save_diagnostic_images(diag_dir: Path, stem: str, rendered, gt, label_left: str, label_right: str):
    from PIL import Image, ImageDraw
    r = Image.fromarray(_to_uint8_hwc(rendered))
    g = Image.fromarray(_to_uint8_hwc(gt))
    r.save(diag_dir / f"{stem}_render.png")
    g.save(diag_dir / f"{stem}_gt.png")
    bar = 28
    w, h = r.size
    comp = Image.new("RGB", (2 * w + 8, h + bar), (32, 32, 32))
    comp.paste(r, (0, bar))
    comp.paste(g, (w + 8, bar))
    draw = ImageDraw.Draw(comp)
    draw.text((6, 7), label_left, fill=(255, 255, 255))
    draw.text((w + 14, 7), label_right, fill=(255, 255, 255))
    comp.save(diag_dir / f"{stem}_comparison.png")


# --------------------------------------------------------------------------- #
# LPIPS (optional, never downloads)
# --------------------------------------------------------------------------- #
LPIPS_REQUIREMENTS = (
    "pip package 'lpips' (bundles the learned linear layers) and the torchvision "
    "AlexNet backbone weights present offline at $TORCH_HOME/hub/checkpoints/"
    "alexnet-owt-7be5be79.pth (default TORCH_HOME=~/.cache/torch)"
)


def _probe_lpips():
    """Return (metric_fn or None, reason). Never triggers a download."""
    try:
        import importlib.util as iu
        if iu.find_spec("lpips") is None:
            return None, "python package 'lpips' is not installed"
        torch_home = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
        weight = Path(torch_home) / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth"
        if not weight.is_file():
            return None, f"AlexNet backbone weights not found offline at {weight}"
        import lpips  # type: ignore
        net = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
        for p in net.parameters():
            p.requires_grad_(False)

        def metric(render, gt):
            # lpips expects [-1, 1]
            return float(net(render[None] * 2 - 1, gt[None] * 2 - 1).item())

        return metric, "lpips alex (offline weights)"
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"lpips unavailable: {exc}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _summarize(rows, *, checkpoint, manifest, cameras, camera_hash, lpips_info, timing, sample_mode):
    def col(k):
        vals = []
        for r in rows:
            try:
                v = float(r.get(k))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                vals.append(v)
        return vals

    psnr, ssim_v, lp, mse = col("psnr"), col("ssim"), col("lpips"), col("mse")
    mean_mse = float(np.mean(mse)) if mse else float("nan")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(manifest.get("checkpoint_iter", manifest.get("iteration", -1))),
        "checkpoint_type": manifest.get("checkpoint_type"),
        "total_points": int(manifest["total_points"]),
        "num_test_cameras": len(rows),
        "camera_sample_mode": sample_mode,
        "camera_protocol_version": proto.PROTOCOL_VERSION,
        "camera_protocol_hash": camera_hash,
        "num_cameras_in_protocol": len(cameras),
        "mean_mse": mean_mse,
        "mean_psnr": float(np.mean(psnr)) if psnr else None,
        "median_psnr": float(np.median(psnr)) if psnr else None,
        "min_psnr": float(np.min(psnr)) if psnr else None,
        "max_psnr": float(np.max(psnr)) if psnr else None,
        "psnr_of_mean_mse": (-10.0 * math.log10(mean_mse)) if mse and mean_mse > 0 else None,
        "num_images_in_mean_psnr": len(psnr),
        "mean_ssim": float(np.mean(ssim_v)) if ssim_v else None,
        "mean_lpips": float(np.mean(lp)) if lp else None,
        "lpips_status": lpips_info,
        "ssim_implementation": "utils.loss_utils.ssim (11x11 gaussian window, sigma 1.5; same as training)",
        "psnr_definition": "mean over views of -10*log10(mean((render-gt)^2)), images in [0,1]",
        "active_sh_degree": int(manifest.get("active_sh_degree", -1)),
        "max_visible_blocks": max((int(r["visible_blocks"]) for r in rows), default=0),
        "evaluation_start_time": timing["start"],
        "evaluation_end_time": timing["end"],
        "evaluation_runtime_seconds": timing["seconds"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--args_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_test_cameras", type=int, default=proto.FULL_VIEWS)
    ap.add_argument("--camera_sample_mode", choices=["linspace", "contiguous", "window"], default="linspace")
    ap.add_argument("--camera_sample_start", type=int, default=0)
    ap.add_argument("--ssd_cache_dir", default="", help="default: <output_dir>/cache (evaluation-owned)")
    ap.add_argument("--diagnostic_views", type=int, default=proto.DEFAULT_DIAGNOSTIC_VIEWS)
    ap.add_argument("--lpips", choices=["auto", "off", "required"], default="auto")
    ap.add_argument("--resume", action="store_true",
                    help="continue a partial evaluation already present in --output_dir")
    ap.add_argument("--keep_cache", action="store_true")
    ap.add_argument("--progress_every", type=int, default=20, help="plain-text progress line interval")
    ns = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out = Path(ns.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(ns.ssd_cache_dir).resolve() if ns.ssd_cache_dir else out / "cache"
    cp = Path(ns.checkpoint).resolve()
    run_dir_guess = cp.parent.parent
    training_cache = (run_dir_guess / "cache").resolve()
    for p, what in ((out, "output_dir"), (cache_dir, "ssd_cache_dir")):
        if p == training_cache or training_cache in p.parents:
            raise RuntimeError(f"refusing: {what} {p} lies inside the training cache {training_cache}")
    if out in cp.parents or out == cp:
        raise RuntimeError(f"refusing: output_dir {out} would contain the checkpoint")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # console.log receives stdout+stderr; tqdm draws on the real stderr only.
    console = open(out / proto.CONSOLE_LOG, "a", encoding="utf-8")
    real_stderr = sys.stderr
    sys.stdout = _Tee(sys.stdout, console)
    sys.stderr = _Tee(sys.stderr, console)

    per_camera_path = out / proto.PER_CAMERA_FILE
    if per_camera_path.exists() and not ns.resume:
        raise RuntimeError(f"refusing to overwrite existing results in {out}; use --resume or a new --output_dir")

    manifest = load_pure_ssd_checkpoint_manifest(cp)
    ckpt_iter = int(manifest.get("checkpoint_iter", manifest.get("iteration", -1)))
    args = SimpleNamespace(**json.loads(Path(ns.args_json).read_text()))
    args.model_path = str(out)
    args.log_folder = args.model_path
    args.eval = True
    args.num_train_cameras = 0
    args.num_test_cameras = int(ns.max_test_cameras)
    args.debug_max_train_cameras = 0
    args.debug_max_test_cameras = int(ns.max_test_cameras)
    args.debug_camera_sample_mode = ns.camera_sample_mode
    args.debug_camera_sample_start = int(ns.camera_sample_start)
    args.decode_dataset_path = getattr(args, "decode_dataset_path", "")
    args.dataset_cache_and_stream_mode = "load_from_disk_on_demand"
    args.pure_ssd_offload = True
    args.pure_ssd_init_backend = "streaming"
    args.pure_ssd_prebuilt_manifest = ""
    args.start_checkpoint = ""
    args._pure_ssd_resume_manifest = manifest
    args.paper_debug_logging = False
    args.tide_debug_logging = False
    args.paper_block_reader_backend = "tiered_cache"
    args.paper_free_unified_params = True
    args.paper_optimizer_backend = "gpu_resident"
    args.paper_optimizer_state_mode = "resident_blocks"
    args.paper_optimizer_deferred_mode = "off"
    args.paper_resident_capacity_blocks = max(1, int(getattr(args, "paper_resident_capacity_blocks", 2048)))
    args.gaussian_block_size = int(manifest["block_size"])
    args.ssd_cache_dir = str(cache_dir)
    args.use_ssd_offload = True
    args.clm_offload = True
    args.naive_offload = False
    args.no_offload = False
    args.quiet = True

    start_wall = time.time()
    start_iso = _now_iso()
    (out / "env.txt").write_text(
        f"start={start_iso}\nhost={socket.gethostname()}\nslurm_job_id={os.environ.get('SLURM_JOB_ID', '')}\n"
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\ngit_head={_git_head()}\n"
        f"python={sys.executable}\ntorch={torch.__version__}\ncheckpoint={cp}\ncheckpoint_iteration={ckpt_iter}\n"
        f"views={ns.max_test_cameras}\ncamera_sample_mode={ns.camera_sample_mode}\ncache_dir={cache_dir}\n"
        f"resume={ns.resume}\nargv={' '.join(sys.argv)}\n")

    log = open(out / "native_eval.log", "a", encoding="utf-8")
    utils.set_args(args); utils.set_log_file(log)
    torch.cuda.set_device(args.gpu)

    model = TideGaussianModel(sh_degree=int(args.sh_degree), args=args, only_for_rendering=True)
    scene = Scene(args, model, shuffle=False, only_for_rendering=True)
    test_infos = scene.getTestCamerasInfo()
    if args.debug_max_test_cameras >= 0:
        test_infos = test_infos[:args.debug_max_test_cameras]
    if ns.camera_sample_mode == "linspace" and len(test_infos) > ns.max_test_cameras:
        ids = np.linspace(0, len(test_infos) - 1, ns.max_test_cameras, dtype=int)
        test_infos = [test_infos[int(i)] for i in ids]

    # ---- camera protocol audit: recompute the selection from transforms_test.json
    # with the reader's own formula and verify it matches what the Scene loaded.
    transforms_test = Path(args.source_path) / "transforms_test.json"
    cameras = proto.select_test_frames(transforms_test, ns.max_test_cameras,
                                       ns.camera_sample_mode, ns.camera_sample_start)
    if len(cameras) != len(test_infos):
        raise RuntimeError(f"camera protocol mismatch: expected {len(cameras)} cameras, Scene loaded {len(test_infos)}")
    for c, info in zip(cameras, test_infos):
        key = getattr(info, "image_cache_key", None) or info.image_name
        if key != c["image_id"] or os.path.basename(c["file"]) != info.image_name:
            raise RuntimeError(f"camera protocol mismatch at eval_index {c['eval_index']}: "
                               f"protocol {c['image_id']} vs loaded {key}")
        c["source_image_path"] = str(info.image_path)
        c["dataset_uid"] = int(info.uid)
    camera_hash = proto.camera_protocol_hash(cameras)
    diag_indices = proto.diagnostic_eval_indices(len(cameras), ns.diagnostic_views)
    selected_payload = {
        "protocol_version": proto.PROTOCOL_VERSION,
        "transforms_test": str(transforms_test),
        "num_test_frames_in_dataset": None,
        "views": len(cameras),
        "camera_sample_mode": ns.camera_sample_mode,
        "camera_sample_start": int(ns.camera_sample_start),
        "camera_protocol_hash": camera_hash,
        "diagnostic_eval_indices": diag_indices,
        "cameras": cameras,
    }
    with open(transforms_test, "r", encoding="utf-8") as f:
        selected_payload["num_test_frames_in_dataset"] = len(json.load(f)["frames"])
    existing_sel = out / proto.SELECTED_CAMERAS_FILE
    if ns.resume and existing_sel.is_file():
        prev = json.loads(existing_sel.read_text())
        if prev.get("camera_protocol_hash") != camera_hash:
            raise RuntimeError("cannot resume: existing selected_cameras.json uses a different camera protocol")
    _write_json_atomic(existing_sel, selected_payload)

    # ---- resume bookkeeping
    done_rows = proto.read_completed_rows(per_camera_path) if ns.resume else {}
    if done_rows:
        print(f"[RESUME] {len(done_rows)}/{len(cameras)} views already evaluated in {out}; continuing")

    status = {
        "status": "running", "checkpoint": str(cp), "checkpoint_iteration": ckpt_iter,
        "expected_views": len(cameras), "completed_views": len(done_rows),
        "camera_protocol_hash": camera_hash, "start_time": start_iso, "end_time": None,
        "resumed": bool(done_rows), "error": None,
    }
    status_path = out / proto.STATUS_FILE
    _write_json_atomic(status_path, status)

    # ---- storage (evaluation-owned cache; checkpoint patches are read in place)
    cfg = StorageConfig(
        ssd_cache_dir=str(cache_dir), block_size=args.gaussian_block_size,
        max_ram_gb=float(getattr(args, "max_ram_gb", 16.0)),
        num_camera_clusters=1, use_6plane=True, skip_camera_clustering=True,
        max_patch_files=32, max_stale_patch_gb=64.0, max_patch_total_gb=128.0,
        min_free_gb=64.0, debug_logging=False, schedule_cache_enabled=False,
    )
    storage = TideStorageAdapter(model, test_infos, cfg)
    model._block_reader = TieredCacheBlockReader(
        cache_manager=storage.cache, total_gaussians=int(manifest["total_points"]),
        block_size=int(manifest["block_size"]), before_read=storage.wait_for_cache_blocks,
        filter_hint=storage.filter_cache_prefetch_candidates,
    )
    # The PREBUILT/base-reuse bind path does not restore the trained SH degree;
    # render with the degree recorded in the checkpoint manifest.
    model.active_sh_degree = int(manifest.get("active_sh_degree", model.max_sh_degree))
    print(f"[EVAL] checkpoint {ckpt_iter} | views {len(cameras)} ({ns.camera_sample_mode}) | "
          f"active_sh_degree {model.active_sh_degree} | protocol {camera_hash[:19]}... | diag {diag_indices}")

    lpips_fn, lpips_info = (None, "disabled by --lpips off")
    if ns.lpips != "off":
        lpips_fn, lpips_info = _probe_lpips()
        if lpips_fn is None:
            msg = f"[INFO] LPIPS disabled: {lpips_info}. Requires: {LPIPS_REQUIREMENTS}"
            if ns.lpips == "required":
                raise RuntimeError(msg)
            print(msg)
    dataset = OffloadSceneDataset(test_infos, args)
    background = torch.ones(3, device="cuda") if args.white_background else None

    diag_dir = out / proto.DIAGNOSTICS_DIR
    diag_dir.mkdir(exist_ok=True)
    diag_set = set(diag_indices)

    # per-camera CSV: append mode, header only when the file is new; flush every row
    new_csv = not per_camera_path.exists() or per_camera_path.stat().st_size == 0
    csv_fh = open(per_camera_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_fh, fieldnames=list(proto.PER_CAMERA_FIELDS))
    if new_csv:
        writer.writeheader(); csv_fh.flush()

    rows_by_index = {int(k): dict(v) for k, v in done_rows.items()}
    todo = [i for i in range(len(cameras)) if i not in rows_by_index]
    # Diagnostic views whose images are missing are re-rendered even when their
    # metrics row already exists (render is deterministic; the row is kept).
    redo_diag = [i for i in diag_indices if i in rows_by_index and
                 not (diag_dir / f"diag{diag_indices.index(i):02d}_{cameras[i]['image_id'].replace('/', '_')}_comparison.png").exists()]
    order = sorted(set(todo) | set(redo_diag))

    from tqdm import tqdm
    run_psnr, run_ssim, run_lp = [], [], []
    for r in rows_by_index.values():
        run_psnr.append(float(r["psnr"])); run_ssim.append(float(r["ssim"]))
        if r.get("lpips") not in (None, "", "nan"):
            run_lp.append(float(r["lpips"]))
    bar = tqdm(total=len(cameras), initial=len(rows_by_index), desc=f"ckpt {ckpt_iter}", file=real_stderr,
               dynamic_ncols=True, mininterval=1.0, unit="view")
    t_loop = time.time()
    n_new = 0
    outcome = "failed"
    try:
        with torch.no_grad():
            for i in order:
                camera = dataset[i]
                _move_camera(camera)
                block_ids = storage.get_visible_blocks(i)
                model.gpu_working_set_manager.load_visible_blocks_with_retention(
                    block_ids, block_reader=model._block_reader, enable_retention=False
                )
                rendered = _render_one(camera, model, scene, background)
                gt = _gt_unit_range(camera)
                mse = float(torch.mean((rendered - gt) ** 2).item())
                psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
                ssim_val = float(gaussian_window_ssim(rendered.unsqueeze(0), gt.unsqueeze(0)).item())
                lp_val = lpips_fn(rendered, gt) if lpips_fn is not None else None
                cam = cameras[i]
                if i in diag_set:
                    stem = f"diag{diag_indices.index(i):02d}_{cam['image_id'].replace('/', '_')}"
                    _save_diagnostic_images(
                        diag_dir, stem, rendered, gt,
                        f"Render | iter {ckpt_iter} | eval {i} | PSNR {psnr:.2f} dB | SSIM {ssim_val:.3f}",
                        f"Ground truth | frame {cam['dataset_frame_index']} | {cam['image_id']}")
                if i not in rows_by_index:
                    row = {"eval_index": i, "dataset_frame_index": cam["dataset_frame_index"],
                           "image_id": cam["image_id"], "image_name": cam["image_name"],
                           "mse": f"{mse:.8g}", "psnr": f"{psnr:.6f}", "ssim": f"{ssim_val:.6f}",
                           "lpips": "" if lp_val is None else f"{lp_val:.6f}",
                           "visible_blocks": len(block_ids)}
                    writer.writerow(row); csv_fh.flush()
                    rows_by_index[i] = row
                    run_psnr.append(psnr); run_ssim.append(ssim_val)
                    if lp_val is not None:
                        run_lp.append(lp_val)
                    n_new += 1
                    status["completed_views"] = len(rows_by_index)
                    _write_json_atomic(status_path, status)
                camera.original_image = None
                bar.update(0 if i in done_rows else 1)
                post = {"PSNR": f"{np.mean(run_psnr):.2f}", "SSIM": f"{np.mean(run_ssim):.3f}"}
                if run_lp:
                    post["LPIPS"] = f"{np.mean(run_lp):.3f}"
                bar.set_postfix(post, refresh=False)
                done = len(rows_by_index)
                if n_new and (n_new % max(1, ns.progress_every) == 0 or done == len(cameras)):
                    el = time.time() - t_loop
                    rate = el / max(1, n_new)
                    eta = rate * (len(cameras) - done)
                    print(f"[PROGRESS] ckpt {ckpt_iter} | {done}/{len(cameras)} | {100.0 * done / len(cameras):.0f}% | "
                          f"PSNR {np.mean(run_psnr):.2f} | SSIM {np.mean(run_ssim):.3f}"
                          + (f" | LPIPS {np.mean(run_lp):.3f}" if run_lp else "")
                          + f" | elapsed {_fmt_hms(el)} | ETA {_fmt_hms(eta)}", flush=True)
        outcome = "complete"
    except KeyboardInterrupt:
        outcome = "interrupted"
        raise
    except Exception as exc:
        outcome = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        bar.close()
        csv_fh.close()
        status["status"] = outcome
        status["completed_views"] = len(rows_by_index)
        status["end_time"] = _now_iso()
        _write_json_atomic(status_path, status)
        try:
            storage.shutdown()
        except Exception as exc:  # do not mask the primary error
            print(f"[WARN] storage shutdown: {exc}")
        log.close()

    rows = [rows_by_index[i] for i in sorted(rows_by_index)]
    end_wall = time.time()
    timing = {"start": start_iso, "end": _now_iso(), "seconds": round(end_wall - start_wall, 1)}
    summary = _summarize(rows, checkpoint=cp, manifest=manifest, cameras=cameras, camera_hash=camera_hash,
                         lpips_info=lpips_info, timing=timing, sample_mode=ns.camera_sample_mode)
    _write_json_atomic(out / proto.SUMMARY_FILE, summary)
    diag_payload = {
        "checkpoint_iteration": ckpt_iter, "camera_protocol_hash": camera_hash,
        "selection": f"np.unique(np.linspace(0, {len(cameras) - 1}, {len(diag_indices)}, dtype=int)) over eval indices",
        "views": [{
            "diagnostic_index": d, "eval_index": i,
            "dataset_frame_index": cameras[i]["dataset_frame_index"], "image_id": cameras[i]["image_id"],
            "psnr": float(rows_by_index[i]["psnr"]) if i in rows_by_index else None,
            "ssim": float(rows_by_index[i]["ssim"]) if i in rows_by_index else None,
            "lpips": (float(rows_by_index[i]["lpips"]) if i in rows_by_index and rows_by_index[i].get("lpips") else None),
            "files": [f"diag{d:02d}_{cameras[i]['image_id'].replace('/', '_')}_{k}.png" for k in ("render", "gt", "comparison")],
        } for d, i in enumerate(diag_indices)],
    }
    _write_json_atomic(diag_dir / proto.DIAGNOSTIC_VIEWS_FILE, diag_payload)
    print(json.dumps({k: summary[k] for k in ("checkpoint_iteration", "num_test_cameras", "mean_psnr", "median_psnr",
                                                 "min_psnr", "max_psnr", "mean_ssim", "mean_lpips",
                                                 "evaluation_runtime_seconds")}, indent=2))

    if not ns.keep_cache:
        # Only the evaluation-owned cache is removed: it must live inside output_dir.
        if cache_dir.exists() and out in cache_dir.parents:
            shutil.rmtree(cache_dir, ignore_errors=True)
        else:
            print(f"[INFO] cache {cache_dir} is outside {out}; left in place")


if __name__ == "__main__":
    main()
