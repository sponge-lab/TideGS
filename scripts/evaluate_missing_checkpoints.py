#!/usr/bin/env python3
"""Evaluate every complete-but-not-yet-evaluated checkpoint of a TideGS run.

Run this manually inside an interactive GPU allocation (tmux -> srun --pty bash):

    python scripts/evaluate_missing_checkpoints.py --run_dir "$RUN"

Defaults: iterations 100000..500000, 200 linspace views, 10 diagnostic views.
The wrapper never submits Slurm jobs, never deletes checkpoints, never touches
the training cache, and never overwrites a complete evaluation.  Each
checkpoint is evaluated by tools/evaluate_pure_ssd_native.py into its own
directory <run>/evaluations/iter<N>_views<V>_<id>/ with an isolated cache.

Storage-contention guard: training and evaluation may share a node on
different GPUs, but a training-side patch compaction (a .tide_compact_*.tmp
file in the run's cache) streams hundreds of GB through Lustre; evaluating
at the same time roughly doubles both durations.  The wrapper therefore waits
for an active compaction to finish before starting a checkpoint (see
--max_wait_minutes / --no_wait_compaction).  Best time to evaluate: right
after a checkpoint completes, since checkpointing itself compacts.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_protocol as proto  # noqa: E402

EVALUATOR = ROOT / "tools" / "evaluate_pure_ssd_native.py"


def _log(tag: str, msg: str) -> None:
    print(f"[{tag:<6}] {msg}", flush=True)


def _unique_id() -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    job = os.environ.get("SLURM_JOB_ID")
    return f"{stamp}_job{job}" if job else f"{stamp}_local"


def wait_for_compaction(watch_dirs, max_wait_minutes: float, poll_seconds: float = 30.0) -> bool:
    """Return True when no compaction is active (possibly after waiting)."""
    deadline = time.time() + max_wait_minutes * 60.0
    last_sizes = {}
    while True:
        files = proto.active_compaction_files(watch_dirs)
        if not files:
            return True
        desc = []
        for f in files:
            try:
                size = f.stat().st_size
            except OSError:
                continue
            growing = "" if f not in last_sizes else (" growing" if size > last_sizes[f] else " NOT growing (stale?)")
            last_sizes[f] = size
            desc.append(f"{f.name} {size / 1e9:.1f} GB{growing}")
        _log("WAIT", "training storage compaction is currently active: " + "; ".join(desc))
        if time.time() >= deadline:
            _log("WAIT", f"compaction still active after {max_wait_minutes:.0f} min; retry later "
                         "(or pass --no_wait_compaction to proceed despite the I/O contention)")
            return False
        time.sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="training run directory (contains checkpoints/ and args.json)")
    ap.add_argument("--iterations", type=int, nargs="+", default=list(proto.DEFAULT_ITERATIONS))
    ap.add_argument("--views", type=int, default=proto.FULL_VIEWS,
                    help="200 = formal protocol; other counts (e.g. 32) are quick checks with a different camera subset")
    ap.add_argument("--camera_sample_mode", choices=["linspace", "contiguous", "window"], default="linspace")
    ap.add_argument("--diagnostic_views", type=int, default=proto.DEFAULT_DIAGNOSTIC_VIEWS)
    ap.add_argument("--lpips", choices=["auto", "off", "required"], default="auto")
    ap.add_argument("--no_resume", action="store_true",
                    help="never continue an incomplete evaluation directory; always start a fresh one")
    ap.add_argument("--no_wait_compaction", action="store_true")
    ap.add_argument("--max_wait_minutes", type=float, default=120.0)
    ap.add_argument("--compaction_watch_dir", action="append", default=[],
                    help="extra run directories whose live cache is checked for an active compaction")
    ap.add_argument("--keep_cache", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="report what would run; evaluate nothing")
    ap.add_argument("--python", default=sys.executable)
    ns = ap.parse_args()

    run_dir = Path(ns.run_dir).resolve()
    args_json = run_dir / "args.json"
    if not args_json.is_file():
        _log("ERROR", f"missing {args_json}")
        return 2
    args = json.loads(args_json.read_text())
    transforms_test = Path(args["source_path"]) / "transforms_test.json"
    if not transforms_test.is_file():
        _log("ERROR", f"missing {transforms_test}")
        return 2
    protocol = proto.select_test_frames(transforms_test, ns.views, ns.camera_sample_mode)
    expected_hash = proto.camera_protocol_hash(protocol)
    formal = ns.views == proto.FULL_VIEWS and ns.camera_sample_mode == "linspace"
    _log("INFO", f"run {run_dir}")
    _log("INFO", f"{'formal' if formal else 'QUICK (not comparable to the 200-view curve)'} protocol: "
                 f"{len(protocol)} {ns.camera_sample_mode} views of {transforms_test.name}, hash {expected_hash[:23]}...")
    if not formal:
        _log("INFO", f"views{ns.views} results are stored separately and never satisfy the 200-view skip rule")
    watch_dirs = [run_dir] + [Path(d) for d in ns.compaction_watch_dir]

    results = []
    failures = 0
    for it in ns.iterations:
        ckpt = proto.find_checkpoint_dir(run_dir, it)
        st = proto.checkpoint_status(ckpt)
        if not st["exists"]:
            _log("WAIT", f"checkpoint {it} not available yet")
            results.append((it, "waiting", None)); continue
        if not st["complete"]:
            _log("WAIT", f"checkpoint {it} incomplete: {st['reasons'][0]}")
            results.append((it, "incomplete-checkpoint", None)); continue
        _log("FOUND", f"checkpoint {it}")

        complete_dir = None
        resumable = []
        for d in proto.find_evaluation_dirs(run_dir, it, ns.views):
            es = proto.evaluation_status(d, ns.views, expected_hash)
            if es["complete"]:
                complete_dir = (d, es["summary"]); break
            if es["resumable"]:
                resumable.append((d, es["completed_views"]))
            else:
                _log("INFO", f"ignoring {d.name}: {es['reasons'][0]} (kept for diagnosis)")
        if complete_dir:
            d, summ = complete_dir
            _log("SKIP", f"checkpoint {it} already has a complete {ns.views}-view evaluation: {d.name} "
                         f"| PSNR={summ.get('mean_psnr'):.2f} | SSIM={summ.get('mean_ssim'):.3f}")
            results.append((it, "skipped", summ)); continue

        resume_dir = None
        if resumable and not ns.no_resume:
            resume_dir = max(resumable, key=lambda x: x[1])[0]
        out_dir = resume_dir or (run_dir / "evaluations" / proto.eval_dir_name(it, ns.views, _unique_id()))
        tag = "RESUME" if resume_dir else "RUN"
        _log(tag, f"checkpoint {it} -> {out_dir.name}")
        if ns.dry_run:
            results.append((it, "would-run", None)); continue

        if not ns.no_wait_compaction and not wait_for_compaction(watch_dirs, ns.max_wait_minutes):
            results.append((it, "deferred-compaction", None))
            break

        cmd = [ns.python, "-B", str(EVALUATOR),
               "--checkpoint", str(ckpt), "--args_json", str(args_json), "--output_dir", str(out_dir),
               "--max_test_cameras", str(ns.views), "--camera_sample_mode", ns.camera_sample_mode,
               "--diagnostic_views", str(ns.diagnostic_views), "--lpips", ns.lpips]
        if resume_dir:
            cmd.append("--resume")
        if ns.keep_cache:
            cmd.append("--keep_cache")
        env = dict(os.environ)
        env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT), env=env)
        es = proto.evaluation_status(out_dir, ns.views, expected_hash)
        if rc == 0 and es["complete"]:
            s = es["summary"]
            lp = s.get("mean_lpips")
            _log("DONE", f"checkpoint {it} | PSNR={s['mean_psnr']:.2f} | SSIM={s['mean_ssim']:.3f}"
                         + (f" | LPIPS={lp:.3f}" if lp is not None else "")
                         + f" | {(time.time() - t0) / 60:.1f} min | {out_dir}")
            results.append((it, "done", s))
        else:
            failures += 1
            reason = es["reasons"][0] if es["reasons"] else f"exit code {rc}"
            _log("FAIL", f"checkpoint {it} | exit {rc} | {reason} | partial results kept in {out_dir}")
            results.append((it, "failed", None))

    print()
    _log("INFO", "summary")
    for it, state, s in results:
        line = f"  {it:>7}  {state:<22}"
        if s:
            line += f" PSNR {s.get('mean_psnr'):.2f}  SSIM {s.get('mean_ssim'):.3f}"
            if s.get("mean_lpips") is not None:
                line += f"  LPIPS {s['mean_lpips']:.3f}"
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
