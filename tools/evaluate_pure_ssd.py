#!/usr/bin/env python3
"""Offline PSNR evaluation for pure SSD manifests and checkpoints.

The evaluator materializes the 59-column SSD layout into a temporary PLY,
loads it through the existing rendering path, and evaluates a fixed test-camera
subset.  It is intended for the 1M-point scale A/B experiment; it is not an
in-training evaluator for billion-point scenes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arguments import (  # noqa: E402
    AuxiliaryParams,
    BenchmarkParams,
    DebugParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
)
from scene import OffloadSceneDataset, Scene  # noqa: E402
from storage.pure_ssd_checkpoint import load_pure_ssd_checkpoint_manifest  # noqa: E402
from strategies.tide_engine.engine import clm_offload_eval_one_cam  # noqa: E402
from strategies.tide_engine.gaussian_model import TideGaussianModel  # noqa: E402
from tools.export_pure_ssd_checkpoint_to_ply import (  # noqa: E402
    PLY_ATTR_DIM,
    SSD_PARAM_DIM,
    ply_attribute_names,
    ssd_rows_to_ply_rows,
    write_binary_little_endian_ply_header,
)
import utils.general_utils as utils  # noqa: E402


def _load_artifact_manifest(artifact: Path) -> tuple[dict, bool]:
    if artifact.is_dir():
        checkpoint_manifest = artifact / "pure_ssd_checkpoint.json"
        if checkpoint_manifest.is_file():
            return load_pure_ssd_checkpoint_manifest(artifact), True
        raise FileNotFoundError(
            f"Expected pure_ssd_checkpoint.json in artifact directory: {artifact}"
        )

    if artifact.name != "streaming_init_manifest.json" and artifact.name != "pure_ssd_checkpoint.json":
        raise ValueError(f"Expected a streaming or checkpoint manifest: {artifact}")
    if artifact.name == "pure_ssd_checkpoint.json":
        return load_pure_ssd_checkpoint_manifest(artifact.parent), True

    with artifact.open("r", encoding="utf-8") as handle:
        return json.load(handle), False


def _write_base_manifest_to_ply(
    manifest: dict,
    output: Path,
    chunk_points: int,
) -> None:
    total_points = int(manifest["total_points"])
    param_dim = int(manifest.get("param_dim", SSD_PARAM_DIM))
    if param_dim != SSD_PARAM_DIM:
        raise ValueError(f"Expected param_dim={SSD_PARAM_DIM}, got {param_dim}")

    base_file = Path(manifest["base_file"])
    expected_size = total_points * SSD_PARAM_DIM * np.dtype(np.float32).itemsize
    if base_file.stat().st_size != expected_size:
        raise RuntimeError(
            f"Base file size mismatch: got {base_file.stat().st_size}, expected {expected_size}"
        )

    attributes = ply_attribute_names()
    if len(attributes) != PLY_ATTR_DIM:
        raise AssertionError(f"PLY attribute count mismatch: {len(attributes)} != {PLY_ATTR_DIM}")

    bytes_per_row = SSD_PARAM_DIM * np.dtype(np.float32).itemsize
    output.parent.mkdir(parents=True, exist_ok=True)
    with base_file.open("rb") as source, output.open("wb") as destination:
        write_binary_little_endian_ply_header(destination, total_points, attributes)
        remaining = total_points
        while remaining:
            rows = min(int(chunk_points), remaining)
            raw = source.read(rows * bytes_per_row)
            if len(raw) != rows * bytes_per_row:
                raise RuntimeError(f"Unexpected EOF in base file with {remaining} rows remaining")
            ssd_rows = np.frombuffer(raw, dtype="<f4").reshape(rows, SSD_PARAM_DIM)
            ply_rows = ssd_rows_to_ply_rows(ssd_rows)
            destination.write(np.ascontiguousarray(ply_rows, dtype="<f4").tobytes())
            remaining -= rows


def _build_default_args() -> SimpleNamespace:
    parser = argparse.ArgumentParser(add_help=False)
    AuxiliaryParams(parser)
    ModelParams(parser)
    OptimizationParams(parser)
    PipelineParams(parser)
    BenchmarkParams(parser)
    DebugParams(parser)
    return parser.parse_args([])


def _load_render_args(args_json: str, source_path: str | None) -> SimpleNamespace:
    if args_json:
        with open(args_json, "r", encoding="utf-8") as handle:
            values = json.load(handle)
        args = SimpleNamespace(**values)
    else:
        args = _build_default_args()

    if source_path:
        args.source_path = source_path
    if not getattr(args, "source_path", ""):
        raise ValueError("--source_path is required when --args_json is not provided")

    return args


def _prepare_render_args(
    args: SimpleNamespace,
    *,
    output_dir: Path,
    ply_path: Path,
    artifact_manifest: dict,
    max_test_cameras: int,
    camera_sample_mode: str,
    dataset_mode: str,
    active_sh_degree: int,
    total_points: int,
) -> SimpleNamespace:
    args.source_path = str(Path(args.source_path).resolve())
    args.model_path = str(output_dir.resolve())
    args.log_folder = str(output_dir.resolve())
    args.load_ply_path = str(ply_path.parent.resolve())
    args.load_pt_path = ""
    args.dense_ply_file = str(ply_path.resolve())
    # Tell the City reader that the artifact already has its own Gaussian
    # source.  This prevents it from loading the original MatrixCity PLY just
    # to build SceneInfo; the evaluator only needs camera metadata there.
    args._pure_ssd_prebuilt_manifest = artifact_manifest
    args.eval = True
    args.dataset_cache_and_stream_mode = dataset_mode
    args.debug_max_train_cameras = 1
    args.debug_max_test_cameras = int(max_test_cameras)
    args.debug_camera_sample_mode = camera_sample_mode
    args.debug_camera_sample_start = 0
    args.use_ssd_offload = False
    args.clm_offload = False
    args.naive_offload = False
    args.no_offload = True
    args.pure_ssd_offload = False
    args.drop_initial_3dgs_p = 0.0
    args.prealloc_capacity = int(total_points)
    args.active_sh_degree = int(active_sh_degree)
    args.debug_frustum = False
    args.check_cpu_memory = False
    args.check_gpu_memory = False
    args.paper_debug_logging = False
    return args


def _move_camera_to_gpu(camera) -> None:
    camera.world_view_transform = camera.world_view_transform.cuda()
    camera.full_proj_transform = camera.full_proj_transform.cuda()
    camera.K = camera.create_k_on_gpu()
    camera.camtoworlds = torch.inverse(
        camera.world_view_transform.transpose(0, 1)
    ).unsqueeze(0)
    camera.original_image = camera.original_image_backup.cuda(non_blocking=True)


def evaluate(
    *,
    artifact: str,
    output_dir: str,
    args_json: str,
    source_path: str | None,
    max_test_cameras: int,
    camera_sample_mode: str,
    dataset_mode: str,
    chunk_points: int,
    active_sh_degree: int | None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Offline PSNR evaluation requires CUDA")

    artifact_path = Path(artifact).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    manifest, is_checkpoint = _load_artifact_manifest(artifact_path)
    total_points = int(manifest["total_points"])

    ply_path = output_path / "evaluation_ply" / "point_cloud.ply"
    if is_checkpoint:
        from tools.export_pure_ssd_checkpoint_to_ply import export_checkpoint_to_ply

        export_checkpoint_to_ply(
            checkpoint=artifact_path,
            output=ply_path,
            chunk_points=chunk_points,
        )
    else:
        _write_base_manifest_to_ply(manifest, ply_path, chunk_points)

    if active_sh_degree is None:
        active_sh_degree = int(manifest.get("active_sh_degree", 0))

    args = _load_render_args(args_json, source_path)
    args = _prepare_render_args(
        args,
        output_dir=output_path,
        ply_path=ply_path,
        artifact_manifest=manifest,
        max_test_cameras=max_test_cameras,
        camera_sample_mode=camera_sample_mode,
        dataset_mode=dataset_mode,
        active_sh_degree=active_sh_degree,
        total_points=total_points,
    )

    log_path = output_path / "evaluator.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        utils.set_args(args)
        utils.set_log_file(log_file)
        torch.manual_seed(0)

        model = TideGaussianModel(
            sh_degree=int(args.sh_degree),
            args=args,
            only_for_rendering=True,
        )
        scene = Scene(args, model, shuffle=False, only_for_rendering=True)
        model.active_sh_degree = int(active_sh_degree)

        test_infos = scene.getTestCamerasInfo()
        if not test_infos:
            raise RuntimeError("No test cameras were loaded")
        test_dataset = OffloadSceneDataset(test_infos, args)
        background = (
            torch.ones(3, dtype=torch.float32, device="cuda")
            if args.white_background
            else None
        )

        results = []
        with torch.no_grad():
            for index in range(len(test_dataset)):
                camera = test_dataset[index]
                _move_camera_to_gpu(camera)
                rendered = clm_offload_eval_one_cam(camera, model, background, scene)
                ground_truth = camera.original_image.float() / 255.0
                rendered = rendered.float().clamp(0.0, 1.0)
                ground_truth = ground_truth.clamp(0.0, 1.0)
                mse = float(torch.mean((rendered - ground_truth) ** 2).item())
                psnr = float("inf") if mse == 0.0 else -10.0 * math.log10(mse)
                results.append(
                    {
                        "index": index,
                        "image_name": camera.image_name,
                        "mse": mse,
                        "psnr": psnr,
                    }
                )
                camera.original_image = None
                if (index + 1) % 10 == 0 or index + 1 == len(test_dataset):
                    print(f"[PSNR] evaluated {index + 1}/{len(test_dataset)} cameras")

        mean_mse = float(np.mean([item["mse"] for item in results]))
        mean_psnr = float("inf") if mean_mse == 0.0 else -10.0 * math.log10(mean_mse)
        summary = {
            "artifact": str(artifact_path),
            "checkpoint": is_checkpoint,
            "total_points": total_points,
            "active_sh_degree": int(active_sh_degree),
            "num_test_cameras": len(results),
            "mean_mse": mean_mse,
            "mean_psnr": mean_psnr,
            "scale_mode": manifest.get("scale_mode", "unknown"),
        }
        with (output_path / "psnr_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "cameras": results}, handle, indent=2)
        with (output_path / "psnr_per_camera.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "image_name", "mse", "psnr"])
            writer.writeheader()
            writer.writerows(results)
        print(json.dumps(summary, indent=2))
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Streaming manifest or pure SSD checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory")
    parser.add_argument("--args_json", default="", help="Training args.json; optional for initial manifests")
    parser.add_argument("--source_path", default="", help="Dataset path when --args_json is omitted")
    parser.add_argument("--max_test_cameras", type=int, default=64)
    parser.add_argument("--camera_sample_mode", choices=["linspace", "contiguous", "window"], default="linspace")
    parser.add_argument("--dataset_mode", choices=["load_from_source_on_demand", "load_from_disk_on_demand"], default="load_from_source_on_demand")
    parser.add_argument("--chunk_points", type=int, default=1_000_000)
    parser.add_argument("--active_sh_degree", type=int, default=None)
    return parser


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    evaluate(
        artifact=cli_args.artifact,
        output_dir=cli_args.output_dir,
        args_json=cli_args.args_json,
        source_path=cli_args.source_path or None,
        max_test_cameras=cli_args.max_test_cameras,
        camera_sample_mode=cli_args.camera_sample_mode,
        dataset_mode=cli_args.dataset_mode,
        chunk_points=cli_args.chunk_points,
        active_sh_degree=cli_args.active_sh_degree,
    )
