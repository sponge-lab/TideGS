import torch
import math
import utils.general_utils as utils
import os
import torch.nn as nn
import gc
from typing import Optional

from strategies.tide_engine.gpu_resident_optimizer import GPUResidentAdam

def get_gpu_resident_optimizer(gaussians, batch_size):
    desired_block_size = int(
        getattr(
            gaussians,
            '_paper_optimizer_block_size',
            getattr(getattr(gaussians, 'args', None), 'gaussian_block_size', 4096),
        )
    )
    current_optimizer = getattr(gaussians, '_paper_gpu_resident_optimizer', None)
    desired_capacity = max(
        0,
        int(getattr(getattr(gaussians, 'args', None), 'paper_resident_capacity_blocks', 0)),
    )
    needs_recreate = (
        current_optimizer is None
        or getattr(current_optimizer, 'batch_size', None) != batch_size
        or getattr(current_optimizer, 'block_size', desired_block_size) != desired_block_size
        or getattr(current_optimizer, 'capacity_blocks', desired_capacity) != desired_capacity
    )
    if needs_recreate:
        gaussians._paper_gpu_resident_optimizer = GPUResidentAdam(
            batch_size=batch_size,
            block_size=desired_block_size,
            capacity_blocks=desired_capacity,
            device='cuda',
        )
    return gaussians._paper_gpu_resident_optimizer


def shutdown_gpu_resident_optimizer(gaussians=None):
    if gaussians is None:
        return
    if hasattr(gaussians, '_paper_gpu_resident_optimizer'):
        gaussians._paper_gpu_resident_optimizer = None


def get_total_gaussians_count(gaussians, storage_adapter=None):
    """Return the full-scene Gaussian count from paper metadata or fallbacks."""
    num_total = getattr(gaussians, '_paper_unified_params_num_total', None)
    if num_total is not None:
        return int(num_total)

    unified_params = getattr(gaussians, '_unified_params', None)
    if unified_params is not None:
        return int(unified_params.shape[0])

    gpu_ws = getattr(gaussians, 'gpu_working_set_manager', None)
    if gpu_ws is not None:
        ws_total = getattr(gpu_ws, 'num_total_gaussians', None)
        if ws_total is not None:
            return int(ws_total)

    if storage_adapter is not None and hasattr(storage_adapter, 'total_gaussians'):
        total_gaussians = getattr(storage_adapter, 'total_gaussians', None)
        if total_gaussians is not None:
            return int(total_gaussians)

    xyz = getattr(gaussians, '_xyz', None)
    if xyz is not None:
        return int(xyz.shape[0])

    grad_buffer = getattr(gaussians, 'parameters_grad_buffer', None)
    if grad_buffer is not None:
        return int(grad_buffer.shape[0])

    raise RuntimeError(
        'Unable to determine total Gaussian count: paper-mode tensors were released, '
        'and no fallback metadata was available.'
    )


def ensure_local_to_global_mapping(gaussians, total_n_gaussians: int, *, log_file=None, context: str = ""):
    """Ensure gpu_working_set_manager.local_to_global_idx exists.

    Some paper-mode branches can preserve resident tensors while dropping the
    compact local->global mapping metadata. Rebuild it from block slices when
    needed so later local-index operations remain valid.
    """
    manager = getattr(gaussians, 'gpu_working_set_manager', None)
    if manager is None:
        raise RuntimeError('gpu_working_set_manager is required to build local_to_global_idx')

    local_to_global = getattr(manager, 'local_to_global_idx', None)
    if local_to_global is not None:
        return local_to_global

    block_size = int(getattr(manager, 'block_size', getattr(getattr(gaussians, 'args', None), 'gaussian_block_size', 4096)))
    device = getattr(manager, 'device', torch.device('cuda'))
    rebuilt_global_ids = []

    block_to_gpu_slice = getattr(manager, 'block_to_gpu_slice', {}) or {}
    if block_to_gpu_slice:
        plans = []
        for block_id, local_slice in block_to_gpu_slice.items():
            if local_slice is None:
                continue
            local_len = max(0, int(local_slice.stop) - int(local_slice.start))
            if local_len <= 0:
                continue
            start_idx = int(block_id) * block_size
            max_rows = max(0, min(start_idx + block_size, total_n_gaussians) - start_idx)
            row_count = min(local_len, max_rows)
            if row_count <= 0:
                continue
            plans.append((int(local_slice.start), start_idx, row_count))

        for _, start_idx, row_count in sorted(plans, key=lambda item: item[0]):
            rebuilt_global_ids.extend(range(start_idx, start_idx + row_count))
    else:
        loaded_blocks = getattr(manager, 'loaded_blocks', []) or []
        for block_id in loaded_blocks:
            start_idx = int(block_id) * block_size
            end_idx = min(start_idx + block_size, total_n_gaussians)
            if end_idx > start_idx:
                rebuilt_global_ids.extend(range(start_idx, end_idx))

    manager.local_to_global_idx = torch.tensor(rebuilt_global_ids, dtype=torch.long, device=device)
    if context:
        utils.log_and_print(
            f"[SSD MAP] Rebuilt local_to_global_idx in {context}: {manager.local_to_global_idx.numel()} entries",
            log_file,
        )
    return manager.local_to_global_idx

# Paper/Tide double-buffer GPU residency support.
from strategies.tide_engine.double_buffer_gpu import (
    DoubleBufferGPUWorkingSet,
)
from strategies.tide_engine.runtime import (
    apply_paper_writeback_payload as _apply_paper_writeback_payload,
    collect_paper_updated_block_ids as _collect_paper_updated_block_ids,
    configure_gpu_resident_optimizer_state as _configure_gpu_resident_optimizer_state,
    initialize_paper_mode_runtime_state as _initialize_paper_mode_runtime_state,
    log_batch_kt_metrics as _log_batch_kt_metrics,
    log_delta_handoff as _log_delta_handoff,
    log_delta_stream_prefetch as _log_delta_stream_prefetch,
    log_double_buffer_stats as _log_double_buffer_stats,
    log_empty_microbatch_camera as _log_empty_microbatch_camera,
    log_empty_paper_projection_cameras as _log_empty_paper_projection_cameras,
    log_empty_stage2_loaded_gaussians as _log_empty_stage2_loaded_gaussians,
    log_paper_block_sets as _log_paper_block_sets,
    log_paper_batch_debug as _log_paper_batch_debug,
    log_paper_block_visibility_debug as _log_paper_block_visibility_debug,
    log_paper_gpu_working_set_parameters as _log_paper_gpu_working_set_parameters,
    log_paper_perf_profile as _log_paper_perf_profile,
    log_paper_warm_layer_metrics as _log_paper_warm_layer_metrics,
    log_ssd_stage1_debug as _log_ssd_stage1_debug,
    log_early_delta_hint as _log_early_delta_hint,
    log_paper_cpu_source_refs_restored as _log_paper_cpu_source_refs_restored,
    log_paper_interbatch_sh_cache_disabled as _log_paper_interbatch_sh_cache_disabled,
    log_paper_order_calculation_skip as _log_paper_order_calculation_skip,
    log_paper_resident_camera_coverage as _log_paper_resident_camera_coverage,
    log_paper_sh_indexed as _log_paper_sh_indexed,
    log_paper_working_set_retained as _log_paper_working_set_retained,
    log_paper_working_set_loaded as _log_paper_working_set_loaded,
    log_paper_writeback_finalize as _log_paper_writeback_finalize,
    log_paper_writeback_staged as _log_paper_writeback_staged,
    load_paper_stage1_working_set as _load_paper_stage1_working_set,
    map_compact_filters_to_global as _map_compact_filters_to_global,
    paper_debug_logging_enabled as _paper_debug_logging_enabled,
    plan_and_start_resident_prefetch as _plan_and_start_resident_prefetch,
    resolve_current_iteration_resident_blocks as _resolve_current_iteration_resident_blocks,
    run_gpu_resident_adam_step as _run_gpu_resident_adam_step,
    seed_paper_active_buffer_from_manager as _seed_paper_active_buffer_from_manager,
    write_paper_phase1_log as _write_paper_phase1_log,
)

# 全局双缓冲 GPU 管理器
_double_buffer_gpu: Optional[DoubleBufferGPUWorkingSet] = None

def get_double_buffer_gpu(
    num_total: int,
    block_size: int,
    device: str = 'cuda',
    verbose: bool = False,
) -> DoubleBufferGPUWorkingSet:
    """获取或创建全局双缓冲 GPU 管理器"""
    global _double_buffer_gpu
    if _double_buffer_gpu is None:
        _double_buffer_gpu = DoubleBufferGPUWorkingSet(
            num_total_gaussians=num_total,
            block_size=block_size,
            device=device,
            verbose=verbose,
        )
    return _double_buffer_gpu

def shutdown_double_buffer_gpu():
    """关闭双缓冲 GPU 管理器"""
    global _double_buffer_gpu
    if _double_buffer_gpu is not None:
        _double_buffer_gpu.clear()
        _double_buffer_gpu = None

import numpy as np
from gsplat import (
    fully_fused_projection,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_pixels,
)
from densification import update_densification_stats_offload_accum_grads
from strategies.base_engine import (
    torch_compiled_loss,
    TILE_SIZE,
    calculate_filters,
    pipeline_forward_one_step,
)




def _collect_updated_block_ids(filters, block_size: int, total_n_gaussians: Optional[int] = None):
    visible_indices_list = []
    for f in filters:
        if f is None or f.numel() == 0:
            continue
        if f.dtype == torch.bool:
            idxs = torch.nonzero(f, as_tuple=False).squeeze(1)
        else:
            idxs = f
        if idxs.device.type != 'cpu':
            idxs = idxs.cpu()
        idxs = idxs.to(dtype=torch.long)
        if total_n_gaussians is not None:
            idxs = idxs[(idxs >= 0) & (idxs < total_n_gaussians)]
        if idxs.numel() > 0:
            visible_indices_list.append(idxs)

    if len(visible_indices_list) == 0:
        return []

    unique_gaussian_ids = torch.cat(visible_indices_list).unique()
    return (unique_gaussian_ids // block_size).unique().cpu().tolist()


def _collect_updated_block_ids_from_indices(
    gaussian_indices: Optional[torch.Tensor],
    block_size: int,
    total_n_gaussians: Optional[int] = None,
):
    if gaussian_indices is None or gaussian_indices.numel() == 0:
        return []

    idxs = gaussian_indices
    if idxs.dtype == torch.bool:
        idxs = torch.nonzero(idxs, as_tuple=False).squeeze(1)
    if idxs.device.type != 'cpu':
        idxs = idxs.cpu()
    idxs = idxs.to(dtype=torch.long)
    if total_n_gaussians is not None:
        idxs = idxs[(idxs >= 0) & (idxs < total_n_gaussians)]
    if idxs.numel() == 0:
        return []
    return torch.unique(torch.div(idxs, block_size, rounding_mode='floor')).tolist()


def visualize_frustum_culling_inline(
    batched_cameras,
    block_bounds,
    visible_block_ids,
    iteration,
    save_dir='debug_frustum',
):
    """Frustum HTML visualization is not included in the release package."""
    return None


def pipeline_forward_one_step_shs_inplace(
    filtered_opacity_gpu,
    filtered_scaling_gpu,
    filtered_rotation_gpu,
    filtered_xyz_gpu,
    filtered_shs,
    camera,
    scene,
    gaussians,
    background,
    pipe_args,
):
    MICRO_BATCH_SIZE = 1  # NOTE: microbatch here only contains one camera.

    viewmat = camera.world_view_transform.transpose(0, 1)  # why transpose
    # K = camera.create_k_on_gpu() # create K now, which may invoke cpu-gpu transfer
    K = camera.K
    n_selected = filtered_xyz_gpu.shape[0]
    image_width = int(utils.get_img_width())
    image_height = int(utils.get_img_height())

    batched_radiis, batched_means2D, batched_depths, batched_conics, _ = (
        fully_fused_projection(
            means=filtered_xyz_gpu,  # (N, 3)
            covars=None,
            quats=filtered_rotation_gpu,
            scales=filtered_scaling_gpu,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=image_width,
            height=image_height,
            packed=False,
        )
    )  # (1, N), (1, N, 2), (1, N), (1, N, 3), (1, N)

    batched_means2D.retain_grad()  # this is only for training.

    sh_degree = gaussians.active_sh_degree
    camtoworlds = camera.camtoworlds
    # camtoworlds = torch.inverse(viewmat.unsqueeze(0)) # (4, 4)
    dirs = filtered_xyz_gpu[None, :, :] - camtoworlds[:, None, :3, 3]
    filtered_shs = filtered_shs.reshape(1, n_selected, 16, 3)

    batched_colors_origin = spherical_harmonics(
        degrees_to_use=sh_degree, dirs=dirs, coeffs=filtered_shs
    )
    batched_colors_detached = batched_colors_origin
    batched_colors = torch.clamp_min(batched_colors_origin + 0.5, 0.0)

    batched_opacities = filtered_opacity_gpu.squeeze(1).unsqueeze(0)  # (N, 1) -> (1, N)

    # NOTE: In the above code, we keep the first batch dimension, even if it is always 1.

    # render
    # Identify intersecting tiles.
    tile_width = math.ceil(image_width / float(TILE_SIZE))
    tile_height = math.ceil(image_height / float(TILE_SIZE))

    # flatten_ids: (C*N)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=batched_means2D,
        radii=batched_radiis,
        depths=batched_depths,
        tile_size=TILE_SIZE,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
    )
    isect_offsets = isect_offset_encode(
        isect_ids, MICRO_BATCH_SIZE, tile_width, tile_height
    )  # (MICRO_BATCH_SIZE, tile_height, tile_width)

    # Rasterize to pixels. batched_rendered_image: (B, image_height, image_width, 3)
    backgrounds = (
        background.repeat(MICRO_BATCH_SIZE, 1) if background is not None else None
    )
    rendered_image, _ = rasterize_to_pixels(
        means2d=batched_means2D,
        conics=batched_conics,
        colors=batched_colors,
        opacities=batched_opacities,
        image_width=image_width,
        image_height=image_height,
        tile_size=TILE_SIZE,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
    )

    rendered_image = rendered_image.squeeze(0).permute(2, 0, 1).contiguous()

    return (
        rendered_image,
        batched_means2D,
        batched_radiis,
        batched_colors_detached,
        dirs,
    )


def clm_offload_train_one_batch(
    gaussians,
    scene,
    batched_cameras,
    background,
    pipe_args,
    comm_stream,
    storage_adapter=None,
    training_schedule=None,
    runtime_args=None,
):
    args = runtime_args
    if args is None:
        raise ValueError("runtime_args is required for TideGS batch training")
    if storage_adapter is None:
        raise RuntimeError("TideGS batch training requires a TideStorageAdapter.")
    if getattr(gaussians, "gpu_working_set_manager", None) is None:
        raise RuntimeError("TideGS batch training requires a GPU working-set manager.")
    if not getattr(gaussians.optimizer, "is_ssd_offload_mode", False):
        raise RuntimeError("TideGS batch training requires ResidentAdamContext.")
    if getattr(gaussians, "_unified_params", None) is not None:
        raise RuntimeError("TideGS batch training cannot use a full RAM parameter table.")
    iteration = utils.get_cur_iter()
    log_file = utils.get_log_file()

    # ========================================================================
    # [PERF PROFILE] Lightweight per-iteration stage timer (prints every 50 iters)
    # ========================================================================
    import time as _time
    _perf_t = {}
    # iterations step by bsz (1, 65, 129, ...), so "% 50 == 1" only fires once
    # use a counter that increments per batch instead
    if not hasattr(clm_offload_train_one_batch, '_batch_count'):
        clm_offload_train_one_batch._batch_count = 0
    clm_offload_train_one_batch._batch_count += 1
    _perf_log = (clm_offload_train_one_batch._batch_count % 5 == 1)  # log every 5 batches
    def _ts(name):
        _perf_t[name] = _time.perf_counter()
    _ts('iter_start')

    current_camera_ids = [cam.global_idx for cam in batched_cameras]
    bsz = len(batched_cameras)

    # ============================================================================
    # STAGE 1: SETUP & PREPROCESSING
    # ============================================================================


    # ========================================================================
    # Get total Gaussians count from a paper-safe source.
    # ========================================================================
    # In paper out-of-core mode the full tensors may already be released before
    # Stage 1.5 materializes the current resident set, so we must not assume
    # gaussians._xyz is available here.
    total_n_gaussians = get_total_gaussians_count(gaussians, storage_adapter)

    # Before resident-set materialization, any coarse-grained bookkeeping that
    # still expects "N" should use the full-scene count rather than dereference
    # the released in-memory tensors.
    n_gaussians = total_n_gaussians

    # ========================================================================
    # Save original parameter references before the resident-set replacement.
    # ========================================================================
    # These will be used to restore parameters after batch completion
    # MUST be saved BEFORE STAGE 1.5 replaces gaussians._xyz with GPU working set!
    original_xyz = gaussians._xyz
    original_scaling = gaussians._scaling
    original_rotation = gaussians._rotation
    original_opacity = gaussians._opacity
    original_features_dc = gaussians._features_dc
    original_features_rest = gaussians._features_rest

    paper_debug_logging = _paper_debug_logging_enabled(args)
    paper_optimizer_deferred_mode, paper_optimizer_backend = _initialize_paper_mode_runtime_state(
        gaussians=gaussians,
        args=args,
        iteration=iteration,
        log_file=log_file,
    )

    _ts('stage1_setup_done')

    # ============================================================================
    # STAGE 1.5: [SSD HOOK] Block-level Culling & Load from SSD
    # ============================================================================
    torch.cuda.nvtx.range_push("SSD: block-level culling and loading")

    _log_paper_batch_debug(
        enabled=paper_debug_logging,
        iteration=iteration,
        batched_cameras=batched_cameras,
        current_camera_ids=current_camera_ids,
    )

    # Step 1: collect coarse block visibility for this camera batch.
    with torch.cuda.nvtx.range("Tide activation: current block culling"):
        current_bounds_generation, current_camera_blocks = storage_adapter.get_visible_blocks_batch(
            current_camera_ids
        )
        visible_block_ids_set = set()
        cam_to_blocks = {}  # Track per-camera block visibility
        for cam_global_idx in current_camera_ids:
            blocks = current_camera_blocks[int(cam_global_idx)]
            visible_block_ids_set.update(blocks)
            cam_to_blocks[cam_global_idx] = len(blocks)

    visible_block_ids = sorted(list(visible_block_ids_set))

    _log_paper_block_visibility_debug(
        enabled=paper_debug_logging,
        iteration=iteration,
        cam_to_blocks=cam_to_blocks,
        visible_block_ids=visible_block_ids,
    )

    # ====================================================================
    # Diagnostic: empty block-level visibility is a geometry/camera issue.
    # ====================================================================
    if len(visible_block_ids) == 0:
        log_file.write(
            f"\n[CRITICAL] Iter {iteration}: NO blocks visible after block-level culling!\n"
            f"  Batch size: {len(batched_cameras)}\n"
            f"  Camera IDs: {current_camera_ids}\n"
            f"  Total blocks in scene: {storage_adapter.culler.num_blocks}\n"
            f"  This suggests either:\n"
            f"    1. Far plane too small\n"
            f"    2. Cameras are far outside scene bounds\n"
            f"    3. Block bounds computed incorrectly\n"
        )

        # Include camera positions to diagnose dataset/culling mismatches.
        for i, cam in enumerate(batched_cameras):
            R = np.array(cam.R).reshape(3, 3)
            T = np.array(cam.T).reshape(3, 1)
            cam_pos = (-R.T @ T).flatten()
            log_file.write(f"  Camera {current_camera_ids[i]}: pos={cam_pos}\n")

    # ====================================================================
    # Optional frustum visualization is disabled in the release package.
    # ====================================================================
    if args.debug_frustum and (iteration == 1 or iteration % 500 == 0):
        block_bounds_for_viz = storage_adapter.block_bounds
        visualize_frustum_culling_inline(
            batched_cameras=batched_cameras,
            block_bounds=block_bounds_for_viz,
            visible_block_ids=visible_block_ids,
            iteration=iteration,
            save_dir=os.path.join(args.model_path, 'debug_frustum'),
        )

    # Lightweight visibility summary for python.log.
    if iteration == 1 or iteration % 100 == 0:
        total_blocks = storage_adapter.culler.num_blocks
        vis_ratio = 100.0 * len(visible_block_ids) / total_blocks if total_blocks > 0 else 0.0
        log_file.write(
            f"[SSD] Iter {iteration}: {len(visible_block_ids)}/{total_blocks} blocks visible ({vis_ratio:.1f}%)\n"
        )

    paper_block_sets = None
    paper_plan_future = None
    paper_ab_buffer_source = None
    should_log_paper_sets = iteration == 1 or _perf_log or iteration % 500 == 0
    resident_recency_scores = dict(
        getattr(gaussians, '_paper_resident_recency_scores', {})
    )

    # Validate block IDs against the full-scene count, not the resident count.
    max_valid_block_id = (total_n_gaussians + args.gaussian_block_size - 1) // args.gaussian_block_size - 1
    invalid_blocks = [bid for bid in visible_block_ids if bid > max_valid_block_id or bid < 0]

    if invalid_blocks:
        raise ValueError(
            f"[SSD ERROR] Invalid block IDs found: {invalid_blocks}\n"
            f"Valid range: [0, {max_valid_block_id}]\n"
            f"total_n_gaussians={total_n_gaussians:,}, block_size={args.gaussian_block_size}\n"
            f"Total valid blocks: {max_valid_block_id + 1}"
        )


    _log_ssd_stage1_debug(
        enabled=paper_debug_logging,
        log_file=log_file,
        iteration=iteration,
        n_gaussians=n_gaussians,
        block_size=args.gaussian_block_size,
        max_valid_block_id=max_valid_block_id,
        visible_block_ids=visible_block_ids,
    )

    # Step 3: materialize the resident block set on GPU.
    with torch.no_grad():
        torch.cuda.nvtx.range_push("SSD→RAM→GPU: Load resident blocks")

        # ============================================================
        # Validate that the selected blocks are available before GPU materialization.
        # ============================================================
        if len(visible_block_ids) == 0:
            log_file.write(
                f"\n[CRITICAL ERROR] Iter {iteration}: visible_block_ids is EMPTY!\n"
                f"  This means block-level frustum culling found NO visible blocks.\n"
                f"  Check:\n"
                f"    1. Far plane setting\n"
                f"    2. Camera positions vs scene bounds\n"
                f"    3. Block bounds computation\n"
            )


        gpu_tensors, retention_stats, used_paper_prefetch_buffer, paper_ab_buffer_source = _load_paper_stage1_working_set(
            gaussians=gaussians,
            args=args,
            iteration=iteration,
            total_n_gaussians=total_n_gaussians,
            visible_block_ids=visible_block_ids,
            current_camera_blocks=current_camera_blocks,
            current_bounds_generation=current_bounds_generation,
            current_camera_ids=current_camera_ids,
            training_schedule=training_schedule,
            storage_adapter=storage_adapter,
            should_log=should_log_paper_sets,
            get_double_buffer_gpu_fn=get_double_buffer_gpu,
            ensure_local_to_global_mapping_fn=ensure_local_to_global_mapping,
            resolve_current_iteration_resident_blocks_fn=_resolve_current_iteration_resident_blocks,
            log_file=log_file,
        )

        # Create nn.Parameters for training (gradients will accumulate here)
        gaussians._xyz = nn.Parameter(gpu_tensors['xyz'].requires_grad_(True))
        gaussians._scaling = nn.Parameter(gpu_tensors['scaling'].requires_grad_(True))
        gaussians._rotation = nn.Parameter(gpu_tensors['rotation'].requires_grad_(True))
        gaussians._opacity = nn.Parameter(gpu_tensors['opacity'].requires_grad_(True))
        gaussians._features_dc = nn.Parameter(gpu_tensors['features_dc'].requires_grad_(True))
        gaussians._features_rest = nn.Parameter(gpu_tensors['features_rest'].requires_grad_(True))

        gaussians.gpu_working_set_manager.gpu_xyz = gaussians._xyz
        gaussians.gpu_working_set_manager.gpu_scaling = gaussians._scaling
        gaussians.gpu_working_set_manager.gpu_rotation = gaussians._rotation
        gaussians.gpu_working_set_manager.gpu_opacity = gaussians._opacity
        gaussians.gpu_working_set_manager.gpu_features_dc = gaussians._features_dc
        gaussians.gpu_working_set_manager.gpu_features_rest = gaussians._features_rest

        double_buffer = get_double_buffer_gpu(
            num_total=total_n_gaussians,
            block_size=args.gaussian_block_size,
            device='cuda'
        )
        storage_adapter.bind_resident_writeback(
            double_buffer,
            gaussians.gpu_working_set_manager,
        )
        _seed_paper_active_buffer_from_manager(
            gaussians,
            double_buffer,
            ensure_local_to_global_mapping,
            preserve_resident_metadata=used_paper_prefetch_buffer,
        )
        actual_current_resident_blocks = list(gaussians.gpu_working_set_manager.loaded_blocks)
        gaussians._paper_loaded_resident_blocks = list(actual_current_resident_blocks)
        active_block_reader = getattr(gaussians, '_block_reader', None)
        paper_plan_future = double_buffer.submit_resident_plan(
            iteration,
            _plan_and_start_resident_prefetch,
            storage_adapter=storage_adapter,
            training_schedule=training_schedule,
            iteration=iteration,
            batch_size=bsz,
            current_block_ids=list(visible_block_ids),
            schedule_ordering=getattr(args, 'ssd_schedule_ordering', 'trajectory'),
            current_resident_blocks=list(actual_current_resident_blocks),
            current_resident_recency_scores=dict(resident_recency_scores),
            resident_selection_policy=args.paper_resident_selection_policy,
            resident_lambda=args.paper_resident_lambda,
            resident_recency_decay=args.paper_resident_recency_decay,
            resident_capacity_blocks=args.paper_resident_capacity_blocks,
            balanced_seed_fraction=float(getattr(args, 'paper_balanced_seed_fraction', 1.0)),
            active_block_reader=active_block_reader,
            double_buffer=double_buffer,
        )
        _configure_gpu_resident_optimizer_state(
            gaussians=gaussians,
            args=args,
            actual_current_resident_blocks=actual_current_resident_blocks,
            iteration=iteration,
            get_gpu_resident_optimizer_fn=get_gpu_resident_optimizer,
            log_file=log_file,
        )
        if should_log_paper_sets:
            _log_batch_kt_metrics(
                log_file=log_file,
                iteration=iteration,
                current_camera_blocks=current_camera_blocks,
                visible_block_ids=visible_block_ids,
                resident_blocks=actual_current_resident_blocks,
            )

        # Keep compact global IDs for later bookkeeping. Do not build a
        # full-scene CUDA mask here: at 1B Gaussians a bool mask alone
        # is ~1 GB and forces an O(N) scan before projection.
        local_to_global = ensure_local_to_global_mapping(
            gaussians,
            total_n_gaussians,
            log_file=log_file,
            context=f"stage1_loaded_ids_iter_{iteration}",
        )

        # ============================================================
        # ============================================================
        _log_paper_resident_camera_coverage(
            iteration=iteration,
            current_camera_blocks=current_camera_blocks,
            loaded_blocks=list(gaussians.gpu_working_set_manager.loaded_blocks),
            should_log=should_log_paper_sets,
            log_file=log_file,
        )

        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()

    _ts('stage1_5_ssd_done')

    # ============================================================================
    # STAGE 2: Gaussian-level Culling (精细剔除)
    # ============================================================================
    with torch.no_grad():
        torch.cuda.nvtx.range_push("compact_and_calculate_filters")

        loaded_gaussian_ids = ensure_local_to_global_mapping(
            gaussians,
            total_n_gaussians,
            log_file=log_file,
            context=f"stage2_loaded_ids_iter_{iteration}",
        )

        num_loaded = loaded_gaussian_ids.shape[0]

        empty_loaded_result = _log_empty_stage2_loaded_gaussians(
            num_loaded=num_loaded,
            iteration=iteration,
            visible_block_ids=visible_block_ids,
            bsz=bsz,
            log_file=log_file,
        )
        if empty_loaded_result is not None:
            torch.cuda.nvtx.range_pop()
            return empty_loaded_result

        # ================================================================
        # Parameters are already on GPU in the compact working set.
        # After load_visible_blocks(), gaussians._xyz/etc point to GPU tensors
        # containing ONLY the visible Gaussians (not all N)
        # ================================================================
        # Use compact resident tensors directly; projection gathers per-camera filters below.
        xyz_compact = gaussians.get_xyz          # GPU tensor, shape (M,3) where M = num visible
        opacity_compact = gaussians.get_opacity  # GPU tensor, shape (M,1)
        scaling_compact = gaussians.get_scaling  # GPU tensor, shape (M,3)
        rotation_compact = gaussians.get_rotation  # GPU tensor, shape (M,4)

        _log_paper_gpu_working_set_parameters(
            iteration=iteration,
            xyz_compact=xyz_compact,
            opacity_compact=opacity_compact,
            scaling_compact=scaling_compact,
            rotation_compact=rotation_compact,
            batched_cameras=batched_cameras,
            log_file=log_file,
        )

        # Sanity check
        assert xyz_compact.is_cuda, "[PAPER MODE] xyz should be on GPU in working set!"
        assert opacity_compact.is_cuda, "[PAPER MODE] opacity should be on GPU!"
        assert scaling_compact.is_cuda, "[PAPER MODE] scaling should be on GPU!"
        assert rotation_compact.is_cuda, "[PAPER MODE] rotation should be on GPU!"
        # ==========================================================

        # Run Gaussian-level culling on the compact GPU working set.
        # Input: Visible Gaussians from blocks (M Gaussians)
        # Output: Subset of M that passes per-camera culling
        # Optional debug-frustum diagnostics before calculate_filters.
        if iteration == 1 and getattr(args, "debug_frustum", False):
            print(f"\n[PRE-PROJECTION DEBUG] Gaussians before calculate_filters:")
            print(f"  Number of Gaussians: {xyz_compact.shape[0]}")
            print(f"  XYZ range: min={xyz_compact.min(dim=0).values.cpu().numpy()}, "
                  f"max={xyz_compact.max(dim=0).values.cpu().numpy()}")
            print(f"  Opacity range: min={opacity_compact.min().item():.4f}, "
                  f"max={opacity_compact.max().item():.4f}, "
                  f"mean={opacity_compact.mean().item():.4f}")
            print(f"  Scaling range: min={scaling_compact.min().item():.4f}, "
                  f"max={scaling_compact.max().item():.4f}, "
                  f"mean={scaling_compact.mean().item():.4f}")

            # Check how many have valid opacity
            valid_opacity = (opacity_compact > 0.01).sum().item()
            print(f"  Gaussians with opacity > 0.01: {valid_opacity}/{xyz_compact.shape[0]}")

            # Check camera positions relative to Gaussians
            cam0_pos = batched_cameras[0].camera_center.cpu().numpy()
            print(f"  Camera[0] position: {cam0_pos}")

            # Check if cameras are inside/outside Gaussian bounds
            xyz_min = xyz_compact.min(dim=0).values.cpu().numpy()
            xyz_max = xyz_compact.max(dim=0).values.cpu().numpy()
            inside = (cam0_pos >= xyz_min).all() and (cam0_pos <= xyz_max).all()
            print(f"  Camera[0] inside Gaussian bounds: {inside}")

        filters_compact, _, _ = calculate_filters(
            batched_cameras,
            xyz_compact,
            opacity_compact,
            scaling_compact,
            rotation_compact,
        )

        # ====================================================================
        # Optional debug-frustum diagnostics after calculate_filters.
        if iteration == 1 and getattr(args, "debug_frustum", False):
            num_visible_per_cam = [len(f) for f in filters_compact]
            print(f"\n[POST-PROJECTION DEBUG] After calculate_filters:")
            print(f"  Gaussians per camera (first 10): {num_visible_per_cam[:10]}")
            print(f"  Total visible entries: {sum(num_visible_per_cam)} / {xyz_compact.shape[0]} unique Gaussians")

            # Check for duplicates in ALL cameras with visible Gaussians
            print(f"\n  Per-camera duplicate analysis:")
            for cam_idx in range(min(10, len(filters_compact))):  # Check first 10 cameras
                if num_visible_per_cam[cam_idx] > 0:
                    unique_count = filters_compact[cam_idx].unique().numel()
                    dup_ratio = num_visible_per_cam[cam_idx] / unique_count
                    print(f"    Camera {cam_idx}: {num_visible_per_cam[cam_idx]} entries, {unique_count} unique → {dup_ratio:.2f}x duplication")
                else:
                    print(f"    Camera {cam_idx}: 0 Gaussians visible")

            # Count cameras with/without Gaussians
            empty_cams = sum(1 for n in num_visible_per_cam if n == 0)
            non_empty_cams = len(num_visible_per_cam) - empty_cams

            print(f"\n  Summary:")
            print(f"    Cameras with Gaussians: {non_empty_cams}/{len(num_visible_per_cam)}")
            print(f"    Cameras without Gaussians: {empty_cams}/{len(num_visible_per_cam)}")

            if empty_cams > 0:
                print(f"\n  ⚠️  {empty_cams}/{len(batched_cameras)} cameras see ZERO Gaussians after projection!")
                print(f"  Possible causes:")
                print(f"    1. Opacity too low (check mean opacity above)")
                print(f"    2. Scale too small (projected radius < radius_clip={args.radius_clip})")
                print(f"    3. Depth culling (Gaussians outside near/far planes)")
                print(f"    4. Camera-Gaussian distance too large")

        # ================================================================
        # Map compact filter indices to global IDs.
        # ================================================================
        # filters_compact contains LOCAL indices (0 to M-1) for GPU working set
        # filters_global contains GLOBAL indices (0 to N-1) for gradient accumulation
        #
        # We need both index spaces:
        # - filters_local: for indexing gaussians._xyz (GPU working set, size M)
        # - filters_global: for gradient accumulation to full parameter tensors (size N)
        local_to_global = ensure_local_to_global_mapping(
            gaussians,
            total_n_gaussians,
            log_file=log_file,
            context=f"projection_filter_map_iter_{iteration}",
        )

        filters_local, filters_global = _map_compact_filters_to_global(
            filters_compact=filters_compact,
            local_to_global=local_to_global,
        )

        # Store both for later use
        # filters_local: used in STAGE 4 for gaussians._xyz[this_filter_local]
        # filters_global: used for gradient scatter_add_ to full tensors
        gaussians.gpu_working_set_manager.filters_local = filters_local
        filters = filters_global
        _log_empty_paper_projection_cameras(
            iteration=iteration,
            current_camera_ids=current_camera_ids,
            filters_global=filters_global,
            log_file=log_file,
        )

        camera_ids = None
        gaussian_ids = None

        # Cleanup - but keep filters_compact info in filters_local
        del xyz_compact, opacity_compact, scaling_compact, rotation_compact, filters_compact

        torch.cuda.nvtx.range_pop()

    # ========================================================================
    # [STATISTICS] Block Visibility Analysis (controlled by --log_block_visibility)
    # ========================================================================
    if args.log_block_visibility:
        with torch.no_grad():
            # Calculate total loaded Gaussians and visible Gaussians
            if gaussians.gpu_working_set_manager.local_to_global_idx is not None:
                loaded_count = len(gaussians.gpu_working_set_manager.local_to_global_idx)
            else:
                loaded_count = total_n_gaussians

            # Count visible Gaussians across all cameras
            visible_count = sum(len(f) for f in filters if f is not None)

            # Calculate per-block visibility
            if loaded_count > 0:
                visibility_ratio = visible_count / loaded_count

                # Detailed per-block analysis (expensive, only do every N iterations)
                if iteration % 100 == 0:
                    # Get all visible Gaussian IDs
                    all_visible_ids = []
                    for f in filters:
                        if f is not None and f.numel() > 0:
                            all_visible_ids.append(f)

                    if len(all_visible_ids) > 0:
                        all_visible_ids = torch.cat(all_visible_ids).unique()

                        # Calculate which blocks have visible Gaussians
                        visible_block_ids_set = set((all_visible_ids // args.gaussian_block_size).cpu().tolist())

                        # Get loaded blocks
                        loaded_block_ids_set = set(gaussians.gpu_working_set_manager.loaded_blocks)

                        # Calculate per-block statistics
                        block_visibility_stats = {}
                        for block_id in loaded_block_ids_set:
                            start_idx = block_id * args.gaussian_block_size
                            end_idx = min(start_idx + args.gaussian_block_size, total_n_gaussians)
                            block_size = end_idx - start_idx

                            # Count visible Gaussians in this block
                            block_visible = ((all_visible_ids >= start_idx) & (all_visible_ids < end_idx)).sum().item()
                            block_ratio = block_visible / block_size if block_size > 0 else 0

                            block_visibility_stats[block_id] = {
                                'total': block_size,
                                'visible': block_visible,
                                'ratio': block_ratio
                            }

                        # Aggregate statistics
                        if block_visibility_stats:
                            ratios = [s['ratio'] for s in block_visibility_stats.values()]
                            avg_ratio = sum(ratios) / len(ratios)
                            min_ratio = min(ratios)
                            max_ratio = max(ratios)

                            # Categorize blocks
                            high_visibility = sum(1 for r in ratios if r > 0.7)
                            medium_visibility = sum(1 for r in ratios if 0.3 <= r <= 0.7)
                            low_visibility = sum(1 for r in ratios if r < 0.3)

                            log_file.write(
                                f"\n[BLOCK VISIBILITY STATS] Iter {iteration}:\n"
                                f"  Total Loaded Gaussians: {loaded_count:,}\n"
                                f"  Total Visible Gaussians: {visible_count:,}\n"
                                f"  Overall Visibility Ratio: {visibility_ratio*100:.1f}%\n"
                                f"  \n"
                                f"  Loaded Blocks: {len(loaded_block_ids_set)}\n"
                                f"  Blocks with Visible Gaussians: {len(visible_block_ids_set)}\n"
                                f"  \n"
                                f"  Per-Block Visibility:\n"
                                f"    Average: {avg_ratio*100:.1f}%\n"
                                f"    Min: {min_ratio*100:.1f}%\n"
                                f"    Max: {max_ratio*100:.1f}%\n"
                                f"  \n"
                                f"  Block Categories:\n"
                                f"    High (>70%): {high_visibility} blocks\n"
                                f"    Medium (30-70%): {medium_visibility} blocks\n"
                                f"    Low (<30%): {low_visibility} blocks\n"
                                f"  \n"
                            )

                            # Print to console for quick monitoring
                            print(f"[Iter {iteration}] Block Visibility: Overall={visibility_ratio*100:.1f}%, "
                                  f"PerBlock={avg_ratio*100:.1f}% (High={high_visibility}, Med={medium_visibility}, Low={low_visibility})")
                else:
                    # Quick stats every iteration (lightweight)
                    log_file.write(
                        f"[BLOCK VISIBILITY] Iter {iteration}: "
                        f"Loaded={loaded_count:,}, Visible={visible_count:,}, "
                        f"Ratio={visibility_ratio*100:.1f}%\n"
                    )

    torch.cuda.nvtx.range_push("sort cameras")
    ordered_cams = list(range(len(batched_cameras)))
    sparsity = [len(f) / float(total_n_gaussians) for f in filters]
    if should_log_paper_sets:
        _log_paper_order_calculation_skip(iteration=iteration, log_file=log_file)
    torch.cuda.nvtx.range_pop()

    microbatch_idx = 0

    # ========================================================================
    # Get local filter indices for GPU working set indexing.
    # ========================================================================
    filters_local = gaussians.gpu_working_set_manager.filters_local

    # ============================================================================
    # STAGE 3: TRAINING STATE INITIALIZATION
    # ============================================================================
    # In paper SSD mode, allocating .grad for the entire GPU working set can still
    # be too large at 1.1B scale. We therefore accumulate only the Gaussians that
    # are actually touched by this batch and snapshot only that sparse subset.
    use_sparse_gpu_grad_accum = (
        filters_local is not None
    )
    sparse_grad_local_ids = None
    sparse_grad_components = None
    sparse_visibility_indices = None

    if use_sparse_gpu_grad_accum:
        with torch.cuda.stream(comm_stream), torch.no_grad():
            touched_local_mask = torch.zeros((gaussians._xyz.shape[0],), dtype=torch.bool, device='cuda')
            for filt_local in filters_local:
                if filt_local is not None and filt_local.numel() > 0:
                    touched_local_mask.scatter_(0, filt_local, True)
            sparse_grad_local_ids = torch.nonzero(touched_local_mask, as_tuple=False).flatten()
            touched_count = int(sparse_grad_local_ids.numel())
            local_to_global = ensure_local_to_global_mapping(
                gaussians,
                total_n_gaussians,
                log_file=log_file,
                context=f"sparse_grad_accum_iter_{iteration}",
            )
            sparse_visibility_indices = local_to_global[sparse_grad_local_ids].cpu()
            sparse_grad_components = {
                'xyz': torch.zeros((touched_count, 3), device='cuda'),
                'opacity': torch.zeros((touched_count, 1), device='cuda'),
                'scaling': torch.zeros((touched_count, 3), device='cuda'),
                'rotation': torch.zeros((touched_count, 4), device='cuda'),
                'features_dc': torch.zeros((touched_count, 3), device='cuda'),
                'features_rest': torch.zeros((touched_count, 45), device='cuda'),
            }
            del touched_local_mask
        utils.print_rank_0(
            f"[SSD GRAD] Touched-only GPU grad buffers: {touched_count}/{gaussians._xyz.shape[0]} working-set Gaussians"
        )
    else:
        gaussians._xyz.grad = torch.zeros_like(gaussians._xyz)
        gaussians._opacity.grad = torch.zeros_like(gaussians._opacity)
        gaussians._scaling.grad = torch.zeros_like(gaussians._scaling)
        gaussians._rotation.grad = torch.zeros_like(gaussians._rotation)

        gaussians._features_dc.grad = torch.zeros_like(gaussians._features_dc)
        gaussians._features_rest.grad = torch.zeros_like(gaussians._features_rest)

    # Stream management: default_stream for compute, comm_stream for CPU<->GPU transfers
    default_stream = torch.cuda.current_stream()

    # Training loop variables
    num_micro_batches = len(batched_cameras)
    losses = []
    # ============================================================================
    _ts('stage2_3_culling_done')

    # STAGE 4: MAIN MICRO-BATCH TRAINING LOOP
    # ============================================================================
    for micro_idx in range(num_micro_batches):
        torch.cuda.nvtx.range_push("micro_batch_idx: " + str(micro_idx))

        # ====================================================================
        # Use correct filter indices based on mode.
        # ====================================================================
        # this_filter_global: Global Gaussian IDs (0 to N-1) - for gradient accumulation
        # this_filter_local: Local working set indices (0 to M-1) - for GPU tensor indexing
        this_filter_global = filters[micro_idx]  # Always global IDs

        if filters_local is not None:
            this_filter_local = filters_local[micro_idx]  # Local indices for GPU working set
        else:
            this_filter_local = this_filter_global  # In non-SSD mode, they're the same

        this_filter_len = this_filter_global.shape[0]

        # ========================================================================
        # Skip cameras whose coarse block visibility has no Gaussian-level hits.
        # ========================================================================
        if this_filter_len == 0:
            _log_empty_microbatch_camera(
                micro_idx=micro_idx,
                log_file=log_file,
            )

            microbatch_idx += 1  # Increment even when skipping
            torch.cuda.nvtx.range_pop()  # Close micro_batch_idx range
            continue

        # ------------------------------------------------------------------------
        # 4.1: Load current SH coefficients
        # ------------------------------------------------------------------------
        if micro_idx == 0:
            # ====================================================================
            # Use direct GPU indexing if features are on GPU.
            # ====================================================================
            torch.cuda.nvtx.range_push("gpu_mode_sh_indexing")

            with torch.no_grad():
                # Directly gather SH features from GPU tensors
                # CRITICAL: Use this_filter_local for GPU working set indexing!
                shs_dc = gaussians._features_dc[this_filter_local]  # (K, 3)
                shs_rest = gaussians._features_rest[this_filter_local]  # (K, 45)

                # Concatenate into single tensor
                shs = torch.cat([shs_dc, shs_rest], dim=1)  # (K, 48)
                shs.requires_grad_(True)


                # Create dummy event for compatibility
                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record()

                log_file = utils.get_log_file()
                _log_paper_sh_indexed(
                    iteration=iteration,
                    filter_len=this_filter_len,
                    log_file=log_file,
                )

            torch.cuda.nvtx.range_pop()

        else:
            # ====================================================================
            # Subsequent micro-batches: different strategies based on mode
            # ====================================================================
            with torch.no_grad():
                # CRITICAL: Use this_filter_local for GPU working set indexing!
                shs_dc = gaussians._features_dc[this_filter_local]
                shs_rest = gaussians._features_rest[this_filter_local]
                shs = torch.cat([shs_dc, shs_rest], dim=1)
                shs.requires_grad_(True)

                cpu2gpu_event = torch.cuda.Event(enable_timing=True)
                cpu2gpu_event.record()

        # ------------------------------------------------------------------------
        # 4.2: Prefetch NEXT micro-batch SH coefficients (overlapped with compute)
        # Only needed by archived CPU→GPU split-feature transfer.
        # ------------------------------------------------------------------------

        # ------------------------------------------------------------------------
        # 4.3: Forward pass - Render image with filtered gaussian parameters
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("forward_pass")
        torch.cuda.nvtx.range_push("prepare filtered parameters")

        # ====================================================================
        # Gather filtered parameters from the compact GPU working set. Local
        # indices address resident tensors; global indices address the full model.
        # Clones create independent tensors whose gradients are scattered back.
        filtered_xyz_gpu = gaussians._xyz[this_filter_local].clone().requires_grad_(True)
        _filtered_opacity_gpu = gaussians._opacity[this_filter_local].clone().requires_grad_(True)
        _filtered_scaling_gpu = gaussians._scaling[this_filter_local].clone().requires_grad_(True)
        _filtered_rotation_gpu = gaussians._rotation[this_filter_local].clone().requires_grad_(True)

        # Retain gradients for cloned non-leaf tensors until scatter-back.
        filtered_xyz_gpu.retain_grad() # retain_grad(): 虽然不是leaf node，但反向传播结束后, 不要销毁它的梯度，把它留在显存中等待使用
        _filtered_opacity_gpu.retain_grad()
        _filtered_scaling_gpu.retain_grad()
        _filtered_rotation_gpu.retain_grad()
        # Apply activation functions to constrain parameter ranges
        filtered_opacity_gpu = gaussians.opacity_activation(_filtered_opacity_gpu)
        filtered_scaling_gpu = gaussians.scaling_activation(_filtered_scaling_gpu)
        filtered_rotation_gpu = gaussians.rotation_activation(_filtered_rotation_gpu)

        torch.cuda.nvtx.range_pop()

        # Wait for SH coefficients. In the GPU-resident path this event has
        # already been recorded after direct GPU indexing.
        cpu2gpu_event.wait(default_stream)

        # ====================================================================
        # Handle SH gradient tracking based on mode.
        # ====================================================================
        filtered_shs = shs.requires_grad_(True)

        # Render image using filtered Gaussian splatting.
        (
            rendered_image,
            batched_means2D,
            batched_radiis,
            batched_colors_detached,
            dirs,
        ) = pipeline_forward_one_step_shs_inplace(
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_xyz_gpu,
            filtered_shs,
            batched_cameras[micro_idx],
            scene,
            gaussians,
            background,
            pipe_args,
        )

        # Compute loss
        loss = torch_compiled_loss(
            rendered_image, batched_cameras[micro_idx].original_image
        )
        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.4: Backward pass - Compute gradients
        # ------------------------------------------------------------------------
        torch.cuda.nvtx.range_push("backward_pass")

        # ====================================================================
        # Optional first-iteration gradient sanity log.
        # ====================================================================
        log_file = utils.get_log_file()
        if iteration == 1 and micro_idx == 0:
            log_file.write(f"\n[GRAD DEBUG] Before backward (iter {iteration}, micro {micro_idx}):\n")
            log_file.write(f"  filtered_xyz_gpu.requires_grad: {filtered_xyz_gpu.requires_grad}\n")
            log_file.write(f"  filtered_xyz_gpu.is_leaf: {filtered_xyz_gpu.is_leaf}\n")
            log_file.write(f"  filtered_xyz_gpu.grad_fn: {filtered_xyz_gpu.grad_fn}\n")
            log_file.write(f"  _filtered_opacity_gpu.requires_grad: {_filtered_opacity_gpu.requires_grad}\n")
            log_file.write(f"  _filtered_opacity_gpu.grad_fn: {_filtered_opacity_gpu.grad_fn}\n")
            log_file.write(f"  filtered_opacity_gpu.requires_grad: {filtered_opacity_gpu.requires_grad}\n")
            log_file.write(f"  filtered_opacity_gpu.grad_fn: {filtered_opacity_gpu.grad_fn}\n")
            log_file.write(f"  rendered_image.requires_grad: {rendered_image.requires_grad}\n")
            log_file.write(f"  loss.requires_grad: {loss.requires_grad}\n")
            log_file.write(f"  loss.grad_fn: {loss.grad_fn}\n")

        # ====================================================================
        # SSD-backed path: one backward is enough; autograd handles gradients.
        # Archived split-feature path: two backward phases handle CPU SH features.
        loss.backward()

        # ====================================================================
        # Optional first-iteration gradient sanity log.
        # ====================================================================
        if iteration == 1 and micro_idx == 0:
            log_file.write(f"\n[GRAD DEBUG] After backward:\n")
            log_file.write(f"  filtered_xyz_gpu.grad is None: {filtered_xyz_gpu.grad is None}\n")
            if filtered_xyz_gpu.grad is not None:
                log_file.write(f"  filtered_xyz_gpu.grad.shape: {filtered_xyz_gpu.grad.shape}\n")
                log_file.write(f"  filtered_xyz_gpu.grad.sum(): {filtered_xyz_gpu.grad.sum().item()}\n")
            log_file.write(f"  _filtered_opacity_gpu.grad is None: {_filtered_opacity_gpu.grad is None}\n")
            if _filtered_opacity_gpu.grad is not None:
                log_file.write(f"  _filtered_opacity_gpu.grad.sum(): {_filtered_opacity_gpu.grad.sum().item()}\n")
            log_file.write("  filtered_opacity_gpu.grad: skipped (non-leaf debug tensor)\n")
            log_file.write("  batched_colors_detached.grad: skipped (non-leaf debug tensor)\n")
            log_file.write("  dirs.grad: skipped (non-leaf debug tensor)\n")

        # ====================================================================
        # Conditional SH gradient computation.
        # ====================================================================
        if iteration == 1 and micro_idx == 0:
            mode_name = 'PAPER GPU WORKING SET'
            _write_paper_phase1_log(
                f"[{mode_name}] Autograd backward completed (no manual computation)\n",
                log_file=log_file,
            )
            log_file.write(f"  filtered_xyz_gpu.grad is not None: {filtered_xyz_gpu.grad is not None}\n")
            if filtered_xyz_gpu.grad is not None:
                log_file.write(f"  filtered_xyz_gpu.grad.sum(): {filtered_xyz_gpu.grad.sum().item()}\n")

        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.5: Accumulate gradients back to full parameter tensors
        # ------------------------------------------------------------------------
        with torch.no_grad():
            torch.cuda.nvtx.range_push("scatter gpu grads back to origin")

            # ================================================================
            # Optional first-iteration scatter sanity log.
            # ================================================================
            if iteration == 1 and micro_idx == 0:
                log_file.write(f"\n[GRAD DEBUG] Before scatter_add_:\n")
                log_file.write(f"  filtered_xyz_gpu.grad: {filtered_xyz_gpu.grad}\n")
                log_file.write(f"  _filtered_opacity_gpu.grad: {_filtered_opacity_gpu.grad}\n")
                log_file.write(f"  _filtered_scaling_gpu.grad: {_filtered_scaling_gpu.grad}\n")
                log_file.write(f"  _filtered_rotation_gpu.grad: {_filtered_rotation_gpu.grad}\n")

            if filtered_xyz_gpu.grad is None:
                log_file.write(f"  [ERROR] filtered_xyz_gpu.grad is None!\n")
                log_file.write(f"  Checking computational graph:\n")
                log_file.write(f"    filtered_xyz_gpu in computation: {filtered_xyz_gpu in loss.grad_fn if hasattr(loss, 'grad_fn') else 'N/A'}\n")
                raise RuntimeError("filtered_xyz_gpu.grad is None - gradient not computed!")

            # ================================================================
            # Scatter operates entirely on GPU.
            # ================================================================
            # In SSD-backed mode, all active parameters are in the GPU working set (size M).
            # Use this_filter_local (0 to M-1) for scatter_add_ indices!
            # 
            # In paper SSD mode we accumulate only the Gaussians
            # touched by this batch, not the entire GPU working set.
            if use_sparse_gpu_grad_accum:
                sparse_filter_idx = torch.searchsorted(sparse_grad_local_ids, this_filter_local)
                sparse_grad_components['xyz'].scatter_add_(
                    dim=0,
                    src=filtered_xyz_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                )
                sparse_grad_components['opacity'].scatter_add_(
                    dim=0,
                    src=_filtered_opacity_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1),
                )
                sparse_grad_components['scaling'].scatter_add_(
                    dim=0,
                    src=_filtered_scaling_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                )
                sparse_grad_components['rotation'].scatter_add_(
                    dim=0,
                    src=_filtered_rotation_gpu.grad,
                    index=sparse_filter_idx.reshape(-1, 1).expand(-1, 4),
                )

                if filtered_shs.grad is not None:
                    shs_dc_grad = filtered_shs.grad[:, :3]
                    shs_rest_grad = filtered_shs.grad[:, 3:48]
                    sparse_grad_components['features_dc'].scatter_add_(
                        dim=0,
                        src=shs_dc_grad,
                        index=sparse_filter_idx.reshape(-1, 1).expand(-1, 3),
                    )
                    sparse_grad_components['features_rest'].scatter_add_(
                        dim=0,
                        src=shs_rest_grad,
                        index=sparse_filter_idx.reshape(-1, 1).expand(-1, 45),
                    )
            else:
                gaussians._xyz.grad.scatter_add_(
                    dim=0,
                    src=filtered_xyz_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                )
                gaussians._opacity.grad.scatter_add_(
                    dim=0, src=_filtered_opacity_gpu.grad, index=this_filter_local.reshape(-1, 1)
                )
                gaussians._scaling.grad.scatter_add_(
                    dim=0,
                    src=_filtered_scaling_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                )
                gaussians._rotation.grad.scatter_add_(
                    dim=0,
                    src=_filtered_rotation_gpu.grad,
                    index=this_filter_local.reshape(-1, 1).expand(-1, 4),
                )

                if filtered_shs.grad is not None:
                    shs_dc_grad = filtered_shs.grad[:, :3]
                    shs_rest_grad = filtered_shs.grad[:, 3:48]
                    gaussians._features_dc.grad.scatter_add_(
                        dim=0,
                        src=shs_dc_grad,
                        index=this_filter_local.reshape(-1, 1).expand(-1, 3),
                    )
                    gaussians._features_rest.grad.scatter_add_(
                        dim=0,
                        src=shs_rest_grad,
                        index=this_filter_local.reshape(-1, 1).expand(-1, 45),
                    )

            torch.cuda.nvtx.range_pop()

        # Cleanup temporary tensors
        del rendered_image, batched_colors_detached, dirs
        shs = None
        del (
            filtered_xyz_gpu,
            filtered_opacity_gpu,
            filtered_scaling_gpu,
            filtered_rotation_gpu,
            filtered_shs,
        )
        del _filtered_opacity_gpu, _filtered_scaling_gpu, _filtered_rotation_gpu

        losses.append(loss.detach())
        del loss

        microbatch_idx += 1
        if micro_idx == num_micro_batches - 1:
            gaussians.block_cache_state["last_shs"] = None
            gaussians.block_cache_state["last_filter"] = None
            gaussians.block_cache_state["last_retention_vec"] = None
            if iteration == 1 or _perf_log:
                _log_paper_interbatch_sh_cache_disabled(
                    iteration=iteration,
                    log_file=log_file,
                )

        # ====================================================================
        # [GPU WORKING SET] NO gradient sync needed in GPU Adam mode
        # ====================================================================
        # In GPU Adam mode, gradients are used directly on GPU for optimizer.step()
        # No need to sync to RAM's parameters_grad_buffer in SSD-backed mode.
        # Gradient sync has been REMOVED - it was redundant

        torch.cuda.nvtx.range_pop()

        # ------------------------------------------------------------------------
        # 4.7: Update densification statistics (for adaptive gaussian control)
        # ------------------------------------------------------------------------
        update_densification_stats_offload_accum_grads(
            scene,
            gaussians,
            int(utils.get_img_height()),
            int(utils.get_img_width()),
            filters[micro_idx],
            batched_means2D.grad.squeeze(0),
            batched_radiis.squeeze(0),
        )

        batched_means2D.grad = None
        del batched_means2D, batched_radiis

    _ts('stage4_train_done')

    # ============================================================================
    # STAGE 5: POST-TRAINING OPTIMIZATION & CLEANUP
    # ============================================================================
    _ts('stage5_optim_start')

    optimizer_updated_global_indices = None
    optimizer_updated_block_ids = None

    assert microbatch_idx == bsz, f"microbatch_idx should be equal to bsz. Got {microbatch_idx} vs {bsz}"

    # ------------------------------------------------------------------------
    # 5.1: Optimizer step (mode-dependent)
    # ------------------------------------------------------------------------


    if paper_optimizer_backend != 'gpu_resident':
        raise RuntimeError(
            "Paper SSD release path only supports GPUResidentAdam. "
            "Set --paper_optimizer_backend gpu_resident."
        )
    optimizer_step_stats = _run_gpu_resident_adam_step(
        gaussians=gaussians,
        args=args,
        iteration=iteration,
        total_n_gaussians=total_n_gaussians,
        sparse_visibility_indices=sparse_visibility_indices,
        sparse_grad_local_ids=sparse_grad_local_ids,
        sparse_grad_components=sparse_grad_components,
        get_gpu_resident_optimizer_fn=get_gpu_resident_optimizer,
        ensure_local_to_global_mapping_fn=ensure_local_to_global_mapping,
        log_prefix="[PAPER SSD MODE]",
        log_file=log_file,
    )
    optimizer_updated_block_ids = optimizer_step_stats.get("updated_block_ids")

    utils.memory_report("after optimizer step")
    _ts('stage5_optim_done')

    # ============================================================================
    # [GPU WORKING SET] Update RAM cache and cleanup
    # ============================================================================
    # Step 3: Restore original parameter references (ALL parameters)
    gaussians._xyz = original_xyz
    gaussians._scaling = original_scaling
    gaussians._rotation = original_rotation
    gaussians._opacity = original_opacity
    gaussians._features_dc = original_features_dc
    gaussians._features_rest = original_features_rest

    log_file = utils.get_log_file()
    _log_paper_cpu_source_refs_restored(log_file=log_file)

    # ============================================================================
    # [SSD WRITEBACK] Persist updated parameters to SSD after optimizer step
    # ============================================================================

    with torch.no_grad():
        torch.cuda.nvtx.range_push("SSD: writeback updated blocks")

        with torch.cuda.nvtx.range("Tide writeback: updated block selection"):
            updated_block_ids = _collect_paper_updated_block_ids(
                iteration=iteration,
                optimizer_updated_block_ids=optimizer_updated_block_ids,
                optimizer_updated_global_indices=optimizer_updated_global_indices,
                filters=filters,
                block_size=args.gaussian_block_size,
                total_n_gaussians=total_n_gaussians,
                collect_from_indices_fn=_collect_updated_block_ids_from_indices,
                collect_from_filters_fn=_collect_updated_block_ids,
                should_log=should_log_paper_sets,
                log_file=log_file,
            )
        double_buffer = get_double_buffer_gpu(
            num_total=total_n_gaussians,
            block_size=args.gaussian_block_size,
            device='cuda',
        )
        with torch.cuda.nvtx.range("Tide writeback: mark dirty blocks"):
            double_buffer.mark_dirty_blocks(updated_block_ids)

        with torch.cuda.nvtx.range("Tide writeback: stage block bounds"):
            pending_bounds = gaussians.gpu_working_set_manager.stage_block_bounds(
                updated_block_ids
            )
            if pending_bounds is not None:
                storage_adapter.submit_bounds_refresh(pending_bounds)

        if paper_plan_future is None:
            raise RuntimeError("Tide resident plan was not submitted")
        with torch.cuda.nvtx.range("Tide writeback: await resident plan"):
            paper_plan_result = paper_plan_future.result()
        with torch.cuda.nvtx.range("Tide writeback: publish resident plan"):
            paper_block_sets = paper_plan_result["block_sets"]
            gaussians._paper_expected_resident_blocks = list(
                paper_block_sets['next_resident_blocks']
            )
            gaussians._paper_resident_recency_scores = dict(
                paper_block_sets.get('updated_recency_scores', {})
            )
            gaussians._paper_plan_bounds_generation = int(
                paper_plan_result["bounds_generation"]
            )
            gaussians._paper_plan_camera_ids = list(paper_plan_result["camera_ids"])

        if should_log_paper_sets:
            _log_paper_block_sets(
                log_file=log_file,
                iteration=iteration,
                current_camera_ids=current_camera_ids,
                block_sets=paper_block_sets,
            )
            stream_in_for_next = list(
                paper_block_sets.get('stream_in_blocks', [])
            )
            _log_early_delta_hint(
                storage_adapter=storage_adapter,
                iteration=iteration,
                submitted=int(paper_plan_result["future_submitted"]),
                requested=len(stream_in_for_next),
                log_file=log_file,
            )
            _log_delta_stream_prefetch(
                iteration=iteration,
                paper_block_sets=paper_block_sets,
                future_submitted=int(paper_plan_result["future_submitted"]),
                future_submitted_late=0,
                storage_adapter=storage_adapter,
                log_file=log_file,
            )
            _log_paper_working_set_loaded(
                iteration=iteration,
                loaded_blocks=list(gaussians.gpu_working_set_manager.loaded_blocks),
                paper_block_sets=paper_block_sets,
                retention_stats=retention_stats,
                load_source=paper_ab_buffer_source or retention_stats.get('source', 'sync_ram_to_gpu'),
                log_file=log_file,
            )
            _log_paper_warm_layer_metrics(
                storage_adapter,
                iteration,
                'prefetch',
                log_file=log_file,
            )

        keep_resident_blocks = list(paper_block_sets['keep_resident_blocks']) if paper_block_sets is not None else []
        evicted_blocks = list(paper_block_sets['evict_blocks']) if paper_block_sets is not None else []
        with torch.cuda.nvtx.range("Tide writeback: dirty eviction selection"):
            writeback_block_ids = double_buffer.dirty_blocks_for_eviction(evicted_blocks)
        with torch.cuda.nvtx.range("Tide writeback: stage eviction payload"):
            staged_writeback_blocks, ready_omega, copied_omega, candidate_blocks = _apply_paper_writeback_payload(
                storage_adapter=storage_adapter,
                double_buffer=double_buffer,
                updated_block_ids=writeback_block_ids,
                omega_blocks=keep_resident_blocks,
            )
        if staged_writeback_blocks != len(writeback_block_ids):
            raise RuntimeError(
                "Dirty eviction writeback was incomplete: "
                f"expected={len(writeback_block_ids)} staged={staged_writeback_blocks}"
            )

        if ready_omega > 0:
            _log_delta_handoff(
                iteration=iteration,
                ready_omega=ready_omega,
                copied_omega=copied_omega,
                context='immediate writeback barrier',
                log_file=log_file,
            )

        _log_paper_writeback_staged(
            iteration=iteration,
            staged_writeback_blocks=staged_writeback_blocks,
            log_file=log_file,
        )

        if should_log_paper_sets:
            _log_paper_writeback_finalize(
                storage_adapter=storage_adapter,
                iteration=iteration,
                optimizer_deferred_mode=paper_optimizer_deferred_mode,
                candidate_blocks=candidate_blocks,
                staged_writeback_blocks=staged_writeback_blocks,
                ready_omega=ready_omega,
                copied_omega=copied_omega,
                log_file=log_file,
            )


        torch.cuda.nvtx.range_pop()

    with torch.cuda.nvtx.range("Tide engine tail: retain working set"):
        gaussians.gpu_working_set_manager.prepare_for_retention()
        _log_paper_working_set_retained(log_file=log_file)

    _ts('stage5_writeback_done')
    # ------------------------------------------------------------------------
    # 5.3: Final synchronization and return
    # ------------------------------------------------------------------------
    _ts('stage5_sync_done')

    # ============================================================================
    # [MEMORY CLEANUP] Periodic cleanup to prevent memory accumulation
    # ============================================================================
    # Clear intermediate variables to help garbage collection
    with torch.cuda.nvtx.range("Tide engine tail: memory maintenance"):
        if iteration % 100 == 0:
            # Force Python garbage collection periodically
            gc.collect()

            # Clear CUDA cache if GPU memory pressure is high
            if torch.cuda.is_available():
                gpu_mem_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() if torch.cuda.max_memory_allocated() > 0 else 0
                if gpu_mem_percent > 0.9:
                    torch.cuda.empty_cache()

    # ============================================================================
    # [STATS] Log double buffer and prefetch statistics periodically
    # ============================================================================
    with torch.cuda.nvtx.range("Tide engine tail: metrics and logging"):
        if iteration % 1000 == 0:
            global _double_buffer_gpu
            _log_double_buffer_stats(
                iteration=iteration,
                double_buffer=_double_buffer_gpu,
                log_file=log_file,
            )

        # [PERF PROFILE] Print stage timings
        _ts('iter_end')
        if _perf_log:
            _log_paper_perf_profile(
                iteration=iteration,
                perf_times=_perf_t,
                log_file=log_file,
                storage_adapter=storage_adapter,
            )
            log_file.flush()

    return losses, ordered_cams, sparsity


def clm_offload_eval_one_cam(camera, gaussians, background, scene):
    # Prepare parameters.
    xyz_gpu = gaussians.get_xyz
    opacity_gpu_origin = gaussians.get_opacity
    scaling_gpu_origin = gaussians.get_scaling
    rotation_gpu_origin = gaussians.get_rotation

    filters, _, _ = calculate_filters(
        [camera], xyz_gpu, opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
    )

    del opacity_gpu_origin, scaling_gpu_origin, rotation_gpu_origin
    this_filter = filters[0]

    filtered_xyz_gpu = torch.gather(
        xyz_gpu, 0, this_filter.reshape(-1, 1).expand(-1, 3)
    )
    filtered_opacity_gpu = torch.gather(
        gaussians._opacity, 0, this_filter.reshape(-1, 1)
    )
    filtered_scaling_gpu = torch.gather(
        gaussians._scaling, 0, this_filter.reshape(-1, 1).expand(-1, 3)
    )
    filtered_rotation_gpu = torch.gather(
        gaussians._rotation, 0, this_filter.reshape(-1, 1).expand(-1, 4)
    )

    filtered_opacity_gpu = gaussians.opacity_activation(filtered_opacity_gpu)
    filtered_scaling_gpu = gaussians.scaling_activation(filtered_scaling_gpu)
    filtered_rotation_gpu = gaussians.rotation_activation(filtered_rotation_gpu)

    this_filter_cpu = this_filter.to("cpu")
    filtered_shs_gpu = torch.gather(
        gaussians._parameters, 0, this_filter_cpu.reshape(-1, 1).expand(-1, 48)
    ).to("cuda")

    # Do rendering.
    rendered_image, _, _ = pipeline_forward_one_step(
        filtered_opacity_gpu=filtered_opacity_gpu,
        filtered_scaling_gpu=filtered_scaling_gpu,
        filtered_rotation_gpu=filtered_rotation_gpu,
        filtered_xyz_gpu=filtered_xyz_gpu,
        filtered_shs=filtered_shs_gpu,
        camera=camera,
        scene=scene,
        gaussians=gaussians,
        background=background,
        pipe_args=None,
        eval=True,
    )

    return rendered_image
