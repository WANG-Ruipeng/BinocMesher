#!/usr/bin/env bash
set -uo pipefail

REPO="${PILOT_REPO:-/home/warpwang/src/BinocMesher}"
INFINIGEN="$REPO/infinigen_binocmesher"
PYTHON="${PILOT_PYTHON:-/home/warpwang/miniforge3/envs/binoc-exp/bin/python}"
SCORER="$REPO/experiments/render_eval/warped_ssim.py"
OUTPUT_ROOT="${1:-/home/warpwang/runs/forest96-four-baseline-pilot-v2-$(date +%Y%m%d-%H%M%S)}"

FRAMES="${PILOT_FRAMES:-96}"
PPC="${PILOT_PPC:-6}"
SAMPLES="${PILOT_SAMPLES:-3072}"
RENDER_START="${PILOT_RENDER_START:-1}"
RENDER_END="${PILOT_RENDER_END:-$FRAMES}"
EXPECTED_FRAMES=$((RENDER_END - RENDER_START + 1))
METHODS=("${@:2}")
if [[ ${#METHODS[@]} -eq 0 ]]; then
    METHODS=(binoc ocmesher96 ocmesher24 spherical8)
fi
for method in "${METHODS[@]}"; do
    case "$method" in
        binoc|ocmesher96|ocmesher24|spherical8) ;;
        *) printf 'Unknown method: %s\n' "$method" >&2; exit 2 ;;
    esac
done
if [[ ! "$FRAMES" =~ ^[0-9]+$ ]] || (( FRAMES < 2 )); then
    printf 'PILOT_FRAMES must be at least 2.\n' >&2
    exit 2
fi

if (( RENDER_START < 1 || RENDER_END > FRAMES || EXPECTED_FRAMES < 2 )); then
    printf 'Render interval must contain at least two frames within [1,FRAMES].\n' >&2
    exit 2
fi

# Never overwrite prior status or retain an old success marker.
mkdir -- "$OUTPUT_ROOT" || exit 2
exec > >(tee -a "$OUTPUT_ROOT/overnight.log") 2>&1

printf 'method\tstart_epoch\tend_epoch\telapsed_seconds\tmanager_rc\tssim_rc\timages\tflows\n' > "$OUTPUT_ROOT/status.tsv"
{
    printf 'experiment=Forest warped-SSIM pilot v2\n'
    printf 'started=%s\n' "$(date --iso-8601=seconds)"
    printf 'repo=%s\n' "$REPO"
    printf 'commit=%s\n' "$(git -C "$REPO" rev-parse HEAD)"
    printf 'frames=%s\n' "$FRAMES"
    printf 'render_frame_start=%s\n' "$RENDER_START"
    printf 'render_frame_end=%s\n' "$RENDER_END"
    printf 'resolution=960x540\n'
    printf 'pixels_per_cube=%s\n' "$PPC"
    printf 'cycles_samples=%s\n' "$SAMPLES"
    printf 'omp_threads=8\n'
    printf 'methods=%s\n' "${METHODS[*]}"
    printf 'binoc_cam_block_size=1\n'
    printf 'reuse_source=%s\n' "${PILOT_REUSE_ROOT:-none}"
    printf 'infinigen_commit=%s\n' "$(git -C "$INFINIGEN" rev-parse HEAD)"
    printf 'note=Exploratory pilot; final paper setting is ppc=3 and samples=8192.\n'
} > "$OUTPUT_ROOT/manifest.txt"

export PATH="/home/warpwang/miniforge3/envs/binoc-exp/bin:$PATH"
export PYTHONPATH="$INFINIGEN${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export BINOC_MAX_SERIALIZED_CACHE_RECORDS=100000000
export BINOC_MAX_SERIALIZED_CACHE_PAYLOAD_BYTES=8589934592

run_method() {
    local method="$1"
    local pipeline="$2"
    local backend="$3"
    local view_block="$4"
    local cam_block="$5"
    shift 5
    local output="$OUTPUT_ROOT/$method"
    local start_epoch end_epoch manager_rc ssim_rc images flows
    local -a mesher_overrides=("$@")
    if [[ "$backend" == BinocMesher && "$cam_block" -ne 1 ]]; then
        printf 'BinocMesher requires cam_block_size=1.\n' >&2
        return 1
    fi

    start_epoch="$(date +%s)"
    printf '[%s] START method=%s pipeline=%s backend=%s view_block=%s cam_block=%s\n' \
        "$(date --iso-8601=seconds)" "$method" "$pipeline" "$backend" "$view_block" "$cam_block"

    (
        cd "$INFINIGEN" || exit 97
        manager_options=()
        if [[ -n "${PILOT_REUSE_ROOT:-}" ]]; then
            "$PYTHON" "$REPO/experiments/render_eval/prepare_resume.py" \
                "$PILOT_REUSE_ROOT" "$output" --method "$method" \
                --frames "$FRAMES" --ppc "$PPC" || exit 98
            manager_options=(--use_existing)
        fi
        "$PYTHON" -m infinigen.datagen.manage_jobs \
            "${manager_options[@]}" \
            --output_folder "$output" \
            --num_scenes 1 \
            --specific_seed 0 \
            --cleanup none \
            --wandb_mode disabled \
            --configs mountain monocular simple no_assets \
            --pipeline_configs local_256GB "$pipeline" \
            --pipeline_overrides \
                "iterate_scene_tasks.frame_range=[1,$FRAMES]" \
                "iterate_scene_tasks.render_frame_range=[$RENDER_START,$RENDER_END]" \
                "iterate_scene_tasks.view_block_size=$view_block" \
                "iterate_scene_tasks.cam_block_size=$cam_block" \
                LocalScheduleHandler.use_gpu=True \
                ground_truth/queue_render.gpus=1 \
                queue_render.gpus=1 \
                manage_datagen_jobs.num_concurrent=1 \
            -p \
                scene.upsidedown_mountains_chance=0 \
                scene.sdf_trees_chance=1 \
                nishita_lighting.sun_elevation=20 \
                nishita_lighting.sun_rotation=0 \
                shader_atmosphere.density=0.0005 \
                shader_atmosphere.anisotropy=0.8 \
                "fine_terrain.mesher_backend=\"$backend\"" \
                'compose_nature.load_cameras="../infinigen_example_scenes/forest.txt"' \
                "${mesher_overrides[@]}" \
                'execute_tasks.generate_resolution=[960,540]' \
                "full/configure_render_cycles.num_samples=$SAMPLES" \
                'Terrain.device="cpu"'
    )
    manager_rc=$?

    ssim_rc=125
    if [[ "$manager_rc" -eq 0 ]]; then
        "$PYTHON" "$SCORER" "$output/0" \
            --output "$output/ssim" --expected-frames "$EXPECTED_FRAMES" \
            --start-frame "$RENDER_START"
        ssim_rc=$?
    fi

    end_epoch="$(date +%s)"
    images="$(find "$output/0/frames/Image/camera_0" -maxdepth 1 -type f -name 'Image_0_0_*.png' 2>/dev/null | wc -l)"
    flows="$(find "$output/0/frames/Flow/camera_0" -maxdepth 1 -type f -name 'Flow_0_0_*.npy' 2>/dev/null | wc -l)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$method" "$start_epoch" "$end_epoch" "$((end_epoch - start_epoch))" \
        "$manager_rc" "$ssim_rc" "$images" "$flows" >> "$OUTPUT_ROOT/status.tsv"
    printf '[%s] END method=%s manager_rc=%s ssim_rc=%s images=%s flows=%s elapsed=%ss\n' \
        "$(date --iso-8601=seconds)" "$method" "$manager_rc" "$ssim_rc" \
        "$images" "$flows" "$((end_epoch - start_epoch))"
    [[ "$manager_rc" -eq 0 && "$ssim_rc" -eq 0 && "$images" -eq "$EXPECTED_FRAMES" && "$flows" -ge "$((EXPECTED_FRAMES - 1))" ]]
}

# Most important direct baseline first; later methods still run if an earlier one fails.
failures=0
for method in "${METHODS[@]}"; do
    case "$method" in
        binoc)
            run_method binoc monocular_video_hyperocmesher BinocMesher "$FRAMES" 1 \
                "BinocMesher.pixels_per_cube=$PPC" || failures=$((failures + 1)) ;;
        ocmesher96)
            run_method ocmesher96 monocular_video_sphericalmesher OcMesher 96 24 \
                "OcMesher.pixels_per_cube=$PPC" || failures=$((failures + 1)) ;;
        ocmesher24)
            run_method ocmesher24 monocular_video_sphericalmesher OcMesher 24 24 \
                "OcMesher.pixels_per_cube=$PPC" || failures=$((failures + 1)) ;;
        spherical8)
            run_method spherical8 monocular_video_sphericalmesher SphericalMesher 8 8 \
                "OpaqueSphericalMesher.pixels_per_cube=$PPC" \
                "TransparentSphericalMesher.pixels_per_cube=$PPC" || failures=$((failures + 1)) ;;
    esac
done

printf 'finished=%s\n' "$(date --iso-8601=seconds)" >> "$OUTPUT_ROOT/manifest.txt"
printf 'failed_methods=%s\n' "$failures" >> "$OUTPUT_ROOT/manifest.txt"
if [[ "$failures" -eq 0 ]]; then
    touch "$OUTPUT_ROOT/FINISHED"
    printf '[%s] ALL METHODS SUCCEEDED\n' "$(date --iso-8601=seconds)"
else
    touch "$OUTPUT_ROOT/FAILED"
    printf '[%s] BATCH FAILED: %s method(s) failed\n' "$(date --iso-8601=seconds)" "$failures"
    exit 1
fi
