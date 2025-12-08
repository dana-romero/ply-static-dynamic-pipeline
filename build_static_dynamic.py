import os
import numpy as np
from gaussian_io import load_ply_gaussians, save_ply_gaussians
from pipeline_utils import load_cameras, load_masks_for_frame
from classify_splats import classify_splats

PLY_DIR = "0448_ply"
OUT_DIR = "output_ply"


def run_pipeline():
    print("=== RUN_PIPELINE START ===")

    print("Loading cameras...")
    cams = load_cameras()
    print("Loaded", len(cams), "cameras")

    os.makedirs(OUT_DIR, exist_ok=True)
    print("Output directory:", OUT_DIR)

    # Step 1: STATIC MASTER
    first_frame_path = os.path.join(PLY_DIR, "time_00000.ply")
    print("Loading first frame:", first_frame_path)

    if not os.path.isfile(first_frame_path):
        print("ERROR: first frame does not exist:", first_frame_path)
        return

    g0 = load_ply_gaussians(first_frame_path)
    print("Loaded g0:", g0.shape)

    masks0 = load_masks_for_frame(0, cams)
    print("Loaded masks for frame 0")

    print("Classifying static/dynamic...")
    static_mask, dynamic_mask0 = classify_splats(g0, cams, masks0)
    print("Static count:", np.sum(static_mask), "Dynamic count:", np.sum(dynamic_mask0))

    static_master = g0[static_mask]
    static_path = os.path.join(OUT_DIR, "Static_Master.ply")
    save_ply_gaussians(static_path, static_master)
    print("Saved Static_Master:", static_path)

    # Step 2: Per-frame dynamic extraction
    num_frames = 58
    print("Processing", num_frames, "frames...")

    for i in range(num_frames):
        fname = f"time_{i:05d}.ply"
        frame_path = os.path.join(PLY_DIR, fname)
        print(f"\nFrame {i}: {frame_path}")

        if not os.path.isfile(frame_path):
            print("WARNING: Missing frame, skipping:", frame_path)
            continue

        g = load_ply_gaussians(frame_path)
        masks_i = load_masks_for_frame(i, cams)

        _, dynamic_mask = classify_splats(g, cams, masks_i)
        dynamic = g[dynamic_mask]

        dyn_path = os.path.join(OUT_DIR, f"Dynamic_{i:05d}.ply")
        save_ply_gaussians(dyn_path, dynamic)
        print("Saved dynamic:", dyn_path, "count:", dynamic.shape[0])

        # Combine static + dynamic
        final = np.concatenate([static_master, dynamic])
        final_path = os.path.join(OUT_DIR, f"Final_{i:05d}.ply")
        save_ply_gaussians(final_path, final)
        print("Saved final:", final_path, "count:", final.shape[0])

    print("=== RUN_PIPELINE COMPLETE ===")


if __name__ == "__main__":
    run_pipeline()