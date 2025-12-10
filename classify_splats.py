import numpy as np
from pipeline_utils import project_points


def classify_splats(gaussians, cams, masks, thresh=2):
    xyz = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=-1)
    N = xyz.shape[0]

    dynamic_votes = np.zeros(N, dtype=np.int32)

    for cam, mask in zip(cams, masks):
        uv, depth = project_points(cam.K, cam.R, cam.T, xyz)

        # === FIX: SCALE UV TO MASK RESOLUTION ===
        H_mask, W_mask = mask.shape
        H_cam, W_cam = cam.height, cam.width
        
        scale_x = W_mask / W_cam  # 640 / 3840 ≈ 0.167
        scale_y = H_mask / H_cam  # 384 / 2160 ≈ 0.178
        
        u_scaled = uv[:, 0] * scale_x
        v_scaled = uv[:, 1] * scale_y
        # === END FIX ===
        
        # USE THE SCALED COORDINATES!
        u_i = np.floor(u_scaled).astype(np.int32)  # Changed from u_f to u_scaled
        v_i = np.floor(v_scaled).astype(np.int32)  # Changed from v_f to v_scaled

        H, W = mask.shape

        # Validity check with mask dimensions
        valid = (
            (depth > 0) &
            (u_i >= 0) & (u_i < W) &
            (v_i >= 0) & (v_i < H)
        )

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        # SAFE indexing
        sample_u = u_i[valid_idx]
        sample_v = v_i[valid_idx]

        # Sample the mask
        sampled_mask = mask[sample_v, sample_u]

        dynamic_votes[valid_idx] += (sampled_mask > 0).astype(np.int32)

    # After all cameras
    max_votes = np.max(dynamic_votes)
    total_dynamic_splats = np.sum(dynamic_votes >= thresh)
    print(f"[DEBUG] Max votes for any splat: {max_votes}")
    print(f"[DEBUG] Splats meeting threshold ({thresh}): {total_dynamic_splats}")
    
    # Add vote distribution
    unique, counts = np.unique(dynamic_votes, return_counts=True)
    print(f"[DEBUG] Vote distribution: {dict(zip(unique, counts))}")

    dynamic_mask = dynamic_votes >= thresh
    static_mask = ~dynamic_mask

    return static_mask, dynamic_mask