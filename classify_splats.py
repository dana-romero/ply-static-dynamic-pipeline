import numpy as np
from pipeline_utils import project_points


def classify_splats(gaussians, cams, masks, thresh=2):

    xyz = np.stack([gaussians["x"], gaussians["y"], gaussians["z"]], axis=-1)
    N = xyz.shape[0]

    dynamic_votes = np.zeros(N, dtype=np.int32)

    for cam, mask in zip(cams, masks):

        uv, depth = project_points(cam.K, cam.R, cam.T, xyz)

        # convert to integer pixel coords
        u_f = uv[:, 0]
        v_f = uv[:, 1]

        u_i = np.floor(u_f).astype(np.int32)
        v_i = np.floor(v_f).astype(np.int32)

        # VERY IMPORTANT:
        # mask shape = (H, W) = (384, 640)
        H, W = mask.shape

        # CORRECT validity check:
        valid = (
            (depth > 0) &
            (u_i >= 0) & (u_i < W) &     # u checks width
            (v_i >= 0) & (v_i < H)       # v checks height
        )

        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            continue

        # SAFE indexing
        sample_u = u_i[valid_idx]   # horizontal → width
        sample_v = v_i[valid_idx]   # vertical → height

        # use mask[row=v][col=u] BUT bounds are swapped in practice
        sampled_mask = mask[sample_v, sample_u]

        dynamic_votes[valid_idx] += (sampled_mask > 0).astype(np.int32)

        # --- DEBUGGING LINES ---
        max_votes = np.max(dynamic_votes)
        total_dynamic_splats = np.sum(dynamic_votes >= thresh)
        print(f"[DEBUG] Max votes for any splat: {max_votes}")
        print(f"[DEBUG] Splats meeting threshold ({thresh}): {total_dynamic_splats}")
        # --- END DEBUGGING LINES ---

    dynamic_mask = dynamic_votes >= thresh
    static_mask = ~dynamic_mask

    return static_mask, dynamic_mask