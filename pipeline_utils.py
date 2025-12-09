import os
import json
import numpy as np
import imageio.v2 as imageio

DATA_ROOT = "dataset_v3"
MASKS_DIR = os.path.join(DATA_ROOT, "masks")
CAM_CFG_PATH = os.path.join(DATA_ROOT, "camera_config.json")


class Camera:
    def __init__(self, K, R, T, width, height, mask_folder):
        self.K = K
        self.R = R
        self.T = T
        self.width = width
        self.height = height
        self.mask_folder = mask_folder


def load_cameras():
    with open(CAM_CFG_PATH, "r") as f:
        data = json.load(f)

    cams = []
    for c in data["cameras"]:
        K = np.array(c["K"], dtype=float)
        R = np.array(c["R"], dtype=float)
        T = np.array(c["T"], dtype=float)
        w, h = int(c["width"]), int(c["height"])
        folder = c["mask_folder"]

        cams.append(Camera(K, R, T, w, h, folder))
    return cams


def load_masks_for_frame(frame_idx, cams):
    """
    frame_idx: integer frame index (0..57)
    cams: list of Camera objects
    """
    frame_str = f"{frame_idx:06d}.png"  # 000000.png, 000001.png, ...
    masks = []
    for cam in cams:
        mask_path = os.path.join(MASKS_DIR, cam.mask_folder, frame_str)
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask missing: {mask_path}")
        m = imageio.imread(mask_path)
        if m.ndim == 3:
            m = m[..., 0]  # convert to grayscale
        masks.append(m.astype(np.uint8))
    return masks


def project_points(K, R, T, xyz):
    """
    xyz: (N,3) world coords
    R, T are assumed to be Camera-to-World (C2W) poses from COLMAP.
    """
    # --- CRITICAL FIX: INVERT POSE TO GET WORLD-TO-CAMERA (W2C) ---
    
    # R_W2C is the transpose of R_C2W
    R_W2C = R.T 
    
    # T_W2C = -R_W2C @ T_C2W (COLMAP convention)
    T_C2W_vector = T.flatten()
    T_W2C_vector = -R_W2C @ T_C2W_vector
    
    # Reshape the new T for stacking
    T_W2C_reshaped = T_W2C_vector.reshape(3, 1)

    # --- Use the inverted W2C matrices for projection ---
    RT_W2C = np.hstack([R_W2C, T_W2C_reshaped])
    
    pts_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    
    # The projection now uses the correct W2C transformation:
    pts_cam = pts_h @ RT_W2C.T
    
    # The rest of the calculation is correct
    pts_img = pts_cam @ K.T

    u = pts_img[:, 0] / pts_img[:, 2]
    v = pts_img[:, 1] / pts_img[:, 2]
    depth = pts_cam[:, 2]

    return np.stack([u, v], axis=-1), depth