import os
import json
import numpy as np

# --------- Helper functions ---------

def qvec2rotmat(qvec):
    """Convert COLMAP quaternion format to rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [
            1 - 2*qy*qy - 2*qz*qz,
            2*qx*qy - 2*qz*qw,
            2*qx*qz + 2*qy*qw
        ],
        [
            2*qx*qy + 2*qz*qw,
            1 - 2*qx*qx - 2*qz*qz,
            2*qy*qz - 2*qx*qw
        ],
        [
            2*qx*qz - 2*qy*qw,
            2*qy*qz + 2*qx*qw,
            1 - 2*qx*qx - 2*qy*qy
        ],
    ])

# --------- Load TA COLMAP files ---------

cams_txt = "TA_sparse_text/cameras.txt"
imgs_txt = "TA_sparse_text/images.txt"

# Load intrinsics
intrinsics = {}
with open(cams_txt, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        cam_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        fx = float(parts[4])
        cx = float(parts[5])
        cy = float(parts[6])
        k1 = float(parts[7])  # not needed for projection

        K = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0,  0,  1]
        ])

        intrinsics[cam_id] = {
            "width": width,
            "height": height,
            "K": K.tolist()
        }

# List of valid mask folders (22 total)
valid_mask_names = {
    "001001", "002001", "003001", "004001", "005001", "006001",
    "008001", "009001", "010001", "012001",
    "101001", "102001", "103001", "104001", "105001",
    "106001", "107001", "108001", "109001", "110001",
    "111001", "112001"
}

cameras_out = []

with open(imgs_txt, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue

        parts = line.split()

        # --- Skip POINTS2D lines (they do NOT end with .png) ---
        if len(parts) < 10 or not parts[-1].endswith(".png"):
            continue

        # Now this is guaranteed to be a header line
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        img_name = parts[9]  # e.g., "001001.png"

        name_no_ext = img_name.replace(".png", "")

        # Skip missing camera
        if name_no_ext not in valid_mask_names:
            print(f"Skipping missing or unused view: {img_name}")
            continue

        # Convert quaternion
        R_c2w = qvec2rotmat([qw, qx, qy, qz])
        t_c2w = np.array([tx, ty, tz])

        # Convert to W2C
        R_w2c = R_c2w.T
        T_w2c = -R_w2c @ t_c2w

        # Intrinsics
        K = intrinsics[cam_id]["K"]
        width = intrinsics[cam_id]["width"]
        height = intrinsics[cam_id]["height"]

        cameras_out.append({
            "cam_index": len(cameras_out),
            "image_name": img_name,
            "mask_folder": name_no_ext,
            "width": width,
            "height": height,
            "K": K,
            "R": R_w2c.tolist(),
            "T": T_w2c.tolist()
        })

# --------- Write final JSON ---------

output = {"cameras": cameras_out}

# Ensure dataset_v3 exists
os.makedirs("dataset_v3", exist_ok=True)

output_path = "dataset_v3/camera_config.json"

with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print("Generated", output_path, "with", len(cameras_out), "cameras.")