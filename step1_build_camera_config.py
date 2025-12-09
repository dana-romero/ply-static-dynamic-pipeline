import os
import json
import numpy as np
from colmap_text_utils import read_cameras_text, read_images_text

#SPARSE_DIR = os.path.join("sparse_small", "0")
SPARSE_DIR = os.path.join("dataset_v3", "sparse")
DATA_ROOT = "dataset_v3"
MASKS_DIR = os.path.join(DATA_ROOT, "masks")
OUTPUT_JSON = os.path.join(DATA_ROOT, "camera_config.json")


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=float)
    return R


def build_intrinsic(model, params):
    """
    Convert COLMAP camera model into 3x3 intrinsic matrix.
    Supports PINHOLE, SIMPLE_PINHOLE, SIMPLE_RADIAL, RADIAL.
    """
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    elif model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
        f, cx, cy = params[:3]
        fx, fy = f, f
    else:
        raise NotImplementedError(f"Camera model {model} not supported")

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,   0, 1]
    ], dtype=float)
    return K


def build_camera_config():
    cameras = read_cameras_text(os.path.join(SPARSE_DIR, "cameras.txt"))
    images = read_images_text(os.path.join(SPARSE_DIR, "images.txt"))

    cameras_cfg = []

    for cam_index, image_id in enumerate(sorted(images.keys())):
        img = images[image_id]
        cam = cameras[img["camera_id"]]

        model_name, width, height, params = cam

        # Build intrinsics and extrinsics
        K = build_intrinsic(model_name, params)
        R = qvec2rotmat(img["qvec"])
        T = img["tvec"]

        image_name = img["name"] 

        #corrected parsing
        # Assuming format is CAMID.png (e.g., "001001.png") and is always Frame 0

        # Check if the file is a standard PNG image
        if not image_name.endswith(".png"):
            # This should only happen if the COLMAP output file has errors
            raise RuntimeError(f"Unexpected image format: {image_name}")

        # Extract the camera ID from the filename (e.g., "001001")
        cam_folder = image_name.split(".")[0] 
        
        # Since COLMAP was run only on the first frame, we hardcode the frame ID.
        frame_id = "00000" 
        
        mask_folder = cam_folder # The mask folder uses the camera ID
        # --- END CORRECTED PARSING ---

        # Check if mask folder exists
        mask_folder_path = os.path.join(MASKS_DIR, mask_folder)
        if not os.path.isdir(mask_folder_path):
            print(f"[WARN] Mask folder not found: {mask_folder_path}")

        cam_info = {
            "cam_index": cam_index,
            "image_name": image_name,
            "mask_folder": mask_folder,
            "width": int(width),
            "height": int(height),
            "K": K.tolist(),
            "R": R.tolist(),
            "T": T.tolist(),
        }
        cameras_cfg.append(cam_info)

        print(f"cam_index={cam_index:2d}  cam='{mask_folder}'  image='{image_name}'")

    # --- CRITICAL FIX: SORT BY MASK FOLDER/CAMERA ID ---
    cameras_cfg = sorted(cameras_cfg, key=lambda x: x['mask_folder'])
    
    # Re-assign cam_index sequentially after sorting (optional, but cleaner)
    for i, cam_info in enumerate(cameras_cfg):
        cam_info['cam_index'] = i
    # --- END CRITICAL FIX ---
    
    os.makedirs(DATA_ROOT, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"cameras": cameras_cfg}, f, indent=2)

    print("\nSaved camera config to:", OUTPUT_JSON)


if __name__ == "__main__":
    build_camera_config()