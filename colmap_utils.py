import os
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R


def read_images_custom_bin(path):
    """
    Custom reader for COLMAP images.bin.
    Returns:
        dict[image_id] = {
            'R': rotation matrix (3x3),
            'T': translation (3,),
            'camera_id': int,
            'name': str,
        }
    """
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # read name until null terminator
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes.append(c)
            name = b"".join(name_bytes).decode("utf-8")

            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()

            images[image_id] = {
                "R": rot,
                "T": np.array([tx, ty, tz], dtype=float),
                "camera_id": camera_id,
                "name": name,
            }

    return images



def read_cameras_binary(path):
    """
    Standard COLMAP cameras.bin reader
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cams):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            # number of parameters depends on model
            if model == 1:      # SIMPLE_PINHOLE
                num_params = 3
            elif model == 2:    # PINHOLE
                num_params = 4
            elif model == 3:    # SIMPLE_RADIAL
                num_params = 4
            else:
                raise RuntimeError(f"Unsupported camera model id {model}")

            params = struct.unpack("<" + "d"*num_params, f.read(8*num_params))

            cameras[cam_id] = {
                "model_id": model,
                "width": width,
                "height": height,
                "params": np.array(params),
            }

    return cameras



def build_K(model_id, params):
    if model_id == 2:  # PINHOLE
        fx, fy, cx, cy = params
    elif model_id in [1, 3]:  # SIMPLE_PINHOLE / SIMPLE_RADIAL
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise RuntimeError(f"Unsupported camera model id {model_id}")

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1],
    ], dtype=float)
    return K



def cameras_from_colmap_bin(sparse_dir):
    """
    Loads cameras.bin and images.bin and returns final camera dict:
      cam_id -> {K, width, height}
      plus images dict
    """
    cams_bin = read_cameras_binary(os.path.join(sparse_dir, "cameras.bin"))
    
    # ✅ FIXED LINE — use the custom reader
    imgs_bin = read_images_custom_bin(os.path.join(sparse_dir, "images.bin"))

    # Build intrinsics for each camera
    for cam_id, cam in cams_bin.items():
        K = build_K(cam["model_id"], cam["params"])
        cam["K"] = K

    return cams_bin, imgs_bin