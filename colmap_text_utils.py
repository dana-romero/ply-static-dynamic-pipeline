import numpy as np


def read_cameras_text(path):
    """
    COLMAP cameras.txt text format.
    Returns dict: camera_id -> (model, width, height, params[np.array])
    """
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "" or line[0] == "#":
                continue
            toks = line.split()
            camera_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = np.array(list(map(float, toks[4:])), dtype=float)
            cameras[camera_id] = (model, width, height, params)
    return cameras


def read_images_text(path):
    """
    COLMAP images.txt text format.
    Returns dict: image_id -> {
        'qvec': np.array(4),
        'tvec': np.array(3),
        'camera_id': int,
        'name': str,
    }
    """
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != "" and l.strip()[0] != "#"]

    # images.txt has pairs of lines: meta line, then 2D points line.
    # We only care about the meta lines (even indices: 0,2,4,...).
    for i in range(0, len(lines), 2):
        toks = lines[i].split()
        if len(toks) < 10:
            continue
        image_id = int(toks[0])
        qvec = np.array(list(map(float, toks[1:5])), dtype=float)
        tvec = np.array(list(map(float, toks[5:8])), dtype=float)
        camera_id = int(toks[8])
        name = " ".join(toks[9:])  # image file name (may contain spaces, but usually doesn't)
        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
        }
    return images