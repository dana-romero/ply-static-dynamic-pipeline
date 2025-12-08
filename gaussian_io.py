import numpy as np
import struct


GAUSSIAN_DTYPE = np.dtype([
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),

    ("nx", "f4"),
    ("ny", "f4"),
    ("nz", "f4"),

    ("f_dc", "f4", (3,)),
    ("opacity", "f4"),

    ("scale", "f4", (3,)),
    ("rot", "f4", (4,)),
])


def load_ply_gaussians(path):
    """Load your Gaussian PLY into a structured NumPy array."""
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break

        # find vertex count
        element_line = [l for l in header if l.startswith("element vertex")][0]
        N = int(element_line.split()[-1])

        # Read all vertex binary data
        data = f.read(N * GAUSSIAN_DTYPE.itemsize)
        arr = np.frombuffer(data, dtype=GAUSSIAN_DTYPE)

    return arr.copy()  # return a writable copy


def save_ply_gaussians(path, arr):
    """Write structured Gaussian array back to PLY."""
    N = arr.shape[0]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.astype(GAUSSIAN_DTYPE).tobytes())