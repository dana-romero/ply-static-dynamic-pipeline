# PLY Static / Dynamic Splitting Pipeline

This repo contains a small pipeline to post-process Gaussian-splat PLY files
for a **58-frame scene** with **22 camera views**.

The goal:

- Use **2D person masks** + **camera poses** to classify each 3D Gaussian as:
  - **Static** (background / environment)
  - **Dynamic** (people)
- Build:
  - One **Static_Master** PLY (stable background)
  - Per-frame **Dynamic** PLYs
  - Optionally: recomposed **Final** PLYs (Static + Dynamic)


## 0. Folder Structure

The pipeline assumes the following layout inside `ply_pipeline/`:

```text
ply_pipeline/
├── 0448_ply/
│   ├── time_00000.ply
│   ├── time_00001.ply
│   ├── ...
│   └── time_00057.ply          # 58 frames total
│
├── dataset_v3/
│   ├── images/                 # reference images per camera (22 views)
│   │   ├── 001001_time_00000.png
│   │   ├── 002001_time_00000.png
│   │   └── ...
│   └── masks/
│       ├── 001001/
│       │   ├── 000000.png
│       │   ├── 000001.png
│       │   └── ...
│       ├── 002001/
│       └── ...
│ 
├── TA_sparse_text/
│   ├── cameras.txt                 
│   ├── images.txt
│   └── points3D.txt
│
├── generate_camera_config.py
├── build_static_dynamic.py
├── classify_splats.py
├── gaussian_io.py / load_gaussians.py
├── pipeline_utils.py
└── ...
```

Notes:

- Each camera ID is something like `001001`, `002001`, …, `112001`.
- Masks follow the same ID:
  - `dataset_v3/masks/001001/000000.png` → first frame mask for cam `001001`
- Masks are binary:
  - **White (255)** = dynamic person
  - **Black (0)** = background


---

## 1. Environment Setup

Inside `ply_pipeline/`:

```bash
python -m venv venv
source venv/bin/activate
```

**Install necessary Python packages**
```bash
$ pip install numpy pillow imageio tqdm opencv-python plyfile scipy
```

## 2. Build Camera Configuration

This step processes the **COLMAP files** (`cameras.bin`, `images.bin`) to create the necessary **camera configuration** for splat projection. 

* **Script:** `generate_camera_config.py`
* **Run:**

```bash
python generate_camera_config.py
```
Expected outcome: 

```bash
cam_index= 0  cam='001001'  image='001001_time_00000.png'
cam_index= 1  cam='002001'  image='002001_time_00000.png'
...
cam_index=21  cam='112001'  image='112001_time_00000.png'
```
Saved camera config to: dataset_v3/camera_config.json


## 3. Static / Dynamic Splitting Pipeline

This is the main execution step that **classifies and splits** the input PLY frames (`0448_ply/`) into static and dynamic components. The process uses the camera configurations generated in Step 2 and the per-frame segmentation masks from the input structure.

After extracting the dynamic splats, the pipeline automatically merges:
```bash
Final[t] = Static_Master + Dynamic[t]
```

* **Run:**

```bash
python build_static_dynamic.py
```

**Output:** The generated PLY files (`Static_Master.ply`, `Dynamic_time_XXXXX.ply`,  `Final_XXXXX.ply` etc.) are saved to the `output_ply/ directory`.


## 4.Visualization

Use a standard Gaussian Splat viewer (e.g., SuperSplat, various web viewers) to inspect the generated PLY outputs (Static_Master.ply, Dynamic_time_XXXXX.ply).
