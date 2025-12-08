from pipeline_utils import load_cameras, load_masks_for_frame

cams = load_cameras()
print("Loaded cameras:", len(cams))

# test the first frame (frame 0)
masks = load_masks_for_frame(0, cams)

for i, (cam, mask) in enumerate(zip(cams, masks)):
    print(f"cam {i:2d} folder={cam.mask_folder} mask_shape={mask.shape}")