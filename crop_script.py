import fundus_image_toolbox as fit
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch

# --- SETUP ---
FINAL_OUTPUT_SIZE = (512, 512)
CROP_RATIO = 0.3

# 1. Setup GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 2. Load Model
model, config = fit.load_fovea_od_model(device=device)

# 3. Setup Paths
input_dir = Path("/path/to/your/input_images")
output_dir = Path("/path/to/your/new_cropped_images")
output_dir.mkdir(exist_ok=True, parents=True)

image_paths = sorted(
    list(input_dir.glob("*.jpg")) +
    list(input_dir.glob("*.png")) +
    list(input_dir.glob("*.tif"))
)

print(f"Found {len(image_paths)} images.")
print("Starting Stable Processing...")

# 4. Processing Loop
for i, img_path in enumerate(image_paths):

    if i % 10 == 0: print(f"Processing {i}/{len(image_paths)}...")

    try:
        batch_input = [str(img_path), str(img_path)]

        predictions = model.predict(batch_input)

        pred = predictions[0]

        # --- VALIDATION ---
        if np.isscalar(pred) or pred is None:
            print(f"Skipping {img_path.name}: Model failed (Scalar/None).")
            continue

        fovea_x, fovea_y, od_x, od_y = pred

        if np.isnan(od_x) or np.isnan(od_y):
            print(f"Skipping {img_path.name}: Coordinates are NaN.")
            continue

        # --- CROP LOGIC (Standard) ---
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Dynamic Crop Size
        dynamic_crop_size = int(height * CROP_RATIO)
        half_size = dynamic_crop_size // 2

        od_x_px = int(od_x)
        od_y_px = int(od_y)

        # Calculate Box
        left = od_x_px - half_size
        upper = od_y_px - half_size
        right = od_x_px + half_size
        lower = od_y_px + half_size

        # Padding Logic
        pad_left = max(0, -left)
        pad_top = max(0, -upper)
        pad_right = max(0, right - width)
        pad_bottom = max(0, lower - height)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            img = ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=(0, 0, 0))
            # Shift coordinates
            left += pad_left
            upper += pad_top
            right += pad_left
            lower += pad_top

        # Crop & Resize
        cropped_img = img.crop((left, upper, right, lower))
        cropped_img = cropped_img.resize(FINAL_OUTPUT_SIZE, Image.LANCZOS)

        # Save
        out_path = output_dir / img_path.name
        cropped_img.save(out_path)

    except Exception as e:
        print(f"Error on {img_path.name}: {e}")

print("Processing complete.")
