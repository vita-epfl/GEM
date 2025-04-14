from PIL import Image
import numpy as np
import os
import random
import numpy as np

# Load images
img1_path = "/var/tmp/europe_videos_converted/train/Driving_Experience_SJsssmcq8U4/frame_0113.png"
img2_path = "/var/tmp/europe_videos_converted/train/Driving_Experience_SJsssmcq8U4/frame_0123.png"

img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

# Convert images to numpy arrays
img1_np = np.array(img1).astype(np.float32) / 255
img2_np = np.array(img2).astype(np.float32) / 255

# Generate 10 timesteps for blending
timesteps = np.linspace(0, 1, 10)

output_dir = "logs/blended_frames"
os.makedirs(output_dir, exist_ok=True)

blended_images = []

# Blend images for each timestep and save
for i, t in enumerate(timesteps):
    blended_image_np = (
        t * img2_np
        + (1 - t) * img1_np
        + t * (1 - t) / 2 * np.random.randn(*img1_np.shape) * 2.5
    )
    blended_image = Image.fromarray(
        np.uint8(np.clip(blended_image_np, a_min=0, a_max=1) * 255)
    )

    # Save the blended image
    output_path = os.path.join(output_dir, f"blended_frame_{i+1:02d}.png")
    blended_image.save(output_path)
    blended_images.append(output_path)
