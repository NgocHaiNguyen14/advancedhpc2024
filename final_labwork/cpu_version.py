import numpy as np
from PIL import Image
import math
import time
import matplotlib.pyplot as plt

# Load the image
img_path = "/home/hai/Github-Repositories/advancedhpc2024/final_labwork/highest_reso.jpeg"
img = Image.open(img_path).convert("RGB")
image_array = np.array(img).astype(np.float32) / 255.0
height, width, _ = image_array.shape

def rgb_to_hsv(image):
    # Convert an RGB image to HSV format
    hsv_image = np.zeros_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            r, g, b = image[y, x]
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val

            v = max_val
            s = delta / max_val if max_val != 0 else 0

            if delta == 0:
                h = 0
            elif max_val == r:
                h = 60 * ((g - b) / delta % 6)
            elif max_val == g:
                h = 60 * ((b - r) / delta + 2)
            else:
                h = 60 * ((r - g) / delta + 4)

            hsv_image[y, x] = [h, s, v]
    return hsv_image

def kuwahara_filter_cpu(image, window_size=4):
    # Apply the Kuwahara filter on an RGB image
    height, width, _ = image.shape
    output_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            min_variance = float('inf')
            best_quadrant_color = [0, 0, 0]

            # Define quadrants and calculate the mean and variance in each
            for quadrant in range(4):
                if quadrant == 0:
                    start_x, end_x = max(x - window_size, 0), x + 1
                    start_y, end_y = max(y - window_size, 0), y + 1
                elif quadrant == 1:
                    start_x, end_x = x, min(x + window_size + 1, width)
                    start_y, end_y = max(y - window_size, 0), y + 1
                elif quadrant == 2:
                    start_x, end_x = max(x - window_size, 0), x + 1
                    start_y, end_y = y, min(y + window_size + 1, height)
                else:
                    start_x, end_x = x, min(x + window_size + 1, width)
                    start_y, end_y = y, min(y + window_size + 1, height)

                # Calculate mean and variance for each quadrant
                region = image[start_y:end_y, start_x:end_x]
                mean_rgb = np.mean(region, axis=(0, 1))
                variance = np.mean((region - mean_rgb) ** 2)

                # Select the quadrant with the lowest variance
                if variance < min_variance:
                    min_variance = variance
                    best_quadrant_color = mean_rgb

            # Assign the color of the selected quadrant to the output image
            output_image[y, x] = best_quadrant_color

    return output_image

# Apply filter and measure time
start_time = time.time()
image_hsv = rgb_to_hsv(image_array)
filtered_image = kuwahara_filter_cpu(image_hsv, window_size=4)
filtered_image_rgb = (filtered_image * 255).astype(np.uint8)
end_time = time.time()

print(f"Filtered image processed in {end_time - start_time} seconds.")
plt.imshow(filtered_image_rgb)
plt.axis('off')
plt.show()
