import numpy as np
from PIL import Image
from numba import cuda
import math
import time
import matplotlib.pyplot as plt

img_path = "/home/hai/Github-Repositories/advancedhpc2024/final_labwork/highest_reso.jpeg"
img = Image.open(img_path).convert("RGB")
image_array = np.array(img).astype(np.float32) / 255.0  
height, width, _ = image_array.shape

filtered_result = np.zeros_like(image_array)

#convert - previous lw
@cuda.jit
def rgb_to_hsv_kernel(src, h, s, v, width, height):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    r, g, b = src[y, x, 0], src[y, x, 1], src[y, x, 2]
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val

    v[y, x] = max_val
    s[y, x] = delta / max_val if max_val != 0 else 0

    if delta == 0:
        h[y, x] = 0
    else:
        if max_val == r:
            h[y, x] = 60 * ((g - b) / delta % 6)
        elif max_val == g:
            h[y, x] = 60 * ((b - r) / delta + 2)
        else:
            h[y, x] = 60 * ((r - g) / delta + 4)

#kuwahara filter
@cuda.jit
def apply_kuwahara_filter_on_gpu(input_rgb, hue, saturation, value, output_rgb, window_size):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    image_height, image_width = input_rgb.shape[0], input_rgb.shape[1]
    if tidx >= image_width or tidy >= image_height:
        return

    min_variance = 1e10
    selected_quadrant = 0
    quadrant_means = cuda.local.array((4, 3), dtype=np.float32)
    for i in range(4):
        for j in range(3):
            quadrant_means[i][j] = 0.0
    # initialize quarant for each pixel
    for quadrant in range(4):
        sum_value = 0.0
        sum_value_squared = 0.0
        sum_rgb = cuda.local.array(3, dtype=np.float32)
        sum_rgb[0], sum_rgb[1], sum_rgb[2] = 0.0, 0.0, 0.0
        count = 0
        # SCATTER 
        if quadrant == 0:   
            start_x, end_x = max(tidx - window_size, 0), tidx + 1 #read from tidx - no shared memory => take time
            start_y, end_y = max(tidy - window_size, 0), tidy + 1
        elif quadrant == 1: 
            start_x, end_x = tidx, min(tidx + window_size + 1, image_width)
            start_y, end_y = max(tidy - window_size, 0), tidy + 1
        elif quadrant == 2:  
            start_x, end_x = max(tidx - window_size, 0), tidx + 1
            start_y, end_y = tidy, min(tidy + window_size + 1, image_height)
        else:               
            start_x, end_x = tidx, min(tidx + window_size + 1, image_width)
            start_y, end_y = tidy, min(tidy + window_size + 1, image_height)
        # read memory mant times - GATHER
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                value_pixel = value[y, x]
                sum_value += value_pixel
                sum_value_squared += value_pixel * value_pixel
                sum_rgb[0] += input_rgb[y, x, 0]
                sum_rgb[1] += input_rgb[y, x, 1]
                sum_rgb[2] += input_rgb[y, x, 2]
                count += 1

        mean_value = sum_value / count
        variance = (sum_value_squared / count) - (mean_value * mean_value)
        std_deviation = math.sqrt(max(variance, 0.0))

        quadrant_means[quadrant][0] = sum_rgb[0] / count 
        quadrant_means[quadrant][1] = sum_rgb[1] / count
        quadrant_means[quadrant][2] = sum_rgb[2] / count

        if std_deviation < min_variance:
            min_variance = std_deviation
            selected_quadrant = quadrant

    output_rgb[tidy, tidx, 0] = quadrant_means[selected_quadrant][0]
    output_rgb[tidy, tidx, 1] = quadrant_means[selected_quadrant][1]
    output_rgb[tidy, tidx, 2] = quadrant_means[selected_quadrant][2]

def convert_to_hsv(rgb_image, window_size=3): #add window size to use for kuwahara filter in the convert function
    image_height, image_width = rgb_image.shape[0], rgb_image.shape[1]
    output = np.zeros_like(rgb_image)

    d_rgb = cuda.to_device(rgb_image)
    d_hue = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_saturation = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_value = cuda.device_array((image_height, image_width), dtype=np.float32)
    d_output = cuda.device_array_like(output)

    block_dim = (16, 16)
    grid_dim = (
        (image_width + block_dim[0] - 1) // block_dim[0],
        (image_height + block_dim[1] - 1) // block_dim[1]
    )

    rgb_to_hsv_kernel[grid_dim, block_dim](d_rgb, d_hue, d_saturation, d_value, image_width, image_height)
    cuda.synchronize()

    apply_kuwahara_filter_on_gpu[grid_dim, block_dim](d_rgb, d_hue, d_saturation, d_value, d_output, window_size)
    cuda.synchronize()

    output = d_output.copy_to_host()
    return output

start_time = time.time()
filtered_image = convert_to_hsv(image_array, window_size=4)
filtered_image_rgb = (filtered_image * 255).astype(np.uint8)
end_time = time.time()

print(f"Filtered image processed in {end_time - start_time} seconds.")
plt.imshow(filtered_image_rgb)
plt.axis('off')
plt.show()
