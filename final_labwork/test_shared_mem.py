import numpy as np
from PIL import Image
from numba import cuda, float32
import time
import matplotlib.pyplot as plt

img_path = "/home/hai/Github-Repositories/advancedhpc2024/final_labwork/highest_reso.jpeg"
img_data = np.array(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
height, width, _ = img_data.shape

hue = np.zeros_like(img_data[:, :, 0], dtype=np.float32)
saturation = np.zeros_like(img_data[:, :, 0], dtype=np.float32)
value = np.zeros_like(img_data[:, :, 0], dtype=np.float32)
output_img = np.zeros_like(img_data)

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

def convert_to_hsv(image_data): #just convert only , no using kuwahar inside convert function, so dont need window size
    height, width, _ = image_data.shape
    h_channel = np.zeros_like(image_data[:, :, 0], dtype=np.float32)
    s_channel = np.zeros_like(image_data[:, :, 0], dtype=np.float32)
    v_channel = np.zeros_like(image_data[:, :, 0], dtype=np.float32)

    d_image = cuda.to_device(image_data)
    d_h = cuda.to_device(h_channel)
    d_s = cuda.to_device(s_channel)
    d_v = cuda.to_device(v_channel)

    block = (16, 16)
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1])

    rgb_to_hsv_kernel[grid, block](d_image, d_h, d_s, d_v, width, height)

    h_channel = d_h.copy_to_host()
    s_channel = d_s.copy_to_host()
    v_channel = d_v.copy_to_host()

    return h_channel, s_channel, v_channel

@cuda.jit
def kuwahara_filter_kernel(input_img, h_in, s_in, v_in, output_img, win_size):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    block_width = cuda.blockDim.x
    block_height = cuda.blockDim.y

    tidx = tx + bx * block_width
    tidy = ty + by * block_height   

    img_height, img_width, _ = input_img.shape

    shared_block = block_width + 2 * win_size
    # Each pixel contains value of rgb and hsv so need to declare for each channel for further computation
    shared_hue = cuda.shared.array((32, 32), dtype=np.float32)
    shared_sat = cuda.shared.array((32, 32), dtype=np.float32)
    shared_val = cuda.shared.array((32, 32), dtype=np.float32) 
    shared_r = cuda.shared.array((32, 32), dtype=np.float32)
    shared_g = cuda.shared.array((32, 32), dtype=np.float32)
    shared_b = cuda.shared.array((32, 32), dtype=np.float32)

    for dy in range(0, shared_block, block_height):
        for dx in range(0, shared_block, block_width):
            shared_y = ty + dy
            shared_x = tx + dx
            global_y = tidy - win_size + dy
            global_x = tidx - win_size + dx

            if (global_y >= 0 and global_y < img_height and
                global_x >= 0 and global_x < img_width):
                shared_hue[shared_y, shared_x] = h_in[global_y, global_x]
                shared_sat[shared_y, shared_x] = s_in[global_y, global_x]
                shared_val[shared_y, shared_x] = v_in[global_y, global_x]
                shared_r[shared_y, shared_x] = input_img[global_y, global_x, 0]
                shared_g[shared_y, shared_x] = input_img[global_y, global_x, 1]
                shared_b[shared_y, shared_x] = input_img[global_y, global_x, 2]
            else:
                shared_hue[shared_y, shared_x] = 0.0
                shared_sat[shared_y, shared_x] = 0.0
                shared_val[shared_y, shared_x] = 0.0
                shared_r[shared_y, shared_x] = 0.0
                shared_g[shared_y, shared_x] = 0.0
                shared_b[shared_y, shared_x] = 0.0

    cuda.syncthreads()

    if tidx < img_width and tidy < img_height:
        means = cuda.local.array((4, 3), dtype=np.float32)
        variances = cuda.local.array(4, dtype=np.float32)

        for quad in range(4):
            sum_r = sum_g = sum_b = sum_v = sum_sq_v = 0.0
            count = 0

            shared_x = tx + win_size
            shared_y = ty + win_size


            if quad == 0:
                y_start, y_end = shared_y - win_size, shared_y + 1 #take from shared memory
                x_start, x_end = shared_x - win_size, shared_x + 1
            elif quad == 1:
                y_start, y_end = shared_y - win_size, shared_y + 1
                x_start, x_end = shared_x, shared_x + win_size + 1
            elif quad == 2:
                y_start, y_end = shared_y, shared_y + win_size + 1
                x_start, x_end = shared_x - win_size, shared_x + 1
            else:
                y_start, y_end = shared_y, shared_y + win_size + 1
                x_start, x_end = shared_x, shared_x + win_size + 1
            #read from shared memory
            for yi in range(y_start, y_end):
                for xi in range(x_start, x_end):
                    count += 1
                    r = shared_r[yi, xi]
                    g = shared_g[yi, xi]
                    b = shared_b[yi, xi]
                    v = shared_val[yi, xi]

                    sum_r += r
                    sum_g += g
                    sum_b += b
                    sum_v += v
                    sum_sq_v += v * v

            if count > 0:
                means[quad, 0] = sum_r / count
                means[quad, 1] = sum_g / count
                means[quad, 2] = sum_b / count
                mean_v = sum_v / count
                variances[quad] = (sum_sq_v / count) - (mean_v * mean_v)
            else:
                variances[quad] = 1e10

        min_variance = variances[0]
        min_idx = 0
        for quad in range(1, 4):
            if variances[quad] < min_variance:
                min_variance = variances[quad]
                min_idx = quad

        output_img[tidy, tidx, 0] = means[min_idx, 0]
        output_img[tidy, tidx, 1] = means[min_idx, 1]
        output_img[tidy, tidx, 2] = means[min_idx, 2]

def apply_kuwahara_filter(input_image, win_size=4):
    h, s, v = convert_to_hsv(input_image)

    img_height, img_width = input_image.shape[:2]
    filtered = np.zeros_like(input_image)

    d_input = cuda.to_device(input_image)
    d_h = cuda.to_device(h)
    d_s = cuda.to_device(s)
    d_v = cuda.to_device(v)
    d_filtered = cuda.to_device(filtered)

    block = (16, 16)
    grid = ((img_width + block[0] - 1) // block[0], (img_height + block[1] - 1) // block[1])

    kuwahara_filter_kernel[grid, block](d_input, d_h, d_s, d_v, d_filtered, win_size)

    filtered = d_filtered.copy_to_host()

    return filtered

start = time.time()
filtered_img = apply_kuwahara_filter(img_data, win_size=4)
end = time.time()

print(f"Filtered image in {end - start} seconds.")

plt.imshow(filtered_img)
plt.title("Filtered Image (Kuwahara)")
plt.axis('off')  # Turn off axis
plt.show()