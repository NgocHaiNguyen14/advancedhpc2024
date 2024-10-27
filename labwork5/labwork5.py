import numpy as np
from numba import cuda
import math
import time
import matplotlib.pyplot as plt
import numba


gaussian_kernel = np.array([
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0]
], dtype=np.float32)

gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel) # normalize the kernel

@cuda.jit
def gaussian_blur_without_shared_memory(img, output_img, kernel):
    x, y = cuda.grid(2)
    img_height, img_width = img.shape
    kernel_radius = 3 
    
    # Check boundaries of the image
    if x >= kernel_radius and y >= kernel_radius and x < img_height - kernel_radius and y < img_width - kernel_radius:
        pixel_sum = 0.0
        for i in range(-kernel_radius, kernel_radius + 1):
            for j in range(-kernel_radius, kernel_radius + 1):
                pixel_sum += img[x + i, y + j] * kernel[i + kernel_radius, j + kernel_radius]
        
        output_img[x, y] = pixel_sum

@cuda.jit
def gaussian_blur_shared_memory(img, output_img, kernel):
    shared_kernel = cuda.shared.array(shape=(7, 7), dtype=cuda.float32)
   
    x, y = cuda.grid(2)
    img_height, img_width = img.shape
    kernel_radius = 3  
    if cuda.threadIdx.x < 7 and cuda.threadIdx.y < 7:
        shared_kernel[cuda.threadIdx.x, cuda.threadIdx.y] = kernel[cuda.threadIdx.x, cuda.threadIdx.y]
    cuda.syncthreads()
    if x >= kernel_radius and y >= kernel_radius and x < img_height - kernel_radius and y < img_width - kernel_radius:
        pixel_sum = 0.0
        for i in range(-kernel_radius, kernel_radius + 1):
            for j in range(-kernel_radius, kernel_radius + 1):
                pixel_sum += img[x + i, y + j] * shared_kernel[i + kernel_radius, j + kernel_radius]
        output_img[x, y] = pixel_sum
        
def process_image_no_shared(img, kernel):
    img_device = cuda.to_device(img)
    output_img_device = cuda.device_array(img.shape, dtype=np.float32)
    kernel_device = cuda.to_device(kernel)

    threads_per_block = (32, 32)
    blocks_per_grid_x = (img.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (img.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    start_gpu = time.time()
    gaussian_blur_without_shared_memory[blocks_per_grid, threads_per_block](img_device, output_img_device, kernel_device)
    cuda.synchronize()
    end_gpu = time.time()

    output_img = output_img_device.copy_to_host()
    return output_img, end_gpu - start_gpu

def process_image_shared(img, kernel):
    shared_kernel = cuda.shared.array(shape=(7, 7), dtype=numba.float32)
    x, y = cuda.grid(2)
    img_height, img_width = img.shape
    kernel_radius = 3 
    if cuda.threadIdx.x < 7 and cuda.threadIdx.y < 7:
        shared_kernel[cuda.threadIdx.x, cuda.threadIdx.y] = kernel[cuda.threadIdx.x, cuda.threadIdx.y]
    cuda.syncthreads()
    if x >= kernel_radius and y >= kernel_radius and x < img_height - kernel_radius and y < img_width - kernel_radius:
        pixel_sum = 0.0
        for i in range(-kernel_radius, kernel_radius + 1):
            for j in range(-kernel_radius, kernel_radius + 1):
                pixel_sum += img[x + i, y + j] * shared_kernel[i + kernel_radius, j + kernel_radius]
        
        output_img[x, y] = pixel_sum   
    
img = plt.imread('low_reso.jpeg')
if img.ndim == 3:
    img = np.mean(img, axis=2)

# Run the Gaussian blur without shared memory
output_img_no_shared, time_no_shared = process_image_no_shared(img, gaussian_kernel)

plt.imshow(output_img_no_shared, cmap='gray')
plt.title(f"Gaussian Blur without shared memory\nTime: {time_no_shared:.4f} seconds")
plt.show()

# Run the Gaussian blur with shared memory
output_img_shared, time_shared = process_image_shared(img, gaussian_kernel)

plt.imshow(output_img_shared, cmap='gray')
plt.title(f"Gaussian Blur with shared memory\nTime: {time_shared:.4f} seconds")
plt.show()
