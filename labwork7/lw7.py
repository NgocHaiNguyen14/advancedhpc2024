import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
from PIL import Image

@cuda.jit
def rgb_to_gray_kernel(image, gray_image):
    height, width = image.shape[0], image.shape[1]
    x, y = cuda.grid(2)
    
    if x < height and y < width:
        r = image[x, y, 0]
        g = image[x, y, 1]
        b = image[x, y, 2]
        gray_image[x, y] = 0.333333 * (r + g + b)

@cuda.jit
def find_min_max_kernel(gray_image, min_max):
    x, y = cuda.grid(2) # retrieves the 2D grid coordinates of the thread that is executing this kernel.
    
    if x < gray_image.shape[0] and y < gray_image.shape[1]: #boundary check
        value = gray_image[x, y]
        cuda.atomic.min(min_max, 0, value)  # atomic operation to find min - cheating should be replaced
        cuda.atomic.max(min_max, 1, value)  # atomic operation to find max

@cuda.jit
def linear_recalc_kernel(gray_image, min_val, max_val, output_image):
    x, y = cuda.grid(2)
    
    if x < gray_image.shape[0] and y < gray_image.shape[1]:
        value = gray_image[x, y]
        output_image[x, y] = (value - min_val) / (max_val - min_val) * 255

def main(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # Allocate memory on the GPU
    gray_image_gpu = cuda.device_array(image_np.shape[:2], dtype=np.float32)
    min_max_gpu = cuda.device_array(2, dtype=np.float32)
    min_max_cpu = np.array([np.inf, -np.inf], dtype=np.float32)  # [min, max]
    output_image_gpu = cuda.device_array(image_np.shape[:2], dtype=np.float32)

    # Launch kernel to convert to grayscale
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(image_np.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(image_np.shape[1] / threads_per_block[1]))
    rgb_to_gray_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](image_np, gray_image_gpu)

    # Launch kernel to find min and max intensity
    find_min_max_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](gray_image_gpu, min_max_gpu)

    # Copy min and max values from GPU to CPU
    min_max_cpu = min_max_gpu.copy_to_host()

    # Launch kernel to linearly recalculate intensity
    linear_recalc_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
        gray_image_gpu, min_max_cpu[0], min_max_cpu[1], output_image_gpu
    )

    # Copy the output image back to CPU
    output_image_np = output_image_gpu.copy_to_host().astype(np.uint8)

    # Display the recalculated intensity image
    plt.imshow(output_image_np, cmap='gray')
    plt.axis('off')
    plt.show()

    # Print min and max intensity
    print(f'Min intensity: {min_max_cpu[0]}')
    print(f'Max intensity: {min_max_cpu[1]}')

# Run the main function
image_path = 'high_reso.jpg'  # Replace with your image path
main(image_path)
