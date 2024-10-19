import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import time
import matplotlib
matplotlib.use('TkAgg')  
def rgb_to_grayscale_cpu(img):
    gray_img = np.zeros(img.shape[0], dtype=np.float32)
    for i in range(img.shape[0]):
        gray_img[i] = 0.299 * img[i, 0] + 0.587 * img[i, 1] + 0.114 * img[i, 2] #Formula from Internet
    return gray_img

@cuda.jit
def rgb_to_grayscale_gpu(img, gray_img):
    i = cuda.grid(1)  #  the grid and block are organized in a 1-dimensional (linear) structure
    if i < img.shape[0]:
        gray_img[i] = 0.299 * img[i, 0] + 0.587 * img[i, 1] + 0.114 * img[i, 2]

def process_image(img):
    if img.shape[2] == 4:
        img = img[:, :, :3]  
    pixel_count = img.shape[0] * img.shape[1] #flat image
    img_flat = img.reshape(pixel_count, 3)

    # --- CPU exe time ---
    start_cpu = time.time()
    gray_img_cpu = rgb_to_grayscale_cpu(img_flat)
    end_cpu = time.time()
    cpu_time = end_cpu - start_cpu

    # --- GPU exe time ---
    gray_img_gpu = np.zeros(pixel_count, dtype=np.float32)
    img_flat_device = cuda.to_device(img_flat) #the data is transferred to the GPU
    gray_img_gpu_device = cuda.to_device(gray_img_gpu) #run code by cuda
    threads_per_block = 256 #Threads per block is set to 256 - based on GPU
    blocks_per_grid = (pixel_count + (threads_per_block - 1)) // threads_per_block 
	#all pixel covered by threads
    start_gpu = time.time()
    rgb_to_grayscale_gpu[blocks_per_grid, threads_per_block](img_flat_device, gray_img_gpu_device)
    gray_img_gpu_device.copy_to_host(gray_img_gpu)
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu

    return gray_img_cpu, gray_img_gpu, cpu_time, gpu_time

# load the images
img1 = plt.imread('low_reso.jpeg') #image with low resoluotion
img2 = plt.imread('high_reso.jpg') #image with medium resolution
img3 = plt.imread('highest_reso.jpeg') #imgae with high resolution

# print resolutions of the images
print(f"Resolution of image 1: {img1.shape[1]} x {img1.shape[0]}")
print(f"Resolution of image 2: {img2.shape[1]} x {img2.shape[0]}")
print(f"Resolution of image 3: {img3.shape[1]} x {img3.shape[0]}")
gray_img_cpu1, gray_img_gpu1, cpu_time1, gpu_time1 = process_image(img1)
gray_img_cpu2, gray_img_gpu2, cpu_time2, gpu_time2 = process_image(img2)
gray_img_cpu3, gray_img_gpu3, cpu_time3, gpu_time3 = process_image(img3)

# ploting graph GPU
plt.figure(figsize=(12, 6))
plt.subplot(3, 2, 1)
plt.imshow(gray_img_cpu1.reshape(img1.shape[0], img1.shape[1]), cmap='gray')
plt.title("Grayscale image 1 (CPU)")
plt.subplot(3, 2, 2)
plt.imshow(gray_img_gpu1.reshape(img1.shape[0], img1.shape[1]), cmap='gray')
plt.title("Grayscale image 1 (GPU)")

plt.subplot(3, 2, 3)
plt.imshow(gray_img_cpu2.reshape(img2.shape[0], img2.shape[1]), cmap='gray')
plt.title("Grayscale image 2 (CPU)")
plt.subplot(3, 2, 4)
plt.imshow(gray_img_gpu2.reshape(img2.shape[0], img2.shape[1]), cmap='gray')
plt.title("Grayscale image 2 (GPU)")

plt.subplot(3, 2, 5)
plt.imshow(gray_img_cpu3.reshape(img3.shape[0], img3.shape[1]), cmap='gray')
plt.title("Grayscale image 3 (CPU)")
plt.subplot(3, 2, 6)
plt.imshow(gray_img_gpu3.reshape(img3.shape[0], img3.shape[1]), cmap='gray')
plt.title("Grayscale image 3 (GPU)")

plt.tight_layout()
plt.show()

#ploting graph CPU
images = ['low_reso_img', 'moderate_reso_img', 'high_reso_img']
cpu_times = [cpu_time1, cpu_time2, cpu_time3]

plt.figure(figsize=(8, 6))
plt.bar(images, cpu_times, width=0.4, label='CPU', color='blue')
plt.xlabel("Images")
plt.ylabel("Execution time (seconds)")
plt.title("CPU execution time")
plt.show()

# Plotting graph GPU
gpu_times = [gpu_time1, gpu_time2, gpu_time3]

plt.figure(figsize=(8, 6))
plt.bar(images, gpu_times, width=0.4, label='GPU', color='green')
plt.xlabel("Images")
plt.ylabel("Execution time (seconds)")
plt.title("GPU execution time")
plt.show()
