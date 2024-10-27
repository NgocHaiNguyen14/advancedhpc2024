import numpy as np
from numba import cuda, float32
from PIL import Image
import matplotlib.pyplot as plt
import time

################################################################
def rgb_to_hsv_cpu(rgb_image):
    h = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    s = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    v = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    
    for x in range(rgb_image.shape[0]):
        for y in range(rgb_image.shape[1]):
            r, g, b = rgb_image[x, y] / 255.0
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            delta = max_val - min_val
            
            # Calculate H
            if delta == 0:
                h[x, y] = 0
            elif max_val == r:
                h[x, y] = (60 * ((g - b) / delta) + 360) % 360
            elif max_val == g:
                h[x, y] = (60 * ((b - r) / delta) + 120) % 360
            elif max_val == b:
                h[x, y] = (60 * ((r - g) / delta) + 240) % 360
                
            # Calculate S and V
            s[x, y] = 0 if max_val == 0 else delta / max_val
            v[x, y] = max_val
            
    return h, s, v

# CPU HSV to RGB conversion
def hsv_to_rgb_cpu(h, s, v):
    rgb_image = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    
    for x in range(h.shape[0]):
        for y in range(h.shape[1]):
            H, S, V = h[x, y], s[x, y], v[x, y]
            C = V * S
            X = C * (1 - abs((H / 60) % 2 - 1))
            m = V - C
            
            if H < 60:
                r, g, b = C, X, 0
            elif H < 120:
                r, g, b = X, C, 0
            elif H < 180:
                r, g, b = 0, C, X
            elif H < 240:
                r, g, b = 0, X, C
            elif H < 300:
                r, g, b = X, 0, C
            else:
                r, g, b = C, 0, X
                
            rgb_image[x, y] = [(r + m) * 255, (g + m) * 255, (b + m) * 255]
    
    return rgb_image

# Timing and testing function
def test_rgb_hsv_conversion(image_path):
    # Load image
    rgb_image = load_image(image_path)
    print("Image loaded successfully:", rgb_image.shape)
    
    # GPU RGB to HSV
    start_gpu_rgb2hsv = time.time()
    h_gpu, s_gpu, v_gpu = rgb_to_hsv_aos_to_soa(rgb_image)
    gpu_rgb2hsv_time = time.time() - start_gpu_rgb2hsv
    print("GPU RGB to HSV time:", gpu_rgb2hsv_time)

    # CPU RGB to HSV
    start_cpu_rgb2hsv = time.time()
    h_cpu, s_cpu, v_cpu = rgb_to_hsv_cpu(rgb_image)
    cpu_rgb2hsv_time = time.time() - start_cpu_rgb2hsv
    print("CPU RGB to HSV time:", cpu_rgb2hsv_time)
    
    # GPU HSV to RGB
    start_gpu_hsv2rgb = time.time()
    converted_rgb_image_gpu = hsv_to_rgb_soa_to_aos(h_gpu, s_gpu, v_gpu, rgb_image.shape)
    gpu_hsv2rgb_time = time.time() - start_gpu_hsv2rgb
    print("GPU HSV to RGB time:", gpu_hsv2rgb_time)

    # CPU HSV to RGB
    start_cpu_hsv2rgb = time.time()
    converted_rgb_image_cpu = hsv_to_rgb_cpu(h_cpu, s_cpu, v_cpu)
    cpu_hsv2rgb_time = time.time() - start_cpu_hsv2rgb
    print("CPU HSV to RGB time:", cpu_hsv2rgb_time)
    

################################################################
@cuda.jit
def RGB2HSV(rgb, H, S, V):
    x, y = cuda.grid(2)
    if x < rgb.shape[0] and y < rgb.shape[1]:
        r = rgb[x, y, 0] / 255.0
        g = rgb[x, y, 1] / 255.0
        b = rgb[x, y, 2] / 255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        delta = max_val - min_val

        # Calculate H
        if delta == 0:
            h = 0
        elif max_val == r:
            h = 60 * ((g - b) / delta) % 6
        elif max_val == g:
            h = 60 * ((b - r) / delta) + 2
        elif max_val == b:
            h = 60 * ((r - g) / delta) + 4
        # Calculate S
        s = 0 if max_val == 0 else delta / max_val

        # Calculate V
        v = max_val

        # Store the HSV values in separate arrays (SoA)
        H[x, y] = h
        S[x, y] = s
        V[x, y] = v

# Kernel for HSV to RGB conversion, from SoA to AoS
@cuda.jit
def HSV2RGB(H, S, V, rgb):
    x, y = cuda.grid(2)
    if x < rgb.shape[0] and y < rgb.shape[1]:
        h = H[x, y]
        s = S[x, y]
        v = V[x, y]

        c = v * s
        x_val = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if h < 60:
            r, g, b = c, x_val, 0
        elif h < 120:
            r, g, b = x_val, c, 0
        elif h < 180:
            r, g, b = 0, c, x_val
        elif h < 240:
            r, g, b = 0, x_val, c
        elif h < 300:
            r, g, b = x_val, 0, c
        else:
            r, g, b = c, 0, x_val

        rgb[x, y, 0] = (r + m) * 255
        rgb[x, y, 1] = (g + m) * 255
        rgb[x, y, 2] = (b + m) * 255

def rgb_to_hsv_aos_to_soa(rgb_image):
    # Allocate arrays for H, S, V
    h = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    s = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
    v = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)

    # Define grid and block sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = (rgb_image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (rgb_image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]

    # Launch the RGB2HSV kernel
    RGB2HSV[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](rgb_image, h, s, v)
    
    return h, s, v

# Function to perform HSV to RGB conversion
def hsv_to_rgb_soa_to_aos(h, s, v, shape):
    rgb_image = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    threads_per_block = (16, 16)
    blocks_per_grid_x = (shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    HSV2RGB[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](h, s, v, rgb_image)
    
    return rgb_image
    

################################################################
def load_image(path):
    image = Image.open(path).convert("RGB")
    return np.array(image, dtype=np.float32)

def display_images(original, hsv_image, converted_image):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(original.astype(np.uint8))
    ax[0].set_title("Original RGB Image")
    ax[0].axis("off")
    
    # Display the HSV image (for visualization, we can use H as hue, S as saturation, V as value in grayscale)
    hsv_display = np.stack((hsv_image[0], hsv_image[1], hsv_image[2]), axis=-1)
    ax[1].imshow(hsv_display, cmap="hsv")
    ax[1].set_title("HSV Image")
    ax[1].axis("off")
    
    ax[2].imshow(converted_image.astype(np.uint8))
    ax[2].set_title("Converted RGB Image")
    ax[2].axis("off")
    
    plt.show()
    

def test_rgb_hsv_conversion(image_path):
    rgb_image = load_image(image_path)
    print("Image loaded successfully:", rgb_image.shape)
    start_gpu_rgb2hsv = time.time()
    h_gpu, s_gpu, v_gpu = rgb_to_hsv_aos_to_soa(rgb_image)
    gpu_rgb2hsv_time = time.time() - start_gpu_rgb2hsv
    print("GPU RGB to HSV time:", gpu_rgb2hsv_time)
    start_cpu_rgb2hsv = time.time()
    h_cpu, s_cpu, v_cpu = rgb_to_hsv_cpu(rgb_image)
    cpu_rgb2hsv_time = time.time() - start_cpu_rgb2hsv
    print("CPU RGB to HSV time:", cpu_rgb2hsv_time)
    start_gpu_hsv2rgb = time.time()
    converted_rgb_image_gpu = hsv_to_rgb_soa_to_aos(h_gpu, s_gpu, v_gpu, rgb_image.shape)
    gpu_hsv2rgb_time = time.time() - start_gpu_hsv2rgb
    print("GPU HSV to RGB time:", gpu_hsv2rgb_time)
    start_cpu_hsv2rgb = time.time()
    converted_rgb_image_cpu = hsv_to_rgb_cpu(h_cpu, s_cpu, v_cpu)
    cpu_hsv2rgb_time = time.time() - start_cpu_hsv2rgb
    print("CPU HSV to RGB time:", cpu_hsv2rgb_time)

    # Display the images and print time comparisons
    display_images(rgb_image, (h_gpu, s_gpu, v_gpu), converted_rgb_image_gpu)
    
    # Plotting execution times
    labels = ['RGB to HSV', 'HSV to RGB']
    gpu_times = [gpu_rgb2hsv_time, gpu_hsv2rgb_time]
    cpu_times = [cpu_rgb2hsv_time, cpu_hsv2rgb_time]
    
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, gpu_times, width, label='GPU')
    bars2 = ax.bar(x + width/2, cpu_times, width, label='CPU')

    # Labels and formatting
    ax.set_xlabel('Conversion Type')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time Comparison: GPU vs. CPU')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Display time values on top of bars
    for bar in bars1 + bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

    plt.show()

################################################################



image_path = "/home/hai/Github-Repositories/advancedhpc2024/labwork3/high_reso.jpg"
test_rgb_hsv_conversion(image_path)

