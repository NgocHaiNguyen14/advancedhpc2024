import numpy as np
from numba import cuda, float32
from PIL import Image
import matplotlib.pyplot as plt

@cuda.jit
def RGB2HSV(rgb, H, S, V):
    x, y = cuda.grid(2)
    if x < rgb.shape[0] and y < rgb.shape[1]:
        # Read RGB values and normalize to [0, 1]
        r = rgb[x, y, 0] / 255.0
        g = rgb[x, y, 1] / 255.0
        b = rgb[x, y, 2] / 255.0

        # Find max, min, and delta
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

        # Scale back to [0, 255] and store in AoS format
        rgb[x, y, 0] = (r + m) * 255
        rgb[x, y, 1] = (g + m) * 255
        rgb[x, y, 2] = (b + m) * 255

# Function to perform RGB to HSV conversion
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
    # Allocate the output RGB array in AoS format
    rgb_image = np.zeros((shape[0], shape[1], 3), dtype=np.float32)

    # Define grid and block sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = (shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (shape[1] + threads_per_block[1] - 1) // threads_per_block[1]

    # Launch the HSV2RGB kernel
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
    # Load image
    rgb_image = load_image(image_path)
    print("Image loaded successfully:", rgb_image.shape)
    
    # Perform RGB (AoS) to HSV (SoA) conversion
    h, s, v = rgb_to_hsv_aos_to_soa(rgb_image)
    print("RGB to HSV conversion complete.")

    # Convert back from HSV (SoA) to RGB (AoS)
    converted_rgb_image = hsv_to_rgb_soa_to_aos(h, s, v, rgb_image.shape)
    print("HSV to RGB conversion complete.")

    # Display the results
    display_images(rgb_image, (h, s, v), converted_rgb_image)

################################################################



image_path = "/home/hai/Github-Repositories/advancedhpc2024/labwork3/high_reso.jpg"
test_rgb_hsv_conversion(image_path)

