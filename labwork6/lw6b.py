import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.image import imread

image_path = 'highest_reso.jpeg'
image_array = imread(image_path)

if image_array.ndim == 3:
    image_array = (0.333 * (image_array[:, :, 0] + image_array[:, :, 1] + image_array[:, :, 2])) * 255
    image_array = image_array.astype(np.uint8)

brightness_offset = -10  # brightness control

@njit
def adjust_brightness(img, offset):
    bright_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            value = img[i, j] + offset
            if value < 0:
                bright_img[i, j] = 0
            elif value > 255:
                bright_img[i, j] = 255
            else:
                bright_img[i, j] = value
    return bright_img

bright_image = adjust_brightness(image_array, brightness_offset)

plt.imshow(bright_image, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()
