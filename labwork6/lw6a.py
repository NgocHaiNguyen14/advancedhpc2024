import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.image import imread

image_path = 'highest_reso.jpeg'
image_array = imread(image_path)

#convert to gray
if image_array.ndim == 3:
    image_array = 0.333 * ( image_array[:, :, 0] + image_array[:, :, 1] +  image_array[:, :, 2]) * 255
    image_array = image_array.astype(np.uint8)

threshold = 128


#function for map on each pixel
@njit
def binarize_image(img, threshold):
    binary_img = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            binary_img[i, j] = 255 if img[i, j] > threshold else 0
    return binary_img


#binarize the image
binary_image = binarize_image(image_array, threshold)

#show the result
plt.imshow(binary_image, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()
