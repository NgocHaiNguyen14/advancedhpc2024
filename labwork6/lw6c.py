import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.transform import resize

# Load the images
image1_path = 'highest_reso.jpeg'  
image2_path = 'high_reso.jpg'  
image1 = imread(image1_path)
image2 = imread(image2_path)

if image1.ndim == 3:
    image1 = (0.333333 * (image1[:, :, 0] + image1[:, :, 1] + image1[:, :, 2])) * 255
    image1 = image1.astype(np.uint8)

if image2.ndim == 3:
    image2 = (0.333333 * (image2[:, :, 0] + image2[:, :, 1] + image2[:, :, 2])) * 255
    image2 = image2.astype(np.uint8)

# Resize image2 to match image1
image2 = resize(image2, image1.shape, anti_aliasing=True, mode='reflect')

blend_factor = 0.5
 
@njit
def blend_images(img1, img2, factor):
    blended_img = np.zeros(img1.shape, dtype=np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            value = img1[i, j] * factor + img2[i, j] * (1 - factor)
            # Manual clipping
            if value < 0:
                blended_img[i, j] = 0
            elif value > 255:
                blended_img[i, j] = 255
            else:
                blended_img[i, j] = value
    return blended_img


# Blend the images
blended_image = blend_images(image1, image2, blend_factor)

plt.imshow(blended_image, cmap='gray')
plt.axis('off')  # Hide axes
plt.show()

