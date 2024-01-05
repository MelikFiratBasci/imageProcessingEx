import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

def contrast_stretching(img, min_out=0, max_out=255):
    min_in = np.min(img)
    max_in = np.max(img)
    stretched_img = (img - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out
    return np.clip(stretched_img, min_out, max_out).astype(np.uint8)

stretched_image = contrast_stretching(image)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.hist(image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Original Histogram')

plt.subplot(2, 2, 3), plt.imshow(stretched_image, cmap='gray'), plt.title('Contrast Stretched Image')
plt.subplot(2, 2, 4), plt.hist(stretched_image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Stretched Histogram')

plt.show()
