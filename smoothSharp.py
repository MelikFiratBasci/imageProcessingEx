import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

blur_kernel_size = (5, 5)
blurred_image = cv2.GaussianBlur(image, blur_kernel_size, 0)

laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
sharpened_image = image - laplacian_image

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(blurred_image, cmap='gray'), plt.title('Blurred Image (Smoothing)')
plt.subplot(1, 3, 3), plt.imshow(sharpened_image, cmap='gray'), plt.title('Sharpened Image')

plt.show()
