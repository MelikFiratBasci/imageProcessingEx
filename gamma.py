import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

def gamma_correction(img, gamma=1.0):
    gamma_corrected = np.power(img / 255.0, gamma)
    return (gamma_corrected * 255).astype(np.uint8)

gamma_value = 0.5
gamma_corrected_image = gamma_correction(image, gamma=gamma_value)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.hist(image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Original Histogram')

plt.subplot(2, 2, 3), plt.imshow(gamma_corrected_image, cmap='gray'), plt.title('Gamma Corrected Image (gamma = 0.5)')
plt.subplot(2, 2, 4), plt.hist(gamma_corrected_image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Corrected Histogram')

plt.show()
