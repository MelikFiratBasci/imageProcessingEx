import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

kernel_opening = np.ones((5, 5), np.uint8)
opening_result = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_opening)

kernel_closing = np.ones((5, 5), np.uint8)
closing_result = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel_closing)

plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1), plt.imshow(binary_image, cmap='gray'), plt.title('Binary Image')
plt.subplot(1, 3, 2), plt.imshow(opening_result, cmap='gray'), plt.title('Opening Result')
plt.subplot(1, 3, 3), plt.imshow(closing_result, cmap='gray'), plt.title('Closing Result')
plt.show()
