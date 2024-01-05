import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

kernel_size = 5
average_filtered_image = cv2.blur(image, (kernel_size, kernel_size))


plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(average_filtered_image, cmap='gray'), plt.title('Average Filtered Image')

plt.show()
