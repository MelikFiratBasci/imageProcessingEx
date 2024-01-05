import cv2
import numpy as np
import matplotlib.pyplot as plt

image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
red_channel, green_channel, blue_channel = cv2.split(image_rgb)

inverse_red_channel = 255 - red_channel
processed_image = cv2.merge([inverse_red_channel, green_channel, blue_channel])

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1), plt.imshow(image_rgb), plt.title('Original Image (RGB)')
plt.subplot(1, 2, 2), plt.imshow(processed_image), plt.title('Processed Image')
plt.show()
