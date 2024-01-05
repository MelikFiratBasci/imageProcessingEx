import cv2
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
import io


image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"  
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

# Görüntüyü 8 bit düzlemine ayır
bit_planes = [np.bitwise_and(image, 2**i) for i in range(8)]

# Bit plane'leri görselleştir
plt.figure(figsize=(10, 6))
plt.subplot(2, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')

for i in range(7, -1, -1):
    plt.subplot(2, 4, 8 - i), plt.imshow(bit_planes[i], cmap='gray'), plt.title(f'Bit {i}')

plt.show()
