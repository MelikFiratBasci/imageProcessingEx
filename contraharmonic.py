import cv2
import numpy as np
import matplotlib.pyplot as plt


image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

def contraharmonic_mean_filter(image, window_size, Q):
    filtered_image = np.zeros_like(image, dtype=np.float32)

    padding = window_size // 2
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)

    for i in range(padding, image.shape[0] + padding):
        for j in range(padding, image.shape[1] + padding):
            window = padded_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            numerator = np.sum(np.power(window, Q + 1))
            denominator = np.sum(np.power(window, Q))
            filtered_image[i - padding, j - padding] = numerator / max(denominator, 1)

    return np.uint8(filtered_image)

window_size = 3 
Q = 1.5 

filtered_image = contraharmonic_mean_filter(image, window_size, Q)


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 2, 2), plt.imshow(filtered_image, cmap='gray'), plt.title('Contraharmonic Mean Filtered Image')
plt.show()
