import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size

  
    num_salt = np.ceil(salt_prob * total_pixels)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

  
    num_pepper = np.ceil(pepper_prob * total_pixels)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image


salt_prob = 0.01
pepper_prob = 0.01
noisy_image = add_salt_and_pepper_noise(image, salt_prob, pepper_prob)


plt.figure(figsize=(15, 6))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy Image (Salt-and-Pepper)')
plt.show()
