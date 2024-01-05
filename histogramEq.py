import cv2
import matplotlib.pyplot as plt


image_url = "https://www.simplilearn.com/ice9/free_resources_article_thumb/what_is_image_Processing.jpg"
image_response = urlopen(image_url)
image_array = np.asarray(bytearray(image_response.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

equ_hist_image = cv2.equalizeHist(image)

plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.hist(image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Original Histogram')

plt.subplot(2, 2, 3), plt.imshow(equ_hist_image, cmap='gray'), plt.title('Equalized Image')
plt.subplot(2, 2, 4), plt.hist(equ_hist_image.flatten(), 256, [0, 256], color='r', histtype='step'), plt.title('Equalized Histogram')

plt.show()
