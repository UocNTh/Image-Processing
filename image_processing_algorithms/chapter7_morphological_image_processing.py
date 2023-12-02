from PIL import Image
import cv2
import numpy as np

# phép co
def dilation_image(image_path):
    original_image = cv2.imread(image_path)
    kernel = np.ones((5, 5), np.uint8)  # Sử dụng kernel lớn hơn
    processedImage = cv2.erode(original_image, kernel, iterations=2)  # Tăng số lần lặp
    pil_image = Image.fromarray(processedImage)
    return pil_image

# phép giãn
def erosion_image(image_path):
    original_image = cv2.imread(image_path)
    kernel = np.ones((5, 5), np.uint8)  # Sử dụng kernel lớn hơn
    processedImage = cv2.dilate(original_image, kernel, iterations=2)  # Tăng số lần lặp
    pil_image = Image.fromarray(processedImage)
    return pil_image
