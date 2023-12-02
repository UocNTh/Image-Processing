from PIL import Image
import cv2
import numpy as np

# Toán tử Roberts
def roberts_operator(image_path):
    original_image = cv2.imread(image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, -1], [1, 0]], dtype=np.float32)

    
    roberts_x_edges = cv2.filter2D(img, -1, roberts_x)
    roberts_y_edges = cv2.filter2D(img, -1, roberts_y)

    roberts_x_edges = cv2.filter2D(img, -1, roberts_x)
    roberts_y_edges = cv2.filter2D(img, -1, roberts_y)

    roberts_x_edges = roberts_x_edges.astype(np.float32)
    roberts_y_edges = roberts_y_edges.astype(np.float32)

    processedImage = cv2.magnitude(roberts_x_edges, roberts_y_edges)
    processedImage = processedImage.astype(np.uint8)

    pil_image = Image.fromarray(processedImage)
    return pil_image

# Toán tử Sobel 
def sobels_operator(image_path):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    processedImage = cv2.magnitude(sobel_x, sobel_y)
    processedImage = processedImage.astype(np.uint8)

    pil_image = Image.fromarray(processedImage)
    return pil_image

# Toán tử Prewitt
def prewitt_operator(image_path):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    prewitt_kernel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
    prewitt_kernel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)

    prewitt_x = cv2.filter2D(
        gray_image, cv2.CV_64F, prewitt_kernel_x[0] * prewitt_kernel_x[1]
    )
    prewitt_y = cv2.filter2D(
        gray_image, cv2.CV_64F, prewitt_kernel_y[0] * prewitt_kernel_y[1]
    )

    processedImage = cv2.magnitude(prewitt_x, prewitt_y)
    processedImage = processedImage.astype(np.uint8)

    pil_image = Image.fromarray(processedImage)
    return pil_image

# Laplacian
def laplacian_operator(image_path):
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    laplace = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplace = laplace.astype(np.uint8)

    pil_image = Image.fromarray(laplace)
    return pil_image


def canny_operator(image_path):
    original_image = cv2.imread(image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img, (3,3), 0) 

    canny = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    canny = canny.astype(np.uint8)

    pil_image = Image.fromarray(canny)
    return pil_image

# Thuật toán OTSU
def otsu_algorithm(image_path):
    original_image = cv2.imread(image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU) 
    thresh1 = thresh1.astype(np.uint8)
    
    pil_image = Image.fromarray(thresh1)
    return pil_image
