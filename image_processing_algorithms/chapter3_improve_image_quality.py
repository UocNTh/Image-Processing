import cv2 
from PIL import Image 
import numpy as np 

# Âm bản
def negative_transform(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    negative_img = 255 - img
    negative_img = cv2.cvtColor(negative_img, cv2.COLOR_GRAY2RGB) 

    # Chuyển đổi NumPy array thành đối tượng Image của Pillow
    pil_image = Image.fromarray(negative_img)

    return pil_image

# Phân ngưỡng 
def threshold(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Xử lý ảnh theo chức năng phân ngưỡng
    return Image.fromarray(img).convert("L").point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')

# Biến đổi Logarith
def log_transform(image_path):
    img = cv2.imread(image_path)
    c = 255 / np.log(1 + np.max(img))
    log_transformed = c * np.log(1 + img)

    # Chuyển đổi kiểu dữ liệu
    log_transformed = np.array(log_transformed, dtype=np.uint8)
    pil_image = Image.fromarray(log_transformed)

    return pil_image

# Biến đổi hàm mũ 
def power_law_transform(image_path, gamma = 2):
    original_img = cv2.imread(image_path)
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype='uint8')

    pil_image = Image.fromarray(gamma_corrected)
    return pil_image

#  Cân bằng lược đồ xám
def gray_histogram_balance(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    equ = cv2.equalizeHist(img)
    
    pil_image = Image.fromarray(equ)
    return pil_image

# Bộ lọc trung bình
def average_filter(image_path):
    original_image = cv2.imread(image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    m, n = img.shape 

    mask = np.ones([3, 3], dtype = int) 
    mask = mask / 9
    img_new = np.zeros([m, n])

    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = img[i-1, j-1]*mask[0, 0]+img[i-1, j]*mask[0, 1]+img[i-1, j + 1]*mask[0, 2]+img[i, j-1]*mask[1, 0]+ img[i, j]*mask[1, 1]+img[i, j + 1]*mask[1, 2]+img[i + 1, j-1]*mask[2, 0]+img[i + 1, j]*mask[2, 1]+img[i + 1, j + 1]*mask[2, 2] 
            
            img_new[i, j]= temp 

    img_new = img_new.astype(np.uint8) 

    pil_image = Image.fromarray(img_new)
    return pil_image

# Bộ lọc trung bị có trọng số
def weighted_averaging(image_path):
    # Định nghĩa ma trận trọng số cho bộ lọc trung bình có trọng số
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16

    # Đọc hình ảnh
    input_image = cv2.imread(image_path)

    # Áp dụng bộ lọc trung bình có trọng số
    processed_image = cv2.filter2D(input_image, -1, kernel)

    # Chuyển đổi kết quả trở lại thành một đối tượng PIL Image
    pil_image = Image.fromarray(processed_image)

    return pil_image

# Bộ lọc trung vị 
def median_filter(image_path):
    original_image = cv2.imread(image_path)
    img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    m, n = img.shape 

    img_new1 = np.zeros([m, n])
    for i in range(1, m-1): 
        for j in range(1, n-1): 
            temp = [img[i-1, j-1], 
                img[i-1, j], 
                img[i-1, j + 1], 
                img[i, j-1], 
                img[i, j], 
                img[i, j + 1], 
                img[i + 1, j-1], 
                img[i + 1, j], 
                img[i + 1, j + 1]] 
            
            temp = sorted(temp) 
            img_new1[i, j]= temp[4] 
    
    img_new1 = img_new1.astype(np.uint8) 
    pil_image = Image.fromarray(img_new1)
    return pil_image
