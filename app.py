import os

from flask import Flask, render_template, request
import cv2 
import numpy as np

from image_processing_algorithms.chapter3_improve_image_quality import * 
from image_processing_algorithms.chapter5_edge_detection import * 
from image_processing_algorithms.chapter7_morphological_image_processing import * 

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            # Lưu file tải lên vào thư mục static
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'original.jpg')
            file.save(filename)

            # Chức năng được chọn từ form
            function = request.form.get('function')

            # Đọc ảnh và áp dụng chức năng tương ứng
            if function == 'threshold':
                processed_image = threshold(filename)
            elif function == 'negative':
                processed_image = negative_transform(filename)
            elif function == 'log_transform':
                processed_image = log_transform(filename)
            elif function == 'power_law_transform':
                processed_image = power_law_transform(filename)
            elif function == 'average_filter':
                processed_image = average_filter(filename)
            elif function == 'weighted_averaging':
                processed_image = weighted_averaging(filename)
            elif function == 'median_filter':
                processed_image = median_filter(filename)
            elif function == 'gray_histogram_balance': 
                processed_image = gray_histogram_balance(filename) 
            elif function == 'sobels_operator':
                processed_image = sobels_operator(filename)
            elif function == 'prewitt_operator':
                processed_image = prewitt_operator(filename)
            elif function == 'laplacian_operator':
                processed_image = laplacian_operator(filename)
            elif function == 'canny_operator':
                processed_image = canny_operator(filename)
            elif function == 'otsu_algorithm':
                processed_image = otsu_algorithm(filename)
            elif function == 'roberts_operator':
                processed_image = roberts_operator(filename)
            elif function == 'erosion_image':
                processed_image = erosion_image(filename)
            elif function == 'dilation_image':
                processed_image = dilation_image(filename)
            # Lưu ảnh đã xử lý
            processed_filename = "processed_" + function + '.jpg'
            processed_filename = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename )
            processed_image.save(processed_filename)

            return render_template('index.html', original=filename, processed=processed_filename)

    return render_template('index.html')

if __name__ == '__main__': 
    app.run(debug = True) 