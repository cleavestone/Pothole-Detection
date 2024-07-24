from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['MODEL_PATH'] = 'model/detector2.pt'

# Load YOLOv8 model
model = torch.hub.load('ultralytics/yolov5', 'yolov8', path=app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    # Load image
    img = Image.open(image_path)
    img = np.array(img)
    
    results = model(img)

    # Save results
    result_image_path = os.path.join('static', 'results.jpg')
    results.save(result_image_path)
    
    return result_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            result_image_path = process_image(file_path)
            
            return render_template('result.html', result_image=result_image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
