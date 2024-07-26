from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}
app.config['MODEL_PATH'] = 'model/detector2.pt'

# Load your YOLOv8 model
model = YOLO(app.config['MODEL_PATH'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)

    # Perform inference
    results = model(img)

    # Extract boxes, scores, and class labels from results
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    labels = results[0].boxes.cls

    # Convert to tensors for PyTorch NMS
    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)

    # Apply Non-Max Suppression (NMS)
    nms_indices = torch.ops.torchvision.nms(boxes_tensor, scores_tensor, 0.4).numpy()

    # Draw bounding boxes for remaining detections
    for i in nms_indices:
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = float(scores[i])
        label_text = f"pothole {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save result image
    result_image_path = os.path.join('static', 'results.jpg')
    cv2.imwrite(result_image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

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

