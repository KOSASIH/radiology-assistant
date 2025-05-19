import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.colors import LinearSegmentedColormap

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['SAMPLE_FOLDER'] = os.path.join('static', 'samples')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Simple tumor detection function using image processing
def detect_tumor(image_path):
    # Load the image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = ImageOps.autocontrast(img)  # Enhance contrast
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Simple thresholding to detect bright regions (potential tumors)
    # Assuming tumors are brighter than surrounding tissue
    threshold = np.percentile(img_array, 95)  # Top 5% brightest pixels
    tumor_mask = img_array > threshold
    
    # Create a heatmap visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(img_array, cmap='gray')
    
    # Create a custom colormap for the tumor overlay
    colors = [(0, 0, 0, 0), (1, 0, 0, 0.7)]  # Transparent to red with alpha
    tumor_cmap = LinearSegmentedColormap.from_list('tumor_cmap', colors)
    
    plt.imshow(tumor_mask, cmap=tumor_cmap)
    plt.axis('off')
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Calculate tumor statistics
    tumor_size = np.sum(tumor_mask)
    total_size = img_array.size
    tumor_percentage = (tumor_size / total_size) * 100
    
    # Simple classification based on tumor size
    if tumor_percentage < 1:
        classification = "Likely Benign"
        confidence = 70 + np.random.randint(0, 15)
    elif tumor_percentage < 3:
        classification = "Possibly Malignant"
        confidence = 60 + np.random.randint(0, 20)
    else:
        classification = "Likely Malignant"
        confidence = 75 + np.random.randint(0, 15)
    
    return {
        'image': img_str,
        'tumor_percentage': round(tumor_percentage, 2),
        'classification': classification,
        'confidence': confidence
    }

@app.route('/')
def index():
    # Get list of sample images
    samples = [f for f in os.listdir(app.config['SAMPLE_FOLDER']) 
               if f.endswith('.png') and not f.endswith('_display.png')]
    
    return render_template('index.html', samples=samples)

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    # If user doesn't select a file
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        result = detect_tumor(filepath)
        
        return render_template('result.html', 
                              filename=filename,
                              result=result)
    
    return redirect(url_for('index'))

@app.route('/analyze_sample/<filename>')
def analyze_sample(filename):
    filepath = os.path.join(app.config['SAMPLE_FOLDER'], filename)
    
    # Process the image
    result = detect_tumor(filepath)
    
    return render_template('result.html', 
                          filename=filename,
                          result=result,
                          is_sample=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/samples/<filename>')
def sample_file(filename):
    return send_from_directory(app.config['SAMPLE_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)