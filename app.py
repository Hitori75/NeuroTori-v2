from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import os
from huggingface_hub import hf_hub_download
import patoolib

# Inisialisasi Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Konfigurasi untuk file model dari Hugging Face
repo_id = "Hitori75/NeuroTori"
rar_file = "Brain Tumor Model.rar"  # Diperbaiki dari "Brain_Tumor_Model.rar"
download_dir = "./downloaded_model"
model_dir = os.path.join(download_dir, "Brain Tumor Model")

# Fungsi untuk mengunduh dan memuat model
def load_model():
    if not os.path.exists(model_dir):
        os.makedirs(download_dir, exist_ok=True)
        rar_path = hf_hub_download(repo_id=repo_id, filename=rar_file, local_dir=download_dir)
        try:
            print(f"Extracting {rar_file}...")
            patoolib.extract_archive(rar_path, outdir=download_dir)
        except Exception as e:
            print(f"Failed to extract {rar_file}: {str(e)}")
            raise

    try:
        model = tf.saved_model.load(model_dir)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Inisialisasi model saat aplikasi mulai
try:
    model = load_model()
    infer = model.signatures['serving_default']
except Exception as e:
    print(f"Initialization failed: {str(e)}")
    infer = None

# Preprocess image function
def preprocess_image(image):
    try:
        img = Image.open(image).convert('RGB')
        if img.size[0] < 100 or img.size[1] < 100:
            return jsonify({'error': 'Image too small, likely not an MRI scan'}), 400

        img_array = np.array(img, dtype=np.float32)
        rgb_diff = np.abs(img_array[:, :, 0] - img_array[:, :, 1]) + np.abs(img_array[:, :, 1] - img_array[:, :, 2])
        if np.mean(rgb_diff) > 10:
            return jsonify({'error': 'Image has excessive color variation, not an MRI scan'}), 400

        gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if np.std(gray_img) < 20:
            return jsonify({'error': 'Image lacks sufficient contrast, not an MRI scan'}), 400

        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea, default=[])) < 10000:
            return jsonify({'error': 'Image lacks significant brain-like structure, not an MRI brain scan'}), 400

        largest_contour = max(contours, key=cv2.contourArea, default=[])
        if len(largest_contour) > 0:
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            if perimeter == 0:
                return jsonify({'error': 'No valid contour perimeter found, likely not an MRI brain scan'}), 400

            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            print(f"Circularity of largest contour: {circularity}")
            if circularity < 0.1:
                return jsonify({'error': 'Image structure not circular enough, likely not an MRI brain scan'}), 400
        else:
            return jsonify({'error': 'No significant contour found, likely not an MRI brain scan'}), 400

        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        peak_intensity = np.argmax(hist)
        print(f"Histogram peak intensity: {peak_intensity}")

        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return tf.constant(img_array)
    except Exception as e:
        return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

@app.route('/')
def index():
    if infer is None:
        return jsonify({'error': 'Model failed to initialize. Please check server logs.'}), 500
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if infer is None:
        return jsonify({'error': 'Model failed to initialize. Please check server logs.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    model_choice = request.form.get('model', 'vgg16')
    
    img_result = preprocess_image(file)
    if isinstance(img_result, tuple):
        return img_result

    img = img_result
    prediction = infer(img)
    print("Full model output:", {k: v.numpy() for k, v in prediction.items()})
    
    output_key = list(prediction.keys())[0]
    prob = prediction[output_key].numpy()[0][0]
    
    if 0.45 <= prob <= 0.55:
        return jsonify({'error': 'Prediction confidence too low, possibly not an MRI brain scan'}), 400

    accuracy = 0.98
    result = "Tumor Detected" if prob > 0.5 else "No Tumor"

    response_data = {
        'prediction': result,
        'accuracy': accuracy * 100
    }
    print("Raw prob:", prob, "Response:", response_data)

    return jsonify(response_data)

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))