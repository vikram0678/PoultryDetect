
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
model = tf.keras.models.load_model('healthy_vs_rotten.h5')
labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def output():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file uploaded"
        file = request.files['image']
        if file.filename == '':
            return "No file selected"
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join('static', 'uploads')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        img_path = os.path.join(upload_dir, file.filename)
        file.save(img_path)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)
        prediction = labels[np.argmax(prediction)]
        return render_template('contact.html', predict=prediction)
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)