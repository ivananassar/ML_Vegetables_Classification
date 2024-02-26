from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
import os
import uuid

# Load your pre-trained model from the .h5 file
model = tf.keras.models.load_model('model.h5')
app=Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST', 'GET'])


def predict():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' in request.files:
            uploaded_image = request.files['image']

            # Generate a unique filename for the uploaded image
            unique_filename = str(uuid.uuid4()) + '.jpg'
            # Save the uploaded image to the "static" folder with the unique filename
            image_path = os.path.join('static', unique_filename)
            uploaded_image.save(image_path)

            # Preprocess the image
            img = Image.open(image_path)
            img = img.resize((32, 32))  # Resize to match the input shape of the model
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0  # Normalize pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Use the loaded model to make predictions
            predictions = model.predict(img_array)

            # Process the predictions (convert them to class names or labels)
            predicted_class_index = np.argmax(predictions)
            class_names = ['apple', 'banana', 'beetroot', 'cabbage', 'capsicum', 'carrot', 'cucumber', 'grape', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'pear', 'pineapple', 'pomegranate', 'potato', 'raddish', 'tomato', 'watermelon']
            predicted_class_name = class_names[predicted_class_index]

            # Save the uploaded image to the "static" folder
            uploaded_image.save(os.path.join('static', uploaded_image.filename))

            return render_template("index.html", image_filename=unique_filename, name=predicted_class_name)


    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
