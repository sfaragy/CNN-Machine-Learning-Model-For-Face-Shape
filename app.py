import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('models/face_shape_classifier_by_cnn.h5')

def classify_face_shape(image):
    image_resized = cv2.resize(image, (128, 128))
    image_resized = image_resized.astype('float32') / 255.0
    image_resized = np.expand_dims(image_resized, axis=0)

    predictions = model.predict(image_resized)
    face_shape_class = np.argmax(predictions)

    face_shape_dict = {
        0: 'Oval',
        1: 'Round',
        2: 'Square',
        3: 'Heart',
        4: 'Diamond'
    }
    return face_shape_dict.get(face_shape_class, 'Unknown')


@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    face_shape = classify_face_shape(img)
    return jsonify({'face_shape': face_shape})


@app.route('/label', methods=['POST'])
def label_image():
    file = request.files['file']
    label = request.form['label']

    output_dir = os.path.join('dataset', label)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_name = os.path.join(output_dir, file.filename)
    cv2.imwrite(img_name, img)

    return jsonify({'message': f'Image saved as {label}'}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
