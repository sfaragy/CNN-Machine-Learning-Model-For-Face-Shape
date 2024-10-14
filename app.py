import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from mtcnn import MTCNN

model = tf.keras.models.load_model('models/face_shape_classifier.h5')

detector = MTCNN()

app = Flask(__name__)


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
        3: 'Heart-shaped',
        4: 'Diamond-shaped'
    }
    return face_shape_dict.get(face_shape_class, 'Unknown')

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
 
    faces = detector.detect_faces(img)
    
    if not faces:
        return jsonify({'error': 'No faces detected'}), 400
    
    results = []
    for face in faces:
        x, y, width, height = face['box']
        face_img = img[y:y+height, x:x+width]
    
        face_shape = classify_face_shape(face_img)
        results.append({
            'shape': face_shape,
            'coordinates': (x, y, width, height)
        })
    
    return jsonify({'faces': results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
