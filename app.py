import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from pyngrok import ngrok
from tensorflow.keras.preprocessing.image import load_img, img_to_array
app = Flask(__name__)

model =load_model('final_model.h5')
# print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}


def getResult(image_path):
    img = load_img(image_path, target_size=(225,225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        predictions=getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return str(predicted_label)
    return None


# ngrok.set_auth_token("")
# # Open a HTTP tunnel on the default Flask port (5000)
# public_url = ngrok.connect(5001)
# print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:6000\"".format(public_url))

if __name__ == '__main__':
    app.run(port = 5000)