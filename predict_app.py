import base64
import numpy as np
import io
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from keras import backend as k
import keras
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask


app = Flask(__name__)

def get_model():
    global model
    model = load_model('vgg19.h5')
    print('*************model loaded bitch!!!!*********')

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

print("*****************Loading keras model************")
get_model()

@app.route("/predict", methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)

    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224,224))
    prediction = model.predict(processed_image).tolist()
    print("predicting on server")

    response = {
    'prediction': {
            'Bike' : prediction[0][0],
            'Car' : prediction[0][1],
            'Cat' : prediction[0][2],
            'Dog' : prediction[0][3],
            'Human' : prediction[0][4]
                }
            }
    return jsonify(response)
