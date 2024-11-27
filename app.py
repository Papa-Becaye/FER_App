from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image

import os
import numpy as np


app = Flask(__name__)
model = load_model('FER_model.h5')


@app.route('/')
def home():
    return render_template('main.html')


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']

        if file:
            filename = file.filename
            file_path = os.path.join('static\images', filename)
            file.save(file_path)

        img = load_img(file_path, target_size=(224, 224))
        x = image.img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images)
        labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        # Conversion en dictionnaire
        result_predict = {label: value for label, value in zip(labels, classes[0])}
        print(result_predict)
        max_key = max(result_predict, key=result_predict.get)

        switch_case = {
            "angry": "You are angry!",
            "disgust": "You are disgusted!",
            "fear": "You are afraid!",
            "happy": "You are happy!",
            "neutral": "You are neutral.",
            "sad": "You are sad.",
            "surprise": "You are surprised!"
        }

        label = switch_case.get(max_key, "Unknown emotion")

        return render_template("result.html", label=label, file_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)