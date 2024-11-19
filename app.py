import numpy as np 
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

app = Flask(__name__)

#load files
model = load_model('weather_model.h5')

scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #get user input
        data = []
        for day in range(1, 6):
            temp = float(request.form[f'temp{day}'])
            humidity =float(request.form[f'humidity{day}'])
            wind_speed = float(request.form[f'wind_speed{day}'])

            data.append([temp, humidity, wind_speed])

        #seperating categorical and numerical columns
        input_data = np.array([[row[0], row[1], row[2]] for row in data])

        #applying scaler
        scaled_data = scaler.transform(input_data)

        #reshaping
        reshaped_data = np.expand_dims(scaled_data, axis=0) #adding batch dimension to convert it into 3D tensor
        print(reshaped_data.shape)

        #predicting
        prediction = model.predict(reshaped_data)
        prediction_scaled = np.array(prediction)

        # Prepare dummy features for inverse transform
        dummy_features = np.zeros((1, scaler.min_.shape[0]))  # (1, 3)
        dummy_features[0, 0] = prediction_scaled[0, 0]  # Assign value to first feature

        #reverse scaling
        prediction_true = scaler.inverse_transform(dummy_features)
        prediction_true = np.round(prediction_true, decimals=2)

        #returning prediction
        return render_template('index.html', prediction=prediction_true[0, 0])

    return render_template('index.html')
if __name__ == "__main__":
    app.run(debug=True)
   