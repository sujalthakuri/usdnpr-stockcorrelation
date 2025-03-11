from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

model = joblib.load('model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        sp500_close = data['SP500_Close']
        nifty500_close = data['Nifty500_Close']
        
        input_data = np.array([[sp500_close, nifty500_close]])

        input_data_scaled = scaler_X.transform(input_data)

        prediction_scaled = model.predict(input_data_scaled)
        
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]

        return jsonify({'Predicted_USD_Sell': round(prediction, 2)}), 200

    except Exception as e:
        return jsonify({'error': str(e) + "damn"}), 400

if __name__ == '__main__':
    app.run(debug=True)
