from flask import Flask, request, jsonify
import joblib  # Assuming you have a model file to load

app = Flask(__name__)

# Load your trained model (replace 'model.pkl' with your model file path)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input
        features = data['features']  # Extract features from the input
        prediction = model.predict([features])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Corrected the typo here
if __name__ == '__main__':
    app.run(debug=True)
