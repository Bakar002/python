from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive input as JSON
    features = data['features']
    prediction = model.predict([features])[0]  # Predict species
    return jsonify({'predicted_class': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
