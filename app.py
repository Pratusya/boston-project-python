from flask import Flask, render_template, request
import pickle
import numpy as np
import os

# Keep the ModelPipeline class in app.py for pickle compatibility
class ModelPipeline:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

app = Flask(__name__)

# Load the model with error handling
def load_model():
    try:
        if not os.path.exists('model.pkl'):
            print("Model file not found! Please run 'python train_model.py' first.")
            return None
        
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load model at startup, but don't fail if it doesn't exist
model = None
try:
    model = load_model()
except Exception as e:
    print(f"Warning: Could not load model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                             prediction_text="Error: Model not loaded. Please ensure model.pkl exists.",
                             show_error=True)
    
    try:
        # Extract features from form
        features = [
            float(request.form['CRIM']),
            float(request.form['ZN']),
            float(request.form['INDUS']),
            float(request.form['CHAS']),
            float(request.form['NOX']),
            float(request.form['RM']),
            float(request.form['AGE']),
            float(request.form['DIS']),
            float(request.form['RAD']),
            float(request.form['TAX']),
            float(request.form['PTRATIO']),
            float(request.form['B']),
            float(request.form['LSTAT'])
        ]
        
        # Reshape for prediction (model expects 2D array)
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        output = round(prediction[0], 2)
        
        # Format output nicely
        return render_template('index.html', 
                             prediction_text=f"Predicted House Price: ${output}K",
                             show_result=True)
    
    except ValueError as ve:
        return render_template('index.html', 
                             prediction_text="Error: Please enter valid numeric values for all fields.",
                             show_error=True)
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}",
                             show_error=True)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)