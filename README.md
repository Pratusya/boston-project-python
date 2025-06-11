# 🏠 Boston Housing Price Predictor

A complete machine learning web application that predicts Boston housing prices using Flask and scikit-learn.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
This will create a `model.pkl` file with the trained machine learning model.

### 3. Run the Application
```bash
python app.py
```
The application will be available at `http://localhost:5000`

## 📁 Project Structure
```
boston-housing-predictor/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── model.pkl             # Trained model (created after running train_model.py)
├── templates/
│   └── index.html        # Web interface
└── README.md            # This file
```

## 🌐 Deployment Options

### Option 1: Heroku Deployment

1. **Install Heroku CLI** and login:
```bash
heroku login
```

2. **Create Procfile**:
```bash
echo "web: gunicorn app:app" > Procfile
```

3. **Initialize Git and deploy**:
```bash
git init
git add .
git commit -m "Initial commit"
heroku create your-app-name
git push heroku main
```

### Option 2: Railway Deployment

1. **Install Railway CLI**:
```bash
npm install -g @railway/cli
```

2. **Deploy**:
```bash
railway login
railway init
railway up
```

### Option 3: Render Deployment

1. **Create `render.yaml`**:
```yaml
services:
  - type: web
    name: boston-housing-predictor
    env: python
    buildCommand: "pip install -r requirements.txt && python train_model.py"
    startCommand: "gunicorn app:app"
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.7
```

2. Connect your GitHub repository to Render and deploy.

### Option 4: Local Network Access

To make your app accessible on your local network:
```bash
python app.py
# App will be available at your local IP address on port 5000
# Find your IP with: ipconfig (Windows) or ifconfig (Mac/Linux)
```

## 🔧 Model Features

The model uses 13 features to predict housing prices:

1. **CRIM** - Per capita crime rate by town
2. **ZN** - Proportion of residential land zoned for lots over 25,000 sq.ft.
3. **INDUS** - Proportion of non-retail business acres per town
4. **CHAS** - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5. **NOX** - Nitric oxides concentration (parts per 10 million)
6. **RM** - Average number of rooms per dwelling
7. **AGE** - Proportion of owner-occupied units built prior to 1940
8. **DIS** - Weighted distances to employment centers
9. **RAD** - Index of accessibility to radial highways
10. **TAX** - Full-value property-tax rate per $10,000
11. **PTRATIO** - Pupil-teacher ratio by town
12. **B** - Black population proportion
13. **LSTAT** - % lower status of the population

## 📊 Model Performance

- **Algorithm**: Random Forest Regressor
- **Features**: 13 housing characteristics
- **Scaling**: StandardScaler for feature normalization
- **Validation**: Train/test split with performance metrics

## 🛠 Customization

### Adding New Features
1. Modify `train_model.py` to include new features
2. Update the form in `templates/index.html`
3. Adjust the feature extraction in `app.py`

### Changing the Model
Edit the `models` dictionary in `train_model.py`:
```python
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Model file not found**:
   - Run `python train_model.py` first

2. **Port already in use**:
   - Change port in `app.py`: `app.run(debug=True, port=5001)`

3. **Import errors**:
   - Make sure virtual environment is activated
   - Install requirements: `pip install -r requirements.txt`

## 📝 API Usage

You can also use the prediction endpoint programmatically:

```python
import requests

data = {
    'CRIM': 0.00632,
    'ZN': 18.0,
    'INDUS': 2.31,
    'CHAS': 0,
    'NOX': 0.538,
    'RM': 6.575,
    'AGE': 65.2,
    'DIS': 4.09,
    'RAD': 1,
    'TAX': 296,
    'PTRATIO': 15.3,
    'B': 396.9,
    'LSTAT': 4.98
}

response = requests.post('http://localhost:5000/predict', data=data)
print(response.text)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

---

**Ready to deploy!** 🚀 Follow the deployment steps above to get your housing price predictor live on the web.