import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from model_pipeline import ModelPipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

# Note: Boston housing dataset was removed from sklearn due to ethical concerns
# We'll use California housing dataset and adapt it to work with your existing features
# Or create a synthetic Boston-like dataset

def create_boston_like_dataset():
    """Create a dataset similar to Boston housing with the same features"""
    np.random.seed(42)
    n_samples = 506  # Same as original Boston dataset
    
    # Generate features similar to Boston housing dataset
    data = {
        'CRIM': np.random.exponential(3, n_samples),  # Crime rate
        'ZN': np.random.choice([0, 12.5, 25, 50, 85], n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05]),  # Zoned land
        'INDUS': np.random.uniform(0.46, 27.74, n_samples),  # Industrial proportion
        'CHAS': np.random.choice([0, 1], n_samples, p=[0.93, 0.07]),  # Charles River
        'NOX': np.random.uniform(0.385, 0.871, n_samples),  # Nitric oxide
        'RM': np.random.normal(6.2, 0.7, n_samples),  # Average rooms
        'AGE': np.random.uniform(2.9, 100, n_samples),  # Age of buildings
        'DIS': np.random.uniform(1.1, 12.1, n_samples),  # Distance to employment
        'RAD': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], n_samples),  # Highway accessibility
        'TAX': np.random.uniform(187, 711, n_samples),  # Tax rate
        'PTRATIO': np.random.uniform(12.6, 22, n_samples),  # Pupil-teacher ratio
        'B': np.random.uniform(0.32, 396.9, n_samples),  # Black population proportion
        'LSTAT': np.random.uniform(1.73, 37.97, n_samples)  # Lower status population
    }
    
    # Create target variable (house prices) based on features
    # This creates a realistic relationship between features and prices
    price = (
        50 - data['CRIM'] * 0.1 +
        data['ZN'] * 0.05 +
        -data['INDUS'] * 0.1 +
        data['CHAS'] * 2 +
        -data['NOX'] * 17 +
        data['RM'] * 4 +
        -data['AGE'] * 0.01 +
        data['DIS'] * 0.3 +
        -data['TAX'] * 0.01 +
        -data['PTRATIO'] * 0.3 +
        data['B'] * 0.01 +
        -data['LSTAT'] * 0.5 +
        np.random.normal(0, 3, n_samples)  # Add some noise
    )
    
    # Ensure prices are positive and in reasonable range
    price = np.clip(price, 5, 50)
    
    df = pd.DataFrame(data)
    return df, price

# Create a pipeline class that includes scaling
class ModelPipeline:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def train_model():
    print("Creating Boston-like housing dataset...")
    X, y = create_boston_like_dataset()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Price range: ${y.min():.2f} - ${y.max():.2f}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models and choose the best one
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"  MSE: {mse:.2f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {np.sqrt(mse):.2f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with R² score: {best_score:.4f}")
    
    # Create a pipeline that includes scaling
    pipeline = ModelPipeline(scaler, best_model)
    
    # Save the model pipeline
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("\nModel saved as 'model.pkl'")
    
    # Test the saved model
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Test with a sample prediction
    sample_data = X_test.iloc[0:1]
    prediction = loaded_model.predict(sample_data)
    actual = y_test[0]
    
    print(f"\nTest prediction:")
    print(f"Predicted: ${prediction[0]:.2f}")
    print(f"Actual: ${actual:.2f}")
    print(f"Difference: ${abs(prediction[0] - actual):.2f}")
    
    return pipeline

if __name__ == "__main__":
    model = train_model()
    print("\nModel training completed successfully!")
    print("You can now run your Flask app with: python app.py")