class ModelPipeline:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)