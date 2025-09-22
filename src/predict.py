import joblib

def predict_data(X):
    """
    Predict the class labels for input data.
    """
    model = joblib.load("../model/health_model.pkl")
    y_pred = model.predict(X)
    return y_pred
