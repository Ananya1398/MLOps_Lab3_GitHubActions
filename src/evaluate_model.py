import joblib
import json
import os
from datetime import datetime
from data import load_data, split_data
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model():
    """Evaluate the trained model and save metrics to a timestamped JSON file."""
    # Load and split the dataset
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load the trained model
    model = joblib.load("../model/health_model.pkl")
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Create the metrics folder if it doesn’t exist
    os.makedirs("../metrics", exist_ok=True)

    # Save metrics file with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"../metrics/metrics_{ts}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1 Score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    evaluate_model()
