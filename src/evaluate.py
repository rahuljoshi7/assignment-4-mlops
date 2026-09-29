import os

import joblib
from sklearn.metrics import accuracy_score

from preprocess import load_and_preprocess_data


def evaluate_model():
    _, X_test, _, y_test = load_and_preprocess_data()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "iris_model.joblib")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    return accuracy


if __name__ == "__main__":
    evaluate_model()
