import os

import joblib
from sklearn.linear_model import LogisticRegression

from preprocess import load_and_preprocess_data


def train_model():
    X_train, _, y_train, _ = load_and_preprocess_data()

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(project_root, "models")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "iris_model.joblib")

    joblib.dump(model, model_path)

    print("Model training completed.")
    print(f"Model saved to: {model_path}")

    return model


if __name__ == "__main__":
    train_model()
