import os
import sys

import joblib

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.preprocess import load_and_preprocess_data  # noqa: E402


def test_data_split():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_model_exists():
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(
        project_root,
        "models",
        "iris_model.joblib"
    )

    assert os.path.exists(model_path)


def test_model_accuracy():
    _, X_test, _, y_test = load_and_preprocess_data()

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    model_path = os.path.join(
        project_root,
        "models",
        "iris_model.joblib"
    )

    model = joblib.load(model_path)

    accuracy = model.score(X_test, y_test)

    assert accuracy >= 0.90
