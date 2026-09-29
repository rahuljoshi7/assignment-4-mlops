from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.python import PythonOperator


def download_data():
    from sklearn.datasets import load_iris

    iris = load_iris()

    print("Dataset downloaded successfully.")
    print(f"Number of samples: {len(iris.data)}")
    print(f"Number of features: {iris.data.shape[1]}")


def preprocess_data():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    print("Data preprocessing completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")


def train_model():
    import os

    import joblib
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    iris = load_iris()

    X_train, _, y_train, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = LogisticRegression(max_iter=200)

    model.fit(X_train, y_train)

    os.makedirs("/opt/airflow/models", exist_ok=True)

    model_path = "/opt/airflow/models/iris_model.joblib"

    joblib.dump(model, model_path)

    print("Model training completed.")
    print(f"Model saved to: {model_path}")


def evaluate_model():
    import joblib
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()

    _, X_test, _, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = joblib.load("/opt/airflow/models/iris_model.joblib")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")

    return accuracy


def check_accuracy(**context):
    import os

    import joblib
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    iris = load_iris()

    _, X_test, _, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    model = joblib.load("/opt/airflow/models/iris_model.joblib")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    best_accuracy_path = "/opt/airflow/data/best_accuracy.txt"

    # First run: no previous accuracy exists.
    if os.path.exists(best_accuracy_path):
        with open(best_accuracy_path, "r", encoding="utf-8") as file:
            content = file.read().strip()

        if content:
            best_accuracy = float(content)
        else:
            best_accuracy = 0.0
    else:
        best_accuracy = 0.0

    print(f"Previous best accuracy: {best_accuracy:.4f}")
    print(f"Current accuracy: {accuracy:.4f}")

    if accuracy > best_accuracy:
        os.makedirs("/opt/airflow/data", exist_ok=True)

        with open(best_accuracy_path, "w", encoding="utf-8") as file:
            file.write(str(accuracy))

        print("Accuracy improved. Deployment will be triggered.")

        return "deploy_model"

    print("Accuracy did not improve. Deployment will be skipped.")

    return "skip_deployment"


def deploy_model():
    import os
    import shutil

    os.makedirs("/opt/airflow/staging", exist_ok=True)

    source_model = "/opt/airflow/models/iris_model.joblib"
    staging_model = "/opt/airflow/staging/iris_model.joblib"

    shutil.copy(source_model, staging_model)

    print("Model deployed to staging successfully.")
    print(f"Staging model: {staging_model}")


with DAG(
    dag_id="mlops_assignment_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["mlops", "assignment-4"],
) as dag:

    download = PythonOperator(
        task_id="download_data",
        python_callable=download_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    check = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=check_accuracy,
    )

    deploy = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    skip = EmptyOperator(
        task_id="skip_deployment",
    )

    download >> preprocess >> train >> evaluate >> check

    check >> deploy
    check >> skip