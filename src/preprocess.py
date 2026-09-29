from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Data preprocessing completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
