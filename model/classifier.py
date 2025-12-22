import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from features.fft_features import extract_fft_features


def load_dataset(real_dir: str, fake_dir: str):
    X, y = [], []

    for label, folder in enumerate([real_dir, fake_dir]):
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            try:
                features = extract_fft_features(path)
                X.append(features)
                y.append(label)
            except Exception:
                continue

    return np.array(X), np.array(y)


def train_and_evaluate(real_dir: str, fake_dir: str):
    X, y = load_dataset(real_dir, fake_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    return model
