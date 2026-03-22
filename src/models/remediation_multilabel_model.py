import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


class RemediationModel:
    def __init__(self):
        self.model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        self.encoders = {}

    def encode_targets(self, y):
        y_encoded = y.copy()
        for col in y_encoded.columns:
            le = LabelEncoder()
            y_encoded[col] = le.fit_transform(y_encoded[col])
            self.encoders[col] = le
        return y_encoded

    def train(self, X_train, y_train):
        y_train_encoded = self.encode_targets(y_train)
        self.model.fit(X_train, y_train_encoded)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):   # ✅ NOW INSIDE CLASS
        y_test_encoded = y_test.copy()

        for col in y_test_encoded.columns:
            le = self.encoders[col]
            y_test_encoded[col] = le.transform(y_test_encoded[col])

        # evaluate each column separately
        for i, col in enumerate(y_test_encoded.columns):
            print(f"\n=== Evaluation for {col} ===")
            print(classification_report(y_test_encoded[col], y_pred[:, i]))