import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalyDetector:
    def __init__(self, contamination=0.05):
        """
        contamination: The proportion of outliers expected in the data.
        """
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def extract_features(self, log_entry: dict) -> list:
        """
        Transforms a raw log json into numerical features.
        Example log: {"ip": "1.1.1.1", "status": 404, "response_size": 2048}
        """
        status_code = log_entry.get("status", 200)
        size = log_entry.get("response_size", 0)
        # Feature engineering: flag client error status codes
        is_error = 1 if 400 <= status_code < 600 else 0
        
        return [status_code, size, is_error]

    def train(self, data_list: list):
        features = [self.extract_features(log) for log in data_list]
        self.model.fit(features)

    def predict(self, log_entry: dict) -> int:
        """
        Returns -1 for an anomaly, 1 for normal data.
        """
        features = np.array(self.extract_features(log_entry)).reshape(1, -1)
        return int(self.model.predict(features)[0])

    def save(self, filepath="model.joblib"):
        joblib.dump(self.model, filepath)
        
    def load(self, filepath="model.joblib"):
        if os.path.exists(filepath):
            self.model = joblib.load(filepath)