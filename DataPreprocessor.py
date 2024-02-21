import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, data, features_to_scale=None):
        self.data = data
        self.features_to_scale = features_to_scale or data.columns  # Scale all if no specific features provided
        self.scaler = MinMaxScaler()
        self.preprocessed_data = self.preprocess()

    def preprocess(self):
        scaled_data = self.data.copy()
        scaled_data[self.features_to_scale] = self.scaler.fit_transform(self.data[self.features_to_scale])
        return scaled_data

    def getPreprocessedData(self):
        return self.preprocessed_data

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
