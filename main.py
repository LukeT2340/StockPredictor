from DataLoader import DataLoader
from DataPreprocessor import DataPreprocessor
from ModelTrainer import ModelTrainer
import tensorflow as tf

if __name__ == "__main__":
    ticker = "AAPL"
    start = "2015-01-01"
    end = "2020-01-01"

    data_loader = DataLoader(ticker, start, end)
    data = data_loader.get_data()

    # Assuming 'Close' as the target variable for simplicity. Adjust as needed.
    data['Next Close'] = data['Close'].shift(-1)  # Predict next day's close price
    data.dropna(inplace=True)  # Remove rows with NaN values resulting from the shift operation

    data_preprocessor = DataPreprocessor(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Next Close']])
    preprocessed_data = data_preprocessor.getPreprocessedData()

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(preprocessed_data.shape[1]-1,)),  # Adjust input_shape based on actual data
        tf.keras.layers.Dense(1)
    ])

    model_trainer = ModelTrainer(model, preprocessed_data)
    model_trainer.train_model()
    model_trainer.evaluate_model()