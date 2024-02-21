import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# This class is responsible for training the model
class ModelTrainer:
    def __init__(self, model, data, test_size=0.2, random_state=None):
        self.model = model
        self.data = data
        self.test_size = test_size
        self.random_state = random_state
        X = data.iloc[:, :-1].values  # Adjusted for pandas DataFrame
        y = data.iloc[:, -1].values   # Adjusted for pandas DataFrame
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_model(self, epochs=100, batch_size=32, verbose=1):
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.1)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R^2 Score:", r2)

    def predict(self, new_data):
        return self.model.predict(new_data)