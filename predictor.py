import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import os

# Constants
CSV_FILE = 'megamillions.csv'
MODEL_FILE = 'megamillions_model.h5'
WINDOW_LENGTH = 7
NUM_FEATURES = 10  # Num1, Num2, Num3, Num4, Num5, Mega_Ball, Megaplier

class MegaMillionsPredictor:
    def __init__(self, csv_file=CSV_FILE):
        # Load data
        self.df = pd.read_csv(csv_file, names=['Game', 'Month', 'Day', 'Year', 
                                               'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 
                                               'Mega_Ball', 'Megaplier'])
        # Create datetime column
        self.df['Date'] = pd.to_datetime(self.df[['Year', 'Month', 'Day']])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Drop unnecessary columns if any
        # Assuming all columns are needed based on the user's initial data
        # If 'Date' exists as a separate column, ensure it's handled
        # self.df.drop(['Date'], axis=1, inplace=True)  # Not needed here
        
        # Feature columns
        self.feature_columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 
                                'Mega_Ball', 'Megaplier']
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Prepare training data
        self._prepare_data()
        
        # Initialize model
        self.model = None

    def _prepare_data(self):
        """Prepare training data with windowing and scaling."""
        # Create windowed samples
        train = self.df.copy()
        train_values = train[self.feature_columns].values
        self.scaler.fit(train_values)
        scaled_values = self.scaler.transform(train_values)
        
        train_rows = scaled_values.shape[0]
        self.x_train = np.empty((train_rows - WINDOW_LENGTH, WINDOW_LENGTH, len(self.feature_columns)), dtype=float)
        self.y_train = np.empty((train_rows - WINDOW_LENGTH, len(self.feature_columns)), dtype=float)
        
        for i in range(train_rows - WINDOW_LENGTH):
            self.x_train[i] = scaled_values[i:i+WINDOW_LENGTH]
            self.y_train[i] = scaled_values[i+WINDOW_LENGTH]
        
        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Feature shape: {self.x_train.shape[1:]}")
        print(f"Label shape: {self.y_train.shape[1:]}")
    
    def build_model(self):
        """Build the deep neural network model."""
        model = Sequential()
        
        # Adding Bidirectional LSTM layers with Dropout
        model.add(Bidirectional(LSTM(240, 
                                     input_shape=(WINDOW_LENGTH, len(self.feature_columns)),
                                     return_sequences=True)))
        model.add(Dropout(0.2))
        
        model.add(Bidirectional(LSTM(240, return_sequences=True)))
        model.add(Dropout(0.2))
        
        model.add(Bidirectional(LSTM(240, return_sequences=True)))
        model.add(Dropout(0.2))
        
        model.add(Bidirectional(LSTM(240, return_sequences=False)))
        model.add(Dropout(0.2))
        
        # Output layers
        model.add(Dense(70, activation='relu'))
        model.add(Dense(len(self.feature_columns), activation='linear'))
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])
        
        self.model = model
        print(model.summary())
    
    def train_model(self, epochs=2400, batch_size=100):
        """Train the model on the prepared data."""
        if self.model is None:
            self.build_model()
        
        self.model.fit(self.x_train, self.y_train, 
                       batch_size=batch_size, epochs=epochs, verbose=2)
        
        # Save the trained model
        self.model.save(MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")
    
    def load_trained_model(self):
        """Load a pre-trained model from disk."""
        if os.path.exists(MODEL_FILE):
            self.model = load_model(MODEL_FILE)
            print(f"Loaded model from {MODEL_FILE}")
        else:
            print(f"No trained model found at {MODEL_FILE}. Please train the model first.")
    
    def predict_next_drawing(self):
        """Predict the next drawing numbers."""
        if self.model is None:
            self.load_trained_model()
            if self.model is None:
                raise ValueError("Model is not loaded. Train the model first.")
        
        # Get the latest window_length data points
        latest_data = self.df.tail(WINDOW_LENGTH)[self.feature_columns].values
        scaled_latest = self.scaler.transform(latest_data)
        x_next = np.array([scaled_latest])
        
        # Predict
        y_pred_scaled = self.model.predict(x_next)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).astype(int)[0]
        
        # Extract predicted numbers
        predicted_regular = sorted(y_pred[:5])
        predicted_mega_ball = y_pred[5]
        predicted_megaplier = y_pred[6]
        
        return {
            'Regular Numbers': predicted_regular,
            'Mega Ball': predicted_mega_ball,
            'Megaplier': predicted_megaplier
        }
    
    def save_scaler(self, filename='scaler.npy'):
        """Save the scaler to disk."""
        np.save(filename, self.scaler.scale_)
        print(f"Scaler saved to {filename}")
    
    def load_scaler(self, filename='scaler.npy'):
        """Load the scaler from disk."""
        if os.path.exists(filename):
            scale = np.load(filename)
            self.scaler.scale_ = scale
            print(f"Scaler loaded from {filename}")
        else:
            print(f"No scaler found at {filename}.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mega Millions Predictor using TensorFlow Keras')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict the next drawing')
    args = parser.parse_args()
    
    predictor = MegaMillionsPredictor()
    
    if args.train:
        predictor.train_model()
    
    if args.predict:
        prediction = predictor.predict_next_drawing()
        print("\nPredicted numbers for next drawing:")
        print(f"Regular Numbers: {prediction['Regular Numbers']}")
        print(f"Mega Ball: {prediction['Mega Ball']}")
        print(f"Megaplier: {prediction['Megaplier']}")
