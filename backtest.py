import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Constants
CSV_FILE = 'megamillions.csv'
MODEL_FILE = 'megamillions_model.h5'
WINDOW_LENGTH = 7
NUM_FEATURES = 10

class MegaMillionsBacktester:
    def __init__(self, csv_file=CSV_FILE, model_file=MODEL_FILE):
        # Load data
        self.df = pd.read_csv(csv_file, names=['Game', 'Month', 'Day', 'Year', 
                                               'Num1', 'Num2', 'Num3', 'Num4', 'Num5', 
                                               'Mega_Ball', 'Megaplier'])
        # Create datetime column
        self.df['Date'] = pd.to_datetime(self.df[['Year', 'Month', 'Day']])
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        
        # Feature columns
        self.feature_columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 
                                'Mega_Ball', 'Megaplier']
        
        # Initialize scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.df[self.feature_columns].values)
        
        # Prepare backtest data
        self._prepare_backtest_data()
        
        # Load model
        self.model = load_model(model_file)
        print(f"Loaded model from {model_file}")
    
    def _prepare_backtest_data(self):
        """Prepare data for backtesting."""
        scaled_values = self.scaler.transform(self.df[self.feature_columns].values)
        self.scaled_values = scaled_values
        self.train_rows = scaled_values.shape[0]
    
    def _get_window(self, index):
        """Get a window of data ending at the given index."""
        return self.scaled_values[index-WINDOW_LENGTH:index]
    
    def _invert_scaling(self, scaled_data):
        """Inverse transform the scaled data."""
        return self.scaler.inverse_transform(scaled_data).astype(int)
    
    def _match_numbers(self, predicted, actual):
        """Calculate the number of matches between predicted and actual numbers."""
        pred_regular = set(predicted[:5])
        pred_mega = predicted[5]
        act_regular = set(actual[:5])
        act_mega = actual[5]
        
        regular_matches = len(pred_regular & act_regular)
        mega_match = (pred_mega == act_mega)
        
        return regular_matches, mega_match
    
    def _calculate_prize(self, regular_matches, mega_match):
        """Calculate prize based on matches."""
        prize_table = {
            (5, True): 1000000000,
            (5, False): 1000000,
            (4, True): 10000,
            (4, False): 500,
            (3, True): 200,
            (3, False): 10,
            (2, True): 10,
            (1, True): 4,
            (0, True): 2
        }
        return prize_table.get((regular_matches, mega_match), 0)
    
    def backtest(self):
        """Perform backtesting of the model."""
        total_draws = self.train_rows - WINDOW_LENGTH
        total_investment = total_draws * 2
        total_winnings = 0
        matches_distribution = {i: 0 for i in range(6)}  # 0 to 5 regular matches
        mega_matches = 0
        prize_distribution = {}
        
        print(f"\nRunning backtest on {total_draws} draws...")
        
        for i in tqdm(range(WINDOW_LENGTH, self.train_rows)):
            window = self._get_window(i)
            window = np.expand_dims(window, axis=0)
            
            # Predict
            y_pred_scaled = self.model.predict(window, verbose=0)
            y_pred = self._invert_scaling(y_pred_scaled)[0]
            
            # Extract predicted numbers
            predicted_regular = sorted(y_pred[:5])
            predicted_mega_ball = y_pred[5]
            predicted_megaplier = y_pred[6]
            predicted = np.concatenate((predicted_regular, [predicted_mega_ball, predicted_megaplier]))
            
            # Actual numbers
            actual = self.df.loc[i, self.feature_columns].values
            
            # Calculate matches
            regular_matches, mega_match = self._match_numbers(predicted, actual)
            matches_distribution[regular_matches] += 1
            if mega_match:
                mega_matches += 1
            
            # Calculate prize
            prize = self._calculate_prize(regular_matches, mega_match)
            total_winnings += prize
            prize_distribution[prize] = prize_distribution.get(prize, 0) + 1
        
        # Calculate ROI
        roi = (total_winnings - total_investment) / total_investment * 100
        
        # Calculate Hit Rate
        hit_rate = sum(1 for prize in prize_distribution.keys() if prize > 0) / total_draws
        
        # Average matches
        avg_matches = sum(k * v for k, v in matches_distribution.items()) / total_draws
        
        # Mega Match Rate
        mega_match_rate = mega_matches / total_draws
        
        # Summary
        summary = {
            'Total_Games': total_draws,
            'Total_Investment': total_investment,
            'Total_Winnings': total_winnings,
            'Net_Profit': total_winnings - total_investment,
            'ROI': roi,
            'Hit_Rate': hit_rate,
            'Average_Matches': avg_matches,
            'Matches_Distribution': matches_distribution,
            'Mega_Matches': mega_matches,
            'Mega_Match_Rate': mega_match_rate,
            'Prize_Distribution': prize_distribution
        }
        
        self._print_summary(summary)
    
    def _print_summary(self, summary):
        """Print the backtest summary."""
        print("\nSimulation Results Summary:")
        print("-" * 50)
        
        print(f"\nPerformance Metrics:")
        print(f"Total Games Simulated: {summary['Total_Games']:,}")
        print(f"Total Investment: ${summary['Total_Investment']:,.2f}")
        print(f"Total Winnings: ${summary['Total_Winnings']:,.2f}")
        print(f"Net Profit: ${summary['Net_Profit']:,.2f}")
        print(f"ROI: {summary['ROI']:.2f}%")
        print(f"Hit Rate: {summary['Hit_Rate']*100:.2f}%")
        print(f"Mega Ball Match Rate: {summary['Mega_Match_Rate']*100:.2f}%")
        
        print(f"\nAverage Statistics:")
        print(f"Average Matches Per Game: {summary['Average_Matches']:.3f}")
        
        print(f"\nMatches Distribution:")
        for matches, count in summary['Matches_Distribution'].items():
            percentage = (count / summary['Total_Games']) * 100
            print(f"{matches} regular numbers matched: {count:,} times ({percentage:.2f}%)")
        
        print(f"\nPrize Distribution:")
        for prize, count in sorted(summary['Prize_Distribution'].items()):
            percentage = (count / summary['Total_Games']) * 100
            if prize >= 1000000:
                prize_str = f"${prize/1000000:,.1f}M"
            else:
                prize_str = f"${prize:,.2f}"
            print(f"{prize_str}: {count:,} times ({percentage:.4f}%)")

if __name__ == "__main__":
    backtester = MegaMillionsBacktester()
    backtester.backtest()