import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings

# Suppress statsmodels warnings for a cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. The Strategy / Indicator Classes
# ==========================================
class ZScorePairsStrategy:
    """
    A specific strategy class that implements the Z-Score mean reversion logic.
    You can build other classes (e.g., BollingerBandsStrategy) that follow this same structure.
    """
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.0, window: int = 60):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window

    def generate_signals(self, data: pd.DataFrame, t1: str, t2: str):
        """
        Calculates the required indicators, determines the day-by-day positions,
        and calculates the underlying spread returns.
        """
        df = data.copy()

        # 1. Calculate Static Hedge Ratio via OLS
        X = sm.add_constant(df[t2])
        y = df[t1]
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params.iloc[1]
        
        # 2. Calculate Spread and Z-Score
        df['spread'] = df[t1] - (hedge_ratio * df[t2])
        df['spread_mean'] = df['spread'].rolling(self.window).mean()
        df['spread_std'] = df['spread'].rolling(self.window).std()
        df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        # 3. Simulate Positions (State Machine)
        positions = np.zeros(len(df))
        current_pos = 0
        
        for i in range(len(df)):
            z = df['zscore'].iloc[i]
            
            if pd.isna(z):
                positions[i] = 0
                continue
                
            if current_pos == 0:
                if z > self.entry_z:
                    current_pos = -1 
                elif z < -self.entry_z:
                    current_pos = 1  
            elif current_pos == 1:
                if z >= -self.exit_z: 
                    current_pos = 0
            elif current_pos == -1:
                if z <= self.exit_z:  
                    current_pos = 0
                    
            positions[i] = current_pos
            
        df['position'] = positions

        # 4. Calculate Spread Returns for the Backtester
        df[f'{t1}_ret'] = df[t1].pct_change()
        df[f'{t2}_ret'] = df[t2].pct_change()
        df['spread_ret'] = df[f'{t1}_ret'] - (hedge_ratio * df[f'{t2}_ret'])

        return df, hedge_ratio
    

class KalmanPairsStrategy:
    """
    A strategy class that uses a dynamic Kalman Filter to estimate the hedge ratio,
    spread, and Z-score tick-by-tick without relying on lookback windows.
    """
    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.0):
        # Notice we don't need a 'window' parameter anymore!
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, data: pd.DataFrame, t1: str, t2: str):
        df = data.copy()
        
        y = df[t1].values
        x = df[t2].values
        
        # 1. Initialize Kalman Filter components
        # Hidden state vector [alpha, beta]
        state_mean = np.zeros(2) 
        state_cov = np.ones((2, 2))
        
        # Noise parameters (Hyperparameters - tune these for faster/slower adaptation)
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2) # How fast the state is allowed to change
        obs_var = 1e-3 # Variance of the measurement noise
        
        # Output arrays
        hedge_ratios = np.zeros(len(df))
        spreads = np.zeros(len(df))
        zscores = np.zeros(len(df))
        
        # 2. Run the Kalman Filter day by day
        for t in range(len(df)):
            # Observation matrix H_t = [1, x_t]
            H = np.array([1, x[t]])
            
            # Prediction Step (Assume state stays the same, uncertainty increases)
            state_cov = state_cov + trans_cov
            
            # Measurement Step
            y_pred = np.dot(H, state_mean)
            
            # The error between prediction and reality is our dynamically calculated spread
            error = y[t] - y_pred
            spreads[t] = error
            
            # Calculate the variance of the prediction (Q)
            Q = np.dot(np.dot(H, state_cov), H.T) + obs_var
            
            # The Z-score is natively derived from the filter's error and variance!
            zscores[t] = error / np.sqrt(Q)
            
            # Kalman Gain (How much should we update our state based on the error?)
            K = np.dot(state_cov, H.T) / Q
            
            # Update the hidden state [alpha, beta] and covariance
            state_mean = state_mean + K * error
            state_cov = state_cov - np.outer(K, np.dot(H, state_cov))
            
            # Store the current dynamic hedge ratio (beta)
            hedge_ratios[t] = state_mean[1]

        # Attach to dataframe
        df['hedge_ratio'] = hedge_ratios
        df['spread'] = spreads
        df['zscore'] = zscores

        # 3. Simulate Positions (State Machine)
        positions = np.zeros(len(df))
        current_pos = 0
        
        for i in range(len(df)):
            z = df['zscore'].iloc[i]
            
            # Give the filter ~10 days to "warm up" and find the true state
            if pd.isna(z) or i < 10:
                positions[i] = 0
                continue
                
            if current_pos == 0:
                if z > self.entry_z:
                    current_pos = -1 
                elif z < -self.entry_z:
                    current_pos = 1  
            elif current_pos == 1:
                if z >= -self.exit_z: 
                    current_pos = 0
            elif current_pos == -1:
                if z <= self.exit_z:  
                    current_pos = 0
                    
            positions[i] = current_pos
            
        df['position'] = positions

        # 4. Calculate Spread Returns for the Backtester
        df[f'{t1}_ret'] = df[t1].pct_change()
        df[f'{t2}_ret'] = df[t2].pct_change()
        
        # CRITICAL: We use yesterday's hedge ratio to calculate today's return.
        # If we used today's hedge ratio, we would introduce lookahead bias.
        df['spread_ret'] = df[f'{t1}_ret'] - (df['hedge_ratio'].shift(1) * df[f'{t2}_ret'])

        # Return the populated dataframe and the final hedge ratio for logging
        return df, hedge_ratios[-1]