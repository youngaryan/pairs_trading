import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress statsmodels warnings for a cleaner output
warnings.filterwarnings("ignore")

class ZScorePairsStrategy:
    """Classic OLS Z-Score Strategy adapted for the universal backtester."""
    def __init__(self, ticker1: str, ticker2: str, entry_z: float = 2.0, exit_z: float = 0.0, window: int = 60):
        self.t1 = ticker1
        self.t2 = ticker2
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.window = window

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()

        # 1. Calculate Static Hedge Ratio via OLS
        X = sm.add_constant(data[self.t2])
        y = data[self.t1]
        model = sm.OLS(y, X).fit()
        hedge_ratio = model.params.iloc[1]
        
        # 2. Spread & Z-Score
        data['spread'] = data[self.t1] - (hedge_ratio * data[self.t2])
        data['spread_mean'] = data['spread'].rolling(self.window).mean()
        data['spread_std'] = data['spread'].rolling(self.window).std()
        data['zscore'] = (data['spread'] - data['spread_mean']) / data['spread_std']
        
        # 3. Positions
        positions = np.zeros(len(data))
        current_pos = 0
        for i in range(len(data)):
            z = data['zscore'].iloc[i]
            if pd.isna(z): continue
            
            if current_pos == 0:
                if z > self.entry_z: current_pos = -1 
                elif z < -self.entry_z: current_pos = 1  
            elif current_pos == 1 and z >= -self.exit_z: current_pos = 0
            elif current_pos == -1 and z <= self.exit_z: current_pos = 0
            positions[i] = current_pos
            
        data['position'] = positions

        # 4. Universal Backtester Expectation: 'instrument_ret'
        data[f'{self.t1}_ret'] = data[self.t1].pct_change()
        data[f'{self.t2}_ret'] = data[self.t2].pct_change()
        # The return of the spread is our "instrument" return
        data['instrument_ret'] = data[f'{self.t1}_ret'] - (hedge_ratio * data[f'{self.t2}_ret'])

        return data


class KalmanPairsStrategy:
    """Dynamic Kalman Filter Strategy adapted for the universal backtester."""
    def __init__(self, ticker1: str, ticker2: str, entry_z: float = 2.0, exit_z: float = 0.0):
        self.t1 = ticker1
        self.t2 = ticker2
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        y = data[self.t1].values
        x = data[self.t2].values
        
        state_mean = np.zeros(2) 
        state_cov = np.ones((2, 2))
        delta = 1e-5
        trans_cov = delta / (1 - delta) * np.eye(2) 
        obs_var = 1e-3 
        
        hedge_ratios = np.zeros(len(data))
        zscores = np.zeros(len(data))
        
        # Kalman Loop
        for t in range(len(data)):
            H = np.array([1, x[t]])
            state_cov = state_cov + trans_cov
            y_pred = np.dot(H, state_mean)
            error = y[t] - y_pred
            Q = np.dot(np.dot(H, state_cov), H.T) + obs_var
            
            zscores[t] = error / np.sqrt(Q)
            K = np.dot(state_cov, H.T) / Q
            state_mean = state_mean + K * error
            state_cov = state_cov - np.outer(K, np.dot(H, state_cov))
            hedge_ratios[t] = state_mean[1]

        data['hedge_ratio'] = hedge_ratios
        data['zscore'] = zscores

        # Positions
        positions = np.zeros(len(data))
        current_pos = 0
        for i in range(len(data)):
            z = data['zscore'].iloc[i]
            if pd.isna(z) or i < 10: continue
            
            if current_pos == 0:
                if z > self.entry_z: current_pos = -1 
                elif z < -self.entry_z: current_pos = 1  
            elif current_pos == 1 and z >= -self.exit_z: current_pos = 0
            elif current_pos == -1 and z <= self.exit_z: current_pos = 0
            positions[i] = current_pos
            
        data['position'] = positions

        # Universal Backtester Expectation: 'instrument_ret'
        data[f'{self.t1}_ret'] = data[self.t1].pct_change()
        data[f'{self.t2}_ret'] = data[self.t2].pct_change()
        # Shift hedge ratio to avoid lookahead bias!
        data['instrument_ret'] = data[f'{self.t1}_ret'] - (data['hedge_ratio'].shift(1) * data[f'{self.t2}_ret'])

        return data
    
class MovingAverageCrossoverStrategy:
    """
    A single-asset trend following strategy.
    Goes Long (+1) when Fast MA > Slow MA.
    Goes Flat (0) when Fast MA < Slow MA.
    """
    def __init__(self, target_ticker: str, fast_window: int = 50, slow_window: int = 200):
        self.target = target_ticker
        self.fast = fast_window
        self.slow = slow_window

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # 1. Calculate Indicators
        data['fast_ma'] = data[self.target].rolling(self.fast).mean()
        data['slow_ma'] = data[self.target].rolling(self.slow).mean()
        
        # 2. Determine Position
        # np.where(condition, value_if_true, value_if_false)
        data['position'] = np.where(data['fast_ma'] > data['slow_ma'], 1, 0)
        
        # 3. Supply the underlying instrument return for the backtester
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data
    
class BollingerBandsStrategy:
    """
    Goes Long when price drops below the Lower Band.
    Goes Short when price spikes above the Upper Band.
    Exits (goes Flat) when price returns to the Moving Average.
    """
    def __init__(self, target_ticker: str, window: int = 20, num_std: float = 2.0):
        self.target = target_ticker
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # 1. Calculate the Bands
        data['sma'] = data[self.target].rolling(self.window).mean()
        data['std'] = data[self.target].rolling(self.window).std()
        data['upper_band'] = data['sma'] + (data['std'] * self.num_std)
        data['lower_band'] = data['sma'] - (data['std'] * self.num_std)
        
        # 2. Simulate Positions (State Machine)
        positions = np.zeros(len(data))
        current_pos = 0
        
        for i in range(len(data)):
            price = data[self.target].iloc[i]
            sma = data['sma'].iloc[i]
            upper = data['upper_band'].iloc[i]
            lower = data['lower_band'].iloc[i]
            
            if pd.isna(sma):
                continue
                
            # Entry Logic
            if current_pos == 0:
                if price < lower:
                    current_pos = 1   # Oversold, buy
                elif price > upper:
                    current_pos = -1  # Overbought, short
            
            # Exit Logic
            elif current_pos == 1 and price >= sma:
                current_pos = 0       # Reverted to mean
            elif current_pos == -1 and price <= sma:
                current_pos = 0       # Reverted to mean
                
            positions[i] = current_pos
            
        data['position'] = positions
        
        # 3. Universal Backtester Expectation: 'instrument_ret'
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data
    
class DonchianBreakoutStrategy:
    """
    Goes Long when price hits a new N-day High.
    Goes Short when price hits a new N-day Low.
    Exits using a faster N/2 day opposite channel.
    """
    def __init__(self, target_ticker: str, entry_window: int = 20):
        self.target = target_ticker
        self.entry_window = entry_window
        # The exit window is usually half the entry window in classic Turtle logic
        self.exit_window = entry_window // 2 

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # 1. Calculate Channels
        # We use .shift(1) so we are comparing TODAY'S price to the high of the PREVIOUS N days
        data['highest_high'] = data[self.target].rolling(self.entry_window).max().shift(1)
        data['lowest_low'] = data[self.target].rolling(self.entry_window).min().shift(1)
        
        data['exit_long'] = data[self.target].rolling(self.exit_window).min().shift(1)
        data['exit_short'] = data[self.target].rolling(self.exit_window).max().shift(1)

        # 2. Simulate Positions
        positions = np.zeros(len(data))
        current_pos = 0
        
        for i in range(len(data)):
            price = data[self.target].iloc[i]
            high = data['highest_high'].iloc[i]
            low = data['lowest_low'].iloc[i]
            exit_l = data['exit_long'].iloc[i]
            exit_s = data['exit_short'].iloc[i]
            
            if pd.isna(high):
                continue
                
            if current_pos == 0:
                if price > high:
                    current_pos = 1
                elif price < low:
                    current_pos = -1
                    
            elif current_pos == 1 and price < exit_l:
                current_pos = 0
                
            elif current_pos == -1 and price > exit_s:
                current_pos = 0
                
            positions[i] = current_pos
            
        data['position'] = positions
        
        # 3. Universal Backtester Expectation
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data
    
    
class MachineLearningStrategy:
    """
    Trains a Random Forest model on an initial training window.
    Uses rolling returns, volatility, and moving averages as features to predict
    tomorrow's price direction.
    """
    def __init__(self, target_ticker: str, train_window_days: int = 1000):
        self.target = target_ticker
        self.train_window = train_window_days  # Number of days reserved purely for training

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # ==========================================
        # 1. Feature Engineering (The Clues)
        # ==========================================
        data['ret_1d'] = data[self.target].pct_change(1)
        data['ret_5d'] = data[self.target].pct_change(5)
        data['ret_20d'] = data[self.target].pct_change(20)
        data['volatility'] = data['ret_1d'].rolling(20).std()
        
        # Distance from the 50-day moving average
        sma_50 = data[self.target].rolling(50).mean()
        data['sma_dist'] = (data[self.target] - sma_50) / sma_50
        
        # ==========================================
        # 2. The Target Variable (The Answer Key)
        # ==========================================
        # If tomorrow's return is > 0, target is 1 (UP). Else, -1 (DOWN).
        # We MUST use .shift(-1) to align TOMORROW'S reality with TODAY'S features.
        data['target'] = np.where(data[self.target].pct_change().shift(-1) > 0, 1, -1)
        
        # Drop NaN values created by rolling windows and shifting
        # We also drop the very last row because we don't know tomorrow's answer yet
        model_data = data.dropna()[:-1] 
        
        # ==========================================
        # 3. Train / Trade Split (Avoiding Lookahead Bias)
        # ==========================================
        train_df = model_data.iloc[:self.train_window]
        trade_df = model_data.iloc[self.train_window:]
        
        features = ['ret_1d', 'ret_5d', 'ret_20d', 'volatility', 'sma_dist']
        
        X_train = train_df[features]
        y_train = train_df['target']
        
        X_trade = trade_df[features]
        
        # ==========================================
        # 4. Train the AI Model
        # ==========================================
        # We restrict max_depth to 5 to prevent "overfitting" (memorizing the past)
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)
        
        # ==========================================
        # 5. Generate Predictions for the Trading Period
        # ==========================================
        predictions = clf.predict(X_trade)
        
        # Map predictions back to the original dataframe
        # We sit in Cash (0) during the training period!
        data['position'] = 0 
        data.loc[trade_df.index, 'position'] = predictions
        
        # Universal Backtester Expectation
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data


class WalkForwardMLStrategy:
    """
    An advanced machine learning strategy that continuously re-trains itself.
    It trains on `train_window` days, predicts the next `step_size` days, 
    and then rolls forward to capture changing market regimes.
    """
    def __init__(self, target_ticker: str, train_window: int = 1000, step_size: int = 60):
        self.target = target_ticker
        self.train_window = train_window  # e.g., Train on the past ~4 years
        self.step_size = step_size        # e.g., Predict the next ~3 months before retraining

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # ==========================================
        # 1. Feature Engineering
        # ==========================================
        data['ret_1d'] = data[self.target].pct_change(1)
        data['ret_5d'] = data[self.target].pct_change(5)
        data['ret_20d'] = data[self.target].pct_change(20)
        data['volatility'] = data['ret_1d'].rolling(20).std()
        
        sma_50 = data[self.target].rolling(50).mean()
        data['sma_dist'] = (data[self.target] - sma_50) / sma_50
        
        # Target: 1 if tomorrow is up, -1 if tomorrow is down
        data['target'] = np.where(data[self.target].pct_change().shift(-1) > 0, 1, -1)
        
        model_data = data.dropna()[:-1] 
        features = ['ret_1d', 'ret_5d', 'ret_20d', 'volatility', 'sma_dist']
        
        # Initialize an empty array to store our walk-forward predictions
        predictions = pd.Series(index=model_data.index, data=0)
        
        # ==========================================
        # 2. Walk-Forward Training Loop
        # ==========================================
        # Start at the end of the first training window, step forward by step_size
        for start_trade_idx in range(self.train_window, len(model_data), self.step_size):
            
            # Define the moving boundaries
            start_train_idx = start_trade_idx - self.train_window
            end_trade_idx = min(start_trade_idx + self.step_size, len(model_data))
            
            # Slice the data for this specific window
            train_chunk = model_data.iloc[start_train_idx:start_trade_idx]
            trade_chunk = model_data.iloc[start_trade_idx:end_trade_idx]
            
            X_train = train_chunk[features]
            y_train = train_chunk['target']
            X_trade = trade_chunk[features]
            
            # Train a fresh model just for this specific time period
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            
            # Predict the next block of days and store them
            chunk_preds = clf.predict(X_trade)
            predictions.iloc[start_trade_idx:end_trade_idx] = chunk_preds
            
        # ==========================================
        # 3. Finalize for the Backtester
        # ==========================================
        data['position'] = predictions
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data



class HurstMetaStrategy:
    """
    A Meta-Strategy that calculates the rolling Hurst Exponent to detect the market regime.
    - If Trending (H > 0.55): Uses a Fast/Slow Moving Average Crossover.
    - If Mean-Reverting (H < 0.45): Uses a Bollinger Band reversal logic.
    - If Random (0.45 <= H <= 0.55): Stays in cash.
    """
    def __init__(self, target_ticker: str, hurst_window: int = 60):
        self.target = target_ticker
        self.hurst_window = hurst_window
        
        # Sub-strategy parameters
        self.ma_fast = 20
        self.ma_slow = 60
        self.bb_window = 20
        self.bb_std = 2.0

    def _calculate_hurst(self, price_series: np.ndarray, max_lag: int = 20) -> float:
        """Helper function to calculate the Hurst Exponent for a given array of prices."""
        lags = range(2, max_lag)
        variances = []
        
        for lag in lags:
            # Calculate the variance of the price differences over the lag
            diff = price_series[lag:] - price_series[:-lag]
            variances.append(np.var(diff))
            
        # If variance is 0 (flat prices), return random walk
        if 0 in variances:
            return 0.5 
            
        # Calculate the slope of the log-log plot
        poly = np.polyfit(np.log(lags), np.log(variances), 1)
        
        # The slope is 2H, so we divide by 2 to get H
        hurst_exponent = poly[0] / 2.0
        return hurst_exponent

    def generate_signals(self, df: pd.DataFrame):
        data = df.copy()
        
        # ==========================================
        # 1. Calculate Regime (Rolling Hurst)
        # ==========================================
        # Note: Rolling apply in pandas can be slow, but it's accurate for backtesting
        data['hurst'] = data[self.target].rolling(window=self.hurst_window).apply(
            lambda x: self._calculate_hurst(x.values), raw=False
        )
        
        # ==========================================
        # 2. Calculate Sub-Strategy Indicators
        # ==========================================
        # Trend Indicators (MA Crossover)
        data['fast_ma'] = data[self.target].rolling(self.ma_fast).mean()
        data['slow_ma'] = data[self.target].rolling(self.ma_slow).mean()
        
        # Mean Reversion Indicators (Bollinger Bands)
        data['bb_sma'] = data[self.target].rolling(self.bb_window).mean()
        data['bb_std'] = data[self.target].rolling(self.bb_window).std()
        data['bb_upper'] = data['bb_sma'] + (data['bb_std'] * self.bb_std)
        data['bb_lower'] = data['bb_sma'] - (data['bb_std'] * self.bb_std)

        # ==========================================
        # 3. Meta-Logic Execution
        # ==========================================
        positions = np.zeros(len(data))
        current_pos = 0
        
        for i in range(len(data)):
            h = data['hurst'].iloc[i]
            price = data[self.target].iloc[i]
            
            # Skip until we have enough data
            if pd.isna(h):
                continue
                
            # REGIME 1: TRENDING (H > 0.55)
            if h > 0.55:
                fast = data['fast_ma'].iloc[i]
                slow = data['slow_ma'].iloc[i]
                
                if fast > slow:
                    current_pos = 1
                elif fast < slow:
                    current_pos = -1
                    
            # REGIME 2: MEAN REVERTING (H < 0.45)
            elif h < 0.45:
                upper = data['bb_upper'].iloc[i]
                lower = data['bb_lower'].iloc[i]
                sma = data['bb_sma'].iloc[i]
                
                # Enter positions
                if current_pos == 0:
                    if price < lower:
                        current_pos = 1
                    elif price > upper:
                        current_pos = -1
                # Exit positions if they revert to the mean
                elif current_pos == 1 and price >= sma:
                    current_pos = 0
                elif current_pos == -1 and price <= sma:
                    current_pos = 0
                    
            # REGIME 3: RANDOM WALK (0.45 <= H <= 0.55)
            else:
                current_pos = 0 # Step aside and hold cash
                
            positions[i] = current_pos
            
        data['position'] = positions
        data['instrument_ret'] = data[self.target].pct_change()
        
        return data