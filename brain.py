import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings

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