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