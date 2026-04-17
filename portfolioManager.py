import pandas as pd
import numpy as np

class PortfolioManager:
    """
    Takes raw signals from a Strategy and translates them into actual 
    capital allocation weights based on risk parameters.
    """
    def __init__(self, max_leverage: float = 1.0, risk_per_trade: float = 0.02):
        self.max_leverage = max_leverage
        self.risk_per_trade = risk_per_trade # e.g., risk max 2% of portfolio per trade

    def allocate_capital(self, strategy_data: pd.DataFrame) -> pd.DataFrame:
        data = strategy_data.copy()
        
        # Ensure the strategy gave us a raw 'signal' (-1, 0, 1) and 'instrument_ret'
        if 'signal' not in data.columns or 'instrument_ret' not in data.columns:
            raise ValueError("Portfolio Manager requires 'signal' and 'instrument_ret' columns.")
            
        # Example Sizing Logic: Volatility Parity
        # Calculate trailing 20-day volatility
        data['volatility'] = data['instrument_ret'].rolling(20).std() * np.sqrt(252)
        
        # Prevent division by zero
        data['volatility'] = data['volatility'].replace(0, np.nan)
        
        # Position Size = (Target Risk) / (Current Volatility)
        # If volatility is high, position size drops. If low, position size increases.
        data['target_weight'] = (self.risk_per_trade / data['volatility']) * data['signal']
        
        # Constrain by maximum allowed leverage
        data['target_weight'] = data['target_weight'].clip(lower=-self.max_leverage, upper=self.max_leverage)
        
        # Shift the weight by 1 day to prevent lookahead bias (trade tomorrow based on today's target)
        data['position'] = data['target_weight'].shift(1).fillna(0)
        
        return data