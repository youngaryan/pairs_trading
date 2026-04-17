import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings

from brain import ZScorePairsStrategy

warnings.filterwarnings("ignore")

class GeneralBacktester:
    """
    A generic backtesting engine. It doesn't know HOW signals are generated, 
    it only cares about taking a Strategy object and evaluating its performance.
    """
    def __init__(self, ticker1: str, ticker2: str, price_series1: pd.Series, price_series2: pd.Series, strategy):
        self.t1 = ticker1
        self.t2 = ticker2
        self.strategy = strategy  # Injects the strategy class!
        
        self.data = pd.DataFrame({self.t1: price_series1, self.t2: price_series2}).dropna()
        self.hedge_ratio = None

    def run_backtest(self):
        """Calculates the cumulative equity curve based on the injected strategy's signals."""
        
        # Ask the injected strategy to calculate the positions and spread returns
        self.data, self.hedge_ratio = self.strategy.generate_signals(self.data, self.t1, self.t2)
        
        # Core Strategy Return: Position from YESTERDAY * Spread Return TODAY (prevents lookahead bias)
        self.data['strategy_ret'] = self.data['position'].shift(1) * self.data['spread_ret']
        self.data['cumulative_ret'] = (1 + self.data['strategy_ret'].fillna(0)).cumprod()
        
        # Calculate Metrics
        total_return = (self.data['cumulative_ret'].iloc[-1] - 1) * 100
        days_in_market = (self.data['position'] != 0).sum()
        pct_time_in_market = (days_in_market / len(self.data)) * 100
        
        print(f"--- Backtest Results: {self.t1} vs {self.t2} ---")
        print(f"Strategy Used:      {self.strategy.__class__.__name__}")
        print(f"Total Return:       {total_return:.2f}%")
        print(f"Time in Market:     {pct_time_in_market:.1f}%")
        print(f"Hedge Ratio (Beta): {self.hedge_ratio:.4f}")
        print("-" * 40)
        
        return self.data

    def plot_equity_curve(self):
        """Visualizes the growth of the strategy over time."""
        if 'cumulative_ret' not in self.data.columns:
            print("Run backtest first!")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data['cumulative_ret'], label=f'{self.t1}/{self.t2} Equity', color='green')
        plt.axhline(1, color='gray', linestyle='--')
        plt.title(f"Cumulative Return: {self.t1} vs {self.t2} ({self.strategy.__class__.__name__})")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Multiplier")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        
        
        
        
        
if __name__ == "__main__":
    print("Downloading data...")
    # Fetching recent hourly data
    prices = yf.download(["JPM", "NVDA"], start="2025-01-01", end="2026-02-01", interval="1h", progress=False)['Close']
    
    # Extract columns carefully to handle potential yfinance MultiIndex formatting
    jpm_prices = prices["JPM"] if isinstance(prices, pd.DataFrame) else prices.xs('JPM', level=1, axis=1)
    nvda_prices = prices["NVDA"] if isinstance(prices, pd.DataFrame) else prices.xs('NVDA', level=1, axis=1)

    # 1. Instantiate the Strategy (The Indicator Logic)
    my_zscore_strategy = ZScorePairsStrategy(entry_z=2.0, exit_z=0.0, window=60)
    
    # 2. Instantiate the Backtester (The Execution Engine), passing the strategy into it
    backtester = GeneralBacktester(
        ticker1="JPM", 
        ticker2="NVDA", 
        price_series1=jpm_prices, 
        price_series2=nvda_prices,
        strategy=my_zscore_strategy  # <--- Dependency Injection happens here!
    )
    
    # 3. Run and Plot
    backtester.run_backtest()
    backtester.plot_equity_curve()