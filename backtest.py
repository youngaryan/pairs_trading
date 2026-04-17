import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings

from brain import ZScorePairsStrategy, KalmanPairsStrategy

warnings.filterwarnings("ignore")

class GeneralBacktester:
    """
    A universal backtesting engine. It evaluates ANY strategy, as long as the strategy's
    `generate_signals` method returns a DataFrame containing 'position' and 'instrument_ret'.
    """
    def __init__(self, data: pd.DataFrame, strategy, name="Strategy"):
        """
        :param data: A pandas DataFrame containing whatever raw data the strategy needs.
        :param strategy: An instantiated strategy class.
        :param name: A display name for the plot.
        """
        self.data = data.copy()
        self.strategy = strategy
        self.name = name

    def run_backtest(self):
        # 1. Ask the strategy to generate positions and underlying returns
        self.data = self.strategy.generate_signals(self.data)
        
        # Guardrail: Ensure the strategy followed the rules
        if 'position' not in self.data.columns or 'instrument_ret' not in self.data.columns:
            raise ValueError("The strategy must output a DataFrame with 'position' and 'instrument_ret' columns.")

        # 2. Calculate Strategy Return: Yesterday's Position * Today's Return
        self.data['strategy_ret'] = self.data['position'].shift(1) * self.data['instrument_ret']
        
        # 3. Calculate Cumulative Equity
        self.data['cumulative_ret'] = (1 + self.data['strategy_ret'].fillna(0)).cumprod()
        self.data['buy_and_hold_ret'] = (1 + self.data['instrument_ret'].fillna(0)).cumprod()

        # 4. Performance Metrics
        total_return = (self.data['cumulative_ret'].iloc[-1] - 1) * 100
        bnh_return = (self.data['buy_and_hold_ret'].iloc[-1] - 1) * 100
        
        days_in_market = (self.data['position'] != 0).sum()
        pct_time_in_market = (days_in_market / len(self.data)) * 100
        
        # Calculate Max Drawdown
        rolling_max = self.data['cumulative_ret'].cummax()
        drawdown = (self.data['cumulative_ret'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        print(f"--- Backtest Results: {self.name} ---")
        print(f"Strategy Model:   {self.strategy.__class__.__name__}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"Buy & Hold Ret:   {bnh_return:.2f}%")
        print(f"Max Drawdown:     {max_dd:.2f}%")
        print(f"Time in Market:   {pct_time_in_market:.1f}%")
        print("-" * 40)
        
        return self.data

    def plot_equity_curve(self):
        if 'cumulative_ret' not in self.data.columns:
            print("Run backtest first!")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data['cumulative_ret'], label=f'{self.name} Strategy', color='green')
        plt.plot(self.data.index, self.data['buy_and_hold_ret'], label='Underlying Asset (Buy & Hold)', color='gray', alpha=0.5, linestyle='--')
        plt.axhline(1, color='black', linestyle=':', alpha=0.3)
        
        plt.title(f"Cumulative Return: {self.name}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Multiplier")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        
        

if __name__ == "__main__":
    
    TICK1 = "LOW"
    TICK2 = "KO" 
       
    print("1. Downloading Data for PEP and KO...")
    # Download data
    raw_data = yf.download(["PEP", "KO"], start="2018-01-01", end="2026-01-01", progress=False)
    
    # Flatten the yfinance MultiIndex dataframe to get clean columns
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_df = raw_data['Close']
    else:
        price_df = raw_data
        
    price_df = price_df.dropna()

    print("\n2. Testing Classic OLS Z-Score Strategy...")
    # Instantiate strategy
    zscore_strat = ZScorePairsStrategy(ticker1="PEP", ticker2="KO", entry_z=2.0, exit_z=0.0, window=60)
    
    # Run backtest
    zscore_bt = GeneralBacktester(data=price_df, strategy=zscore_strat, name="PEP/KO OLS Z-Score")
    zscore_bt.run_backtest()
    zscore_bt.plot_equity_curve()


    print("\n3. Testing Dynamic Kalman Filter Strategy...")
    # Instantiate strategy (no window needed!)
    kalman_strat = KalmanPairsStrategy(ticker1="PEP", ticker2="KO", entry_z=2.0, exit_z=0.0)
    
    # Run backtest
    kalman_bt = GeneralBacktester(data=price_df, strategy=kalman_strat, name="PEP/KO Kalman Filter")
    kalman_bt.run_backtest()
    kalman_bt.plot_equity_curve()