import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import warnings

from brain import HMMRegimeStrategy, TargetVolatilityStrategy, ZScorePairsStrategy, KalmanPairsStrategy, MovingAverageCrossoverStrategy, BollingerBandsStrategy, DonchianBreakoutStrategy, MachineLearningStrategy,WalkForwardMLStrategy,HurstMetaStrategy

warnings.filterwarnings("ignore")


class GeneralBacktester:
    """
    A universal backtesting engine with built-in Risk Management.
    Enforces Stop-Loss and Take-Profit limits independently of the Strategy's logic.
    """
    def __init__(self, data: pd.DataFrame, strategy, name="Strategy", stop_loss: float = None, take_profit: float = None):
        """
        :param data: Raw data dataframe.
        :param strategy: Instantiated strategy class.
        :param name: Display name.
        :param stop_loss: Float representing max loss per trade (e.g., 0.05 for 5%).
        :param take_profit: Float representing target profit per trade (e.g., 0.10 for 10%).
        """
        self.data = data.copy()
        self.strategy = strategy
        self.name = name
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run_backtest(self):
        # 1. Ask the strategy for its raw, unfiltered signals
        self.data = self.strategy.generate_signals(self.data)
        
        if 'position' not in self.data.columns or 'instrument_ret' not in self.data.columns:
            raise ValueError("The strategy must output 'position' and 'instrument_ret'.")

        # 2. Risk Management Engine (Path-Dependent Loop)
        if self.stop_loss is not None or self.take_profit is not None:
            raw_positions = self.data['position'].values
            returns = self.data['instrument_ret'].values
            
            managed_positions = np.zeros(len(raw_positions))
            current_pos = 0
            trade_pnl = 0.0  # Tracks the cumulative return of the active trade
            
            for i in range(len(raw_positions)):
                strat_signal = raw_positions[i]
                today_ret = returns[i]
                
                # If we are currently in a trade, update its PnL
                if current_pos != 0:
                    # Compounding the trade's PnL
                    trade_pnl = (1 + trade_pnl) * (1 + (today_ret * current_pos)) - 1
                    
                    # Check Risk Limits
                    hit_sl = self.stop_loss is not None and trade_pnl <= -self.stop_loss
                    hit_tp = self.take_profit is not None and trade_pnl >= self.take_profit
                    
                    if hit_sl or hit_tp:
                        current_pos = 0  # Force exit (Stop-out / Take-profit)
                        trade_pnl = 0.0  # Reset trade PnL
                    else:
                        # If strategy wants to flip direction or exit naturally
                        if strat_signal != current_pos:
                            current_pos = strat_signal
                            trade_pnl = 0.0
                
                # If we are flat, just listen to the strategy
                elif strat_signal != 0:
                    current_pos = strat_signal
                    trade_pnl = 0.0
                
                managed_positions[i] = current_pos
                
            # Overwrite the strategy's raw positions with our risk-managed positions
            self.data['position'] = managed_positions

        # 3. Calculate Strategy Return: Yesterday's Position * Today's Return
        self.data['strategy_ret'] = self.data['position'].shift(1) * self.data['instrument_ret']
        
        # 4. Calculate Cumulative Equity
        self.data['cumulative_ret'] = (1 + self.data['strategy_ret'].fillna(0)).cumprod()
        self.data['buy_and_hold_ret'] = (1 + self.data['instrument_ret'].fillna(0)).cumprod()

        # 5. Performance Metrics
        total_return = (self.data['cumulative_ret'].iloc[-1] - 1) * 100
        bnh_return = (self.data['buy_and_hold_ret'].iloc[-1] - 1) * 100
        
        days_in_market = (self.data['position'] != 0).sum()
        pct_time_in_market = (days_in_market / len(self.data)) * 100
        
        rolling_max = self.data['cumulative_ret'].cummax()
        drawdown = (self.data['cumulative_ret'] - rolling_max) / rolling_max
        max_dd = drawdown.min() * 100

        print(f"--- Backtest Results: {self.name} ---")
        print(f"Risk Params:      SL: {self.stop_loss} | TP: {self.take_profit}")
        print(f"Total Return:     {total_return:.2f}%")
        print(f"Buy & Hold Ret:   {bnh_return:.2f}%")
        print(f"Max Drawdown:     {max_dd:.2f}%")
        print(f"Time in Market:   {pct_time_in_market:.1f}%")
        print("-" * 40)
        
        return self.data

    def plot_equity_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data.index, self.data['cumulative_ret'], label=f'{self.name} Strategy', color='green')
        plt.plot(self.data.index, self.data['buy_and_hold_ret'], label='Underlying Asset', color='gray', alpha=0.5, linestyle='--')
        plt.axhline(1, color='black', linestyle=':', alpha=0.3)
        plt.title(f"Equity Curve: {self.name}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Multiplier")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    
    TICK = "AAPL"
    print(f"1. Downloading Data for {TICK}...")
    raw_data = yf.download(TICK, start="2025-01-01",interval="1h", end="2026-04-01", progress=False)
    
    if isinstance(raw_data.columns, pd.MultiIndex):
        price_df = raw_data['Close']
    else:
        price_df = raw_data[['Close']].rename(columns={'Close': TICK})
        
    price_df = price_df.dropna()

    print("\n2. Testing Bollinger Bands Strategy...")
    bb_strat = BollingerBandsStrategy(target_ticker=TICK, window=20, num_std=2.0)
    bb_bt = GeneralBacktester(data=price_df, strategy=bb_strat, name="TICK Bollinger Bands")
    bb_bt.run_backtest()
    bb_bt.plot_equity_curve()

    print("\n3. Testing Donchian Breakout Strategy...")
    turtle_strat = DonchianBreakoutStrategy(target_ticker=TICK, entry_window=20)
    turtle_bt = GeneralBacktester(data=price_df, strategy=turtle_strat, name=f"{TICK} Donchian Breakout")
    turtle_bt.run_backtest()
    turtle_bt.plot_equity_curve()
    
    print("\n4. Testing Moving Average Strategy...")
    ma_strategy = MovingAverageCrossoverStrategy(target_ticker=TICK, fast_window=50, slow_window=200)

    # 3. Instantiate and run the Backtester
    backtester = GeneralBacktester(
        data=price_df, 
        strategy=ma_strategy, 
        name="SPY Trend Follower"
    )
    
    backtester.run_backtest()
    backtester.plot_equity_curve()
    
    
    print("\n5. Training and Backtesting the Random Forest Model...")
    # Train on the first 1000 days (~4 years), then trade on the remaining time
    ml_strat = MachineLearningStrategy(target_ticker=TICK, train_window_days=1000)
    
    ml_bt = GeneralBacktester(data=price_df, strategy=ml_strat, name=f"{TICK} Random Forest AI")
    ml_bt.run_backtest()
    ml_bt.plot_equity_curve()
    
    print("\n6. Backtesting Walk-Forward ML Strategy (This may take a moment)...")
    # Retrains a fresh AI model every 60 trading days (~3 months)
    wf_ml_strat = WalkForwardMLStrategy(target_ticker=TICK, train_window=1000, step_size=60)
    
    wf_bt = GeneralBacktester(data=price_df, strategy=wf_ml_strat, name=f"{TICK} Walk-Forward AI")
    wf_bt.run_backtest()
    wf_bt.plot_equity_curve()
    
    print("\n7. Backtesting the Hurst Meta-Strategy (This will take a minute or two)...")
    hurst_strat = HurstMetaStrategy(target_ticker=TICK, hurst_window=60)
    
    hurst_bt = GeneralBacktester(data=price_df, strategy=hurst_strat, name=f"{TICK} Hurst Regime Switcher")
    hurst_bt.run_backtest()
    hurst_bt.plot_equity_curve()
    
    print("\n8. Backtesting the HMMRegimeStrategy (This will take a minute or two)...")
    hurst_strat = HMMRegimeStrategy(target_ticker=TICK, train_window=100)
    
    hurst_bt = GeneralBacktester(data=price_df, strategy=hurst_strat, name=f"{TICK} HMMRegimeStrategy")
    hurst_bt.run_backtest()
    hurst_bt.plot_equity_curve()
    
    print("\n9. Backtesting the TargetVolatilityStrategy (This will take a minute or two)...")
    hurst_strat = TargetVolatilityStrategy(target_ticker=TICK)
    
    hurst_bt = GeneralBacktester(data=price_df, strategy=hurst_strat, name=f"{TICK} TargetVolatilityStrategy")
    hurst_bt.run_backtest()
    hurst_bt.plot_equity_curve()