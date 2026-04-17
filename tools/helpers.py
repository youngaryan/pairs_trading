import yfinance as yf
import pandas as pd
import itertools
from statsmodels.tsa.stattools import coint
import warnings

# Suppress statsmodels warnings for a cleaner console output
warnings.filterwarnings("ignore")

def find_cointegrated_pairs(tickers, start_date, end_date, p_value_threshold=0.05):
    """
    Downloads historical data for a list of tickers, tests all unique pairs 
    for cointegration, and returns a sorted dataframe of the results.
    """
    print(f"1. Downloading historical data for {len(tickers)} tickers...")
    
    # Bulk download all tickers at once for speed
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    # Extract just the 'Close' prices
    if isinstance(raw_data.columns, pd.MultiIndex):
        prices = raw_data['Close']
    else:
        prices = raw_data
        
    # Generate all unique combinations (pairs) from the ticker list
    ticker_columns = prices.columns.tolist()
    pairs = list(itertools.combinations(ticker_columns, 2))
    
    print(f"2. Generating and testing {len(pairs)} unique pairs...")
    
    results = []
    
    for t1, t2 in pairs:
        # Isolate the two tickers and drop any rows where either has NaN values
        # We do this per pair so one missing stock doesn't ruin the whole dataset
        pair_data = prices[[t1, t2]].dropna()
        
        # Ensure we have enough data points for a statistically valid test
        if len(pair_data) < 100:
            continue
            
        try:
            # Run the Engle-Granger cointegration test
            score, p_value, _ = coint(pair_data[t1], pair_data[t2])
            
            results.append({
                'Pair': f"{t1} - {t2}",
                'Ticker_1': t1,
                'Ticker_2': t2,
                'P_Value': p_value,
                'Is_Cointegrated': p_value < p_value_threshold
            })
        except Exception as e:
            # Catch errors if a time series is perfectly flat or otherwise invalid
            continue

    # Convert to a DataFrame and sort by the most cointegrated (lowest p-value)
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        results_df = results_df.sort_values(by='P_Value', ascending=True).reset_index(drop=True)
        
    return results_df

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # A mix of competitors across different sectors to scan
    sector_tickers = [
        "PEP", "KO", "KDP",         # Beverages
        "XOM", "CVX", "COP",        # Oil & Gas
        "JPM", "BAC", "WFC", "C",   # Banking
        "AMD", "NVDA", "INTC",      # Semiconductors
        "HD", "LOW"                 # Home Improvement
    ]
    
    print("Starting Pairs Screener...")
    
    # Run the screener
    cointegration_results = find_cointegrated_pairs(
        tickers=sector_tickers, 
        start_date="2012-01-01",
        end_date="2024-12-30"
    )
    
    # Display the Top 10 most cointegrated pairs
    print("\n--- TOP 10 MOST COINTEGRATED PAIRS ---")
    print(cointegration_results.head(10).to_string(index=False))
    
    # Optionally, you can filter to show ONLY the viable trading pairs
    viable_pairs = cointegration_results[cointegration_results['Is_Cointegrated'] == True]
    print(f"\nTotal viable pairs found (P-Value < 0.05): {len(viable_pairs)}")