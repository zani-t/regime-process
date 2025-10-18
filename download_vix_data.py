import os
import sys

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Error: yfinance not installed")
    print("Install with: pip install yfinance")
    sys.exit(1)

START_DATE = '2024-01-01'

def download_vix_data(
    start_date=START_DATE,
    end_date=None,
    output_path='data/vix_data.csv'
):
    """
    Download VIX data from Yahoo Finance.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        output_path: Output CSV path
    """
    print(f"Downloading VIX data from {start_date} to {end_date or 'today'}...")
    
    # Download data
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    
    if vix is None or len(vix) == 0:
        print("Error: No data downloaded")
        return False
    
    # Prepare dataframe
    vix = vix.reset_index()
    vix = vix[['Date', 'Close']]
    vix.columns = ['Date', 'VIX']
    
    # Remove NaN values
    vix = vix.dropna()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    vix.to_csv(output_path, index=False)
    
    print(f"\n✓ Downloaded {len(vix)} observations")
    print(f"  Date range: {vix['Date'].min()} to {vix['Date'].max()}")
    print(f"  VIX range: {vix['VIX'].min():.2f} to {vix['VIX'].max():.2f}")
    print(f"  Mean VIX: {vix['VIX'].mean():.2f}")
    print(f"  Saved to: {output_path}")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download VIX data from Yahoo Finance')
    parser.add_argument('--start', type=str, default=START_DATE,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--output', type=str, default='data/vix_data.csv',
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    success = download_vix_data(args.start, args.end, args.output)
    sys.exit(0 if success else 1)
