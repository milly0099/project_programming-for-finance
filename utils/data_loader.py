import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def load_example_data(dataset_type="Stock Price Data"):
    """
    Generates example financial datasets for demo purposes.
    
    Parameters:
    -----------
    dataset_type : str
        Type of example dataset to generate:
        - "Stock Price Data": Daily stock prices with indicators
        - "Financial Metrics": Company financial metrics
        - "Trading Signals": Trading signal data with binary outcomes
    
    Returns:
    --------
    pandas.DataFrame
        Example dataset
    """
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == "Stock Price Data":
        # Create a stock price time series dataset
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        try:
            # Try to fetch real data
            df = yf.download('AAPL', start=start_date, end=end_date)
            
            # Add some technical indicators
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Relative Strength Index (simplified)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * std
            df['BB_Lower'] = df['BB_Middle'] - 2 * std
            
            # Moving Average Convergence Divergence
            df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Trading volume features
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
            
            # Price momentum
            df['Return_1d'] = df['Close'].pct_change()
            df['Return_5d'] = df['Close'].pct_change(periods=5)
            df['Return_20d'] = df['Close'].pct_change(periods=20)
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Drop NaN values
            df.dropna(inplace=True)
            
            return df
        
        except Exception as e:
            print(f"Error fetching stock data: {e}. Creating synthetic data instead.")
            # If fetching fails, create synthetic data
    
    # If we're here, either another dataset was requested or the API fetch failed
    if dataset_type == "Stock Price Data" or dataset_type == "Trading Signals":
        # Create 500 days of simulated stock data
        dates = pd.date_range(end=datetime.now(), periods=500).tolist()
        
        # Start with a base price and add random walks
        base_price = 100
        prices = [base_price]
        for i in range(1, 500):
            # Random walk with slight upward drift
            rw = np.random.normal(0.0005, 0.015)
            prices.append(prices[-1] * (1 + rw))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Open': [price * (1 + np.random.normal(0, 0.005)) for price in prices],
            'High': [price * (1 + abs(np.random.normal(0, 0.01))) for price in prices],
            'Low': [price * (1 - abs(np.random.normal(0, 0.01))) for price in prices],
            'Close': prices,
            'Volume': [int(np.random.normal(1000000, 300000)) for _ in range(500)]
        })
        
        # Add technical indicators
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (simplified)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * std
        df['BB_Lower'] = df['BB_Middle'] - 2 * std
        
        # Moving Average Convergence Divergence
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume Change
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Returns
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(periods=5)
        
        if dataset_type == "Trading Signals":
            # Add trading signals
            # Buy signal when price crosses above 20 SMA
            df['SMA_Signal'] = np.where(df['Close'] > df['SMA_20'], 1, 0)
            
            # RSI overbought/oversold signals
            df['RSI_Signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
            
            # MACD crossover signal
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            df['MACD_Signal'] = np.where(df['MACD_Hist'] > 0, 1, -1)
            
            # Bollinger Band signals
            df['BB_Signal'] = np.where(df['Close'] < df['BB_Lower'], 1, 
                                    np.where(df['Close'] > df['BB_Upper'], -1, 0))
            
            # Combined signal
            df['Combined_Signal'] = df['SMA_Signal'] + df['RSI_Signal'] + df['MACD_Signal'] + df['BB_Signal']
            
            # Binary trade decision (buy/sell)
            df['Trade_Decision'] = np.where(df['Combined_Signal'] > 0, 1, 0)
            
            # Next day return (target)
            df['Next_Return'] = df['Return_1d'].shift(-1)
            
            # Profitable trade (binary target)
            df['Profitable'] = np.where((df['Trade_Decision'] == 1) & (df['Next_Return'] > 0), 1, 
                                      np.where((df['Trade_Decision'] == 0) & (df['Next_Return'] < 0), 1, 0))
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    elif dataset_type == "Financial Metrics":
        # Create a synthetic financial metrics dataset for 100 companies
        np.random.seed(42)
        companies = [f"Company_{i}" for i in range(1, 101)]
        
        # Sectors
        sectors = ['Technology', 'Finance', 'Healthcare', 'Consumer', 'Energy', 
                  'Materials', 'Industrials', 'Utilities', 'Real Estate', 'Telecom']
        
        # Company financials
        revenue = np.random.normal(1000, 300, 100)  # Million $
        profit_margin = np.random.normal(0.15, 0.08, 100)
        profit = revenue * profit_margin
        
        assets = np.random.normal(5000, 1500, 100)  # Million $
        debt = assets * np.random.normal(0.3, 0.15, 100)
        equity = assets - debt
        
        # Financial ratios
        pe_ratio = np.random.normal(20, 8, 100)
        pb_ratio = np.random.normal(2.5, 1.2, 100)
        dividend_yield = np.random.normal(0.02, 0.01, 100)
        roe = profit / equity  # Return on Equity
        roa = profit / assets  # Return on Assets
        debt_equity = debt / equity
        
        # Growth metrics
        revenue_growth = np.random.normal(0.08, 0.05, 100)
        profit_growth = np.random.normal(0.06, 0.07, 100)
        
        # Market data
        market_cap = revenue * np.random.normal(3, 1, 100)  # Million $
        beta = np.random.normal(1.1, 0.4, 100)
        volatility = np.random.normal(0.25, 0.1, 100)
        
        # ESG scores (Environmental, Social, Governance)
        esg_score = np.random.normal(70, 15, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Company': companies,
            'Sector': np.random.choice(sectors, 100),
            'Revenue_M': revenue,
            'Profit_Margin': profit_margin,
            'Profit_M': profit,
            'Assets_M': assets,
            'Debt_M': debt,
            'Equity_M': equity,
            'PE_Ratio': np.abs(pe_ratio),  # Ensure positive
            'PB_Ratio': np.abs(pb_ratio),  # Ensure positive
            'Dividend_Yield': np.abs(dividend_yield),  # Ensure positive
            'ROE': roe,
            'ROA': roa,
            'Debt_to_Equity': debt_equity,
            'Revenue_Growth': revenue_growth,
            'Profit_Growth': profit_growth,
            'Market_Cap_M': market_cap,
            'Beta': beta,
            'Volatility': volatility,
            'ESG_Score': np.clip(esg_score, 0, 100)  # Clip to 0-100 range
        })
        
        return df
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
