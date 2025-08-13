# src/data_loader.py
import yfinance as yf
import pandas as pd
import streamlit as st

def load_data(symbol: str, start_date, end_date):
    """
    Load historical stock/index data from Yahoo Finance.
    Returns a DataFrame with 'Date' and 'Close' columns.
    """
    try:
        # Download the full dataset from yfinance
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            st.warning(f"No data downloaded for {symbol}. Check the ticker and date range.")
            return pd.DataFrame()
        
        # 1. Reset the index to turn the 'Date' index into a column
        df = df.reset_index()
        
        # 2. Keep only the 'Date' and 'Close' columns
        # This guarantees the output format.
        df = df[["Date", "Close"]]
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()