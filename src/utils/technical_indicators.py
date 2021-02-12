import pandas as pd

TECHNICAL_INDICATORS = ["Closing Price", "SMA", "EMA", "ROC", "MACD", "Fast K", "Slow D", "Upper Band", "Lower Band"]

N = 20

INDICATOR_FUNCTIONS = {
    "Closing Price": lambda df: df,
    "SMA": lambda df: df.rolling(window=5).mean(),
    "EMA": lambda df: df.ewm(span=N).mean(),
    "MACD": lambda df: df.ewm(span=12, adjust=False).mean() - df.ewm(span=26, adjust=False).mean(),
    "ROC": lambda df: df.pct_change(periods=1),
}
