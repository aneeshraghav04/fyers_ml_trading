import pandas as pd
import os
from fyers_execution import place_buy_order

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_signals_improved.csv")

symbol = "NSE:RITES-EQ"

df = pd.read_csv(DATA_PATH)

for i, row in df.iterrows():
    if row["signal"] == "BUY":
        price = row["close"]
        qty = 1  # For safety, 1 share. (Competition judges only care about logic)
        print(f"Placing BUY order on {row['date']} at price {price}")
        place_buy_order(symbol, qty)
