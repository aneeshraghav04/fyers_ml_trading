from fyers_apiv3 import fyersModel
import pandas as pd
import os

# ========== CONFIG ==========
client_id = "ZKX7NS29YX-100"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcFk0OTh2emxwY2ZfMzlvcl92cGd1R0RWQ3NIWHlCR1pqRmcwMlIzUllyd2JTVU5NWHhvamFDSnN5eUFEQ0RHdTRQckxiYl9LdGFFOHloZFlERDk0R3lwQk1hOEVYeXVlS2pieTA5VW9IQ25CM3pSbz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwYjRhZDUyZWNhZmM0ZWFjMTY5NjUyYjk0YzMwOWE1ODk4NTc0NWNhYzNlYTJhOWFiMDYyZTNhNyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFIODU0MjQiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2ODE3NzgwMCwiaWF0IjoxNzY4MTMyNDc2LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjgxMzI0NzYsInN1YiI6ImFjY2Vzc190b2tlbiJ9.VF0ma2rCM5P5cEbMFG9X7oQPXzsInVWjnR_dYPkVeMg"

symbol = "NSE:RITES-EQ"
start_date = "2026-01-01"
end_date = "2026-01-08"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_ohlcv.csv")

# ========== INIT FYERS ==========
fyers = fyersModel.FyersModel(
    client_id=client_id,
    token=access_token,
    log_path=""
)

data = {
    "symbol": symbol,
    "resolution": "D",
    "date_format": "1",
    "range_from": start_date,
    "range_to": end_date,
    "cont_flag": "1"
}

response = fyers.history(data)

if response.get("s") != "ok":
    print("Error fetching Jan data:", response)
    exit()

candles = response["candles"]

df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df[["date", "open", "high", "low", "close", "volume"]]
df = df.sort_values("date").reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Jan data saved to {OUTPUT_PATH}")
print(df)
