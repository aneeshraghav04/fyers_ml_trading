from fyers_apiv3 import fyersModel
import pandas as pd
from datetime import datetime

# ========== CONFIG ==========
client_id = "ZKX7NS29YX-100"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcFlUbURzcWxQQkFoalQ2ZWNjdXBTUTVHUHhHd3EtZzdfUG9lUFZISUd3d2RHcmRYblVRWHROVHJtdV96ZUJrSTJHUjVCbEF0UEgwX1FNc201UmMwajUwOFA1WExXazFoYVMxcklzSjFoQmFBcFpMYz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiJlY2I1MWYwYzAxOTU2YjZjN2NhOGIyNDJlY2I5YTdjZjgwNmQ2MzY5MGY5MGZhMmViMzM2YjA5NiIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFIODU0MjQiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2ODAwNTAwMCwiaWF0IjoxNzY3OTc5Mzk1LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3Njc5NzkzOTUsInN1YiI6ImFjY2Vzc190b2tlbiJ9.IavVBr2W3epq2EXAXzuUlGeWbvsB1wsDvOTNosJTWZM"

symbol = "NSE:RITES-EQ"   # Change to NSE:IRCON-EQ or NSE:SONATSOFTW-EQ if needed
start_date = "2025-11-01"
end_date = "2025-12-31"
output_file = "rites_ohlcv.csv"
# ============================

# Initialize FYERS
fyers = fyersModel.FyersModel(
    client_id=client_id,
    token=access_token,
    log_path=""
)

# Prepare request data
data = {
    "symbol": symbol,
    "resolution": "D",        # Daily candles
    "date_format": "1",       # 1 = yyyy-mm-dd
    "range_from": start_date,
    "range_to": end_date,
    "cont_flag": "1"
}

# Fetch data
response = fyers.history(data)

print("Raw API response:")
print(response)

# Check success
if response.get("s") != "ok":
    print("Error fetching data:", response)
    exit()

candles = response["candles"]

# Convert to DataFrame
df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])

# Convert timestamp to date
df["date"] = pd.to_datetime(df["timestamp"], unit="s")
df = df[["date", "open", "high", "low", "close", "volume"]]

# Sort by date (important)
df = df.sort_values("date").reset_index(drop=True)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"\nSaved OHLCV data to {output_file}")
print(df.head())
print("\nData summary:")
print(df.describe())
