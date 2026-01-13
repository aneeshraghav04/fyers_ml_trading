from fyers_apiv3 import fyersModel
import os

client_id = "ZKX7NS29YX-100"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCcFk0OTh2emxwY2ZfMzlvcl92cGd1R0RWQ3NIWHlCR1pqRmcwMlIzUllyd2JTVU5NWHhvamFDSnN5eUFEQ0RHdTRQckxiYl9LdGFFOHloZFlERDk0R3lwQk1hOEVYeXVlS2pieTA5VW9IQ25CM3pSbz0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiIwYjRhZDUyZWNhZmM0ZWFjMTY5NjUyYjk0YzMwOWE1ODk4NTc0NWNhYzNlYTJhOWFiMDYyZTNhNyIsImlzRGRwaUVuYWJsZWQiOiJOIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiRkFIODU0MjQiLCJhcHBUeXBlIjoxMDAsImV4cCI6MTc2ODE3NzgwMCwiaWF0IjoxNzY4MTMyNDc2LCJpc3MiOiJhcGkuZnllcnMuaW4iLCJuYmYiOjE3NjgxMzI0NzYsInN1YiI6ImFjY2Vzc190b2tlbiJ9.VF0ma2rCM5P5cEbMFG9X7oQPXzsInVWjnR_dYPkVeMg"

fyers = fyersModel.FyersModel(
    client_id=client_id,
    token=access_token,
    log_path=""
)

def place_buy_order(symbol, qty):
    order = {
        "symbol": symbol,
        "qty": int(qty),
        "type": 2,            # Market order
        "side": 1,            # Buy
        "productType": "CNC",
        "limitPrice": 0,
        "stopPrice": 0,
        "validity": "DAY",
        "disclosedQty": 0,
        "offlineOrder": "False"
    }

    response = fyers.place_order(order)
    print("Order Response:", response)
    return response
