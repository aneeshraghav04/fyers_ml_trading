import pandas as pd
import numpy as np
import os

print("=" * 70)
print("RUNNING IMPROVED BACKTEST ON JANUARY DATA")
print("=" * 70)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SIGNALS_PATH = os.path.join(BASE_DIR, "..", "data", "rites_jan_signals_improved.csv")

df = pd.read_csv(SIGNALS_PATH, parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

numeric_cols = ["stop_loss_pct", "take_profit_pct", "position_size", "rsi_14"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


INITIAL_CAPITAL = 100000
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005

# ========== BACKTEST ENGINE ==========
class ImprovedBacktestEngine:
    def __init__(self, data, initial_capital=100000):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.position_entry_price = 0
        self.position_entry_date = None
        self.days_in_position = 0
        self.trades = []
        self.portfolio_values = []
    
    def execute_buy(self, row, shares):
        if shares <= 0:
            return False
        
        price = row["close"] * (1 + SLIPPAGE)
        cost = price * shares
        transaction_cost = (price * shares) * TRANSACTION_COST
        total_cost = cost + transaction_cost
        
        if total_cost <= self.cash:
            self.cash -= total_cost
            self.position = shares
            self.position_entry_price = price
            self.position_entry_date = row["date"]
            self.days_in_position = 0
            
            self.trades.append({
                "date": row["date"],
                "action": "BUY",
                "price": price,
                "shares": shares,
                "cost": total_cost
            })
            return True
        return False
    
    def execute_sell(self, row, shares):
        if shares > self.position:
            shares = self.position
        if shares <= 0:
            return False
        
        price = row["close"] * (1 - SLIPPAGE)
        proceeds = price * shares
        transaction_cost = proceeds * TRANSACTION_COST
        net_proceeds = proceeds - transaction_cost
        
        self.cash += net_proceeds
        
        pnl = (price - self.position_entry_price) * shares - transaction_cost
        pnl_pct = (price / self.position_entry_price - 1) * 100 if self.position_entry_price > 0 else 0
        
        self.position -= shares
        
        if self.position == 0:
            self.position_entry_price = 0
            self.days_in_position = 0
        
        self.trades.append({
            "date": row["date"],
            "action": "SELL",
            "price": price,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })
        return True
    
    def run(self):
        print(f"\nInitial Capital: ₹{self.initial_capital:,.2f}")
        print(f"Period: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Total days: {len(self.data)}")
        print("\nTrade log:")
        print("-" * 80)
        
        for idx, row in self.data.iterrows():
            if self.position > 0:
                self.days_in_position += 1
            
            # RISK MANAGEMENT (HIGHEST PRIORITY)
            if self.position > 0:
                current_price = row["close"]
                loss_pct = (current_price - self.position_entry_price) / self.position_entry_price
                
                # Stop loss
                if loss_pct <= -row["stop_loss_pct"]:
                    self.execute_sell(row, self.position)
                    print(f"{row['date']}: STOP LOSS HIT (Loss: {loss_pct*100:.2f}%)")
                    continue
                
                # Take profit
                elif loss_pct >= row["take_profit_pct"]:
                    self.execute_sell(row, self.position)
                    print(f"{row['date']}: TAKE PROFIT HIT (Gain: {loss_pct*100:.2f}%)")
                    continue
                
                # Max hold days
                elif self.days_in_position >= row["max_hold_days"]:
                    self.execute_sell(row, self.position)
                    print(f"{row['date']}: MAX HOLD DAYS EXCEEDED")
                    continue
            
            # PROCESS SIGNALS
            signal = row["signal"]
            
            if signal in ["STRONG_BUY", "BUY"]:
                if self.position == 0:
                    capital_to_use = self.cash * row["position_size"]
                    shares = int(capital_to_use / row["close"])
                    if shares > 0:
                        self.execute_buy(row, shares)
                        print(f"{row['date']}: {signal:10s} | Regime: {row['market_regime']:20s} | "
                              f"RSI: {row['rsi_14']:5.1f} | Position: {shares} shares")
            
            elif signal in ["STRONG_SELL", "SELL"]:
                if self.position > 0:
                    self.execute_sell(row, self.position)
                    print(f"{row['date']}: {signal:10s} | Regime: {row['market_regime']:20s}")
            
            # RECORD PORTFOLIO VALUE
            portfolio_value = self.cash + (self.position * row["close"])
            self.portfolio_values.append({
                "date": row["date"],
                "portfolio_value": portfolio_value,
                "cash": self.cash,
                "position_value": self.position * row["close"],
                "position": self.position
            })
        
        # CLOSE REMAINING POSITIONS
        if self.position > 0:
            last_row = self.data.iloc[-1]
            self.execute_sell(last_row, self.position)
            print(f"{last_row['date']}: FINAL SELL (End of period)")
        
        print("-" * 80)
        return self.analyze_performance()
    
    def analyze_performance(self):
        portfolio_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades)
        
        final_value = portfolio_df["portfolio_value"].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Daily returns
        portfolio_df["daily_return"] = portfolio_df["portfolio_value"].pct_change()
        
        # Sharpe ratio
        risk_free_rate = 0.06 / 252
        excess_returns = portfolio_df["daily_return"] - risk_free_rate
        
        if excess_returns.std() > 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cumulative = (1 + portfolio_df["daily_return"]).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate
        sell_trades = trades_df[trades_df["action"] == "SELL"]
        winning_trades = sell_trades[sell_trades["pnl"] > 0]
        win_rate = (len(winning_trades) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
        
        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown,
            "num_trades": len(trades_df[trades_df["action"] == "BUY"]),
            "num_sells": len(sell_trades),
            "win_rate_pct": win_rate,
            "portfolio_df": portfolio_df,
            "trades_df": trades_df
        }

# ========== RUN BACKTEST ==========
engine = ImprovedBacktestEngine(df, INITIAL_CAPITAL)
results = engine.run()

# ========== DISPLAY RESULTS ==========
print("\n" + "=" * 70)
print("BACKTEST RESULTS - IMPROVED SYSTEM")
print("=" * 70)
print(f"\nInitial Capital:        ₹{results['initial_capital']:,.2f}")
print(f"Final Value:            ₹{results['final_value']:,.2f}")
print(f"Total Return:           {results['total_return_pct']:+.2f}%")
print(f"\nSharpe Ratio:           {results['sharpe_ratio']:+.2f}")
print(f"Max Drawdown:           {results['max_drawdown_pct']:.2f}%")
print(f"\nNumber of Trades:       {results['num_trades']}")
print(f"Number of Exits:        {results['num_sells']}")
print(f"Win Rate:               {results['win_rate_pct']:.1f}%")

print("\n" + "=" * 70)
if results['sharpe_ratio'] >= 1.5:
    print("✓✓✓ EXCELLENT! Sharpe >= 1.5 TARGET ACHIEVED! ✓✓✓")
elif results['sharpe_ratio'] >= 1.2:
    print("✓✓ VERY GOOD! Sharpe >= 1.2 ✓✓")
elif results['sharpe_ratio'] >= 0:
    print("✓ POSITIVE! Sharpe is positive")
else:
    print("○ In progress: Sharpe still negative, consider refinements")
print("=" * 70)

# Save results
results['trades_df'].to_csv(os.path.join(BASE_DIR, "..", "data", "trades_log.csv"), index=False)
results['portfolio_df'].to_csv(os.path.join(BASE_DIR, "..", "data", "portfolio_values.csv"), index=False)

print("\n✓ Results saved to:")
print(f"  • trades_log.csv")
print(f"  • portfolio_values.csv")
