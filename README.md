# EUR/USD Mean-Reversion Trading Bot

An automated intraday trading bot for EUR/USD built in Python, deployed on AWS EC2, and connected to OANDA's v20 REST API. Built as a portfolio project to demonstrate Python automation, trading strategy implementation, and AWS cloud infrastructure.

---

## What This Does

The bot runs 24/5 on an AWS EC2 instance, evaluating EUR/USD every hour and placing trades when a specific set of conditions are met simultaneously. It logs every trade to a CSV file and sends alerts through AWS SNS.

---

## Trading Strategy

**Instrument:** EUR/USD via OANDA (intraday only, all positions closed before 5pm NY rollover)

**Entry logic (all conditions must be true simultaneously):**
1. 4h ADX < 25 — market must be in a ranging, non-trending regime
2. No high-impact economic event within 2 hours (NFP, CPI, ECB, etc.)
3. Price has touched the ATR band (daily open +/- 0.4 x 10-day ATR)
4. RSI(14) confirms oversold (< 35 for longs) or overbought (> 65 for shorts)

**Position sizing:** OANDA fractional units, max 3% account risk per trade

**Exit rules:**
- Target: return to daily open (mean-reversion anchor)
- Stop: band edge + 0.2 x ATR beyond entry (exchange-native stop-market order)
- Hard account stop: pause all trading if account drops to $120

**Paper trading pass criteria (8 weeks minimum before going live):**
- Positive net P&L after simulated spreads
- Win rate above 45%
- Minimum 30 trades before pass/fail verdict
- Expectancy per trade > 0 after costs
- No single week loss exceeding $15

---

## Python Modules

| Module | Purpose |
|---|---|
| `trade_logger.py` | Appends one row per closed trade to trades.csv with full P&L breakdown |
| `atr_rsi.py` | Wilder ATR(10) on FX-day daily bars + RSI(14) on 1h candles |
| `adx.py` | Wilder ADX(14) on 4h candles, matches MT4/TradingView behavior |
| `news_filter.py` | Blocks signals within 2h window of high-impact economic events |
| `entry_signal.py` | Ties all indicators together, returns signal dict or None |

---

## AWS Infrastructure

| Service | What it does |
|---|---|
| EC2 t3.micro | Runs the bot 24/5 on Amazon Linux, free tier |
| IAM Role | Grants EC2 permission to access Secrets Manager and CloudWatch without hardcoded credentials |
| Secrets Manager | Stores OANDA API token and account ID securely |
| SNS Topic | Central notification hub for cost alerts and bot events |
| AWS Budgets | Monitors monthly spend, triggers SNS alert at 80% of $10 threshold |
| CloudWatch | Logs bot activity and monitors instance health |

---

## Key Design Decisions

- **No hardcoded credentials** — OANDA API key is stored in AWS Secrets Manager, EC2 reads it via IAM role at runtime
- **Look-ahead bias prevention** — all candle data filtered to current_time before any indicator is computed, safe for backtesting
- **Structured AI debate framework — Claude and GPT-4 alternated as proposer and critic to stress-test strategy logic before any code was written, the same way a quant uses backtesting to validate before going live. All directional decisions, instrument selection, risk rules, and approval gates were made by the developer.
- **Daily bar ATR** — ATR computed on FX-day daily OHLC bars (5pm NY to 5pm NY), not hourly candles, matching the strategy spec
- **fcntl file locking** — trade log is race-safe for concurrent writes on Linux
- **Paper trading first** — bot runs on OANDA practice account for minimum 8 weeks before any real money is deployed

---

## Project Status

- [x] Trade logger
- [x] ATR and RSI calculations
- [x] ADX regime filter
- [x] News blackout filter
- [x] Entry signal evaluator
- [ ] OANDA executor (places actual orders)
- [ ] Main bot loop
- [ ] EC2 deployment and systemd service
- [ ] CloudWatch logging integration
- [ ] Paper trading phase

---

## Tech Stack

- Python 3.11
- pandas, numpy
- OANDA v20 REST API
- AWS EC2, IAM, Secrets Manager, SNS, Budgets, CloudWatch
- Amazon Linux 2023

---

## About

Built by Fadli Ali as a portfolio project while studying for AWS Solutions Architect Associate (SAA-C03). Demonstrates real-world application of cloud infrastructure, Python automation, and quantitative trading concepts.

GitHub: github.com/fadli-ali
LinkedIn: linkedin.com/in/fadli-ali-063a13386
