# ML Trading Pipeline — Directional Classifier + Backtest

A modular machine-learning research pipeline for predicting **next-day price direction** and evaluating it with a **realistic trading backtest**.

This project is designed to be **modified and extended** for personal trading research.

---

## What it does

The script:
- Downloads daily market data (default: **AAPL**) via `yfinance`
- Builds **no-lookahead features**
- Creates **next-day direction labels**
- Filters out tiny price moves to reduce noise
- Selects features using **filter + wrapper + embedded** methods
- Trains a **time-series-validated SVM classifier**
- Finds the **optimal probability threshold**
- Runs a **same-day MOC trading backtest**
- Compares performance vs **buy & hold**

---

## Pipeline

1. **Data**
   - 5 years of daily adjusted OHLCV

2. **Labels**
   - `1` if tomorrow’s return > 0  
   - `0` otherwise  
   - `FwdRet` = tomorrow’s return

3. **Noise control**
   - Ignore small moves for metrics:  
     `|FwdRet| < 0.25%`  
   - Backtest still uses full data

4. **Features**
   - Moving-average ratios
   - Momentum
   - Volatility
   - Lagged returns
   - Normalized high-low ranges  
   *(all computed using past data only)*

5. **Feature selection**
   - Correlation filter  
   - Redundancy removal  
   - Time-series wrapper test  
   - Gradient-boosting importance  

6. **Model**
   - `StandardScaler → SVM`
   - Grid search with `TimeSeriesSplit`
   - Optimized by ROC-AUC

7. **Threshold**
   - Out-of-fold probabilities
   - F1-optimized cutoff

8. **Evaluation**
   - ROC-AUC, ROC curve
   - Confusion matrix
   - Precision / Recall / F1

9. **Backtest**
   - Trade at **market-on-close**
   - Long when `P(up) > threshold`
   - 10 bps transaction cost per position change
   - Equity curve vs buy & hold

---

## Run

```bash
pip install -r requirements.txt
python Binary Classifier.py