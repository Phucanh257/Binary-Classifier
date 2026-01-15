# =====================================================
# Binary Classifier with MOC Backtest
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score
)
from sklearn.base import clone
import yfinance as yf

RANDOM_STATE = 42

# -----------------------------
# 1) Data & Targets 
# -----------------------------
df = yf.download("AAPL", period="5y", interval="1d", auto_adjust=True).sort_index()
price = "Close"  # adjusted close when auto_adjust=True

# Daily simple returns
df["Return"] = df[price].pct_change()
df = df.dropna()

# Binary label and forward return at calendar t+1
df["Label"]  = (df["Return"].shift(-1) > 0).astype(int)
df["FwdRet"] = df["Return"].shift(-1)

# Near-zero band on the dependent leg 
EPS = 0.0025  # 25 bps
band_mask = df["FwdRet"].abs() >= EPS  

# -----------------------------
# 2) Features on the full, continuous tape (info <= t-1)
# -----------------------------
C1 = df[price].shift(1)  # last known close at decision time t

# Stationary ratio/vol/mom features
df["RelMA10"] = C1 / C1.rolling(10).mean() - 1
df["RelMA20"] = C1 / C1.rolling(20).mean() - 1
df["RelMA50"] = C1 / C1.rolling(50).mean() - 1

df["Vol5"]  = df["Return"].shift(1).rolling(5).std()
df["Vol10"] = df["Return"].shift(1).rolling(10).std()
df["Vol20"] = df["Return"].shift(1).rolling(20).std()

df["Mom5"]  = C1 / df[price].shift(6)  - 1
df["Mom10"] = C1 / df[price].shift(11) - 1

df["Lag1"]  = df["Return"].shift(1)
df["Lag5"]  = df["Return"].shift(5)

rng_norm = (df["High"].shift(1) - df["Low"].shift(1)) / C1
df["Range5"]  = rng_norm.rolling(5).mean()
df["Range10"] = rng_norm.rolling(10).mean()

# Drop NaNs from rolling/shift (target NaNs remain only in the last row)
df = df.dropna()

feats = [
    "RelMA10","RelMA20","RelMA50",
    "Vol5","Vol10","Vol20",
    "Mom5","Mom10",
    "Lag1","Lag5",
    "Range5","Range10"
]

# -----------------------------
# 3) Chronological split first
# -----------------------------
sp = int(len(df) * 0.70)
train_df = df.iloc[:sp].copy()
test_df  = df.iloc[sp:].copy()

# Dates & basic diagnostics 
print("Train dates:", train_df.index[0].strftime("%Y-%m-%d"), "->", train_df.index[-1].strftime("%Y-%m-%d"))
print("Test  dates:", test_df.index[0].strftime("%Y-%m-%d"),  "->", test_df.index[-1].strftime("%Y-%m-%d"))

# Apply near-zero band to classification sets 
train_mask = band_mask.loc[train_df.index]
test_mask  = band_mask.loc[test_df.index]  

X_train_all = train_df.loc[train_mask, feats]
y_train     = train_df.loc[train_mask, "Label"]

# Test: keep two views
X_test_all_metrics = test_df.loc[test_mask, feats]  
y_test_metrics     = test_df.loc[test_mask, "Label"]
X_test_all_full    = test_df[feats]                
y_test_full        = test_df["Label"]               


assert (X_train_all.index.equals(y_train.index))
assert set(X_test_all_metrics.index) <= set(test_df.index)     
assert set(test_df.index) == set(test_df["FwdRet"].dropna().index)  

# Class balance 
print("Class balance (train):", float(y_train.mean()))
print("Class balance (test, metrics subset):", float(y_test_metrics.mean()) if len(y_test_metrics) else np.nan)

# -----------------------------
# 4) Feature Selection on train only: Filter -> Wrapper -> Embedded
# -----------------------------
# Build correlation frame aligned to (masked) training rows
corr_frame = X_train_all.copy()
corr_frame["Label"] = y_train.values

corr_train = corr_frame.corr(numeric_only=True)
rho_to_lbl = corr_train["Label"].drop("Label").sort_values(ascending=False)

# Filter (relevance)
keep = rho_to_lbl[rho_to_lbl.abs() > 0.02].index.tolist()
if len(keep) == 0:
    # Fallback: take top 6 by absolute correlation to ensure progress
    keep = list(rho_to_lbl.reindex(rho_to_lbl.abs().sort_values(ascending=False).index).index[:6])

# Filter (redundancy)
cf = corr_train.loc[keep, keep]
drop = set()
for i in range(len(cf.columns)):
    for j in range(i):
        if abs(cf.iloc[i, j]) > 0.90:
            drop.add(cf.columns[i])
filtered = [f for f in keep if f not in drop]
if len(filtered) == 0:
    filtered = keep[:]  # fallback if all pruned

X_train_fs = X_train_all[filtered]

# Wrapper (TimeSeriesSplit ROC-AUC with Logistic Regression)
tscv = TimeSeriesSplit(n_splits=3)

def ts_cv_auc_lr(cols):
    scores = []
    for tr, va in tscv.split(X_train_fs):
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train_fs.iloc[tr][cols], y_train.iloc[tr])
        proba = lr.predict_proba(X_train_fs.iloc[va][cols])[:, 1]
        scores.append(roc_auc_score(y_train.iloc[va], proba))
    return float(np.mean(scores)) if len(scores) else np.nan

wrapper_scores = {"all": ts_cv_auc_lr(filtered)}
for c in filtered:
    subset = [f for f in filtered if f != c]
    wrapper_scores[f"drop_{c}"] = ts_cv_auc_lr(subset)

best_key = max(wrapper_scores, key=wrapper_scores.get)
wrapper = filtered if best_key == "all" else [f for f in filtered if f != best_key.replace("drop_", "")]

# Embedded (GB importances on train only)
gb = GradientBoostingClassifier(random_state=RANDOM_STATE)
gb.fit(X_train_all[wrapper], y_train)
imp = pd.Series(gb.feature_importances_, index=wrapper).sort_values(ascending=False)

top_k = min(5, len(imp))
final_feats = list(imp.index[:top_k]) if top_k > 0 else wrapper  # fallback if needed

X_train = X_train_all[final_feats].copy()
X_test_metrics = X_test_all_metrics[final_feats].copy()
X_test_full    = X_test_all_full[final_feats].copy()

# -----------------------------
# 5) SVM Pipeline + TimeSeriesSplit CV (ROC-AUC)
# -----------------------------
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(probability=True, random_state=RANDOM_STATE))
])

param_grid = {
    "svc__kernel": ["rbf", "linear"],
    "svc__C": [0.1, 1, 10],
    "svc__gamma": ["scale", 0.01, 0.001],
    "svc__class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=3),
    scoring="roc_auc",
    n_jobs=-1,
    refit=True
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Best params:", grid.best_params_)


# -----------------------------
# 6) Probability threshold from train OOF (F1-opt)
# -----------------------------
oof_prob = pd.Series(index=X_train.index, dtype=float)
for tr, va in TimeSeriesSplit(n_splits=3).split(X_train):
    est = clone(best)
    est.fit(X_train.iloc[tr], y_train.iloc[tr])
    oof_prob.iloc[va] = est.predict_proba(X_train.iloc[va])[:, 1]

candidates = np.linspace(0.00, 1.00, 101)
f1s = [f1_score(y_train, (oof_prob > t).astype(int)) for t in candidates]
thr = float(candidates[int(np.argmax(f1s))])
print("Chosen prob threshold (train OOF, F1-opt):", round(thr, 3))


# -----------------------------
# 7) Test-set evaluation (classification metrics on masked subset)
# -----------------------------
if len(X_test_metrics) > 0:
    y_prob_metrics = best.predict_proba(X_test_metrics)[:, 1]
    y_pred_metrics = (y_prob_metrics > thr).astype(int)
    auc = roc_auc_score(y_test_metrics, y_prob_metrics)
    print(f"Test ROC-AUC (EPS-masked): {auc:.4f}")
    fpr, tpr, _ = roc_curve(y_test_metrics, y_prob_metrics)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0,1], [0,1], '--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - SVM (Test, EPS-masked)")
    plt.legend(); plt.tight_layout(); plt.show()

    cm = confusion_matrix(y_test_metrics, y_pred_metrics)
    print("Confusion matrix (test, masked):\n", cm)
    print("Classification report (test, masked):\n", classification_report(y_test_metrics, y_pred_metrics, digits=4))
else:
    print("NOTE: Test metrics subset is empty after EPS mask; skipping classifier metrics.")

# -----------------------------
# 8) Backtest — same-day MOC (no signal shift) on full test tape
#     Decision at t uses info <= t-1, submit MOC at t, realize FwdRet (t->t+1).
#     Costs: 10 bps per position change.
# -----------------------------
y_prob_full = best.predict_proba(X_test_full)[:, 1]
signals = pd.Series((y_prob_full > thr).astype(int), index=X_test_full.index).sort_index()
fwd     = test_df["FwdRet"].reindex(signals.index)

TC_BPS = 10  # per flip (enter/exit each count)
flips = (signals != signals.shift(1)).astype(int).fillna(0)
tc = (TC_BPS / 10000.0)
strat_ret = signals * fwd - flips * tc

cum_bh    = (1 + fwd).cumprod().rename("Buy & Hold (FwdRet)")
cum_strat = (1 + strat_ret).cumprod().rename("SVM Strategy (MOC, 10bps)")

combo = pd.concat([cum_bh, cum_strat], axis=1).dropna()
ax = combo["Buy & Hold (FwdRet)"].plot(lw=2, linestyle="--", label="Buy & Hold (FwdRet)", zorder=3)
combo["SVM Strategy (MOC, 10bps)"].plot(ax=ax, lw=2, label="SVM Strategy (MOC, 10bps)", zorder=4)
plt.title("Cumulative Returns - Out-of-Sample (Full Tape, MOC, 10 bps)")
plt.ylabel("Growth of $1"); plt.legend(); plt.tight_layout(); plt.show()

print(f"Buy & Hold Total Return: {combo['Buy & Hold (FwdRet)'].iloc[-1]-1:.2%}")
print(f"SVM Strategy Total Ret.: {combo['SVM Strategy (MOC, 10bps)'].iloc[-1]-1:.2%}")

# -----------------------------
# 9) Report artifacts
# -----------------------------
print("\n[Section B] Correlation to Label (TRAIN):\n", rho_to_lbl)
print("\n[Section B] Filter kept:", filtered)
print("\n[Section B] Wrapper ROC-AUC scores:", wrapper_scores)
print("\n[Section B] Embedded importances:\n", imp)
print("\n[Section B] Final features:", final_feats)
