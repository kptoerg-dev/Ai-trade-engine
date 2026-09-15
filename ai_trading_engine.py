"""
AI Trading Engine v12
Research-grade long-only ML backtester.

Design goals:
- No same-bar lookahead: signal at close t -> entry at next available open.
- Purged walk-forward validation with fold-local feature selection.
- Robust per-asset panel handling (no cross-asset label shifts).
- Triple-barrier-inspired labels using future path, not only terminal close.
- Probability calibration from a validation split inside each training fold.
- Cross-sectional ranking of OOS opportunities.
- Portfolio/risk controls, ATR exits, costs and slippage.
- Full daily equity curve and diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")


@dataclass
class Config:
    tickers: List[str]
    benchmark: str = "SPY"
    start: str = "2015-01-01"
    end: str = "2026-01-01"
    initial_capital: float = 10000.0

    train_days: int = 756
    test_days: int = 63
    step_days: int = 63
    purge_days: int = 5
    embargo_days: int = 5
    horizon_days: int = 10

    min_probability: float = 0.56
    min_expected_return: float = 0.005
    max_positions: int = 5
    max_single_weight: float = 0.30
    max_total_exposure: float = 0.95

    risk_fraction_min: float = 0.003
    risk_fraction_max: float = 0.025
    kelly_fraction: float = 0.20
    drawdown_risk_floor: float = 0.35

    atr_stop: float = 2.0
    atr_target: float = 4.0
    max_hold_bars: int = 15

    commission_bps: float = 5.0
    slippage_bps: float = 3.0

    min_train_rows: int = 250
    min_test_rows: int = 5
    random_state: int = 42
    max_iter: int = 500


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance can return (field, ticker) or (ticker, field).
        level0 = [str(x) for x in df.columns.get_level_values(0)]
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        if any(x in fields for x in level0):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        else:
            df = df.copy()
            df.columns = df.columns.get_level_values(-1)
    return df


def download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    raw = _flatten_yf_columns(raw)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing}")
    out = raw[needed].copy()
    out.columns = [c.lower() for c in out.columns]
    out = out.dropna(subset=["open", "high", "low", "close"]).copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    out["ticker"] = ticker
    out["date"] = out.index
    return out.reset_index(drop=True)


def calculate_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def make_features(df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    x = df.sort_values("date").copy()
    c, h, l, o, v = x["close"], x["high"], x["low"], x["open"], x["volume"]

    x["ret_1"] = c.pct_change()
    x["ret_3"] = c.pct_change(3)
    x["ret_5"] = c.pct_change(5)
    x["ret_10"] = c.pct_change(10)
    x["ret_20"] = c.pct_change(20)
    x["ret_60"] = c.pct_change(60)

    x["vol_10"] = x["ret_1"].rolling(10).std()
    x["vol_20"] = x["ret_1"].rolling(20).std()
    x["vol_60"] = x["ret_1"].rolling(60).std()

    x["atr_14"] = calculate_atr(x, 14)
    x["atr_pct"] = x["atr_14"] / c.replace(0, np.nan)

    ma10, ma20, ma50, ma100, ma200 = [c.rolling(n).mean() for n in [10,20,50,100,200]]
    x["dist_ma10"] = c / ma10 - 1
    x["dist_ma20"] = c / ma20 - 1
    x["dist_ma50"] = c / ma50 - 1
    x["dist_ma100"] = c / ma100 - 1
    x["dist_ma200"] = c / ma200 - 1

    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi_14"] = 100 - 100 / (1 + rs)

    x["range_pct"] = (h - l) / c.replace(0, np.nan)
    x["body_pct"] = (c - o) / o.replace(0, np.nan)
    x["upper_wick"] = (h - np.maximum(o, c)) / c.replace(0, np.nan)
    x["lower_wick"] = (np.minimum(o, c) - l) / c.replace(0, np.nan)

    logv = np.log1p(v.clip(lower=0))
    x["volume_z20"] = (logv - logv.rolling(20).mean()) / logv.rolling(20).std()
    x["volume_change"] = v.pct_change().replace([np.inf, -np.inf], np.nan)

    x["momentum_accel"] = x["ret_5"] - x["ret_20"] / 4
    x["trend_strength"] = (ma20 - ma100) / ma100
    x["breakout_20"] = c / h.shift(1).rolling(20).max() - 1
    x["drawdown_60"] = c / c.rolling(60).max() - 1

    dow = x["date"].dt.dayofweek
    month = x["date"].dt.month
    x["dow_sin"] = np.sin(2 * np.pi * dow / 5)
    x["dow_cos"] = np.cos(2 * np.pi * dow / 5)
    x["month_sin"] = np.sin(2 * np.pi * month / 12)
    x["month_cos"] = np.cos(2 * np.pi * month / 12)

    if benchmark is not None:
        b = benchmark[["date", "close"]].rename(columns={"close": "bench_close"}).copy()
        b = b.sort_values("date")
        x = pd.merge_asof(
            x.sort_values("date"),
            b.sort_values("date"),
            on="date",
            direction="backward",
        )
        x["bench_ret_5"] = x["bench_close"].pct_change(5)
        x["bench_ret_20"] = x["bench_close"].pct_change(20)
        asset_ret = x["close"].pct_change()
        bench_ret = x["bench_close"].pct_change()
        x["rel_strength_20"] = x["ret_20"] - x["bench_ret_20"]
        x["beta_60"] = asset_ret.rolling(60).cov(bench_ret) / bench_ret.rolling(60).var()
        x["corr_60"] = asset_ret.rolling(60).corr(bench_ret)

    return x.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int, atr_stop: float, atr_target: float) -> pd.DataFrame:
    """
    Path-aware long label:
    Entry reference = next bar open.
    Over the next horizon bars, target/stop are checked in chronological order.
    If neither barrier is hit, terminal close is used.
    Label = 1 when realized barrier/terminal return is positive.
    label_exit is the last future bar touched/used and is used for purging.
    """
    x = df.sort_values(["ticker", "date"]).copy()
    parts = []
    for ticker, g in x.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True).copy()
        opens = g["open"].to_numpy(float)
        highs = g["high"].to_numpy(float)
        lows = g["low"].to_numpy(float)
        closes = g["close"].to_numpy(float)
        atr = g["atr_14"].to_numpy(float)
        dates = g["date"].to_numpy()

        label = np.full(len(g), np.nan)
        target_ret = np.full(len(g), np.nan)
        exit_dates = np.full(len(g), np.datetime64("NaT"), dtype="datetime64[ns]")

        for i in range(len(g) - 1):
            entry = opens[i + 1]
            if not np.isfinite(entry) or not np.isfinite(atr[i]) or entry <= 0:
                continue
            end = min(len(g) - 1, i + horizon + 1)
            stop = entry - atr_stop * atr[i]
            target = entry + atr_target * atr[i]
            result = None
            exit_i = end
            for j in range(i + 1, end + 1):
                # Conservative ordering if both barriers occur in the same bar.
                if lows[j] <= stop:
                    result = stop / entry - 1
                    exit_i = j
                    break
                if highs[j] >= target:
                    result = target / entry - 1
                    exit_i = j
                    break
            if result is None:
                result = closes[end] / entry - 1
            label[i] = 1.0 if result > 0 else 0.0
            target_ret[i] = result
            exit_dates[i] = dates[exit_i]
        g["label"] = label
        g["target_return"] = target_ret
        g["label_exit"] = pd.to_datetime(exit_dates)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "date", "ticker", "open", "high", "low", "close", "volume",
        "label", "target_return", "label_exit", "exec_date", "bench_close"
    }
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _select_features(X: pd.DataFrame, y: pd.Series, min_features: int = 8) -> List[str]:
    features = list(X.columns)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", LogisticRegression(
            penalty="l1", solver="liblinear", C=0.08,
            class_weight="balanced", max_iter=1500, random_state=42
        )),
    ])
    pipe.fit(X[features], y.astype(int))
    coef = np.abs(pipe.named_steps["selector"].coef_[0])
    if not np.isfinite(coef).any():
        return features[:min(16, len(features))]
    threshold = np.nanmedian(coef[coef > 0]) if np.any(coef > 0) else 0
    selected = [f for f, w in zip(features, coef) if w >= threshold and w > 0]
    if len(selected) < min_features:
        order = np.argsort(-coef)
        selected = [features[i] for i in order[:min(min_features, len(features))]]
    return selected


def _fit_calibrator(raw: np.ndarray, y: np.ndarray):
    raw = np.asarray(raw, float)
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2 or len(raw) < 30:
        return None
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw, y)
        return iso
    except Exception:
        return None


def train_models(train: pd.DataFrame, features: List[str], cfg: Config):
    tr = train.dropna(subset=["label", "target_return"]).copy()
    if len(tr) < cfg.min_train_rows or tr["label"].nunique() < 2:
        raise ValueError(f"Insufficient training data ({len(tr)} rows / {tr['label'].nunique()} classes).")

    # Chronological calibration split inside the training fold.
    split = max(int(len(tr) * 0.82), 1)
    base = tr.iloc[:split].copy()
    cal = tr.iloc[split:].copy()
    if len(cal) < 30:
        base, cal = tr.iloc[:-30].copy(), tr.iloc[-30:].copy()

    selected = _select_features(base[features], base["label"])
    Xb, yb = base[selected], base["label"].astype(int)

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=cfg.max_iter, learning_rate=0.045, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=cfg.random_state
        ))
    ])
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=cfg.max_iter, learning_rate=0.045, max_leaf_nodes=15,
            l2_regularization=1.0, loss="huber", random_state=cfg.random_state
        ))
    ])
    clf.fit(Xb, yb)
    reg.fit(Xb, base["target_return"].astype(float))

    calibrator = None
    if len(cal) >= 30:
        raw = clf.predict_proba(cal[selected])[:, 1]
        calibrator = _fit_calibrator(raw, cal["label"].astype(int).to_numpy())

    return {
        "clf": clf,
        "reg": reg,
        "features": selected,
        "calibrator": calibrator,
    }


def predict_models(model_bundle, data: pd.DataFrame) -> pd.DataFrame:
    x = data.copy()
    f = model_bundle["features"]
    raw = model_bundle["clf"].predict_proba(x[f])[:, 1]
    if model_bundle["calibrator"] is not None:
        prob = model_bundle["calibrator"].predict(raw)
    else:
        prob = raw
    expret = model_bundle["reg"].predict(x[f])
    vol = x["vol_20"].clip(lower=0.002).fillna(0.03).to_numpy()
    x["prob_up"] = np.clip(prob, 0.001, 0.999)
    x["raw_prob_up"] = raw
    x["expected_return"] = expret
    x["score"] = expret * (0.5 + x["prob_up"]) / vol
    return x


def _fold_starts(dates: pd.DatetimeIndex, cfg: Config) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    out = []
    pos = cfg.train_days
    while pos < len(dates):
        train_start = dates[max(0, pos - cfg.train_days)]
        test_start = dates[pos]
        test_end_pos = min(pos + cfg.test_days - 1, len(dates) - 1)
        test_end = dates[test_end_pos]
        out.append((train_start, test_start, test_end, dates[max(0, test_end_pos + 1)] if test_end_pos + 1 < len(dates) else test_end))
        pos += cfg.step_days
    return out


def walk_forward(panel: pd.DataFrame, cfg: Config):
    panel = panel.sort_values(["date", "ticker"]).copy()
    all_dates = pd.DatetimeIndex(sorted(panel["date"].dropna().unique()))
    folds = _fold_starts(all_dates, cfg)
    predictions, diagnostics = [], []

    for fold_id, (train_start, test_start, test_end, next_date) in enumerate(folds, 1):
        # Training observations must be fully resolved before the OOS test begins.
        tr = panel[(panel["date"] >= train_start) & (panel["date"] < test_start)].copy()
        tr = tr[tr["label_exit"] < test_start]

        # Embargo protects the first observations after the OOS block when a later
        # fold reuses the same rolling history.
        test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
        if fold_id < len(folds):
            embargo_end = test_end + pd.Timedelta(days=cfg.embargo_days)
            # This removes any rows in the training candidate that would otherwise
            # fall inside the embargo; with strict pre-test windows it is usually a no-op,
            # but it is explicit and auditable.
            tr = tr[~((tr["date"] > test_end) & (tr["date"] <= embargo_end))]

        diag = {
            "fold": fold_id,
            "train_start": train_start,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": len(tr),
            "test_rows": len(test),
            "status": "skipped",
            "reason": "",
            "selected_features": 0,
            "auc": np.nan,
        }

        if len(tr) < cfg.min_train_rows:
            diag["reason"] = f"training rows {len(tr)} < min_train_rows {cfg.min_train_rows}"
            diagnostics.append(diag)
            continue
        if len(test) < cfg.min_test_rows:
            diag["reason"] = f"test rows {len(test)} < min_test_rows {cfg.min_test_rows}"
            diagnostics.append(diag)
            continue

        try:
            features = get_feature_columns(tr)
            bundle = train_models(tr, features, cfg)
            diag["selected_features"] = len(bundle["features"])
            pred = predict_models(bundle, test)
            pred["fold"] = fold_id

            # Signal generated at t; execution happens at next available bar of same ticker.
            pred["exec_date"] = pred.groupby("ticker")["date"].shift(-1)
            pred = pred.dropna(subset=["exec_date"]).copy()

            if len(pred):
                diagnostics.append({**diag, "status": "ok", "reason": ""})
                y = pred["label"].dropna()
                if y.nunique() == 2:
                    try:
                        diag_auc = roc_auc_score(y, pred.loc[y.index, "prob_up"])
                        diagnostics[-1]["auc"] = diag_auc
                    except Exception:
                        pass
                predictions.append(pred)
            else:
                diag["reason"] = "no rows with a next available execution bar"
                diagnostics.append(diag)
        except Exception as e:
            diag["reason"] = f"{type(e).__name__}: {e}"
            diagnostics.append(diag)

    if not predictions:
        d = pd.DataFrame(diagnostics)
        reason = "; ".join(d["reason"].dropna().astype(str).head(5).tolist())
        raise RuntimeError(
            "No walk-forward predictions were produced. "
            f"Fold diagnostics: {reason or 'no valid folds'}"
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(diagnostics)


def _cost_rate(cfg: Config) -> float:
    return (cfg.commission_bps + cfg.slippage_bps) / 10000.0


def _kelly_risk(recent_r: List[float], edge_hint: float, cfg: Config) -> float:
    if len(recent_r) < 10:
        base = 0.012
    else:
        wins = [r for r in recent_r if r > 0]
        losses = [-r for r in recent_r if r < 0]
        if not wins or not losses:
            base = 0.012
        else:
            p = len(wins) / len(recent_r)
            b = np.mean(wins) / max(np.mean(losses), 1e-9)
            k = p - (1 - p) / max(b, 1e-9)
            base = max(0.0, k) * cfg.kelly_fraction
    base *= np.clip(1 + edge_hint * 5, 0.5, 1.5)
    return float(np.clip(base, cfg.risk_fraction_min, cfg.risk_fraction_max))


def backtest(predictions: pd.DataFrame, panel: pd.DataFrame, cfg: Config):
    pred = predictions.copy()
    pred["exec_date"] = pd.to_datetime(pred["exec_date"])
    bars = panel.sort_values(["date", "ticker"]).copy()

    # Full market-date equity curve.
    market_dates = pd.DatetimeIndex(sorted(bars["date"].unique()))
    cash = float(cfg.initial_capital)
    positions: Dict[str, Dict] = {}
    trades = []
    equity_rows = []
    recent_r: List[float] = []

    grouped = {t: g.sort_values("date").reset_index(drop=True) for t, g in bars.groupby("ticker")}
    pred_by_date = {d: g for d, g in pred.groupby("exec_date")}

    for day in market_dates:
        # Mark-to-market using latest available close for each held asset.
        for ticker, pos in list(positions.items()):
            g = grouped[ticker]
            row = g[g["date"] == day]
            if row.empty:
                continue
            r = row.iloc[0]
            pos["last_price"] = float(r["close"])

        equity = cash + sum(
            p["shares"] * p["last_price"] for p in positions.values()
        )

        # Manage existing positions using today's bar.
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            g = grouped[ticker]
            row = g[g["date"] == day]
            if row.empty:
                continue
            r = row.iloc[0]
            high, low, close = float(r["high"]), float(r["low"]), float(r["close"])
            exit_reason = None
            exit_price = None

            if low <= pos["stop"]:
                exit_reason, exit_price = "stop", pos["stop"]
            elif high >= pos["target"]:
                exit_reason, exit_price = "target", pos["target"]
            elif pos["bars_held"] >= cfg.max_hold_bars:
                exit_reason, exit_price = "time", close

            pos["bars_held"] += 1
            if exit_reason:
                gross = (exit_price - pos["entry_price"]) * pos["shares"]
                cost = (pos["entry_price"] + exit_price) * pos["shares"] * _cost_rate(cfg)
                pnl = gross - cost
                cash += pos["shares"] * exit_price - exit_price * pos["shares"] * _cost_rate(cfg)
                r_trade = pnl / max(pos["entry_value"], 1e-9)
                recent_r.append(float(r_trade))
                recent_r = recent_r[-50:]
                trades.append({
                    "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "return": r_trade,
                    "reason": exit_reason, "fold": pos["fold"],
                })
                del positions[ticker]

        # Recompute equity after exits before new entries.
        equity = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
        dd = 1 - equity / max(max([x["equity"] for x in equity_rows], default=equity), equity)
        dd_factor = max(cfg.drawdown_risk_floor, 1 - dd * 1.8)

        candidates = pred_by_date.get(day)
        if candidates is not None and len(positions) < cfg.max_positions:
            c = candidates[
                (candidates["prob_up"] >= cfg.min_probability) &
                (candidates["expected_return"] >= cfg.min_expected_return)
            ].copy()
            c = c.sort_values("score", ascending=False).drop_duplicates("ticker")
            for _, s in c.iterrows():
                if s["ticker"] in positions or len(positions) >= cfg.max_positions:
                    continue
                g = grouped.get(s["ticker"])
                if g is None:
                    continue
                row = g[g["date"] == day]
                if row.empty:
                    continue
                r = row.iloc[0]
                entry_price = float(r["open"])
                atr = float(r["atr_14"]) if np.isfinite(r["atr_14"]) else entry_price * 0.02
                stop_distance = max(cfg.atr_stop * atr, entry_price * 0.005)
                stop = entry_price - stop_distance
                target = entry_price + cfg.atr_target * atr
                if stop <= 0:
                    continue

                risk_frac = _kelly_risk(recent_r, float(s["expected_return"]), cfg) * dd_factor
                risk_cash = equity * risk_frac
                shares_risk = risk_cash / stop_distance
                max_value = equity * cfg.max_single_weight
                shares_cap = max_value / entry_price
                remaining = max(0.0, equity * cfg.max_total_exposure -
                                sum(p["shares"] * p["last_price"] for p in positions.values()))
                shares_exposure = remaining / entry_price
                shares = math.floor(max(0.0, min(shares_risk, shares_cap, shares_exposure)))
                if shares < 1:
                    continue

                entry_cost = entry_price * shares * _cost_rate(cfg)
                total = entry_price * shares + entry_cost
                if total > cash:
                    shares = math.floor(cash / (entry_price * (1 + _cost_rate(cfg))))
                    total = entry_price * shares * (1 + _cost_rate(cfg))
                if shares < 1:
                    continue

                cash -= total
                positions[s["ticker"]] = {
                    "shares": shares,
                    "entry_price": entry_price,
                    "entry_value": entry_price * shares,
                    "entry_date": day,
                    "last_price": entry_price,
                    "stop": stop,
                    "target": target,
                    "bars_held": 0,
                    "fold": int(s["fold"]),
                }

        equity = cash + sum(p["shares"] * p["last_price"] for p in positions.values())
        equity_rows.append({"date": day, "equity": equity, "cash": cash, "positions": len(positions)})

    # Force-close at each asset's last available bar.
    for ticker, pos in list(positions.items()):
        g = grouped[ticker]
        r = g.iloc[-1]
        day = r["date"]
        exit_price = float(r["close"])
        gross = (exit_price - pos["entry_price"]) * pos["shares"]
        cost = (pos["entry_price"] + exit_price) * pos["shares"] * _cost_rate(cfg)
        pnl = gross - cost
        trades.append({
            "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": day,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "shares": pos["shares"], "pnl": pnl,
            "return": pnl / max(pos["entry_value"], 1e-9),
            "reason": "end_of_data", "fold": pos["fold"],
        })

    return pd.DataFrame(equity_rows), pd.DataFrame(trades)


def performance_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial: float) -> Dict[str, float]:
    if equity.empty:
        return {}
    e = equity.sort_values("date").copy()
    curve = e["equity"].astype(float)
    total_return = curve.iloc[-1] / initial - 1
    days = max((e["date"].iloc[-1] - e["date"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = (curve.iloc[-1] / initial) ** (1 / years) - 1 if curve.iloc[-1] > 0 else -1
    peak = curve.cummax()
    dd = curve / peak - 1
    rets = curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = np.sqrt(252) * rets.mean() / rets.std() if len(rets) > 1 and rets.std() > 0 else np.nan

    if trades.empty:
        pf = np.nan
        win = np.nan
        avg = np.nan
    else:
        wins = trades.loc[trades["pnl"] > 0, "pnl"]
        losses = trades.loc[trades["pnl"] < 0, "pnl"]
        pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
        win = (trades["pnl"] > 0).mean()
        avg = trades["return"].mean()

    return {
        "End Capital": float(curve.iloc[-1]),
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Max Drawdown": float(dd.min()),
        "Sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "Profit Factor": float(pf) if np.isfinite(pf) else pf,
        "Win Rate": float(win) if np.isfinite(win) else np.nan,
        "Trades": int(len(trades)),
        "Avg Trade Return": float(avg) if np.isfinite(avg) else np.nan,
    }


def build_benchmarks(panel: pd.DataFrame, benchmark_df: pd.DataFrame, initial: float):
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    curves = pd.DataFrame(index=dates)

    # Universe equal-weight buy-and-hold approximation using normalized closes.
    piv = panel.pivot_table(index="date", columns="ticker", values="close").reindex(dates).ffill()
    norm = piv / piv.iloc[0]
    curves["Equal Weight"] = norm.mean(axis=1) * initial

    b = benchmark_df.set_index("date")["close"].reindex(dates).ffill()
    curves["Benchmark"] = b / b.iloc[0] * initial
    return curves.reset_index(names="date")


def run_engine(cfg: Config):
    if not cfg.tickers:
        raise ValueError("At least one ticker is required.")

    bench = download_ohlcv(cfg.benchmark, cfg.start, cfg.end)
    assets = []
    errors = []
    for ticker in cfg.tickers:
        try:
            d = download_ohlcv(ticker, cfg.start, cfg.end)
            if len(d) >= 250:
                f = make_features(d, bench)
                assets.append(f)
            else:
                errors.append(f"{ticker}: only {len(d)} bars")
        except Exception as e:
            errors.append(f"{ticker}: {type(e).__name__}: {e}")

    if not assets:
        raise RuntimeError("No asset data could be loaded. " + " | ".join(errors))

    panel = make_labels(pd.concat(assets, ignore_index=True), cfg.horizon_days, cfg.atr_stop, cfg.atr_target)
    predictions, diagnostics = walk_forward(panel, cfg)
    equity, trades = backtest(predictions, panel, cfg)
    metrics = performance_metrics(equity, trades, cfg.initial_capital)
    benchmarks = build_benchmarks(panel, bench, cfg.initial_capital)
    return {
        "config": asdict(cfg),
        "panel": panel,
        "predictions": predictions,
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "diagnostics": diagnostics,
        "errors": errors,
    }
