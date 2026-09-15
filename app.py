import streamlit as st
import pandas as pd
import numpy as np

from ai_trading_engine import Config, run_engine

st.set_page_config(page_title="AI Trading Engine v12", page_icon="📈", layout="wide")

st.title("📈 AI Trading Engine v12")
st.caption("Purged Walk-Forward • Fold-local Feature Selection • Calibrated Probability • Expected Return • Portfolio Risk")

with st.expander("⚙️ Strategy & Backtest Settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        tickers_text = st.text_input("Assets", "SPY,QQQ,IWM,GLD,TLT")
        benchmark = st.text_input("Benchmark", "SPY")
        start = st.date_input("Start", pd.Timestamp("2015-01-01"))
        end = st.date_input("End", pd.Timestamp("2026-01-01"))
        capital = st.number_input("Initial capital", 1000.0, 10_000_000.0, 10000.0, 500.0)
    with c2:
        train_days = st.number_input("Training window (calendar observations)", 252, 3000, 756, 21)
        test_days = st.number_input("OOS test window", 21, 504, 63, 21)
        step_days = st.number_input("Walk-forward step", 21, 504, 63, 21)
        purge_days = st.number_input("Purge days", 0, 60, 5)
        embargo_days = st.number_input("Embargo days", 0, 60, 5)
        horizon = st.number_input("Label horizon", 2, 60, 10)
    with c3:
        min_prob = st.slider("Minimum probability", 0.50, 0.80, 0.56, 0.01)
        min_er = st.number_input("Minimum expected return", -0.10, 0.50, 0.005, 0.001, format="%.3f")
        max_pos = st.number_input("Max positions", 1, 20, 5)
        max_weight = st.slider("Max single weight", 0.05, 1.0, 0.30, 0.05)
        max_exposure = st.slider("Max total exposure", 0.10, 1.0, 0.95, 0.05)

with st.expander("🛡️ Risk, Exits & Costs", expanded=False):
    r1, r2, r3 = st.columns(3)
    with r1:
        risk_min = st.number_input("Min risk / trade", 0.001, 0.05, 0.003, 0.001, format="%.3f")
        risk_max = st.number_input("Max risk / trade", 0.005, 0.15, 0.025, 0.001, format="%.3f")
        kelly = st.slider("Fractional Kelly", 0.05, 0.75, 0.20, 0.05)
    with r2:
        atr_stop = st.number_input("ATR stop", 0.5, 6.0, 2.0, 0.25)
        atr_target = st.number_input("ATR target", 1.0, 10.0, 4.0, 0.25)
        max_hold = st.number_input("Max hold bars", 2, 100, 15)
    with r3:
        commission = st.number_input("Commission (bps)", 0.0, 100.0, 5.0, 0.5)
        slippage = st.number_input("Slippage (bps)", 0.0, 200.0, 3.0, 0.5)
        seed = st.number_input("Random seed", 1, 999999, 42)

tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

st.info("Das Signal entsteht am Schlusskurs t. Die Position wird frühestens am nächsten verfügbaren Open desselben Assets eröffnet. Der Backtest ist Research-Zweck, keine Anlageberatung.")

run = st.button("🚀 Run Research Backtest", type="primary", use_container_width=True)

if run:
    cfg = Config(
        tickers=tickers, benchmark=benchmark.upper(), start=str(start), end=str(end),
        initial_capital=capital, train_days=int(train_days), test_days=int(test_days),
        step_days=int(step_days), purge_days=int(purge_days), embargo_days=int(embargo_days),
        horizon_days=int(horizon), min_probability=min_prob, min_expected_return=min_er,
        max_positions=int(max_pos), max_single_weight=max_weight, max_total_exposure=max_exposure,
        risk_fraction_min=risk_min, risk_fraction_max=risk_max, kelly_fraction=kelly,
        atr_stop=atr_stop, atr_target=atr_target, max_hold_bars=int(max_hold),
        commission_bps=commission, slippage_bps=slippage, random_state=int(seed)
    )
    with st.spinner("Daten laden, Features bauen und purged Walk-Forward berechnen …"):
        try:
            result = run_engine(cfg)
            st.session_state["result"] = result
        except Exception as e:
            st.error(f"Backtest fehlgeschlagen: {e}")
            st.stop()

result = st.session_state.get("result")

if result:
    m = result["metrics"]
    cols = st.columns(5)
    cols[0].metric("End Capital", f"{m.get('End Capital', np.nan):,.2f}")
    cols[1].metric("Total Return", f"{m.get('Total Return', np.nan)*100:.2f}%")
    cols[2].metric("CAGR", f"{m.get('CAGR', np.nan)*100:.2f}%")
    cols[3].metric("Max Drawdown", f"{m.get('Max Drawdown', np.nan)*100:.2f}%")
    cols[4].metric("Sharpe", f"{m.get('Sharpe', np.nan):.2f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Equity", "💹 Trades", "🧪 Walk-Forward", "🎯 Signals", "📊 Data"])

    with tab1:
        eq = result["equity"].copy()
        bm = result["benchmarks"].copy()
        chart = eq.merge(bm, on="date", how="left").set_index("date")
        st.line_chart(chart[["equity", "Equal Weight", "Benchmark"]])
        st.dataframe(chart.tail(100), use_container_width=True)

    with tab2:
        trades = result["trades"].copy()
        if trades.empty:
            st.warning("Keine Trades unter den aktuellen Filtern.")
        else:
            st.dataframe(trades.sort_values("exit_date", ascending=False), use_container_width=True)
            st.download_button("⬇️ Trades CSV", trades.to_csv(index=False), "trades_v12.csv", "text/csv")

    with tab3:
        diag = result["diagnostics"].copy()
        st.dataframe(diag, use_container_width=True)
        st.metric("Successful folds", int((diag["status"] == "ok").sum()))
        if result["errors"]:
            st.warning("Datenhinweise: " + " | ".join(result["errors"]))

    with tab4:
        p = result["predictions"].copy()
        show = ["date","exec_date","ticker","fold","prob_up","expected_return","score"]
        st.dataframe(p[show].sort_values(["exec_date","score"], ascending=[False,False]).head(500), use_container_width=True)
        st.download_button("⬇️ Predictions CSV", p.to_csv(index=False), "predictions_v12.csv", "text/csv")

    with tab5:
        panel = result["panel"]
        st.write(f"Panel rows: {len(panel):,} • Assets: {panel['ticker'].nunique()}")
        st.dataframe(panel.tail(200), use_container_width=True)
        st.download_button("⬇️ Panel CSV", panel.to_csv(index=False), "panel_v12.csv", "text/csv")

else:
    st.markdown("### Bereit")
    st.write("Parameter oben einstellen und **Run Research Backtest** drücken.")
    st.markdown("""
**v12-Schwerpunkte**
- Purged Walk-Forward mit sichtbarer Fold-Diagnostik
- Fold-lokale Feature Selection
- Path-aware Barrier-Labels
- Kalibrierte Wahrscheinlichkeiten
- Expected Return + Cross-Sectional Ranking
- ATR Stop/Target
- Fractional Kelly + Drawdown-Adjustment
- Positions-, Exposure- und Kostenlimits
- Vollständige Equity-Kurve
""")
