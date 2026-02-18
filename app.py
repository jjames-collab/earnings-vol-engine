import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import time

st.set_page_config(
    page_title="Earnings Volatility Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# PROFESSIONAL STYLING
# -------------------------

st.markdown("""
    <style>
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #111827;
        color: white;
        margin-bottom: 10px;
    }
    .big-font {
        font-size:18px !important;
        font-weight:600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Earnings Volatility Scanner")

# -------------------------
# SIDEBAR CONTROLS
# -------------------------

st.sidebar.header("Scan Settings")

min_underpricing = st.sidebar.slider(
    "Minimum Underpricing Ratio",
    0.5, 3.0, 1.0, 0.1
)

min_prob = st.sidebar.slider(
    "Minimum Squeeze / Cascade Probability",
    0.4, 0.9, 0.5, 0.05
)

max_symbols = st.sidebar.slider(
    "Max Symbols To Scan",
    20, 500, 200, 20
)

# -------------------------
# GET S&P 500 LIST
# -------------------------

@st.cache_data
def get_sp500():
    df = pd.read_csv("sp500.csv")
    return df["Symbol"].tolist()


UNIVERSE = get_sp500()[:max_symbols]
TODAY = datetime.today().date()

# -------------------------
# CORE FUNCTIONS
# -------------------------

def reports_today(ticker):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal is None or cal.empty:
            return False
        
        earnings_date = cal.loc["Earnings Date"][0].date()
        return earnings_date == TODAY
    except:
        return False

def get_options_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        spot = stock.history(period="1d")["Close"].iloc[-1]
        expirations = stock.options
        if not expirations:
            return None
        
        expiry = expirations[0]
        chain = stock.option_chain(expiry)
        return spot, expiry, chain.calls, chain.puts
    except:
        return None

def implied_move(spot, calls, puts):
    calls["dist"] = abs(calls["strike"] - spot)
    puts["dist"] = abs(puts["strike"] - spot)
    atm_call = calls.loc[calls["dist"].idxmin()]
    atm_put = puts.loc[puts["dist"].idxmin()]
    return (atm_call["lastPrice"] + atm_put["lastPrice"]) / spot

def hist_move(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")
        df["ret"] = df["Close"].pct_change()
        return df["ret"].abs().mean()
    except:
        return np.nan

def skew_proxy(calls, puts):
    return puts["impliedVolatility"].mean() - calls["impliedVolatility"].mean()

def oi_imbalance(calls, puts):
    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()
    return (call_oi - put_oi) / (call_oi + put_oi + 1)

def score_model(implied_move, hist_move, skew, oi):
    underpricing = hist_move / (implied_move + 1e-6)
    tilt = 0.3*np.tanh(skew) + 0.2*np.tanh(oi)
    squeeze = 0.5 + tilt
    cascade = 0.5 - tilt
    return underpricing, squeeze, cascade

# -------------------------
# SCAN ENGINE
# -------------------------

if st.button("Run Earnings Scan"):
    results = []

    with st.spinner("Scanning earnings universe..."):
        for ticker in UNIVERSE:
            if reports_today(ticker):
                data = get_options_data(ticker)
                if data:
                    spot, expiry, calls, puts = data
                    im = implied_move(spot, calls, puts)
                    hm = hist_move(ticker)
                    skew = skew_proxy(calls, puts)
                    oi = oi_imbalance(calls, puts)
                    
                    underpricing, squeeze, cascade = score_model(im, hm, skew, oi)
                    
                    if underpricing >= min_underpricing and (squeeze >= min_prob or cascade >= min_prob):
                        results.append({
                            "Ticker": ticker,
                            "Spot": round(spot,2),
                            "Implied Move %": round(im*100,2),
                            "Hist Avg Move %": round(hm*100,2),
                            "Underpricing": round(underpricing,2),
                            "Squeeze Prob": round(squeeze,2),
                            "Cascade Prob": round(cascade,2)
                        })
            time.sleep(0.3)

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("Underpricing", ascending=False)

        st.subheader("Ranked Earnings Opportunities")

        st.dataframe(
            df.style.background_gradient(
                subset=["Underpricing","Squeeze Prob","Cascade Prob"]
            ),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            "earnings_rankings.csv",
            "text/csv"
        )

    else:
        st.warning("No qualifying earnings opportunities found.")
