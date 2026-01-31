import yfinance as yf
import pandas as pd
import numpy as np
import os
import nltk
from flask import Flask, jsonify, request
from flask_cors import CORS
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ================== NLP ==================
try:
    sia = SentimentIntensityAnalyzer()
except:
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# ================== APP ==================
app = Flask(__name__)
CORS(app)

# ================== STOCK UNIVERSE ==================
STOCKS = [
    'RELIANCE.NS','TCS.NS','HDFCBANK.NS','ICICIBANK.NS','SBIN.NS',
    'INFY.NS','BHARTIARTL.NS','ITC.NS','HINDUNILVR.NS','AXISBANK.NS',
    'KOTAKBANK.NS','LT.NS','BAJFINANCE.NS','ASIANPAINT.NS','MARUTI.NS',
    'SUNPHARMA.NS','TITAN.NS','ULTRACEMCO.NS','NTPC.NS','POWERGRID.NS',
    'COALINDIA.NS','ONGC.NS','TECHM.NS','HCLTECH.NS','WIPRO.NS',
    'JSWSTEEL.NS','TATASTEEL.NS','GRASIM.NS','BPCL.NS','DIVISLAB.NS',
    'DRREDDY.NS','CIPLA.NS','BRITANNIA.NS','EICHERMOT.NS','HEROMOTOCO.NS',
    'SBILIFE.NS','HDFCLIFE.NS','APOLLOHOSP.NS','TATACONSUM.NS','ADANIPORTS.NS',

    # ---- Banking & Finance ----
    'BANKBARODA.NS','PNB.NS','CANBK.NS','IDFCFIRSTB.NS','FEDERALBNK.NS',
    'INDUSINDBK.NS','AUBANK.NS','CHOLAFIN.NS','MUTHOOTFIN.NS','RECLTD.NS',
    'PFC.NS','IRFC.NS','HUDCO.NS',

    # ---- Infra / Energy / Metals ----
    'ADANIENT.NS','ADANIPOWER.NS','JSWENERGY.NS','TATAPOWER.NS','NHPC.NS',
    'IOC.NS','GAIL.NS','HINDALCO.NS','NMDC.NS','SAIL.NS',

    # ---- FMCG / Consumer ----
    'DABUR.NS','GODREJCP.NS','COLPAL.NS','MARICO.NS','UBL.NS',
    'MCDOWELL-N.NS','VBL.NS',

    # ---- Pharma / Chemicals ----
    'LUPIN.NS','AUROPHARMA.NS','BIOCON.NS','ALKYLAMINE.NS','PIDILITIND.NS',

    # ---- Others / Growth ----
    'TRENT.NS','DLF.NS','INDIGO.NS','BALKRISIND.NS','EXIDEIND.NS'
]


# ================== INDICATORS ==================
def add_indicators(df):
    df['EMA_9'] = df['Close'].ewm(span=9).mean()
    df['EMA_21'] = df['Close'].ewm(span=21).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    df['EMA_200'] = df['Close'].ewm(span=200).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    return df

# ================== LOGICS ==================
def intraday_logic(df):
    last = df.iloc[-1]
    score = 50
    reasons = []

    if last['Close'] > last['EMA_9'] > last['EMA_21']:
        score += 25
        reasons.append("EMA 9 > 21 (Momentum)")

    if 45 <= last['RSI'] <= 65:
        score += 15
        reasons.append("RSI healthy (45-65)")

    bias = "BULLISH" if score > 65 else "NEUTRAL"
    sl = round(last['Close'] - last['ATR'], 2)
    tgt = round(last['Close'] + (last['ATR'] * 2), 2)

    return score, bias, reasons, sl, tgt

def swing_logic(df):
    last = df.iloc[-1]
    score = 50
    reasons = []

    if last['EMA_21'] > last['EMA_50']:
        score += 20
        reasons.append("EMA 21 > 50")

    if last['Close'] > last['EMA_200']:
        score += 20
        reasons.append("Above 200 EMA")

    bias = "BULLISH" if score > 60 else "NEUTRAL"
    sl = round(last['Close'] * 0.95, 2)
    tgt = round(last['Close'] * 1.1, 2)

    return score, bias, reasons, sl, tgt

def longterm_logic(df):
    last = df.iloc[-1]
    score = 50
    reasons = []

    if last['Close'] > last['EMA_200']:
        score += 30
        reasons.append("Macro uptrend (200 EMA)")

    if 40 <= last['RSI'] <= 60:
        score += 20
        reasons.append("Accumulation RSI")

    bias = "BULLISH" if score > 65 else "NEUTRAL"
    sl = round(last['Close'] * 0.85, 2)
    tgt = round(last['Close'] * 1.3, 2)

    return score, bias, reasons, sl, tgt

# ================== SUPPORT / RESISTANCE ==================
def find_support_resistance(df, window=10):
    supports, resistances = [], []

    for i in range(window, len(df) - window):
        if df['Low'].iloc[i] == min(df['Low'].iloc[i-window:i+window]):
            supports.append(round(df['Low'].iloc[i], 2))
        if df['High'].iloc[i] == max(df['High'].iloc[i-window:i+window]):
            resistances.append(round(df['High'].iloc[i], 2))

    return supports[-3:], resistances[-3:]

# ================== SCAN API ==================
@app.route('/scan', methods=['POST'])
def scan():
    mode = request.json.get('mode')

    if mode == "INTRADAY":
        period, interval = "5d", "15m"
    else:
        period, interval = "2y", "1d"

    data = yf.download(STOCKS, period=period, interval=interval, group_by='ticker', progress=False)
    results = []

    for s in STOCKS:
        try:
            df = data[s].dropna()
            if len(df) < 50:
                continue

            df = add_indicators(df)

            if mode == "INTRADAY":
                score,bias,reasons,sl,tgt = intraday_logic(df)
            elif mode == "SWING":
                score,bias,reasons,sl,tgt = swing_logic(df)
            else:
                score,bias,reasons,sl,tgt = longterm_logic(df)

            if score > 55:
                results.append({
                    "symbol": s.replace(".NS",""),
                    "ltp": round(df['Close'].iloc[-1],2),
                    "bias": bias,
                    "score": score,
                    "reason": reasons,
                    "execution": {
                        "entry": round(df['Close'].iloc[-1],2),
                        "sl": sl,
                        "target1": tgt
                    }
                })
        except:
            continue

    final = sorted(results, key=lambda x: x['score'], reverse=True)[:5]
    return jsonify({"status":"success","data":final})

# ================== CHART DATA (OPTION A) ==================
@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    symbol = request.json.get('symbol') + ".NS"
    mode = request.json.get('mode')

    if mode == "INTRADAY":
        period, interval = "5d", "15m"
    elif mode == "SWING":
        period, interval = "6mo", "1d"
    else:
        period, interval = "2y", "1wk"

    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.dropna(inplace=True)

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            "time": int(idx.timestamp()),
            "open": round(row['Open'],2),
            "high": round(row['High'],2),
            "low": round(row['Low'],2),
            "close": round(row['Close'],2)
        })

    support, resistance = find_support_resistance(df)

    return jsonify({
        "status": "success",
        "candles": candles,
        "support": support,
        "resistance": resistance
    })

# ================== DETAILS ==================
@app.route('/get_stock_details', methods=['POST'])
def details():
    symbol = request.json.get('symbol') + ".NS"
    stock = yf.Ticker(symbol)

    info = stock.info
    fundamentals = {
        "sector": info.get('sector','N/A'),
        "high52": info.get('fiftyTwoWeekHigh','N/A')
    }

    news_payload = []
    try:
        for n in stock.news[:3]:
            sentiment = sia.polarity_scores(n['title'])['compound']
            tag = "POSITIVE" if sentiment > 0.1 else "NEGATIVE" if sentiment < -0.1 else "NEUTRAL"
            news_payload.append({
                "title": n['title'],
                "publisher": n.get('publisher','News'),
                "tag": tag,
                "link": n.get('link','#')
            })
    except:
        pass

    return jsonify({"status":"success","fundamentals":fundamentals,"news":news_payload})

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

