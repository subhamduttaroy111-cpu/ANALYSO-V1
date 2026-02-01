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
    'LUPIN.NS','AUROPHARMO.NS','BIOCON.NS','ALKYLAMINE.NS','PIDILITIND.NS',

    # ---- Others / Growth ----
    'TRENT.NS','DLF.NS','INDIGO.NS','BALKRISIND.NS','EXIDEIND.NS'
]


# ================== ADVANCED INDICATORS ==================
def add_indicators(df):
    """Enhanced technical indicators with MACD, Bollinger Bands, Volume analysis"""
    
    # Moving Averages
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    # Volume Analysis
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    return df

# ================== IMPROVED LOGICS ==================
def intraday_logic(df):
    """Enhanced intraday logic with MACD, volume confirmation, and tighter risk management"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 40
    reasons = []

    # Trend alignment (EMA crossovers)
    if last['Close'] > last['EMA_9'] > last['EMA_21']:
        score += 20
        reasons.append("✓ Strong momentum (9/21 EMA aligned)")
    elif last['EMA_9'] > last['EMA_21']:
        score += 10
        reasons.append("→ Building momentum")

    # RSI for intraday (more sensitive range)
    if 50 <= last['RSI'] <= 70:
        score += 15
        reasons.append("✓ RSI in bullish zone (50-70)")
    elif 40 <= last['RSI'] < 50:
        score += 8
        reasons.append("→ RSI neutral-bullish")

    # MACD confirmation
    if last['MACD'] > last['MACD_Signal'] and last['MACD_Hist'] > prev['MACD_Hist']:
        score += 18
        reasons.append("✓ MACD bullish crossover with rising histogram")
    elif last['MACD'] > last['MACD_Signal']:
        score += 10
        reasons.append("→ MACD above signal")

    # Volume confirmation (critical for intraday)
    if last['Volume_Ratio'] > 1.3:
        score += 12
        reasons.append("✓ High volume breakout (1.3x avg)")
    elif last['Volume_Ratio'] > 1.0:
        score += 6
        reasons.append("→ Above avg volume")

    # Price position relative to Bollinger Bands
    if last['Close'] > last['SMA_20'] and last['Close'] < last['BB_Upper']:
        score += 5
        reasons.append("✓ In upper BB zone (expansion room)")

    # Determine bias
    bias = "BULLISH" if score >= 70 else "NEUTRAL" if score >= 55 else "BEARISH"
    
    # Dynamic SL/Target based on ATR (better risk management)
    atr_multiplier_sl = 1.2  # Tighter SL for intraday
    atr_multiplier_tgt = 2.5  # Better R:R ratio
    
    sl = round(last['Close'] - (last['ATR'] * atr_multiplier_sl), 2)
    tgt = round(last['Close'] + (last['ATR'] * atr_multiplier_tgt), 2)
    
    # Calculate risk-reward
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio

def swing_logic(df):
    """Enhanced swing logic with trend strength and multi-timeframe confirmation"""
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 40
    reasons = []

    # Primary trend (EMA alignment)
    if last['EMA_21'] > last['EMA_50'] > last['EMA_200']:
        score += 25
        reasons.append("✓ Perfect EMA stack (21>50>200)")
    elif last['EMA_21'] > last['EMA_50']:
        score += 15
        reasons.append("✓ Mid-term uptrend (21>50)")

    # Long-term trend
    if last['Close'] > last['EMA_200']:
        score += 15
        reasons.append("✓ Above 200 EMA (macro uptrend)")
    
    # RSI for swing
    if 45 <= last['RSI'] <= 65:
        score += 12
        reasons.append("✓ RSI healthy zone (45-65)")
    elif last['RSI'] > 40:
        score += 6
        reasons.append("→ RSI bullish territory")

    # MACD trend confirmation
    if last['MACD'] > last['MACD_Signal'] and last['MACD'] > 0:
        score += 15
        reasons.append("✓ MACD bullish and positive")
    elif last['MACD'] > last['MACD_Signal']:
        score += 8
        reasons.append("→ MACD crossover active")

    # Momentum check
    if last['Momentum'] > 0:
        score += 8
        reasons.append("✓ Positive 10-period momentum")

    # Volume trend (sustained interest)
    if last['Volume_Ratio'] > 1.1:
        score += 10
        reasons.append("✓ Strong volume participation")

    # Determine bias
    bias = "BULLISH" if score >= 70 else "NEUTRAL" if score >= 55 else "BEARISH"
    
    # Swing SL/Target (wider stops, bigger targets)
    sl = round(last['Close'] - (last['ATR'] * 2.0), 2)
    tgt = round(last['Close'] + (last['ATR'] * 4.0), 2)
    
    # Calculate risk-reward
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio

def longterm_logic(df):
    """Enhanced long-term logic focusing on macro trends and fundamental strength"""
    last = df.iloc[-1]
    score = 40
    reasons = []

    # Macro trend (200 EMA is king)
    if last['Close'] > last['EMA_200']:
        score += 30
        reasons.append("✓ Strong macro uptrend (above 200 EMA)")
        
        # Distance from 200 EMA (not overextended)
        distance_pct = ((last['Close'] - last['EMA_200']) / last['EMA_200']) * 100
        if distance_pct < 15:
            score += 10
            reasons.append(f"✓ Not overextended ({distance_pct:.1f}% from 200 EMA)")

    # Long-term EMA structure
    if last['EMA_50'] > last['EMA_200']:
        score += 15
        reasons.append("✓ Intermediate trend aligned (50>200)")

    # RSI for accumulation zone
    if 35 <= last['RSI'] <= 60:
        score += 15
        reasons.append("✓ RSI in accumulation zone (35-60)")
    elif last['RSI'] < 35:
        score += 8
        reasons.append("→ Oversold opportunity (RSI<35)")

    # MACD long-term positioning
    if last['MACD'] > 0 and last['MACD'] > last['MACD_Signal']:
        score += 12
        reasons.append("✓ MACD in positive territory")

    # Price vs Bollinger (for long-term entries)
    if last['Close'] < last['BB_Upper'] and last['Close'] > last['BB_Lower']:
        score += 8
        reasons.append("✓ Within Bollinger range (not extreme)")

    # Determine bias
    bias = "BULLISH" if score >= 65 else "NEUTRAL" if score >= 50 else "BEARISH"
    
    # Long-term SL/Target (much wider)
    sl = round(last['Close'] * 0.88, 2)  # 12% stop
    tgt = round(last['Close'] * 1.35, 2)  # 35% target
    
    # Calculate risk-reward
    risk = last['Close'] - sl
    reward = tgt - last['Close']
    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    return score, bias, reasons, sl, tgt, rr_ratio

# ================== SUPPORT / RESISTANCE ==================
def find_support_resistance(df, window=10):
    """Find key support and resistance levels"""
    supports, resistances = [], []

    for i in range(window, len(df) - window):
        # Support: local minimum
        if df['Low'].iloc[i] == min(df['Low'].iloc[i-window:i+window+1]):
            supports.append(round(df['Low'].iloc[i], 2))
        # Resistance: local maximum
        if df['High'].iloc[i] == max(df['High'].iloc[i-window:i+window+1]):
            resistances.append(round(df['High'].iloc[i], 2))

    # Return most recent 3 levels
    return supports[-3:] if len(supports) >= 3 else supports, \
           resistances[-3:] if len(resistances) >= 3 else resistances

# ================== SCAN API ==================
@app.route('/scan', methods=['POST'])
def scan():
    """Main scanning endpoint with improved filtering"""
    mode = request.json.get('mode')

    # Adjust timeframes based on mode
    if mode == "INTRADAY":
        period, interval = "5d", "15m"
    elif mode == "SWING":
        period, interval = "3mo", "1d"
    else:  # LONG_TERM
        period, interval = "2y", "1d"

    data = yf.download(STOCKS, period=period, interval=interval, group_by='ticker', progress=False)
    results = []

    for s in STOCKS:
        try:
            df = data[s].dropna()
            if len(df) < 200:  # Need enough data for 200 EMA
                continue

            df = add_indicators(df)

            # Apply appropriate logic
            if mode == "INTRADAY":
                score, bias, reasons, sl, tgt, rr = intraday_logic(df)
                min_score = 55  # Balanced threshold
            elif mode == "SWING":
                score, bias, reasons, sl, tgt, rr = swing_logic(df)
                min_score = 55
            else:  # LONG_TERM
                score, bias, reasons, sl, tgt, rr = longterm_logic(df)
                min_score = 50

            # Only include if score meets threshold AND decent risk-reward
            if score >= min_score and rr >= 1.2:
                current_price = round(df['Close'].iloc[-1], 2)
                
                results.append({
                    "symbol": s.replace(".NS", ""),
                    "ltp": current_price,
                    "bias": bias,
                    "score": score,
                    "reason": reasons,
                    "execution": {
                        "entry": current_price,
                        "sl": sl,
                        "target1": tgt,
                        "rr_ratio": rr
                    },
                    "indicators": {
                        "rsi": round(df['RSI'].iloc[-1], 1),
                        "macd": "BUY" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "SELL",
                        "volume": "HIGH" if df['Volume_Ratio'].iloc[-1] > 1.2 else "NORMAL"
                    }
                })
        except Exception as e:
            print(f"Error processing {s}: {e}")
            continue

    # Sort by score and return top 10
    final = sorted(results, key=lambda x: x['score'], reverse=True)[:10]
    return jsonify({"status": "success", "data": final})

# ================== CHART DATA ==================
@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    """Get chart data with support/resistance levels"""
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
            "open": round(row['Open'], 2),
            "high": round(row['High'], 2),
            "low": round(row['Low'], 2),
            "close": round(row['Close'], 2)
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
    """Get stock fundamentals and news sentiment"""
    symbol = request.json.get('symbol') + ".NS"
    stock = yf.Ticker(symbol)

    info = stock.info
    fundamentals = {
        "sector": info.get('sector', 'N/A'),
        "high52": info.get('fiftyTwoWeekHigh', 'N/A'),
        "low52": info.get('fiftyTwoWeekLow', 'N/A'),
        "marketCap": info.get('marketCap', 'N/A'),
        "pe": info.get('trailingPE', 'N/A')
    }

    news_payload = []
    try:
        for n in stock.news[:3]:
            sentiment = sia.polarity_scores(n['title'])['compound']
            tag = "POSITIVE" if sentiment > 0.1 else "NEGATIVE" if sentiment < -0.1 else "NEUTRAL"
            news_payload.append({
                "title": n['title'],
                "publisher": n.get('publisher', 'News'),
                "tag": tag,
                "link": n.get('link', '#')
            })
    except:
        pass

    return jsonify({
        "status": "success",
        "fundamentals": fundamentals,
        "news": news_payload
    })

# ================== RUN ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)