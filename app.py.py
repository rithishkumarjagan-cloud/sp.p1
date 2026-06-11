# ============================================================
#   ADVANCED STOCK MARKET PREDICTION
#   Features: Live Prices, Buy/Sell Signals, ML Prediction
#   Run: python app.py  →  http://localhost:5000
# ============================================================

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import threading
import time

app = Flask(__name__)

# ─────────────────────────────────────────
# STOCKS CONFIG
# ─────────────────────────────────────────
STOCKS = {
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS':      'Tata Consultancy',
    'INFY.NS':     'Infosys',
    'HDFCBANK.NS': 'HDFC Bank',
    'WIPRO.NS':    'Wipro'
}

# Cache for live prices
price_cache = {}
model_store = {}

# ─────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────
def download_data(ticker, period='2y'):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def add_features(df):
    df = df.copy()
    for col in ['Open','High','Low','Close','Volume']:
        if isinstance(df[col], pd.DataFrame):
            df[col] = df[col].squeeze()

    close  = df['Close'].squeeze()
    high   = df['High'].squeeze()
    low    = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # Moving Averages
    df['MA_10']  = close.rolling(10).mean()
    df['MA_20']  = close.rolling(20).mean()
    df['MA_50']  = close.rolling(50).mean()
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    df['MACD']   = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    # RSI
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    rs       = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    std20          = close.rolling(20).std()
    df['BB_mid']   = close.rolling(20).mean()
    df['BB_upper'] = df['BB_mid'] + 2 * std20
    df['BB_lower'] = df['BB_mid'] - 2 * std20
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # Volume indicators
    df['Volume_MA'] = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / (df['Volume_MA'] + 1e-10)

    # Price features
    df['Price_Change']  = close.pct_change()
    df['High_Low_Pct']  = (high - low) / (close + 1e-10)
    df['Close_Open_Pct'] = (close - df['Open'].squeeze()) / (df['Open'].squeeze() + 1e-10)

    # Stochastic Oscillator
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Target
    df['Target'] = close.shift(-1)
    df.dropna(inplace=True)
    return df

# ─────────────────────────────────────────
# BUY / SELL SIGNAL LOGIC
# ─────────────────────────────────────────
def compute_signal(df):
    last = df.iloc[-1]
    score = 0
    reasons = []

    # RSI signal
    rsi = float(last['RSI'])
    if rsi < 35:
        score += 2
        reasons.append(f"RSI oversold ({rsi:.1f})")
    elif rsi > 65:
        score -= 2
        reasons.append(f"RSI overbought ({rsi:.1f})")

    # MACD signal
    macd  = float(last['MACD'])
    sigln = float(last['Signal_Line'])
    if macd > sigln:
        score += 1
        reasons.append("MACD bullish crossover")
    else:
        score -= 1
        reasons.append("MACD bearish crossover")

    # MA signal
    close = float(last['Close'])
    ma20  = float(last['MA_20'])
    ma50  = float(last['MA_50'])
    if close > ma20 > ma50:
        score += 2
        reasons.append("Price above MA20 & MA50")
    elif close < ma20 < ma50:
        score -= 2
        reasons.append("Price below MA20 & MA50")

    # Bollinger Band signal
    bb_upper = float(last['BB_upper'])
    bb_lower = float(last['BB_lower'])
    if close < bb_lower:
        score += 1
        reasons.append("Near lower Bollinger Band")
    elif close > bb_upper:
        score -= 1
        reasons.append("Near upper Bollinger Band")

    # Stochastic signal
    stoch_k = float(last['Stoch_K'])
    if stoch_k < 20:
        score += 1
        reasons.append(f"Stochastic oversold ({stoch_k:.1f})")
    elif stoch_k > 80:
        score -= 1
        reasons.append(f"Stochastic overbought ({stoch_k:.1f})")

    if score >= 3:
        signal = "STRONG BUY"
        color  = "#00e676"
    elif score >= 1:
        signal = "BUY"
        color  = "#69f0ae"
    elif score <= -3:
        signal = "STRONG SELL"
        color  = "#ff1744"
    elif score <= -1:
        signal = "SELL"
        color  = "#ff5252"
    else:
        signal = "HOLD"
        color  = "#ffd740"

    return {"signal": signal, "score": score, "color": color, "reasons": reasons}

# ─────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────
FEATURES = ['Open','High','Low','Close','Volume',
            'MA_10','MA_20','MA_50','RSI','MACD',
            'BB_width','Volume_Ratio','Price_Change',
            'High_Low_Pct','Stoch_K','Close_Open_Pct']

def train_all_models():
    print("\nTraining models for all stocks...")
    for ticker in STOCKS:
        try:
            raw = download_data(ticker, '2y')
            if raw.empty:
                continue
            df = add_features(raw)

            X = df[FEATURES]
            y = df['Target']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=150, max_depth=12,
                                       random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            rf_acc = max(0, round(r2_score(y_test, rf.predict(X_test)) * 100, 1))

            # Gradient Boosting
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                           learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            gb_acc = max(0, round(r2_score(y_test, gb.predict(X_test)) * 100, 1))

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_acc = max(0, round(r2_score(y_test, lr.predict(X_test)) * 100, 1))

            model_store[ticker] = {
                'random_forest':       {'model': rf, 'acc': rf_acc},
                'gradient_boosting':   {'model': gb, 'acc': gb_acc},
                'linear_regression':   {'model': lr, 'acc': lr_acc},
                'df': df,
                'raw': raw
            }
            print(f"  {ticker}: RF={rf_acc}% | GB={gb_acc}% | LR={lr_acc}%")

        except Exception as e:
            print(f"  Error training {ticker}: {e}")

    print("All models ready!\n")

# ─────────────────────────────────────────
# LIVE PRICE CACHE (refresh every 60s)
# ─────────────────────────────────────────
def refresh_prices():
    while True:
        for ticker in STOCKS:
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                price_cache[ticker] = {
                    'price':   round(float(info.last_price), 2),
                    'change':  round(float(info.last_price) - float(info.previous_close), 2),
                    'pct':     round(((float(info.last_price) - float(info.previous_close))
                                      / float(info.previous_close)) * 100, 2),
                    'high':    round(float(info.day_high), 2),
                    'low':     round(float(info.day_low), 2),
                    'volume':  int(info.three_month_average_volume or 0),
                    'updated': datetime.now().strftime('%H:%M:%S')
                }
            except:
                price_cache[ticker] = {
                    'price': 0, 'change': 0, 'pct': 0,
                    'high': 0, 'low': 0, 'volume': 0, 'updated': '--'
                }
        time.sleep(60)

# ─────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html', stocks=STOCKS)


@app.route('/api/prices')
def api_prices():
    result = {}
    for ticker, name in STOCKS.items():
        result[ticker] = {**price_cache.get(ticker, {}), 'name': name}
    return jsonify(result)


@app.route('/api/chart/<ticker>')
def api_chart(ticker):
    range_ = request.args.get('range', '1m')
    periods = {'1w': 7, '1m': 30, '3m': 90, '6m': 180, '1y': 365}
    days = periods.get(range_, 30)

    if ticker in model_store:
        raw = model_store[ticker]['raw']
    else:
        raw = download_data(ticker, '1y')

    df = raw.tail(days).reset_index()
    close = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

    labels = df['Date'].dt.strftime('%b %d').tolist()
    prices = [round(float(x), 2) for x in close.tolist()]

    # Simple moving averages for chart
    ma20 = pd.Series(prices).rolling(20).mean().round(2).tolist()
    ma50 = pd.Series(prices).rolling(50).mean().round(2).tolist()

    return jsonify({'labels': labels, 'prices': prices, 'ma20': ma20, 'ma50': ma50})


@app.route('/api/signal/<ticker>')
def api_signal(ticker):
    if ticker not in model_store:
        return jsonify({'signal': 'N/A', 'score': 0, 'color': '#888', 'reasons': []})
    df = model_store[ticker]['df']
    sig = compute_signal(df)
    return jsonify(sig)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data     = request.json
    ticker   = data.get('ticker', 'RELIANCE.NS')
    model_key = data.get('model', 'random_forest')
    open_p   = float(data.get('open',   0))
    high_p   = float(data.get('high',   0))
    low_p    = float(data.get('low',    0))
    volume   = float(data.get('volume', 0))

    if ticker not in model_store:
        return jsonify({'error': 'Model not ready'}), 400

    store = model_store[ticker]
    df    = store['df']
    last  = df.iloc[-1]

    # Use latest indicators + user input
    close_est = (open_p + high_p + low_p) / 3 if open_p > 0 else float(last['Close'])
    input_row = {
        'Open':           open_p  if open_p  > 0 else float(last['Open']),
        'High':           high_p  if high_p  > 0 else float(last['High']),
        'Low':            low_p   if low_p   > 0 else float(last['Low']),
        'Close':          close_est,
        'Volume':         volume * 100000 if volume > 0 else float(last['Volume']),
        'MA_10':          float(last['MA_10']),
        'MA_20':          float(last['MA_20']),
        'MA_50':          float(last['MA_50']),
        'RSI':            float(last['RSI']),
        'MACD':           float(last['MACD']),
        'BB_width':       float(last['BB_width']),
        'Volume_Ratio':   float(last['Volume_Ratio']),
        'Price_Change':   float(last['Price_Change']),
        'High_Low_Pct':   float(last['High_Low_Pct']),
        'Stoch_K':        float(last['Stoch_K']),
        'Close_Open_Pct': float(last['Close_Open_Pct'])
    }
    input_df = pd.DataFrame([input_row])

    mdl = store.get(model_key, store['random_forest'])
    predicted = float(mdl['model'].predict(input_df)[0])
    accuracy  = mdl['acc']
    signal    = compute_signal(df)

    return jsonify({
        'predicted_price': round(predicted, 2),
        'current_price':   round(close_est, 2),
        'change':          round(predicted - close_est, 2),
        'change_pct':      round(((predicted - close_est) / close_est) * 100, 2),
        'accuracy':        accuracy,
        'signal':          signal['signal'],
        'signal_color':    signal['color'],
        'model':           model_key.replace('_', ' ').title()
    })


@app.route('/api/indicators/<ticker>')
def api_indicators(ticker):
    if ticker not in model_store:
        return jsonify({})
    df   = model_store[ticker]['df']
    last = df.iloc[-1]
    return jsonify({
        'rsi':       round(float(last['RSI']), 2),
        'macd':      round(float(last['MACD']), 2),
        'signal_ln': round(float(last['Signal_Line']), 2),
        'ma20':      round(float(last['MA_20']), 2),
        'ma50':      round(float(last['MA_50']), 2),
        'bb_upper':  round(float(last['BB_upper']), 2),
        'bb_lower':  round(float(last['BB_lower']), 2),
        'stoch_k':   round(float(last['Stoch_K']), 2),
        'stoch_d':   round(float(last['Stoch_D']), 2),
        'vol_ratio': round(float(last['Volume_Ratio']), 2),
    })


# ─────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────
print("=" * 50)
print("  Advanced Stock Market Prediction")
print("  Loading data & training models...")
print("=" * 50)

train_all_models()

# Start live price refresh thread
t = threading.Thread(target=refresh_prices, daemon=True)
t.start()

# Initial price fetch
refresh_prices_once = threading.Thread(target=lambda: [
    price_cache.update({tk: {
        'price': 0,'change': 0,'pct': 0,
        'high': 0,'low': 0,'volume': 0,'updated': 'Loading...'
    }}) for tk in STOCKS
], daemon=True)
refresh_prices_once.start()

if __name__ == '__main__':
    print(f"\n  Open: http://localhost:5000\n{'='*50}")  
    app.run(debug=True, port=5000, use_reloader=False)