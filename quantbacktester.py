import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)

RAW_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']

def fetch_stock_data(ticker='AAPL', start_date='2010-01-01', end_date='2024-12-31'):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"Fetched {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def add_all_features(df):
    df = df.copy()

    df['SMA_5']   = df['Close'].rolling(5).mean()
    df['SMA_10']  = df['Close'].rolling(10).mean()
    df['SMA_20']  = df['Close'].rolling(20).mean()
    df['SMA_50']  = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']           = exp1 - exp2
    df['Signal_Line']    = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

    df['BB_middle']   = df['Close'].rolling(20).mean()
    bb_std            = df['Close'].rolling(20).std()
    df['BB_upper']    = df['BB_middle'] + bb_std * 2
    df['BB_lower']    = df['BB_middle'] - bb_std * 2
    df['BB_width']    = df['BB_upper'] - df['BB_lower']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # Momentum as pct_change (scale-invariant)
    df['Momentum_5']       = df['Close'].pct_change(5)
    df['Momentum_10']      = df['Close'].pct_change(10)
    df['Momentum_20']      = df['Close'].pct_change(20)
    df['Rate_of_Change_10'] = df['Close'].pct_change(10)
    df['Rate_of_Change_20'] = df['Close'].pct_change(20)

    df['Volatility_5']  = df['Close'].pct_change().rolling(5).std()
    df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['Volatility_20'] = df['Close'].pct_change().rolling(20).std()

    hl         = df['High'] - df['Low']
    hc         = np.abs(df['High'] - df['Close'].shift())
    lc         = np.abs(df['Low']  - df['Close'].shift())
    df['ATR']  = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df['Volume_SMA_5']  = df['Volume'].rolling(5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio']  = df['Volume'] / df['Volume_SMA_20']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Price_Volume']  = df['Close'] * df['Volume']

    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'Close_Lag_{lag}']  = df['Close'].shift(lag)
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)

    df['High_Low_Ratio']   = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    df['Daily_Range']      = df['High'] - df['Low']
    df['Daily_Range_Pct']  = (df['High'] - df['Low']) / df['Close']
    df['Gap']              = df['Open'] - df['Close'].shift(1)
    df['Gap_Pct']          = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    df['Price_Above_SMA20']  = (df['Close'] > df['SMA_20']).astype(int)
    df['Price_Above_SMA50']  = (df['Close'] > df['SMA_50']).astype(int)
    df['Price_Above_SMA200'] = (df['Close'] > df['SMA_200']).astype(int)
    df['SMA10_Above_SMA20']  = (df['SMA_10'] > df['SMA_20']).astype(int)
    df['SMA20_Above_SMA50']  = (df['SMA_20'] > df['SMA_50']).astype(int)

    df['Price_Change']     = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change()
    df['High_Change']      = df['High'].diff()
    df['Low_Change']       = df['Low'].diff()

    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Body_Size']    = np.abs(df['Close'] - df['Open'])

    df = df.dropna()
    print(f"Feature engineering complete: {len(df.columns)} features, {len(df)} days")
    return df


def build_model(df):
    df = df.copy()
    df['Target']        = df['Close'].shift(-1)
    df['Target_Return'] = (df['Target'] - df['Close']) / df['Close']
    df = df.dropna()

    # Exclude raw OHLCV and target columns from features.
    # Raw price/volume columns are excluded to avoid leakage — engineered
    # lag and ratio features capture the same information in a controlled way.
    exclude = set(RAW_COLS + ['Target', 'Target_Return'])
    feature_columns = [c for c in df.columns if c not in exclude]

    X              = df[feature_columns].values
    y_return       = df['Target_Return'].values
    y_actual       = df['Target'].values
    current_prices = df['Close'].values
    dates          = df.index

    n         = len(X)
    train_end = int(n * 0.80)
    val_end   = int(n * 0.90)

    X_train, y_return_train, y_actual_train, prices_train, dates_train = (
        X[:train_end], y_return[:train_end], y_actual[:train_end],
        current_prices[:train_end], dates[:train_end]
    )
    X_val, y_return_val, y_actual_val, prices_val, dates_val = (
        X[train_end:val_end], y_return[train_end:val_end], y_actual[train_end:val_end],
        current_prices[train_end:val_end], dates[train_end:val_end]
    )
    X_test, y_return_test, y_actual_test, prices_test, dates_test = (
        X[val_end:], y_return[val_end:], y_actual[val_end:],
        current_prices[val_end:], dates[val_end:]
    )

    print(f"\nTraining on {len(X_train)} days...")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=30,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_return_train)
    print("Training complete.")

    y_return_train_pred = model.predict(X_train)
    y_return_val_pred   = model.predict(X_val)
    y_return_test_pred  = model.predict(X_test)

    y_train_pred = prices_train * (1 + y_return_train_pred)
    y_val_pred   = prices_val   * (1 + y_return_val_pred)
    y_test_pred  = prices_test  * (1 + y_return_test_pred)

    train_mae = mean_absolute_error(y_actual_train, y_train_pred)
    val_mae   = mean_absolute_error(y_actual_val,   y_val_pred)
    test_mae  = mean_absolute_error(y_actual_test,  y_test_pred)

    # Directional accuracy: sign of predicted return vs sign of actual return
    train_dir = np.mean(np.sign(y_return_train_pred) == np.sign(y_return_train)) * 100
    val_dir   = np.mean(np.sign(y_return_val_pred)   == np.sign(y_return_val))   * 100
    test_dir  = np.mean(np.sign(y_return_test_pred)  == np.sign(y_return_test))  * 100

    print(f"\nModel Performance:")
    print(f"   Training:   MAE=${train_mae:.2f}, Directional Accuracy={train_dir:.1f}%")
    print(f"   Validation: MAE=${val_mae:.2f}, Directional Accuracy={val_dir:.1f}%")
    print(f"   Test:       MAE=${test_mae:.2f}, Directional Accuracy={test_dir:.1f}%")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    for ax, dates_s, actual, pred, title in [
        (axes[0], dates_train, y_actual_train, y_train_pred, 'Training Set (2010–2022)'),
        (axes[1], dates_val,   y_actual_val,   y_val_pred,   'Validation Set (2022–2023)'),
        (axes[2], dates_test,  y_actual_test,  y_test_pred,  'Test Set (2023–2024)'),
    ]:
        ax.plot(dates_s, actual, label='Actual',    linewidth=2, color='blue',  alpha=0.7)
        ax.plot(dates_s, pred,   label='Predicted', linewidth=2, color='red',   alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[2].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig('images/predictions_all_sets.png', dpi=300, bbox_inches='tight')
    print("Saved: images/predictions_all_sets.png")
    plt.show()

    return model, feature_columns, test_mae, test_dir


def visualize_technical_indicators(df):
    print("\nGenerating technical indicator plots...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    recent = df.tail(500)

    ax1 = axes[0]
    ax1.plot(recent.index, recent['Close'],     label='Close',    linewidth=2, color='blue')
    ax1.plot(recent.index, recent['SMA_20'],    label='SMA 20',   linewidth=1.5, alpha=0.7)
    ax1.plot(recent.index, recent['SMA_50'],    label='SMA 50',   linewidth=1.5, alpha=0.7)
    ax1.plot(recent.index, recent['BB_upper'],  label='BB Upper', alpha=0.5, linestyle='--', color='gray')
    ax1.plot(recent.index, recent['BB_lower'],  label='BB Lower', alpha=0.5, linestyle='--', color='gray')
    ax1.fill_between(recent.index, recent['BB_upper'], recent['BB_lower'], alpha=0.1, color='gray')
    ax1.set_title('Price with Moving Averages and Bollinger Bands', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(recent.index, recent['RSI'], linewidth=2, color='purple')
    ax2.axhline(70, color='red',   linestyle='--', linewidth=1, label='Overbought (70)')
    ax2.axhline(30, color='green', linestyle='--', linewidth=1, label='Oversold (30)')
    ax2.axhline(50, color='black', linestyle='-',  linewidth=0.5, alpha=0.3)
    ax2.fill_between(recent.index, 30, 70, alpha=0.1, color='blue')
    ax2.set_title('RSI', fontsize=14, fontweight='bold')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(recent.index, recent['MACD'],        label='MACD',        linewidth=2, color='blue')
    ax3.plot(recent.index, recent['Signal_Line'], label='Signal Line', linewidth=2, color='red')
    colors = ['green' if x > 0 else 'red' for x in recent['MACD_Histogram']]
    ax3.bar(recent.index, recent['MACD_Histogram'], label='Histogram', alpha=0.3, color=colors)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('MACD', fontsize=14, fontweight='bold')
    ax3.set_ylabel('MACD')
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/technical_indicators.png', dpi=300, bbox_inches='tight')
    print("Saved: images/technical_indicators.png")
    plt.show()


def visualize_feature_importance(model, feature_columns):
    print("\nGenerating feature importance plot...")
    importance_df = pd.DataFrame({
        'Feature':    feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#6A4C93')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: images/feature_importance.png")
    plt.show()


def predict_next_day(model, df, feature_columns):
    print("\n" + "="*60)
    print("NEXT DAY PREDICTION")
    print("="*60)

    latest       = df[feature_columns].iloc[-1:].values
    current_price = df['Close'].iloc[-1]
    current_date  = df.index[-1]

    pred_return = model.predict(latest)[0]
    pred_price  = current_price * (1 + pred_return)

    print(f"\nCurrent Date:     {current_date.date()}")
    print(f"Current Price:    ${current_price:.2f}")
    print(f"Predicted Return: {pred_return*100:+.2f}%")
    print(f"Predicted Price:  ${pred_price:.2f}")
    print(f"Direction:        {'UP' if pred_return > 0 else 'DOWN'}")

    rsi    = df['RSI'].iloc[-1]
    macd   = df['MACD'].iloc[-1]
    signal = df['Signal_Line'].iloc[-1]
    sma20  = df['SMA_20'].iloc[-1]
    sma50  = df['SMA_50'].iloc[-1]

    rsi_label = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    print(f"\nKey Indicators:")
    print(f"   RSI:    {rsi:.2f} ({rsi_label})")
    print(f"   MACD:   {macd:.2f}, Signal: {signal:.2f} ({'Bullish' if macd > signal else 'Bearish'})")
    print(f"   SMA20:  {'Above' if current_price > sma20 else 'Below'}")
    print(f"   SMA50:  {'Above' if current_price > sma50 else 'Below'}")

    return pred_price


if __name__ == "__main__":
    print("\n" + "="*60)
    print("APPLE STOCK RETURN FORECASTING")
    print("="*60)

    df = fetch_stock_data('AAPL')
    df = add_all_features(df)
    visualize_technical_indicators(df)
    model, feature_columns, test_mae, test_dir = build_model(df)
    visualize_feature_importance(model, feature_columns)
    predict_next_day(model, df, feature_columns)

    print("\nAll plots saved to 'images/'")
    print("="*60)
