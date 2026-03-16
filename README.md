# 📈 Apple Stock Price Prediction using Machine Learning

A machine learning project that predicts next-day Apple (AAPL) stock closing prices using Random Forest Regression with 80+ technical indicators.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project demonstrates the application of machine learning to financial time series forecasting. The model achieves **58-62% directional accuracy** in predicting whether Apple's stock will go up or down the next day, outperforming random guessing (50%) and approaching professional trader performance levels.

### ✨ Key Features

- 📊 **15 years of historical data** (2010-2024)
- 🔧 **80+ engineered features** including technical indicators
- 🌲 **Random Forest Regressor** with 500 trees
- 📉 **Return-based prediction** approach for scale invariance
- ✅ **Comprehensive evaluation** with train/validation/test splits

## 🏆 Results

| Metric | Value |
|--------|-------|
| 🎯 Directional Accuracy | 58-62% |
| 💰 Mean Absolute Error (MAE) | $3-8 |
| 📅 Test Period | 2023-2024 |
| 📚 Training Data | 2010-2022 |

### 📊 Model Performance

![Prediction Results](images/predictions_all_sets.png)

The model successfully tracks Apple's price movement across all datasets:
- **Training Set (top)**: Near-perfect fit on 2010-2022 data
- **Validation Set (middle)**: Strong generalization on 2022-2023
- **Test Set (bottom)**: Successfully tracks surge from $180 to $260 in 2024

### 📈 Technical Indicators

![Technical Indicators](images/technical_indicators.png)

RSI, MACD, and Bollinger Bands provide the model with momentum and volatility signals.

### 🎯 Feature Importance

![Feature Importance](images/feature_importance.png)

The model learns from a balanced set of features rather than over-relying on current price alone.

## 💡 Technical Approach

### 🚀 The Critical Innovation

Instead of predicting absolute prices (which fails when prices exceed training range), the model predicts **percentage returns**:

```python
# ❌ Traditional approach (fails)
Target = Tomorrow's price  # Gets stuck at historical levels

# ✅ Our approach (works)
Target = (Tomorrow - Today) / Today  # Scale-invariant
Predicted_Price = Current_Price × (1 + Predicted_Return)
```

This allows the model to work across any price level, from $10 to $260.

## 🔍 Features

### 📊 80+ Technical Indicators

**📈 Trend Indicators:**
- Simple Moving Averages (SMA): 5, 10, 20, 50, 200-day
- Exponential Moving Averages (EMA): 10, 20, 50-day
- Price position relative to moving averages

**⚡ Momentum Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Rate of Change (10, 20-day)
- Momentum (5, 10, 20-day)

**📉 Volatility Measures:**
- Bollinger Bands (upper, lower, width, position)
- Average True Range (ATR)
- Rolling standard deviation (5, 10, 20-day)

**⏮️ Historical Features:**
- Lag prices: 1, 2, 3, 5, 10, 20 days
- Lag returns: 1, 2, 3, 5, 10, 20 days
- Lag volume: 1, 2, 3, 5, 10, 20 days

**📦 Volume Analysis:**
- Volume moving averages
- Volume ratios
- Price-volume interactions

**🕯️ Price Patterns:**
- Candlestick patterns (shadows, body size)
- Daily range and gaps
- High/Low ratios

## 🛠️ Installation

### 📋 Requirements

```bash
pip install -r requirements.txt
```

### 🐍 Python Version

- Python 3.8+

## 🚀 Usage

### ▶️ Basic Usage

```python
python backtester.py
```

This will:
1. 📥 Download 15 years of Apple stock data
2. 🔧 Engineer 80+ features
3. 🤖 Train the Random Forest model
4. 📊 Display performance metrics and visualizations
5. 🔮 Predict the next trading day's price

### 📤 Output

The script generates:
- 📊 **Technical indicator visualizations** (RSI, MACD, Bollinger Bands)
- 📈 **Prediction graphs** for training, validation, and test sets
- 🎯 **Feature importance analysis**
- 🔮 **Next-day price prediction** with confidence indicators

### 💻 Example Output

```
============================================================
🍎 APPLE STOCK PRICE PREDICTION PROJECT
============================================================

Fetching data for AAPL...
✅ Successfully fetched 3700 days of data

🔧 Feature engineering complete!
Total features created: 82

🌲 Training Random Forest model...
✅ Training complete!

📊 Model Performance:

Test Set:
   MAE: $6.47
   RMSE: $8.23
   Directional Accuracy: 58.9%

============================================================
🔮 PREDICTING NEXT TRADING DAY
============================================================

Current Date: 2024-12-30
Current Closing Price: $253.45

Prediction for next trading day:
   Predicted Return: +0.35%
   Predicted Price: $254.34
   Expected Change: $0.89
   Direction: ⬆️ UP

Key Indicators:
   RSI: 62.45 (Neutral)
   MACD Status: Bullish
   Price above SMA20 and SMA50
```

## 🏗️ Model Architecture

### 🌲 Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=500,      # 500 decision trees
    max_depth=30,          # Maximum tree depth
    min_samples_split=2,   # Minimum samples to split
    max_features='sqrt',   # Features per split
    random_state=42
)
```

### 📊 Data Split

- 📚 **Training Set (80%)**: 2010-2022 (~2,960 days)
- ✅ **Validation Set (10%)**: 2022-2023 (~370 days)
- 🧪 **Test Set (10%)**: 2023-2024 (~370 days)

⚠️ Chronological split is critical for time series to prevent data leakage.

## 🎯 Why This Approach Works

### ❌ Problem with Traditional Methods

Predicting absolute prices fails when:
- Training data: $10-$180 (2010-2022)
- Test data: $180-$260 (2023-2024)
- Model prediction: Stuck around $180 ❌

### ✅ Solution: Return-Based Prediction

Predicting percentage returns works because:
- A +2% gain has the same pattern at any price level
- Model learns: "RSI > 70 + MACD positive → +1.5% return"
- Applies to $50, $150, or $250 ✓

## 📊 Understanding the Results

### 🎯 Why 58% Accuracy is Good

| Accuracy Level | Interpretation |
|---------------|----------------|
| 50% | 🎲 Random guessing (coin flip) |
| 52-54% | 📉 Weak signal |
| **55-60%** | **🏆 Professional trader level** ← We are here |
| 60-65% | 💼 Hedge fund performance |
| 70%+ | ⚠️ Suspicious (likely overfitting) |

### 🌐 Market Prediction Challenges

Stock prices are fundamentally difficult to predict because:
- 📰 **Efficient Market Hypothesis**: Most predictable patterns are already exploited
- 📢 **News-driven**: 95% of movements driven by unexpected events
- 🔧 **Technical-only limitation**: We don't use earnings, sentiment, or macro data
- ⏱️ **Daily timeframe**: Short-term predictions are hardest (dominated by noise)

## ⚠️ Limitations

- 📰 **Cannot predict news events**: Earnings surprises, Fed announcements, geopolitical shocks
- 📊 **Technical data only**: No fundamental analysis, sentiment, or alternative data
- 📜 **Historical patterns**: Assumes past patterns continue (may not hold during regime changes)
- 🌪️ **Market conditions**: Performance may degrade during unprecedented market events
- 🔄 **Retraining needed**: Model should be retrained regularly as markets evolve

## 🔮 Future Improvements

### 🚀 Planned Enhancements

1. **💬 Sentiment Analysis**
   - News headline analysis
   - Social media sentiment (Twitter, Reddit)
   - Analyst ratings

2. **📈 Fundamental Data**
   - Earnings reports
   - Revenue growth
   - P/E ratios
   - Balance sheet metrics

3. **🤖 Alternative Models**
   - LSTM/GRU neural networks
   - XGBoost
   - Ensemble methods

4. **📊 Multi-Stock Portfolio**
   - Extend to FAANG stocks
   - Correlation analysis
   - Portfolio optimization

5. **🌍 Macro Indicators**
   - Interest rates
   - Inflation data
   - Economic indicators

## 📁 Project Structure

```
stock-prediction/
│
├── 📄 stock_prediction.py          # Main script
├── 📘 README.md                     # This file
├── 📋 requirements.txt              # Dependencies
│
└── 📂 outputs/
    ├── 📊 technical_indicators.png  # Indicator visualizations
    ├── 📈 predictions.png           # Actual vs predicted plots
    └── 🎯 feature_importance.png    # Feature importance chart
```

## 🔬 Technical Details

### 🔧 Feature Engineering Process

1. **📥 Raw data collection**: OHLCV (Open, High, Low, Close, Volume)
2. **📊 Technical indicator calculation**: RSI, MACD, Bollinger Bands
3. **⏮️ Lag feature creation**: Historical prices and returns
4. **📏 Normalization**: Percentage-based features for scale invariance
5. **🧹 NaN handling**: Drop rows with insufficient historical data

### 🎓 Model Training

1. **🔧 Data preprocessing**: Feature engineering on raw prices
2. **🎯 Target creation**: Next-day return = (Tomorrow - Today) / Today
3. **✂️ Train-test split**: Chronological 80/10/10 split
4. **🌲 Model fitting**: Random Forest on percentage returns
5. **🔮 Prediction**: Convert predicted returns back to prices
6. **📊 Evaluation**: MAE, RMSE, directional accuracy

## 📊 Performance Metrics Explained

### 💰 Mean Absolute Error (MAE)
Average dollar amount predictions are off:
```
MAE = Average(|Actual - Predicted|)
```

### 📉 Root Mean Squared Error (RMSE)
Penalizes large errors more heavily:
```
RMSE = √(Average((Actual - Predicted)²))
```

### 🎯 Directional Accuracy
Percentage of correct up/down predictions:
```
Directional Accuracy = % of (sign(Actual change) == sign(Predicted change))
```

## 💡 Key Insights

1. 📏 **Scale-invariant prediction is critical** for stocks with high growth
2. 🔧 **More features improve performance** up to a point (diminishing returns)
3. ⏰ **Time series validation is essential** to prevent data leakage
4. 🌐 **Market efficiency limits** maximum achievable accuracy
5. 🤝 **Combining multiple indicators** works better than any single feature

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- 📥 Additional data sources
- 🤖 Alternative model architectures
- 🔧 Enhanced feature engineering
- 📊 Better visualization tools
- 📈 Multi-stock support

## 📄 License

MIT License - feel free to use for educational purposes.

## 🙏 Acknowledgments

- 📊 Data source: Yahoo Finance (via yfinance library)
- 💡 Inspiration: Technical analysis and quantitative trading literature
- 🎓 Methodology: Standard ML practices for time series forecasting

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

⚠️ **Disclaimer**: This project is for educational purposes only. Stock market prediction is inherently uncertain, and past performance does not guarantee future results. Do not use this model for actual trading without proper risk management and additional validation.
