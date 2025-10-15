import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import yfinance as yf
from textblob import TextBlob

# ==========================================================
# Streamlit Page Setup
# ==========================================================
st.set_page_config(page_title="📈 AAPL Stock Prediction Dashboard", layout="wide")
st.title("📊 Apple Stock Prediction with LSTM + Sentiment Analysis")
st.markdown("This dashboard visualizes **actual vs predicted stock prices**, error distribution, and sentiment insights.")

# ==========================================================
# Load Stock Data
# ==========================================================
st.header("📥 Data Loading")
data_load_state = st.text("Downloading AAPL stock data...")
data = yf.download('AAPL', start='2018-01-01', end='2025-10-01', group_by='ticker')

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns]
elif any('AAPL' in c for c in data.columns):
    data.columns = [c.replace('AAPL_', '') for c in data.columns]

data = data.reset_index()
data_load_state.text("✅ Data Loaded Successfully!")

# ==========================================================
# Sentiment Data Simulation
# ==========================================================
news_data = pd.DataFrame({
    'date': pd.date_range(start='2018-01-01', end='2025-10-01', freq='D'),
    'title': [
        'Apple releases new iPhone with strong sales' if i % 5 == 0 else
        'Investors worried about Apple supply chain' if i % 7 == 0 else
        'Apple stock remains steady amid market volatility'
        for i in range(len(pd.date_range(start='2018-01-01', end='2025-10-01', freq='D')))
    ]
})

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

news_data['cleaned'] = news_data['title'].apply(clean_text)
news_data['sentiment_polarity'] = news_data['cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity)

sentiment_daily = news_data.groupby('date')['sentiment_polarity'].mean().reset_index()
sentiment_daily.rename(columns={'sentiment_polarity': 'Sentiment'}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])
data = pd.merge(data, sentiment_daily, left_on='Date', right_on='date', how='left')
data['Sentiment'] = data['Sentiment'].fillna(0)
data.drop(columns=['date'], inplace=True)

# ==========================================================
# Feature Engineering
# ==========================================================
data['MA_10'] = data['Close'].rolling(window=10).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Returns'] = data['Close'].pct_change()
data = data.dropna()

features = ['Close', 'MA_10', 'MA_50', 'Returns', 'Volume', 'Sentiment']
dataset = data[features].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

training_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, :])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# ==========================================================
# Model Definition
# ==========================================================
learning_rate = 0.0005
batch_size = 32
epochs = 100
dropout_rate = 0.3
units = 100

model = Sequential([
    LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(dropout_rate),
    LSTM(units, return_sequences=True),
    Dropout(dropout_rate),
    LSTM(units//2, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1)
])

optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='mean_squared_error')

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=0)

# ==========================================================
# Model Training
# ==========================================================
st.header("⚙️ Model Training Progress")
with st.spinner("Training the model... This may take a few minutes."):
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[es, mc, lr],
        verbose=0
    )
st.success("✅ Model Training Completed!")

# ==========================================================
# Predictions and Metrics
# ==========================================================
predictions = model.predict(X_test)
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
predicted_prices = close_scaler.inverse_transform(predictions)
real_prices = close_scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = math.sqrt(mean_squared_error(real_prices, predicted_prices))
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)
mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
accuracy = 100 - mape

st.header("📊 Model Performance Metrics")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.write(f"**Accuracy:** {accuracy:.2f}%")

# ==========================================================
# Visualizations
# ==========================================================

# Actual vs Predicted
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(real_prices, label='Actual', color='#1f77b4')
ax1.plot(predicted_prices, label='Predicted', color='#ff7f0e')
ax1.set_title("📈 Actual vs Predicted Prices")
ax1.legend()
st.pyplot(fig1)

# Error Distribution
errors = real_prices - predicted_prices
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.hist(errors, bins=40, color='#9467bd', alpha=0.75, edgecolor='black')
ax2.set_title('📊 Prediction Error Distribution')
ax2.set_xlabel('Error')
st.pyplot(fig2)

# Scatter Plot
fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.scatter(real_prices, predicted_prices, alpha=0.6, color='#17becf', edgecolor='black')
ax3.plot([real_prices.min(), real_prices.max()], [real_prices.min(), real_prices.max()],
         color='orange', linestyle='--', linewidth=2)
ax3.set_title('🎯 Actual vs Predicted Scatter Plot')
st.pyplot(fig3)

# Sentiment Pie Chart
sentiment_counts = (
    news_data['sentiment_polarity']
    .apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')
    .value_counts()
)
fig4, ax4 = plt.subplots(figsize=(8,6))
ax4.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
        startangle=120, colors=['#2ecc71', '#e74c3c', '#95a5a6'],
        wedgeprops={'edgecolor': 'white'})
ax4.set_title("🧠 Sentiment Composition (Positive, Negative, Neutral)")
st.pyplot(fig4)

st.success("✅ Dashboard Rendering Complete!")
