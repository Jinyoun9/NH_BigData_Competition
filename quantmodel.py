import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

# 1. 데이터 수집 (주식 데이터 다운로드)
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def get_multiple_stock_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        stock_data = get_stock_data(ticker, start_date, end_date)
        stock_data['Ticker'] = ticker  # 종목 구분을 위한 열 추가
        all_data.append(stock_data)
    return pd.concat(all_data)

# 2. 데이터 전처리
def preprocess_data(data):
    data['Return'] = data['Adj Close'].pct_change()
    data['10_day_momentum'] = data['Return'].rolling(window=10).mean()
    data['20_day_momentum'] = data['Return'].rolling(window=20).mean()
    data['30_day_momentum'] = data['Return'].rolling(window=30).mean()
    data = data.dropna()
    data['Target'] = np.where(data['Return'] > 0, 1, 0)
    
    # 피처 정규화
    scaler = StandardScaler()
    data[['10_day_momentum', '20_day_momentum', '30_day_momentum']] = scaler.fit_transform(
        data[['10_day_momentum', '20_day_momentum', '30_day_momentum']]
    )
    
    return data

# 3. 데이터 변환 (LSTM 입력 형태로 변환)
def prepare_lstm_data(data, timesteps):
    features = data[['10_day_momentum', '20_day_momentum', '30_day_momentum']].values
    target = data['Target'].values

    X, y = [], []
    for i in range(len(features) - timesteps):
        X.append(features[i:i + timesteps])
        y.append(target[i + timesteps])
        
    return np.array(X), np.array(y)

# 4. 모델 정의 및 훈련
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(100),
        Dropout(0.3),
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
   
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])
    return model

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        alpha_t = y_true * alpha + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.keras.backend.ones_like(y_true) - y_true) * (1 - y_pred)
        fl = -alpha_t * tf.math.pow(1. - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_sum(fl, axis=1)
    return focal_loss_fixed

def train_and_evaluate_lstm(data, timesteps, model=None):
    X, y = prepare_lstm_data(data, timesteps)
    if len(X) == 0 or len(y) == 0:
        raise ValueError("No data available after preparing LSTM input.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if model is None:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')
    
    return model

# 5. 모델 저장 및 불러오기
def save_model(model, filepath):
    model.save(filepath)
    print(f"Model saved as '{filepath}'")

def load_model_from_file(filepath):
    return load_model(filepath)


if __name__ == "__main__":
    # 에너지 섹터에 포함된 주요 종목 티커들
    energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'HAL', 
                      'BP', 'OXY', 'MPC', 'KMI', 'BKR', 'FANG', 'WMB', 'HES', 'ET']

    # 여러 종목 데이터를 2020년부터 2024년까지 다운로드
    data = get_multiple_stock_data(energy_tickers, '2020-01-01', '2024-09-04')
    print(data)
    # 데이터 전처리
    processed_data = preprocess_data(data)
    
    for i in range(5):
        try:
            # 기존 모델 불러오기
            trained_lstm_model = load_model_from_file('lstm_model.keras')
            print("Loaded existing model.")
            
            # 기존 모델에 추가 학습
            trained_lstm_model = train_and_evaluate_lstm(processed_data, timesteps=10, model=trained_lstm_model)
            
        except Exception as e:
            print(f"No existing model found. Training a new model. Error: {e}")
            # 새로운 모델 훈련
            trained_lstm_model = train_and_evaluate_lstm(processed_data, timesteps=10)
        
        # 모델 저장
        save_model(trained_lstm_model, 'lstm_model.keras')
        
        # 모델 불러오기
        loaded_model = load_model_from_file('lstm_model.keras')
        print("Loaded model for further use.")