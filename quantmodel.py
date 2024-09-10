import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from my_xgboost_script import XGBClassifier
from sklearn.metrics import accuracy_score

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

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),  # 유닛 수를 128로 증가
        Dropout(0.2),  # Dropout 비율을 0.2로 조정
        LSTM(128),  # 두 번째 LSTM 레이어도 유닛 수 조정
        Dropout(0.2),
        Dense(64, activation='relu'),  # 더 작은 Dense 레이어 추가
        Dense(1, activation='sigmoid')  # 출력층
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# 5. LSTM 모델에서 피처 추출
def get_lstm_features(model, X_data):
    # 모델이 데이터를 처리한 후 중간 레이어에서 출력값을 추출
    lstm_model = Sequential([
        model.layers[0],  # 첫 번째 LSTM 레이어
        model.layers[1],  # 첫 번째 Dropout 레이어
        model.layers[2],  # 두 번째 LSTM 레이어
    ])
    
    lstm_features = lstm_model.predict(X_data)
    return lstm_features


# 6. XGBoost 모델 학습
def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
    xgb_model.fit(X_train, y_train)
    
    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Accuracy: {accuracy * 100:.2f}%")
    
    return xgb_model

# 7. LSTM + XGBoost 결합
def combine_lstm_xgboost(lstm_model, X_train, X_test, y_train, y_test):
    lstm_train_features = get_lstm_features(lstm_model, X_train)
    lstm_test_features = get_lstm_features(lstm_model, X_test)
    
    # XGBoost로 추가 학습 및 예측
    xgb_model = train_xgboost(lstm_train_features, y_train, lstm_test_features, y_test)
    
    return xgb_model

# 8. 모델 훈련 및 평가
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
    print(f'LSTM Loss: {loss:.4f}, LSTM Accuracy: {accuracy * 100:.2f}%')
    
    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # 에너지 섹터에 포함된 주요 종목 티커들
    energy_tickers = ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'HAL', 
                      'BP', 'OXY', 'MPC', 'KMI', 'BKR', 'FANG', 'WMB', 'HES', 'ET']

    
    
    # 여러 종목 데이터를 2020년부터 2024년까지 다운로드
    data = get_multiple_stock_data(energy_tickers, '2020-01-01', '2024-09-04')
    processed_data = preprocess_data(data)
    for i in range(10):
        # LSTM 모델 학습 및 평가
        lstm_model, X_train, X_test, y_train, y_test = train_and_evaluate_lstm(processed_data, timesteps=10)
        
        # LSTM 모델에서 특징 추출 후 XGBoost 적용
        xgb_model = combine_lstm_xgboost(lstm_model, X_train, X_test, y_train, y_test)
