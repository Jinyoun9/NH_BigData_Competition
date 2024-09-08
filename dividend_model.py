import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# 1. 데이터 로드 및 병합
dividend_data = pd.read_csv('NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv')
performance_data = pd.read_csv('NH_CONTEST_ETF_SOR_IFO.csv')    

# 열 이름 소문자로 변환
dividend_data.columns = [col.lower() for col in dividend_data.columns]
performance_data.columns = [col.lower() for col in performance_data.columns]

# 공백 제거 및 대문자 통일
dividend_data['etf_tck_cd'] = dividend_data['etf_tck_cd'].str.strip().str.upper()
performance_data['etf_tck_cd'] = performance_data['etf_tck_cd'].str.strip().str.upper()

# 'bse_dt'를 datetime 형식으로 변환
performance_data['bse_dt'] = pd.to_datetime(performance_data['bse_dt'], format='%Y%m%d')

# 2. 과거 데이터를 고려한 가중 평균 계산
# 최신일 기준으로 가중치 계산 (exponential decay)
performance_data['days_from_recent'] = (performance_data['bse_dt'].max() - performance_data['bse_dt']).dt.days
performance_data['weight'] = np.exp(-0.01 * performance_data['days_from_recent'])

# 가중 평균을 계산할 지표 (예: 'yr1_tot_pft_rt', 'shpr_z_sor' 등)
weighted_columns = ['yr1_tot_pft_rt', 'shpr_z_sor', 'vty_z_sor', 'mxdd_z_sor']
weighted_avg = performance_data.groupby('etf_tck_cd').apply(
    lambda x: pd.Series({
        col: np.average(x[col], weights=x['weight']) for col in weighted_columns
    })
).reset_index()
# 데이터 병합
merged_data = pd.merge(dividend_data, weighted_avg, on='etf_tck_cd', how='inner')

# 2. 필요한 열만 추출
X = merged_data[['yr1_tot_pft_rt', 'shpr_z_sor', 'vty_z_sor', 'mxdd_z_sor']]
y = merged_data['ddn_amt']  # 배당금 예측 목표

# 3. 데이터 분할 (훈련 데이터와 테스트 데이터로 나누기)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 랜덤 포레스트 회귀 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 6. 성과 평가 (RMSE 계산)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# 7. 전체 데이터로 배당금 예측
merged_data['predicted_ddn_amt'] = model.predict(X)

# 8. 중복 ETF 코드 제거 및 상위 5개의 ETF 추출
# 각 etf_tck_cd별로 예측 배당금이 가장 높은 것만 선택
top_etfs_by_predicted = merged_data.loc[merged_data.groupby('etf_tck_cd')['predicted_ddn_amt'].idxmax()]

# 9. 성과 지표 기반 상위 5개 ETF 추출 (예측 배당금이 아닌 성과 지표 기준으로 선택)
# 예: 1년 총 수익률(yr1_tot_pft_rt)을 기준으로 상위 5개의 ETF 선택
top_5_etfs_by_performance = top_etfs_by_predicted.nlargest(5, 'yr1_tot_pft_rt')

# 10. 결과 출력
print("Top 5 ETFs based on predicted dividend amount (model-based):")
print(top_etfs_by_predicted[['etf_tck_cd', 'predicted_ddn_amt', 'ddn_amt']].nlargest(5, 'predicted_ddn_amt'))

print("\nTop 5 ETFs based on performance (yr1_tot_pft_rt):")
print(top_5_etfs_by_performance[['etf_tck_cd', 'yr1_tot_pft_rt', 'ddn_amt']])
# ETF 유형에 따라 구분
def categorize_etf(row):
    if row['yr1_tot_pft_rt'] > 50 and row['predicted_ddn_amt'] < 1:
        return '성장형 ETF'
    elif row['predicted_ddn_amt'] > 1:
        return '배당형 ETF'
    else:
        return '혼합형 ETF'

# 새로운 열을 추가하여 ETF 유형 구분
merged_data['etf_type'] = merged_data.apply(categorize_etf, axis=1)

# 1. 성장형 ETF에 가중 평균 적용 (80% 수익률, 20% 배당금)
merged_data['combined_score_growth'] = np.where(
    merged_data['etf_type'] == '성장형 ETF',
    0.8 * merged_data['yr1_tot_pft_rt'] + 0.2 * merged_data['ddn_amt'],
    np.nan  # 다른 유형은 계산하지 않음
)

# 2. 배당형 ETF에 가중 평균 적용 (30% 수익률, 70% 배당금)
merged_data['combined_score_dividend'] = np.where(
    merged_data['etf_type'] == '배당형 ETF',
    0.3 * merged_data['yr1_tot_pft_rt'] + 0.7 * merged_data['ddn_amt'],
    np.nan  # 다른 유형은 계산하지 않음
)

# 3. 혼합형 ETF에 가중 평균 적용 (50% 수익률, 50% 배당금)
merged_data['combined_score_mixed'] = np.where(
    merged_data['etf_type'] == '혼합형 ETF',
    0.5 * merged_data['yr1_tot_pft_rt'] + 0.5 * merged_data['ddn_amt'],
    np.nan  # 다른 유형은 계산하지 않음
)

# 각 ETF 코드별로 성장형, 배당형, 혼합형에 대해 combined_score가 가장 높은 항목을 선택
top_etfs_growth = merged_data.loc[merged_data.groupby('etf_tck_cd')['combined_score_growth'].idxmax().dropna()]
top_etfs_dividend = merged_data.loc[merged_data.groupby('etf_tck_cd')['combined_score_dividend'].idxmax().dropna()]
top_etfs_mixed = merged_data.loc[merged_data.groupby('etf_tck_cd')['combined_score_mixed'].idxmax().dropna()]

# 3개의 결과를 모두 합친 후, 중복 제거
all_etfs_combined = pd.concat([top_etfs_growth, top_etfs_dividend, top_etfs_mixed]).drop_duplicates(subset='etf_tck_cd')

# 최종적으로 상위 5개의 ETF 추출 (combined_score를 통합하여 가장 큰 값 기준으로)
all_etfs_combined['final_combined_score'] = all_etfs_combined[['combined_score_growth', 'combined_score_dividend', 'combined_score_mixed']].max(axis=1)

# 상위 5개의 ETF 추출
top_5_final_etfs = all_etfs_combined.nlargest(5, 'predicted_ddn_amt')

# 결과 출력
print("Top 5 ETFs based on combined score across all ETF types:")
print(top_5_final_etfs[['etf_tck_cd', 'yr1_tot_pft_rt', 'ddn_amt', 'predicted_ddn_amt']])


# 1. 중복된 ETF_TCK_CD 제거 (첫 번째 값만 유지)
unique_etf_data = merged_data.drop_duplicates(subset='etf_tck_cd', keep='first')

# 2. 실제 배당금과 예측 배당금 비교 (중복 제거 후)
actual = unique_etf_data['ddn_amt']
predicted = unique_etf_data['predicted_ddn_amt']

# 3. 평균 절대 오차 (MAE) - 중복 제거 후
mae = mean_absolute_error(actual, predicted)
print(f'Mean Absolute Error (MAE) after removing duplicates: {mae}')

# 4. 평균 제곱 오차 (MSE) - 중복 제거 후
mse = mean_squared_error(actual, predicted)
print(f'Mean Squared Error (MSE) after removing duplicates: {mse}')

# 5. 루트 평균 제곱 오차 (RMSE) - 중복 제거 후
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error (RMSE) after removing duplicates: {rmse}')

# 6. 중복 제거 후 실제 값과 예측 값 비교 (상위 5개)
comparison_df = pd.DataFrame({
    'ETF_TCK_CD': unique_etf_data['etf_tck_cd'],
    'Actual_Dividend': actual,
    'Predicted_Dividend': predicted
})

# 예측값과 실제값의 차이 계산 (절대값)
comparison_df['Difference'] = abs(comparison_df['Actual_Dividend'] - comparison_df['Predicted_Dividend'])

# 간극이 작은 것 상위 5개
smallest_differences = comparison_df.nsmallest(5, 'Difference')

# 간극이 큰 것 상위 5개
largest_differences = comparison_df.nlargest(5, 'Difference')

# 결과 출력
print("Top 5 ETFs with smallest difference between Actual and Predicted Dividends:")
print(smallest_differences)

print("\nTop 5 ETFs with largest difference between Actual and Predicted Dividends:")
print(largest_differences)