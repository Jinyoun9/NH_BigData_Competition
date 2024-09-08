import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 1. 데이터 로드 및 전처리
performance_data = pd.read_csv('NH_CONTEST_ETF_SOR_IFO.csv')
stk_data = pd.read_csv('NH_CONTEST_NW_FC_STK_IEM_IFO.csv', encoding='cp949')
dividend_data = pd.read_csv('NH_CONTEST_DATA_HISTORICAL_DIVIDEND.csv')

# 2. 필요한 컬럼만 추출
performance_data = performance_data[['etf_iem_cd', 'mm1_tot_pft_rt', 'mm3_tot_pft_rt', 'yr1_tot_pft_rt', 
                                     'shpr_z_sor', 'vty_z_sor', 'mxdd_z_sor', 'crr_z_sor', 'trk_err_z_sor', 'bse_dt']]
stk_data = stk_data[['tck_iem_cd', 'mkt_pr_tot_amt', 'ser_cfc_nm', 'ids_nm']]
dividend_data = dividend_data[['etf_tck_cd', 'ddn_amt', 'ediv_dt', 'ddn_pym_fcy_cd']]

# 3. 열 이름 통일
stk_data.rename(columns={'tck_iem_cd': 'etf_tck_cd'}, inplace=True)
performance_data.rename(columns={'etf_iem_cd': 'etf_tck_cd'}, inplace=True)

# 4. 데이터의 'etf_tck_cd' 값 대소문자 통일 및 공백 제거
performance_data['etf_tck_cd'] = performance_data['etf_tck_cd'].str.strip().str.upper()
stk_data['etf_tck_cd'] = stk_data['etf_tck_cd'].str.strip().str.upper()
dividend_data['etf_tck_cd'] = dividend_data['etf_tck_cd'].str.strip().str.upper()

# 5. 'bse_dt'를 datetime 형식으로 변환
performance_data['bse_dt'] = pd.to_datetime(performance_data['bse_dt'], format='%Y%m%d')

# 6. 과거 데이터를 고려한 가중 평균 계산
performance_data['days_from_recent'] = (performance_data['bse_dt'].max() - performance_data['bse_dt']).dt.days
performance_data['weight'] = np.exp(-0.01 * performance_data['days_from_recent'])

# 가중 평균을 계산할 지표
weighted_columns = ['yr1_tot_pft_rt', 'shpr_z_sor', 'vty_z_sor', 'mxdd_z_sor']

# 7. 가중 평균 계산
weighted_avg = performance_data.groupby('etf_tck_cd', group_keys=False).apply(
    lambda x: pd.Series({
        col: np.average(x[col], weights=x['weight']) for col in weighted_columns
    })
).reset_index()

# 중간 데이터 확인 (필수)
print("Weighted Average Data:")
print(weighted_avg.head())

# 8. 데이터 병합
merged_data = pd.merge(dividend_data, weighted_avg, on='etf_tck_cd', how='inner')
final_data = pd.merge(merged_data, stk_data, on='etf_tck_cd', how='inner')

# 9. 리스크 성향을 결정하는 타겟 변수 생성
final_data['risk_preference'] = (
    (final_data['shpr_z_sor'] * (-1) + final_data['vty_z_sor'] + final_data['mxdd_z_sor'] > 
     final_data['shpr_z_sor'].mean() * (-0.5) + final_data['vty_z_sor'].mean() + final_data['mxdd_z_sor'].mean())
).astype(float)

final_data['stock_type'] = (
    (final_data['yr1_tot_pft_rt'] / final_data['mkt_pr_tot_amt'] > 
     (final_data['yr1_tot_pft_rt'] / final_data['mkt_pr_tot_amt']).quantile(0.8))
).astype(float)

# 10. risk_preference와 stock_type을 결합한 새로운 타겟 변수 생성
# 위험형 성장주(1), 안전형 가치주(0)
final_data['risk_growth_combined'] = (
    (final_data['risk_preference'] > 0.65) | (final_data['stock_type'] > 0.65)
).astype(int)

final_data['risk_return_score'] = (final_data['shpr_z_sor'] * (-0.5)+
                            final_data['vty_z_sor'] * 0.3  +
                            final_data['mxdd_z_sor'] * 0.2  +
                            final_data['yr1_tot_pft_rt'] * 0.5)

# 타겟 변수 분포 확인
print("Target Class Distribution:")
print(final_data['risk_growth_combined'].value_counts())

X_combined = final_data[['yr1_tot_pft_rt', 'shpr_z_sor', 'vty_z_sor', 'mxdd_z_sor']]  # 입력 데이터
y_combined = final_data['risk_growth_combined']  # 결합된 타겟 변수

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# 11. 저장된 모델 불러오기 (파일 이름을 'portfolio_model_optimized.joblib'로 지정)
clf_loaded = joblib.load('portfolio_model_optimized.joblib')

# 12. 불러온 모델을 사용하여 예측
y_pred_loaded = clf_loaded.predict(X_test_combined)

# 13. 모델 성능 평가 (정확도, F1 스코어)
accuracy_loaded = accuracy_score(y_test_combined, y_pred_loaded)
f1_loaded = f1_score(y_test_combined, y_pred_loaded, pos_label=1, zero_division=1)

print(f"Loaded Model Test Accuracy: {accuracy_loaded}")
print(f"Loaded Model Test F1 Score: {f1_loaded}")

# 14. 혼동 행렬 시각화
cm_loaded = confusion_matrix(y_test_combined, y_pred_loaded)

plt.figure(figsize=(6, 4))
sns.heatmap(cm_loaded, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Safe (0)', 'Risk-Taking (1)'], 
            yticklabels=['Safe (0)', 'Risk-Taking (1)'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Loaded Model')
plt.show()

# 15. 분류 리포트 출력 (Precision, Recall, F1 Score 등)
print("\nClassification Report:")
print(classification_report(y_test_combined, y_pred_loaded, target_names=['Safe (0)', 'Risk-Taking (1)']))

# 16. StratifiedKFold를 이용한 교차 검증
skf = StratifiedKFold(n_splits=5)

# F1 스코어를 위한 커스텀 스코어러 생성
f1_scorer = make_scorer(f1_score, pos_label=1, zero_division=1)

# 교차 검증 수행 (F1 Score 및 Accuracy 평가)
f1_scores = cross_val_score(clf_loaded, X_combined, y_combined, cv=skf, scoring=f1_scorer)
accuracy_scores = cross_val_score(clf_loaded, X_combined, y_combined, cv=skf, scoring='accuracy')

# 교차 검증 결과 출력
print(f"Cross-Validation F1 Scores: {f1_scores}")
print(f"Mean F1 Score: {np.mean(f1_scores)}")
print(f"Cross-Validation Accuracy Scores: {accuracy_scores}")
print(f"Mean Accuracy: {np.mean(accuracy_scores)}")

# 2. 성장주 5개 추천 (겹치지 않게, risk_return_score 기준 상위 5개)
growth_stocks = final_data[(final_data['risk_growth_combined'] == 1)]
growth_stocks_recommendations = growth_stocks[['etf_tck_cd', 'yr1_tot_pft_rt', 'risk_return_score']]\
    .drop_duplicates()\
    .sort_values(by='risk_return_score', ascending=False)\
    .head(5)

print("\nTop 5 Growth Stocks Recommendations (Based on Risk-Return Score):")
print(growth_stocks_recommendations)

# 3. 가치주 5개 추천 (겹치지 않게, risk_return_score 기준 상위 5개)
value_stocks = final_data[(final_data['risk_growth_combined'] == 0)]
value_stocks_recommendations = value_stocks[['etf_tck_cd', 'yr1_tot_pft_rt', 'risk_return_score']]\
    .drop_duplicates()\
    .sort_values(by='risk_return_score', ascending=False)\
    .head(5)

print("\nTop 5 Value Stocks Recommendations (Based on Risk-Return Score):")
print(value_stocks_recommendations)