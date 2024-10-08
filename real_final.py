import requests
import google.generativeai as genai
import numpy as np
import yfinance as yf
from IPython.display import display, Markdown
import random  # 무작위 점수 생성을 위해 추가

# Gemini API 키 설정
genai.configure(api_key='AIzaSyCK1V7X1xWcynJGp-Ke98G_kd_pZse0f7I')

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# ETF 및 주식 티커 목록
etf_tickers = ['SPY', 'XLF', 'QQQ', 'IWM', 'EEM', 'TZA', 'VWO', 'EFA', 'FXI', 
               'SDS', 'XLI', 'GDX', 'EWZ', 'FAZ', 'XLE', 'EWJ', 'TNA', 'SLV', 
               'UNG', 'XLK', 'GLD', 'UVXY', 'XLB', 'XHB', 'XLY', 'XRT', 'FAS']

# 더미 포트폴리오 데이터
portfolio = [
    {'TCK_CD': 'SPY', 'Value': 1000},  # ETF
    {'TCK_CD': 'AAPL', 'Value': 2000}   # 주식
]

# 사용자 입력: 목표하는 연간 수익률, 변동성, 총 매수할 금액
target_return = float(input("목표하는 연간 수익률을 입력하세요 (예: 0.10 = 10%): "))
max_volatility = float(input("최대 감수할 수 있는 연간 변동성을 입력하세요 (예: 0.15 = 15%): "))
total_investment = float(input("총 매수할 금액을 입력하세요 (예: 5000USD): "))

def calculate_annual_performance(ticker):
    data = yf.download(ticker, start='2023-09-24', end='2024-09-24')['Adj Close']
    daily_returns = data.pct_change().dropna()
    annual_return = daily_returns.mean() * 252
    annual_volatility = daily_returns.std() * (252 ** 0.5)
    return annual_return, annual_volatility

# ETF 및 주식 데이터 가져오기
def fetch_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_expected_returns_and_covariance(data):
    returns = data.pct_change().dropna()
    expected_returns = returns.mean().values
    cov_matrix = returns.cov().values
    return expected_returns, cov_matrix

# 포트폴리오 성과 계산
def calculate_portfolio_performance(weights, expected_returns, cov_matrix):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_risk

# ETF 후보의 연간 수익률 및 변동성 계산
def get_etf_candidates_performance(etf_tickers):
    etf_candidates = []
    for ticker in etf_tickers:
        annual_return, annual_volatility = calculate_annual_performance(ticker)
        etf_candidates.append((ticker, annual_return, annual_volatility))
    return etf_candidates

# Gemini 추천 사항 받기
def get_gemini_recommendations(portfolio, etf_candidates, total_investment):
    portfolio_summary = '\n'.join([f"{p['TCK_CD']}: {p['Value']} USD" for p in portfolio])
    etf_summary = '\n'.join([f"{etf[0]}: 연간 수익률 {etf[1]:.2%}, 변동성 {etf[2]:.2%}" for etf in etf_candidates])

    total_value = sum(p['Value'] for p in portfolio) + total_investment

    prompt = f"""
    현재 포트폴리오에 대한 할당량:
    {portfolio_summary}

    다음 ETF 후보들을 고려하여:
    {etf_summary}

    포트폴리오 리밸런싱을 위해 전체 금액이 {total_value:.2f} USD를 넘지 않도록 하면서 %의 합은 100%이어야 하고,
    목표 수익률은 {target_return:.2%} 이상, 최대 변동성은 {max_volatility:.2%} 이하가 되도록,
    매수 및 매도의 비율을 포함하여 어떤 ETF를 얼마나 매수하고 얼마나 매도해야 하는지에 대한
    권장 사항과 포트폴리오의 정보 요약 및 연간 수익율, 연간 변동성 그리고 포트폴리오에 대한 설명을 제공하십시오.
    """
    
    response = chat.send_message(prompt)
    
    if response and response.candidates:
        recommendations = response.candidates[0].content.parts[0].text
        display(Markdown(recommendations))
        print(recommendations)
        return recommendations
    
    return {}

# 포트폴리오 평가 수행
def perform_portfolio_evaluation(portfolio):
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    # ETF와 주식의 티커 목록 생성
    all_tickers = [p['TCK_CD'] for p in portfolio] + etf_tickers
    data = fetch_data(all_tickers, start_date, end_date)

    expected_returns, cov_matrix = calculate_expected_returns_and_covariance(data)

    total_value = sum(p['Value'] for p in portfolio)
    weights = np.array([p['Value'] / total_value for p in portfolio])

    portfolio_return, portfolio_risk = calculate_portfolio_performance(weights, expected_returns[:len(weights)], cov_matrix[:len(weights), :len(weights)])
    print(f"예상 포트폴리오 수익률: {portfolio_return:.2%}, 포트폴리오 위험 (표준편차): {portfolio_risk:.2%}")

    # ETF 후보 성과 계산
    etf_candidates = get_etf_candidates_performance(etf_tickers)
    target_allocation = get_gemini_recommendations(portfolio, etf_candidates, total_investment)
    return target_allocation

# 메인 실행
if __name__ == "__main__":
    # 포트폴리오 평가 수행
    perform_portfolio_evaluation(portfolio)
