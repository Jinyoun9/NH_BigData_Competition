import pandas as pd
import yfinance as yf
import numpy as np

# 포트폴리오 입력 (배열 형태)
def input_portfolio():
    # 배열 형태로 포트폴리오를 직접 입력
    return [
        {'Ticker': 'SPY', 'Value': 1000, 'Weight': 0.4},
        {'Ticker': 'QQQ', 'Value': 3000, 'Weight': 0.3},
        {'Ticker': 'IVV', 'Value': 2000, 'Weight': 0.2},
        {'Ticker': 'IWM', 'Value': 500, 'Weight': 0.05},
        {'Ticker': 'VEA', 'Value': 500, 'Weight': 0.05},
    ]

# 예상 수익률 및 변동성 계산 함수
def calculate_performance(portfolio):
    # 모든 티커를 수집
    tickers = [item['Ticker'] for item in portfolio]
    
    # yfinance를 통해 데이터 다운로드
    data = yf.download(tickers, start='2023-09-24', end='2024-09-24')['Adj Close']
    
    # 일일 수익률 계산
    daily_returns = data.pct_change().dropna()
    
    # 포트폴리오의 예상 수익률 및 변동성 계산
    expected_returns = daily_returns.mean() * 252  # 연간 수익률
    annual_volatility = daily_returns.std() * (252**0.5)  # 연간 변동성

    # 포트폴리오 수익률 및 변동성 계산
    portfolio_return = sum(item['Weight'] * expected_returns[item['Ticker']] for item in portfolio)
    portfolio_volatility = np.sqrt(sum((item['Weight'] ** 2) * (annual_volatility[item['Ticker']] ** 2) for item in portfolio))

    return portfolio_return, portfolio_volatility

# 메인 프로그램 실행
if __name__ == "__main__":
    # 포트폴리오 배열을 입력
    user_portfolio = input_portfolio()
    
    # 포트폴리오 성과 계산
    portfolio_return, portfolio_volatility = calculate_performance(user_portfolio)
    
    # 결과 출력
    print(f"\n포트폴리오의 예상 연간 수익률: {portfolio_return:.2%}")
    print(f"포트폴리오의 연간 변동성: {portfolio_volatility:.2%}")
