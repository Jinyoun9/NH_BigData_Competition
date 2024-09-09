import yfinance as yf



def get_index_info(tic1, tic2, tic3):
    # ETF 티커 코드 목록 (원자재나 기술과 연관된 ETF)
    etf_tickers = [tic1, tic2, tic3]  # 여기에 원하는 ETF 티커를 추가
    etf_infos = []
    # 각 ETF에 대한 정보 가져오기
    for ticker in etf_tickers:
        etf = yf.Ticker(ticker)
        info = etf.info
        etf_infos.append(info)
        print(f"\n{ticker} ETF Info:")
        print(f"추종하는 자산/산업: {info.get('category', '알 수 없음')}")
        print(f"섹터: {info.get('sector', '알 수 없음')}")
        print(f"ETF의 목표/전략: {info.get('longBusinessSummary', '알 수 없음')}")
    
    return etf_tickers, etf_infos


def get_economic_indicators():
    economic_indicators = {
    "10-Year Treasury Yield": "^TNX",   # 10년 만기 미국 국채 금리
    "Gold": "GC=F",                    # 금 가격
    "Silver": "SI=F",                  # 은 가격
    "Crude Oil": "CL=F",               # 원유 가격 (서부 텍사스산 원유)
    "USD/EUR Exchange Rate": "EURUSD=X",  # 달러/유로 환율
    "USD/JPY Exchange Rate": "JPY=X",     # 달러/엔 환율
    "S&P 500": "^GSPC",                # S&P 500 지수
    "NASDAQ 100": "^NDX",              # NASDAQ 100 지수
    "Dow Jones": "^DJI",               # 다우존스 지수
    }
    indicator_history = []
    # 각 경제 지표의 데이터를 불러와서 출력
    for name, ticker in economic_indicators.items():
        data = yf.Ticker(ticker)
        hist = data.history(period="1y")  # 지난 1년간의 데이터
        #print(f"\n{name} ({ticker}) Historical Data:")
        #print(hist.head())  # 각 경제 지표의 데이터 일부 출력
        indicator_history.append((name, hist))
        
    return indicator_history
    