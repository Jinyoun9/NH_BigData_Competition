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
        "10-Year Treasury Yield": "^TNX",    # 10년 만기 미국 국채 금리
        "Gold": "GC=F",                     # 금 가격
        "Silver": "SI=F",                   # 은 가격
        "Crude Oil": "CL=F",                # 원유 가격 (서부 텍사스산 원유)
        "Brent Crude Oil": "BZ=F",          # 브렌트유 가격
        "Copper": "HG=F",                   # 구리 가격
        "USD/EUR Exchange Rate": "EURUSD=X", # 달러/유로 환율
        "USD/JPY Exchange Rate": "JPY=X",    # 달러/엔 환율
        "S&P 500": "^GSPC",                 # S&P 500 지수
        "NASDAQ 100": "^NDX",               # NASDAQ 100 지수
        "Dow Jones": "^DJI",                # 다우존스 지수
        "Russell 2000": "^RUT",             # 러셀 2000 지수 (소형주 지수)
        "FTSE 100": "^FTSE",                # FTSE 100 (영국 지수)
        "DAX": "^GDAXI",                    # 독일 DAX 지수
        "Nikkei 225": "^N225",              # 일본 니케이 225 지수
        "Shanghai Composite": "000001.SS",  # 상하이 종합 지수
        "Bloomberg Commodity Index": "^BCOM",# 블룸버그 원자재 지수
        "S&P GSCI": "^SPGSCI",              # S&P 골드만삭스 원자재 지수
        "Dow Jones U.S. Real Estate": "IYR", # 미국 부동산 섹터
        "Technology Sector ETF (XLK)": "XLK",    # 기술 섹터 ETF
        "Global Energy Sector ETF (IXC)": "IXC", # 글로벌 에너지 섹터 ETF
        "Global Industrials Sector ETF (EXI)": "EXI", # 글로벌 산업재 섹터 ETF
        "Philadelphia Semiconductor": "^SOX"   # 필라델피아 반도체 지수
    }
    indicator_history = []
    # 각 경제 지표의 데이터를 불러와서 출력
    for name, ticker in economic_indicators.items():
        data = yf.Ticker(ticker)
        hist = data.history(period="5d")  # 지난 1년간의 데이터
        #print(f"\n{name} ({ticker}) Historical Data:")
        #print(hist.head())  # 각 경제 지표의 데이터 일부 출력
        indicator_history.append((name, hist))
        
    return indicator_history
