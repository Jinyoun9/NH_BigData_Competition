from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='41aea5c3290540008ad38df95ac5b448')


def get_news_headlines():
    # 특정 키워드에 대한 최신 뉴스 가져오기
    top_headlines = newsapi.get_top_headlines(language='en')
    headlines = []
    # 결과 출력
    for article in top_headlines['articles']:
        # 기사 제목과 출처를 딕셔너리로 저장
        headlines.append({
            'title': article['title'],
            'source': article['source']['name']
        })
    return headlines

