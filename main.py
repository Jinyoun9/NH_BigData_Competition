import json
import boto3
import streamlit as st
import base64  
from finance_index import *

# AWS Bedrock 클라이언트 설정
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

# AI에 요청하여 ETF 추천을 받는 함수
def get_etf_predictions(tic1, tic2, tic3):
    etf_tickers, etf_infos = get_index_info(tic1, tic2, tic3)  # ETF 정보 가져오기
    indicator_history = get_economic_indicators()  # 경제 지표 정보 가져오기
    
    prompt = "These are the ETFs and their current information:\n\n"
    
    for etf, info in zip(etf_tickers, etf_infos):
        prompt += f"{etf} - {info.get('longBusinessSummary', 'No summary available')}\n"
    
    prompt += "\nThese are some relevant economic indicators:\n\n"
    
    # 경제 지표 정보를 텍스트로 추가
    for name, hist in indicator_history:
        prompt += f"\n{name}:\n"
        prompt += hist.tail(100).to_string()  # 최근 100일간의 데이터를 텍스트로 추가
    
    prompt += """
    너가 만약 워렌 버핏과 같은 가치 투자자라면 고객에게 어떤 ETF를 추천해주겠어? 
    다음 조건을 충족하는 대답을 해줘.
    1. 제시한 경제 지수 데이터를 기반으로 할 것
    2. 제시한 ETF 데이터를 기반으로 할 것
    3. 경제 지수와 ETF의 특성을 연관지어 근거로 설명할 것.
    4. 대답은 워렌 버핏과 비슷한 나이대의 말투로 할 것.
    5. 한국어로 할 것.
    """

    # AI에 요청하여 미래 성과 예측
    response = get_response(prompt)
    return response

# get_response 함수 정의
def get_response(prompt, image_data=None):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ]

        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_message = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_base64,
                }
            }
            messages[0]["content"].insert(0, image_message)

        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages,
            }
        )

        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-opus-20240229-v1:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response.get("body").read())
        output_text = response_body["content"][0]["text"]
        return output_text
    except Exception as e:
        print(e)

# 웹 앱 제목 설정
st.title("Chatbot powered by Bedrock")

# ETF 티커 입력 폼
with st.form("etf_form"):
    ticker1 = st.text_input("Enter the first ETF ticker:")
    ticker2 = st.text_input("Enter the second ETF ticker:")
    ticker3 = st.text_input("Enter the third ETF ticker:")
    submit_button = st.form_submit_button("Submit")  # 폼 제출 버튼

# 세션 상태에 메시지 없으면 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 폼 제출 버튼이 클릭되면
if submit_button:
    if ticker1 and ticker2 and ticker3:
        # ETF 예측을 위한 AI 요청
        etf_predictions = get_etf_predictions(ticker1, ticker2, ticker3)
        # AI 응답을 세션 상태에 추가하고 표시
        st.session_state.messages.append({"role": "assistant", "content": etf_predictions})
        with st.chat_message("assistant"):
            st.markdown(etf_predictions)

# 세션 상태에 저장된 메시지 순회하며 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 채팅 입력 폼 (ETF 예측과는 별도로, 추가적으로 대화할 때)
if prompt := st.chat_input("Message Bedrock..."):
    # 사용자 메시지를 세션 상태에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 다른 입력에 대한 AI 요청 (여기서는 폼과 별도로 사용)
    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
