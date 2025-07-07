# 환경변수
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import streamlit as st
from chromadb import PersistentClient

# 환경변수 로드
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

#OpenAI 환경설정
def get_openai_client(messages):
    # OpenAI API 호출 예시
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.4
        )
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return f"Error: {e}"

# chroma_db에서 유사 문단 검색 함수
AZURE_EMBEDDING_API_KEY = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_API_VERSION = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("TEXT_EMBEDDING_DEPLOYMENT_NAME")

# 임베딩 생성 함수 (Azure OpenAI)
def get_query_embedding(query):
    embedding_client = AzureOpenAI(
        api_key=AZURE_EMBEDDING_API_KEY,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_version=AZURE_EMBEDDING_API_VERSION
    )
    response = embedding_client.embeddings.create(
        input=query,
        model=EMBEDDING_DEPLOYMENT_NAME
    )
    return response.data[0].embedding

def search_chroma_db(query, persist_dir="/Users/minho/Desktop/mvp_project_kmh/chroma_db", top_k=3):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    query_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []

# Streamlit UI 설정
st.title("Azure OpenAI Chat Interface")
st.write("Azure OpenAI API를 사용한 모델과 대화해 보세요 ㅎㅎ^^")

# 채팅 기록의 초기화 및 리셋 버튼 추가
if 'messages' not in st.session_state:
    st.session_state.messages = []

if st.button('채팅 기록 초기화'):
    st.session_state.messages = []
    st.rerun()

# 채팅 메시지 표시 (system 메시지는 표시하지 않음)
for message in st.session_state.messages:
    if message["role"] != "system":
        st.chat_message(message["role"]).write(message["content"])

# 사용자 입력 받기
if user_input := st.chat_input("메시지를 입력하세요:"):
    # 기존 system 메시지 제거
    st.session_state.messages = [m for m in st.session_state.messages if m["role"] != "system"]
    # chroma_db에서 유사 문단 검색
    similar_chunks = search_chroma_db(user_input)
    if similar_chunks:
        context_text = "\n\n".join(similar_chunks)
        system_prompt = context_text
        # system 메시지는 세션에만 추가
        st.session_state.messages.append({"role": "system", "content": system_prompt})
        st.session_state.messages.append({"role": "user", "content": user_input})
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# 답변 생성
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.spinner("응답 생성 중..."):
        response = get_openai_client(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()