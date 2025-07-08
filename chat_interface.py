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

def search_chroma_db(query, persist_dir="./chroma_db", top_k=3):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    query_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []

# Streamlit UI 설정
st.set_page_config(layout="centered")
st.markdown("""
<style>
    .block-container {
        max-width: 1000px !important;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

title_col, reset_col = st.columns([8, 1])
with title_col:
    st.markdown("<div style='display:flex; align-items:center; gap:12px;'>"
                "<span style='font-size:2.1rem; font-weight:700;'>K-ICIS 오더 VOC 전문 상담 챗봇</span>"
                "</div>", unsafe_allow_html=True)
with reset_col:
    pass  # 상단에서 초기화 버튼 제거

col1, col2 = st.columns([1, 4], gap="small")

# 세션 상태 초기화 (항상 보장)
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with col1:
    # 초기화 버튼 영역
    with st.container():
        st.markdown("""
        <style>
        .reset_button {
        width: 100%;
        border: 2px solid #1976d2;
        border-radius: 12px;
        padding: 12px 8px 8px 8px;
        background: #f5f7fa;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif';
        font-size: 17px;
        letter-spacing: -0.01em;
        }
        .reset_button:hover {
        opacity: 0.8;
        }
        </style>
        """, unsafe_allow_html=True)
        if st.button('초기화', key='reset_chat_col1', use_container_width=True):
            st.session_state['messages'] = []
            st.session_state['pdf_applied'] = False
            st.rerun()

    # 파일첨부 영역
    with st.container():
        st.markdown("""        
        <div style="border:2px dashed #1976d2; border-radius:10px; padding:32px 8px; background:#f7fafd; text-align:center;">
            <b>📎 파일 첨부</b><br><br>
            <span style="color:#888; font-size:14px;">여기로 PDF 파일을 드래그하거나 클릭하여 업로드하세요.</span>
        </div>
        """, unsafe_allow_html=True)
        uploaded_pdf = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            st.success(f"업로드된 파일: {uploaded_pdf.name}")
            if 'pdf_applied' not in st.session_state:
                st.session_state['pdf_applied'] = False
            if not st.session_state['pdf_applied']:
                if st.button("적용", key="apply_pdf"):
                    temp_path = f"/tmp/{uploaded_pdf.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_pdf.getbuffer())
                    try:
                        from pdf_to_vectordb import extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma
                        text = extract_text_from_pdf(temp_path)
                        chunks = split_text(text)
                        embeddings = get_azure_embeddings(chunks)
                        save_to_chroma(chunks, embeddings, persist_dir="./chroma_db", pdf_path=temp_path)
                        st.session_state['pdf_applied'] = True
                        st.success("PDF가 벡터 DB에 성공적으로 적용되었습니다!")
                    except Exception as e:
                        st.error(f"PDF 벡터화 중 오류 발생: {e}")
            else:
                st.success("PDF가 벡터 DB에 성공적으로 적용되었습니다!")

with col2:
    # 채팅 메시지 영역 (고정 높이, 스크롤, 가로폭 900px)
    chat_html = '''
    <div id="chat-area" style="max-width: 765px;
        min-width: 765px;
        max-height: 500px;
        min-height: 500px;
        overflow-y: auto;
        border: 2px solid #1976d2;
        border-radius: 12px;
        padding: 12px 8px 8px 8px;
        background: #f5f7fa;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif';
        font-size: 17px;
        letter-spacing: -0.01em;">
    '''
    
    for message in st.session_state.get('messages', []):
        if message["role"] == "user":
            chat_html += f"<div style='text-align:right; margin:8px 0;'><span style='display:inline-block; background:#e6f0ff; color:#222; padding:8px 14px; border-radius:16px 16px 2px 16px;'>{message['content']}</span></div>"
        elif message["role"] == "assistant":
            chat_html += f"<div style='text-align:left; margin:8px 0;'><span style='display:inline-block; background:#f3f3f3; color:#222; padding:8px 14px; border-radius:16px 16px 16px 2px;'>{message['content']}</span></div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)
    
    user_input = st.chat_input("메시지를 입력하세요:")
    if user_input:
        # 기존 system 메시지 제거
        st.session_state.messages = [m for m in st.session_state.messages if m["role"] != "system"]
        # chroma_db에서 유사 문단 검색
        similar_chunks = search_chroma_db(user_input)
        if similar_chunks:
            context_text = "\n\n".join(similar_chunks)
            system_prompt = context_text
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
