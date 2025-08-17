import streamlit as st
from chat_core import get_openai_client, search_all_content, get_agent_system_prompt, save_to_faiss
from pdf_to_vectordb import extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma

def main():
    # Streamlit UI 설정
    st.set_page_config(layout="centered", page_title="K-ICIS AI 에이전트")
    st.markdown("""
    <style>
        body, .block-container {
            background: linear-gradient(135deg, #e3f0ff 0%, #f7fafd 100%) !important;
        }
        .block-container {
            max-width: 1000px !important;
            margin-left: auto;
            margin-right: auto;
            background: #f7fafd !important;
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(25, 118, 210, 0.07);
            padding-bottom: 32px;
        }
        /* 타이틀 영역 */
        .stMarkdown > div[style*='display:flex'] {
            background: linear-gradient(90deg, #e3f0ff 60%, #f7fafd 100%);
            border-radius: 16px;
            padding: 18px 24px 12px 24px;
            margin-bottom: 8px;
            box-shadow: 0 2px 8px rgba(25, 118, 210, 0.06);
        }
        /* 에이전트 선택 영역 */
        .agent-selector {
            border: 2px solid #90caf9;
            border-radius: 14px;
            padding: 16px 12px;
            background: #e3f0ff;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(25, 118, 210, 0.04);
        }
        .agent-selector h4 {
            color: #1976d2;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
            text-align: center;
        }
        /* 초기화 버튼 */
        .reset_button {
            width: 100%;
            border: 2px solid #90caf9;
            border-radius: 12px;
            padding: 12px 8px 8px 8px;
            background: #e3f0ff;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(25, 118, 210, 0.04);
            font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif';
            font-size: 17px;
            letter-spacing: -0.01em;
            color: #1976d2;
            font-weight: 600;
            transition: background 0.2s, border 0.2s;
        }
        .reset_button:hover {
            background: #bbdefb;
            border-color: #1976d2;
            color: #1565c0;
            opacity: 0.95;
        }
        /* 파일 업로드 영역 */
        .pdf-upload-area {
            border:2px dashed #90caf9;
            border-radius:14px;
            padding:32px 8px;
            background: #e3f0ff;
            text-align:center;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(25, 118, 210, 0.04);
        }
        .pdf-upload-area b {
            color: #1976d2;
            font-size: 18px;
        }
        .pdf-upload-area span {
            color:#789;
            font-size:14px;
        }
        /* 채팅 영역 */
        #chat-area {
            max-width: 765px;
            min-width: 765px;
            max-height: 500px;
            min-height: 500px;
            overflow-y: auto;
            border: 2px solid #90caf9;
            border-radius: 16px;
            padding: 16px 12px 12px 12px;
            background: #f7fafd;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(25, 118, 210, 0.06);
            font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif';
            font-size: 17px;
            letter-spacing: -0.01em;
        }
        /* 채팅 버블 */
        .chat-bubble-user {
            display:inline-block;
            background:#e3f0ff;
            color:#1976d2;
            padding:8px 16px;
            border-radius:18px 18px 4px 18px;
            margin: 4px 0;
            font-weight: 500;
            box-shadow: 0 1px 4px rgba(25, 118, 210, 0.04);
        }
        .chat-bubble-assistant {
            display:inline-block;
            background:#f3f3f3;
            color:#333;
            padding:8px 16px;
            border-radius:18px 18px 18px 4px;
            margin: 4px 0;
            font-weight: 500;
            box-shadow: 0 1px 4px rgba(25, 118, 210, 0.04);
        }
        /* 입력창 */
        section[data-testid="stChatInput"] textarea {
            background: #e3f0ff;
            border: 1.5px solid #90caf9;
            border-radius: 10px;
            font-size: 17px;
            color: #1976d2;
            font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'sans-serif';
        }
        /* 적용 버튼 */
        button[kind="secondary"] {
            background: #e3f0ff !important;
            color: #1976d2 !important;
            border: 1.5px solid #90caf9 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        button[kind="secondary"]:hover {
            background: #bbdefb !important;
            color: #1565c0 !important;
            border-color: #1976d2 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    title_col, reset_col = st.columns([8, 1])
    with title_col:
        st.markdown("<div style='display:flex; align-items:center; gap:12px;'>"
                    "<span style='font-size:2.1rem; font-weight:700;'>K-ICIS AI 에이전트</span>"
                    "</div>", unsafe_allow_html=True)
    with reset_col:
        pass  # 상단에서 초기화 버튼 제거

    col1, col2 = st.columns([1, 4], gap="small")

    # 세션 상태 초기화 (항상 보장)
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'selected_agent' not in st.session_state:
        st.session_state['selected_agent'] = 'voc_agent'
    
    # 에이전트별 초기 메시지
    agent_messages = {
        'voc_agent': {"role": "assistant", "content": "안녕하세요! K-ICIS 오더 VOC 전문 상담 챗봇입니다. <br>고객 문의사항에 대해 도움을 드리겠습니다."},
        'knowledge_agent': {"role": "assistant", "content": "안녕하세요! K-ICIS 지식자산화 에이전트입니다. <br>문서 분석 및 지식 정리를 도와드리겠습니다."}
    }
    
    # 메시지가 비어있으면 선택된 에이전트의 초기 메시지 추가
    if not st.session_state['messages']:
        st.session_state['messages'] = [agent_messages[st.session_state['selected_agent']]]

    with col1:
        # 에이전트 선택 영역
        with st.container():
            st.markdown('''
            <div class="agent-selector">
                <h4>🤖 에이전트 선택</h4>
            </div>
            ''', unsafe_allow_html=True)
            
            # 에이전트 선택
            agent_options = {
                'voc_agent': '🎧 VOC 상담 챗봇',
                'knowledge_agent': '📚 지식자산화 에이전트'
            }
            
            selected_agent = st.radio(
                "",
                options=list(agent_options.keys()),
                format_func=lambda x: agent_options[x],
                key='agent_selector',
                label_visibility="collapsed"
            )
            
            # 에이전트가 변경되면 메시지 초기화
            if selected_agent != st.session_state.get('selected_agent'):
                st.session_state['selected_agent'] = selected_agent
                st.session_state['messages'] = [agent_messages[selected_agent]]
                st.session_state['pdf_applied'] = False
                st.rerun()
        
        # 초기화 버튼 영역
        with st.container():
            if st.button('초기화', key='reset_chat_col1', use_container_width=True):
                st.session_state['messages'] = []
                st.session_state['pdf_applied'] = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        # 파일첨부 영역
        with st.container():
            # 선택된 에이전트에 따른 파일 업로드 안내 메시지
            current_agent = st.session_state.get('selected_agent', 'voc_agent')
            if current_agent == 'voc_agent':
                upload_guide = "VOC 관련 문서나 매뉴얼을 업로드하세요."
            else:  # knowledge_agent
                upload_guide = "분석할 문서나 자료를 업로드하세요."
            
            st.markdown(f'''        
            <div class="pdf-upload-area">
                <b>📎 파일 첨부</b><br><br>
                <span>{upload_guide}</span>
            </div>
            ''', unsafe_allow_html=True)
            uploaded_pdf = st.file_uploader(" ", type=["pdf"], label_visibility="collapsed")
            if uploaded_pdf is not None:
                st.success(f"업로드된 파일: {uploaded_pdf.name}")
                # 파일명이 바뀌면 pdf_applied 자동 초기화
                if st.session_state.get('last_uploaded_pdf') != uploaded_pdf.name:
                    st.session_state['pdf_applied'] = False
                    st.session_state['last_uploaded_pdf'] = uploaded_pdf.name
                if 'pdf_applied' not in st.session_state:
                    st.session_state['pdf_applied'] = False
                if not st.session_state['pdf_applied']:
                    if st.button("적용", key="apply_pdf"):
                        temp_path = f"/tmp/{uploaded_pdf.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_pdf.getbuffer())
                        try:
                            text = extract_text_from_pdf(temp_path)
                            chunks = split_text(text)
                            embeddings = get_azure_embeddings(chunks)
                            
                            # 에이전트별 벡터 DB 저장
                            current_agent = st.session_state.get('selected_agent', 'voc_agent')
                            if current_agent == 'voc_agent':
                                # VOC 에이전트: ChromaDB 사용
                                success, message = save_to_chroma(chunks, embeddings, pdf_path=temp_path)
                                db_name = "ChromaDB"
                            else:  # knowledge_agent
                                # 지식자산화 에이전트: FAISS DB 사용
                                success, message = save_to_faiss(chunks, embeddings, pdf_path=temp_path)
                                db_name = "FAISS DB"
                            
                            if success:
                                st.session_state['pdf_applied'] = True
                                st.success(f"✅ PDF가 {db_name}에 성공적으로 적용되었습니다!\n📊 {message}")
                            else:
                                st.session_state['pdf_applied'] = True  # 이미 처리된 경우도 적용된 것으로 간주
                                st.info(f"ℹ️ {message}\n이미 처리된 파일이므로 추가 임베딩을 생략합니다.")
                        except Exception as e:
                            st.error(f"PDF 벡터화 중 오류 발생: {e}")
                elif st.session_state['pdf_applied']:
                    current_agent = st.session_state.get('selected_agent', 'voc_agent')
                    db_name = "ChromaDB" if current_agent == 'voc_agent' else "FAISS DB"
                    st.success(f"✅ PDF가 {db_name}에 성공적으로 적용되었습니다!")

    with col2:
        # 채팅 메시지 영역 (고정 높이, 스크롤, 가로폭 900px)
        chat_html = '''
        <div id="chat-area">
        '''
        for message in st.session_state.get('messages', []):
            if message["role"] == "user":
                chat_html += f"<div style='text-align:right; margin:8px 0;'><span class='chat-bubble-user'>{message['content']}</span></div>"
            elif message["role"] == "assistant":
                chat_html += f"<div style='text-align:left; margin:8px 0;'><span class='chat-bubble-assistant'>{message['content']}</span></div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        # 에이전트 상태 표시
        current_agent = st.session_state.get('selected_agent', 'voc_agent')
        agent_info = {
            'voc_agent': {'name': '🎧 VOC 상담 챗봇', 'color': '#1976d2', 'bg': '#e3f0ff'},
            'knowledge_agent': {'name': '📚 지식자산화 에이전트', 'color': '#7b1fa2', 'bg': '#f3e5f5'}
        }
        
        current_info = agent_info[current_agent]
        st.markdown(f"""
        <div style='background: {current_info['bg']}; border: 2px solid {current_info['color']}; 
                    border-radius: 10px; padding: 8px 12px; margin-bottom: 12px; text-align: center;'>
            <strong style='color: {current_info['color']};'>현재 활성: {current_info['name']}</strong>
        </div>
        """, unsafe_allow_html=True)

        user_input = st.chat_input("메시지를 입력하세요:")
        if user_input:
            st.session_state.messages = [m for m in st.session_state.messages if m["role"] != "system"]
            
            # 선택된 에이전트 정보 가져오기
            current_agent = st.session_state.get('selected_agent', 'voc_agent')
            
            # 통합 검색 (PDF만) - 에이전트별 맞춤 검색
            search_result = search_all_content(user_input, pdf_top_k=8, agent_type=current_agent)
            
            # 에이전트별 시스템 프롬프트 생성
            if search_result['context_text']:
                agent_system_prompt = get_agent_system_prompt(current_agent, search_result['context_text'])
                st.session_state.messages.append({"role": "system", "content": agent_system_prompt})
            else:
                # 검색 결과가 없어도 에이전트별 기본 시스템 프롬프트 추가
                agent_system_prompt = get_agent_system_prompt(current_agent, "")
                st.session_state.messages.append({"role": "system", "content": agent_system_prompt})
            
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

        # 답변 생성
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("응답 생성 중..."):
                response = get_openai_client(st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.rerun()

if __name__ == "__main__":
    main()