from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 질문 추론 및 답변 위해 추가
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.chat import ChatMessage


from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, List

import unicodedata
import os
from dotenv import load_dotenv

import streamlit as st

# 환경변수 로드
load_dotenv()

st.title("INTERNET 지침 검색")

# 처음 1번만 실행하기 위한 용도
if "messages" not in st.session_state:
    # 대화 내용을 저장하기 위한 용도
    st.session_state["messages"] = []

with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")


def load_documents():
    """
    @function       load_documents
    @description    모든 md 파일 로드 및 전처리
    """
    markdown_dir = Path("processed")
    md_file_paths = list(markdown_dir.glob("*.md"))

    all_docs = []
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("####", "subsection"),
            ("#####", "detail"),
        ]
    )

    for path in md_file_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            header_chunks = splitter.split_text(text)

            # 각 chunk에 대해 다시 문장 기준으로 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=20
            )
            for doc in header_chunks:
                # 🔽 metadata 정보를 page_content에 prepend
                meta = doc.metadata
                meta_str = f"{meta.get('title', '')} {meta.get('section', '')} {meta.get('subsection', '')} {meta.get('detail', '')}"
                doc.page_content = meta_str + "\n\n" + doc.page_content

                split_docs = text_splitter.split_documents([doc])
                for sd in split_docs:
                    sd.metadata["source_file"] = unicodedata.normalize(
                        "NFC", str(path.name)
                    )
                all_docs.extend(split_docs)

    print(f"===== 문서 총 {len(all_docs)}개 chunk 생성 완료")
    return all_docs


def get_vectorstore(embedding, all_docs) -> FAISS:
    """
    @function    get_vectorstore
    @description FAISS 벡터db가 없으면 만들고 있으면 가져와서 쓰기
    """
    if Path("db/faiss_index").exists():
        vectordb = FAISS.load_local(
            "db/faiss_index", embedding, allow_dangerous_deserialization=True
        )
    else:
        vectordb = FAISS.from_documents(all_docs, embedding)
        vectordb.save_local("db/faiss_index")
    return vectordb


# @st.cache_resource는 Streamlit이 비싼 연산 결과(모델, 체인 등)를 한 번만 실행하고 캐싱해줘서
# 같은 함수가 다시 호출될 때는 결과를 재사용해.
# 예를 들어 LangChain 체인을 매번 새로 만들면:
# 문서 로딩, 벡터 임베딩, FAISS 로딩 등 시간 오래 걸리고
# 쓸데없이 API 호출해서 비용 날아가고
# 응답 속도도 느려져
@st.cache_resource
def create_qa_chain():
    """
    @function    create_qa_chain
    @description 체인 생성하기
    """
    question = "결제안심 인터넷 가입 가능 대상 고객 종류 알려줘"
    all_docs = load_documents()
    embedding = OpenAIEmbeddings()
    vectordb = get_vectorstore(embedding, all_docs)
    retriever = vectordb.as_retriever()  # 추론된 카테고리로 문서 필터
    llm = ChatOpenAI(model="gpt-4")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # 또는 "map_reduce", "refine" 등
        return_source_documents=True,
    )
    return qa_chain


@st.cache_resource
def create_streaming_chain():
    all_docs = load_documents()
    embedding = OpenAIEmbeddings()
    vectordb = get_vectorstore(embedding, all_docs)
    retriever = vectordb.as_retriever()
    llm = ChatOpenAI(model="gpt-4", streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "아래 문서를 참고하여 질문에 친절하고 간결하게 답변해주세요.\n\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    chain = (
        {
            "question": lambda x: x["question"],
            "context": lambda x: "\n\n".join(
                [
                    doc.page_content
                    for doc in retriever.get_relevant_documents(x["question"])
                ]
            ),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# class QAState(TypedDict):
#     """
#     @class       QAState
#     @description 상태 변수 설정 클래스
#     """

#     question: str
#     answer: str
#     source_documents: List[str]
#     feedback: str


# 2. 질문 입력 노드
# def ask_question(state: QAState) -> QAState:
#     print(f"\n👤 사용자 질문: {state['question']}")
#     return state


# # 3. 검색 + 답변 노드
# def retrieve_and_answer(state: QAState) -> QAState:
#     result = qa_chain.invoke({"query": state["question"]})
#     return {
#         "question": state["question"],
#         "answer": result["result"],
#         "source_documents": result["source_documents"],
#         "feedback": "",
#     }


# 4. 피드백 받는 노드
# def show_answer_and_get_feedback(state: QAState) -> QAState:
#     print("\n🤖 GPT 답변:")
#     print(state["answer"])
#     feedback = input("\n👉 이 답변 괜찮아? (y/n): ").strip().lower()
#     state["feedback"] = feedback
#     return state


# 5. 흐름 분기 조건
# def decide_next_step(state: QAState) -> str:
#     return "regenerate" if state["feedback"] == "n" else END


# 6. LangGraph 정의
# graph = StateGraph(QAState)

# # 노드 생성
# graph.add_node("ask", RunnableLambda(ask_question))
# graph.add_node("generate_answer", RunnableLambda(retrieve_and_answer))
# graph.add_node("feedback", RunnableLambda(show_answer_and_get_feedback))

# # 그래프 생성
# graph.set_entry_point("ask")
# graph.add_edge("ask", "generate_answer")
# graph.add_edge("generate_answer", "feedback")

# # 조건부 생성
# graph.add_conditional_edges(
#     "feedback", decide_next_step, {"regenerate": "generate_answer", END: END}
# )

# app = graph.compile()

# 7. 실행
# init_state = {
#     "question": question,
#     "answer": "",
#     "source_documents": [],
#     "feedback": "",
# }

# app.invoke(init_state)


def add_message(role, message):
    """
    @function    add_message
    @description 새로운 메세지를 추가
    """
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def print_messages():
    """
    @function    print_messages
    @description 이전 대화 출력
    """
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def handle_input(user_input):
    """
    @function    handle_input
    @description 사용자 입력 처리 및 응답 출력
    """
    st.chat_message("user").write(user_input)

    # QAChain 방식
    # qa_chain = create_qa_chain()
    # response = qa_chain.invoke({"query": user_input})

    # ai_answer = response["result"]
    # st.chat_message("assistant").markdown(ai_answer)

    # stream Chain 방식
    chain = create_streaming_chain()
    response = chain.stream({"question": user_input})

    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for chunk in response:
            ai_answer += chunk
            container.markdown(ai_answer)

    # 대화 기록 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)


def main():
    """
    @function    main
    @description 프로그램 실행
    """
    # 초기화 버튼 처리
    if clear_btn:
        st.session_state["messages"] = []
    else:
        print_messages()

    # 사용자 입력 받기
    user_input = st.chat_input("궁금한 내용을 물어보세요")

    # 입력 처리
    if user_input:
        handle_input(user_input)


if __name__ == "__main__":

    # @function    __main__
    # @description 프로그램 실행
    main()
    # 랭그래프로 변경해야 함. zzonetest_remove_cate.py 보고 메모리 저장되게 변경하기
