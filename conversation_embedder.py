import os
import time
from dotenv import load_dotenv
from openai import AzureOpenAI
from chromadb import PersistentClient

# 환경변수 로드
load_dotenv()

# Azure OpenAI 임베딩 환경변수
AZURE_EMBEDDING_API_KEY = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_KEY")
AZURE_EMBEDDING_ENDPOINT = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_API_VERSION = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT_NAME = os.getenv("TEXT_EMBEDDING_DEPLOYMENT_NAME")

# ChromaDB 저장 경로 (Azure Web App 호환)
def get_chroma_db_path():
    """
    Azure Web App 환경에 맞는 ChromaDB 경로를 반환합니다.
    Azure에서는 /home/site/wwwroot가 영구 저장소입니다.
    """
    # Azure Web App 환경 감지
    if os.getenv("WEBSITE_SITE_NAME"):  
        base_path = "/home/site/wwwroot/chroma_db"
        print(f"Azure Web App 환경 감지: {base_path}")
        return base_path
    else:
        # 로컬 개발 환경
        base_path = os.path.join(os.getcwd(), "chroma_db")
        print(f"로컬 개발 환경: {base_path}")
        return base_path

PERSIST_DIR = get_chroma_db_path()

# 대화 임베딩 생성 함수
def get_conversation_embedding(text):
    client = AzureOpenAI(
        api_key=AZURE_EMBEDDING_API_KEY,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_version=AZURE_EMBEDDING_API_VERSION
    )
    
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT_NAME
    )
    return response.data[0].embedding

# 대화 내용을 ChromaDB에 저장하는 함수
def save_conversation_to_chroma(user_message, assistant_message):
    """
    사용자 메시지와 AI 답변을 ChromaDB에 저장합니다.
    PDF와 같은 컬렉션(pdf_collection)에 저장됩니다.
    Azure Web App 환경에서도 안정적으로 작동합니다.
    """
    persist_dir = get_chroma_db_path()  # 동적 경로 사용
    
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(persist_dir, exist_ok=True)
        print(f"ChromaDB 저장 경로 확인/생성: {persist_dir}")
    except Exception as e:
        print(f"ChromaDB 디렉토리 생성 실패: {e}")
        # 실패해도 계속 진행 (이미 존재할 수 있음)
    
    try:
        client = PersistentClient(path=persist_dir)
        
        # PDF와 같은 컬렉션 사용
        collection = client.get_or_create_collection("pdf_collection")
        
        # 타임스탬프 생성
        ts = int(time.time())
        
        # 사용자 메시지 저장
        user_embedding = get_conversation_embedding(user_message)
        user_id = f"conversation_user_{ts}_{hash(user_message) % 10000}"
        
        collection.add(
            documents=[user_message],
            embeddings=[user_embedding],
            ids=[user_id],
            metadatas=[{
                "type": "conversation",
                "role": "user",
                "timestamp": ts,
                "source": "chat_history"
            }]
        )
        
        # AI 답변 저장
        assistant_embedding = get_conversation_embedding(assistant_message)
        assistant_id = f"conversation_assistant_{ts}_{hash(assistant_message) % 10000}"
        
        collection.add(
            documents=[assistant_message],
            embeddings=[assistant_embedding],
            ids=[assistant_id],
            metadatas=[{
                "type": "conversation",
                "role": "assistant",
                "timestamp": ts,
                "source": "chat_history",
                "related_user_id": user_id
            }]
        )
        
        print(f"대화 내용 저장 완료! (사용자: {user_id}, AI: {assistant_id})")
        
    except Exception as e:
        print(f"대화 내용 저장 중 오류: {e}")
        raise

# 대화 내용에서 유사한 내용 검색하는 함수
def search_conversation_history(query, top_k=3):
    """
    PDF와 같은 컬렉션에서 대화 기록만 검색합니다.
    Azure Web App 환경에서도 안정적으로 작동합니다.
    """
    persist_dir = get_chroma_db_path()  # 동적 경로 사용
    
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception as e:
        print(f"ChromaDB 디렉토리 생성 실패: {e}")
    
    try:
        client = PersistentClient(path=persist_dir)
        
        collection = client.get_or_create_collection("pdf_collection")

        # 컬렉션이 비어있는지 확인
        if collection.count() == 0:
            return []
        
        query_embedding = get_conversation_embedding(query)
        
        try:
            # 메타데이터 필터링으로 대화 기록만 검색 - 검색 범위 확대
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 6,  # 더 많이 가져와서 필터링 (3배 -> 6배)
                include=["documents", "metadatas"],
                where={"type": "conversation"}  # 대화 기록만 검색
            )
        except Exception as filter_error:
            print(f"메타데이터 필터링 실패, 전체 검색으로 fallback: {filter_error}")
            # 메타데이터 필터링이 실패하면 전체 검색 후 필터링
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 10, 150),  # 더 많이 가져와서 수동 필터링 (6배 -> 10배)
                include=["documents", "metadatas"]
            )
        
        if results["documents"] and results["documents"][0]:
            # 대화 기록만 수동으로 필터링하여 반환
            conversation_docs = []
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results["metadatas"] and results["metadatas"][0] else [{}] * len(docs)
            
            for doc, meta in zip(docs, metas):
                # 메타데이터가 None이거나 빈 딕셔너리일 경우 처리
                if meta and isinstance(meta, dict) and meta.get("type") == "conversation":
                    conversation_docs.append(doc)
                elif not meta or not isinstance(meta, dict):
                    # 메타데이터가 없는 경우, ID로 대화 기록인지 판단
                    continue  # 대화 기록이 아니라고 가정
                
                if len(conversation_docs) >= top_k:
                    break
            
            return conversation_docs
        else:
            return []
            
    except Exception as e:
        print(f"대화 기록 검색 중 오류: {e}")
        return []

# 대화 기록 통계 조회 함수
def get_conversation_stats():
    """
    PDF와 같은 컬렉션에서 대화 기록 통계를 조회합니다.
    Azure Web App 환경에서도 안정적으로 작동합니다.
    """
    persist_dir = get_chroma_db_path()  # 동적 경로 사용
    
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception as e:
        print(f"ChromaDB 디렉토리 생성 실패: {e}")
    
    try:
        client = PersistentClient(path=persist_dir)
        
        collection = client.get_or_create_collection("pdf_collection")
        total_count = collection.count()
        
        if total_count > 0:
            # 전체 데이터 조회 (오류 처리 강화)
            try:
                all_data = collection.get(include=["metadatas"])
            except Exception as get_error:
                print(f"데이터 조회 중 오류: {get_error}")
                # 메타데이터 없이 기본 정보만 반환
                return {
                    "total": total_count,
                    "pdf_chunks": total_count,  # 정확하지 않지만 대략적인 값
                    "conversation_total": 0,
                    "user_messages": 0,
                    "assistant_messages": 0
                }
            
            # 메타데이터 안전하게 처리
            metadatas = all_data.get("metadatas", [])
            if not metadatas:
                print("메타데이터가 없습니다.")
                return {
                    "total": total_count,
                    "pdf_chunks": total_count,
                    "conversation_total": 0,
                    "user_messages": 0,
                    "assistant_messages": 0
                }
            
            # 대화 기록만 필터링 (안전하게)
            conversation_data = []
            pdf_data = []
            
            for meta in metadatas:
                if meta and isinstance(meta, dict):
                    if meta.get("type") == "conversation":
                        conversation_data.append(meta)
                    else:
                        pdf_data.append(meta)
                else:
                    # 메타데이터가 None이거나 dict가 아닌 경우
                    pdf_data.append({})  # PDF 데이터로 가정
            
            user_messages = sum(1 for meta in conversation_data if meta.get("role") == "user")
            assistant_messages = sum(1 for meta in conversation_data if meta.get("role") == "assistant")
            pdf_chunks = len(pdf_data)
            
            print(f"전체 컬렉션 통계:")
            print(f"- 총 문서: {total_count}개")
            print(f"- PDF 청크: {pdf_chunks}개")
            print(f"- 대화 기록: {len(conversation_data)}개 (사용자: {user_messages}, AI: {assistant_messages})")
            
            return {
                "total": total_count,
                "pdf_chunks": pdf_chunks,
                "conversation_total": len(conversation_data),
                "user_messages": user_messages,
                "assistant_messages": assistant_messages
            }
        else:
            print("저장된 데이터가 없습니다.")
            return {"total": 0, "pdf_chunks": 0, "conversation_total": 0, "user_messages": 0, "assistant_messages": 0}
            
    except Exception as e:
        print(f"통계 조회 중 오류: {e}")
        return {"total": 0, "pdf_chunks": 0, "conversation_total": 0, "user_messages": 0, "assistant_messages": 0}

if __name__ == "__main__":
    test_user_msg = "주문 취소는 어떻게 하나요?"
    test_assistant_msg = "주문 취소는 주문 상세 페이지에서 '주문 취소' 버튼을 클릭하시면 됩니다."
    
    save_conversation_to_chroma(test_user_msg, test_assistant_msg)
    get_conversation_stats()
