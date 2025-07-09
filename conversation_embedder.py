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

# ChromaDB 저장 경로
PERSIST_DIR = "./chroma_db"

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
    저장 경로는 ./chroma_db로 고정됩니다.
    """
    persist_dir = "./chroma_db"
    client = PersistentClient(path=persist_dir)
    
    # 대화용 별도 컬렉션 생성
    conversation_collection = client.get_or_create_collection("conversation_collection")
    
    # 타임스탬프 생성
    ts = int(time.time())
    
    # 사용자 메시지 저장
    user_embedding = get_conversation_embedding(user_message)
    user_id = f"user_{ts}_{hash(user_message) % 10000}"
    
    conversation_collection.add(
        documents=[user_message],
        embeddings=[user_embedding],
        ids=[user_id],
        metadatas=[{
            "type": "user_message",
            "timestamp": ts,
            "role": "user"
        }]
    )
    
    # AI 답변 저장
    assistant_embedding = get_conversation_embedding(assistant_message)
    assistant_id = f"assistant_{ts}_{hash(assistant_message) % 10000}"
    
    conversation_collection.add(
        documents=[assistant_message],
        embeddings=[assistant_embedding],
        ids=[assistant_id],
        metadatas=[{
            "type": "assistant_message",
            "timestamp": ts,
            "role": "assistant",
            "related_user_id": user_id
        }]
    )
    
    print(f"대화 내용 저장 완료! (사용자: {user_id}, AI: {assistant_id})")

# 대화 내용에서 유사한 내용 검색하는 함수
def search_conversation_history(query, top_k=3):
    persist_dir = "./chroma_db"
    client = PersistentClient(path=persist_dir)
    
    try:
        conversation_collection = client.get_or_create_collection("conversation_collection")

        # 컬렉션이 비어있는지 확인
        if conversation_collection.count() == 0:
            return []
        
        query_embedding = get_conversation_embedding(query)
        
        results = conversation_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if results["documents"] and results["documents"][0]:
            return results["documents"][0]
        else:
            return []
            
    except Exception as e:
        print(f"대화 기록 검색 중 오류: {e}")
        return []

# 대화 기록 통계 조회 함수
def get_conversation_stats():
    persist_dir = "./chroma_db"
    client = PersistentClient(path=persist_dir)
    
    try:
        conversation_collection = client.get_or_create_collection("conversation_collection")
        total_count = conversation_collection.count()
        
        if total_count > 0:
            # 최근 대화 조회
            all_data = conversation_collection.get(include=["metadatas"])
            user_messages = sum(1 for meta in all_data["metadatas"] if meta.get("role") == "user")
            assistant_messages = sum(1 for meta in all_data["metadatas"] if meta.get("role") == "assistant")
            
            print(f"총 저장된 대화: {total_count}개")
            print(f"사용자 메시지: {user_messages}개")
            print(f"AI 답변: {assistant_messages}개")
            
            return {
                "total": total_count,
                "user_messages": user_messages,
                "assistant_messages": assistant_messages
            }
        else:
            print("저장된 대화 기록이 없습니다.")
            return {"total": 0, "user_messages": 0, "assistant_messages": 0}
            
    except Exception as e:
        print(f"대화 통계 조회 중 오류: {e}")
        return {"total": 0, "user_messages": 0, "assistant_messages": 0}

if __name__ == "__main__":
    test_user_msg = "주문 취소는 어떻게 하나요?"
    test_assistant_msg = "주문 취소는 주문 상세 페이지에서 '주문 취소' 버튼을 클릭하시면 됩니다."
    
    save_conversation_to_chroma(test_user_msg, test_assistant_msg)
    get_conversation_stats()
