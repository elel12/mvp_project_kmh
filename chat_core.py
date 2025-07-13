import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import AzureOpenAI as EmbeddingOpenAI
from chromadb import PersistentClient
from pdf_to_vectordb import extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma
from conversation_embedder import search_conversation_history

load_dotenv()

# OpenAI 챗 클라이언트
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

# 임베딩 클라이언트
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

# OpenAI 챗 함수
def get_openai_client(messages):
    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# 임베딩 생성 함수
def get_query_embedding(query):
    embedding_client = EmbeddingOpenAI(
        api_key=AZURE_EMBEDDING_API_KEY,
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,
        api_version=AZURE_EMBEDDING_API_VERSION
    )
    response = embedding_client.embeddings.create(
        input=query,
        model=EMBEDDING_DEPLOYMENT_NAME
    )
    return response.data[0].embedding

# ChromaDB 검색 함수 (저장 경로 고정: ./chroma_db)
def search_chroma(query, top_k=10):
    """
    ChromaDB에서 PDF 문서만 검색합니다.
    더 많은 PDF 내용을 검색하여 포괄적인 답변이 가능합니다.
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
        query_emb = get_query_embedding(query)
        
        try:
            # PDF 문서를 더 많이 검색 (top_k * 3으로 확장)
            results = collection.query(
                query_embeddings=[query_emb],
                n_results=min(top_k * 3, 50),  # 최대 50개까지 검색
                include=["documents", "metadatas"],
                where={"type": "pdf"}  # PDF 문서만 검색
            )
            
            if results["documents"] and results["documents"][0]:
                # PDF 문서만 필터링하여 반환
                pdf_docs = []
                for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                    if meta.get("type") == "pdf":
                        pdf_docs.append(doc)
                    if len(pdf_docs) >= top_k:
                        break
                return pdf_docs
            else:
                return []
        except Exception as e:
            print(f"PDF 검색 중 오류: {e}")
            # 메타데이터 필터링이 실패하면 기존 방식으로 fallback
            try:
                results = collection.query(
                    query_embeddings=[query_emb],
                    n_results=min(top_k * 2, 30),  # fallback에서도 더 많이 검색
                    include=["documents"]
                )
                return results["documents"][0][:top_k] if results["documents"] else []
            except Exception as e2:
                print(f"Fallback 검색 중 오류: {e2}")
                return []
                
    except Exception as e:
        print(f"ChromaDB 초기화 중 오류: {e}")
        return []

# PDF 관련 함수는 pdf_to_vectordb.py에서 import하여 그대로 사용
# extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma

# 통합 검색 함수 (PDF + 대화 기록)
def search_all_content(query, pdf_top_k=10, conversation_top_k=3):
    """
    PDF 내용과 대화 기록을 통합하여 검색합니다.
    
    Args:
        query: 검색할 쿼리
        pdf_top_k: PDF에서 검색할 최대 결과 수 (기본값: 10개로 증가)
        conversation_top_k: 대화 기록에서 검색할 최대 결과 수 (기본값: 3개로 증가)
    
    Returns:
        dict: {'pdf_chunks': [], 'conversation_history': [], 'context_text': str}
    """
    result = {
        'pdf_chunks': [],
        'conversation_history': [],
        'context_text': ''
    }
    
    try:
        # 1. PDF 내용 검색
        pdf_chunks = search_chroma(query, top_k=pdf_top_k)
        result['pdf_chunks'] = pdf_chunks
        
        # 2. 대화 기록 검색
        conversation_history = search_conversation_history(query, top_k=conversation_top_k)
        result['conversation_history'] = conversation_history
        
        # 3. 통합 컨텍스트 구성 (더 상세하고 체계적으로)
        context_parts = []
        
        if pdf_chunks:
            context_parts.append("=== 📄 PDF 문서 관련 정보 ===")
            context_parts.append(f"검색된 관련 내용 {len(pdf_chunks)}개:")
            for i, chunk in enumerate(pdf_chunks, 1):
                context_parts.append(f"\n[정보 {i}]")
                context_parts.append(chunk.strip())
        
        if conversation_history:
            context_parts.append("\n=== 💬 관련 대화 기록 ===")
            context_parts.append(f"과거 유사한 대화 {len(conversation_history)}개:")
            for i, conv in enumerate(conversation_history, 1):
                context_parts.append(f"\n[대화 {i}]")
                context_parts.append(conv.strip())
        
        if context_parts:
            # 더 명확한 지시사항 추가 - 대화 기록 활용 강화
            instruction = """
🔍 **답변 지침**:
1. **이전 대화 기록 우선 활용**: 과거에 동일하거나 유사한 질문에 대한 답변이 있다면, 그 정보를 우선적으로 참고하여 일관성 있는 답변을 제공하세요.
2. **PDF 문서 정보 보완**: PDF 문서의 관련 정보로 답변을 보완하고 더 상세한 내용을 제공하세요.
3. **정확성과 일관성**: 이전에 제공한 답변과 모순되지 않도록 주의하며, 새로운 정보가 있다면 명확히 구분하여 설명하세요.
4. **구체적 정보 제공**: 접수번호, 시스템명, 날짜 등 구체적인 정보가 있다면 반드시 포함하세요.

아래 정보를 모두 검토하여 종합적이고 정확한 답변을 제공해주세요:
"""
            result['context_text'] = instruction + "\n".join(context_parts)
        
        return result
        
    except Exception as e:
        print(f"통합 검색 중 오류: {e}")
        return result
