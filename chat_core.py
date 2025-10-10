import os
import time
import numpy as np
import pickle
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import AzureOpenAI as EmbeddingOpenAI
from chromadb import PersistentClient
from pdf_to_vectordb import extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma

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

# ChromaDB 검색 함수 (안전한 오류 처리 포함)
def search_chroma(query, top_k=10):
    """
    ChromaDB에서 PDF 문서를 검색합니다.
    ID 관련 오류를 방지하고 안전한 검색을 제공합니다.
    """
    persist_dir = get_chroma_db_path()
    
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except Exception as e:
        print(f"ChromaDB 디렉토리 생성 실패: {e}")
        return []
    
    try:
        client = PersistentClient(path=persist_dir)
        
        # 컬렉션 존재 여부 확인
        try:
            collection = client.get_collection("pdf_collection")
            # 컬렉션이 비어있는지 확인
            if collection.count() == 0:
                print("ChromaDB 컬렉션이 비어있습니다.")
                return []
        except ValueError:
            # 컬렉션이 존재하지 않음
            print("ChromaDB 컬렉션이 존재하지 않습니다.")
            return []
        except Exception as e:
            print(f"ChromaDB 컬렉션 확인 중 오류: {e}")
            return []
        
        query_emb = get_query_embedding(query)
        
        # 여러 단계의 안전한 검색 시도
        search_attempts = [
            # 1단계: 메타데이터 필터링 포함 검색
            {
                "query_embeddings": [query_emb],
                "n_results": min(top_k * 2, 30),
                "include": ["documents", "metadatas"],
                "where": {"type": "pdf"}
            },
            # 2단계: 메타데이터 필터링 없이 검색
            {
                "query_embeddings": [query_emb],
                "n_results": min(top_k * 2, 30),
                "include": ["documents", "metadatas"]
            },
            # 3단계: 최소한의 정보로 검색
            {
                "query_embeddings": [query_emb],
                "n_results": min(top_k, 20),
                "include": ["documents"]
            }
        ]
        
        for attempt_num, search_params in enumerate(search_attempts, 1):
            try:
                print(f"검색 시도 {attempt_num}/{len(search_attempts)}")
                results = collection.query(**search_params)
                
                if results["documents"] and results["documents"][0]:
                    documents = results["documents"][0]
                    
                    # 메타데이터가 있으면 PDF 타입만 필터링
                    if "metadatas" in results and results["metadatas"] and results["metadatas"][0]:
                        pdf_docs = []
                        for doc, meta in zip(documents, results["metadatas"][0]):
                            if meta and meta.get("type") == "pdf":
                                pdf_docs.append(doc)
                            elif not meta:  # 메타데이터가 None인 경우도 포함
                                pdf_docs.append(doc)
                            if len(pdf_docs) >= top_k:
                                break
                        if pdf_docs:
                            print(f"검색 성공 ({attempt_num}단계): {len(pdf_docs)}개 문서")
                            return pdf_docs
                    else:
                        # 메타데이터 없이 문서만 반환
                        result_docs = documents[:top_k]
                        if result_docs:
                            print(f"검색 성공 ({attempt_num}단계): {len(result_docs)}개 문서")
                            return result_docs
                            
            except Exception as e:
                print(f"검색 시도 {attempt_num} 실패: {e}")
                continue
        
        # 모든 시도가 실패한 경우
        print("모든 검색 시도가 실패했습니다.")
        return []
                
    except Exception as e:
        print(f"ChromaDB 초기화 중 치명적 오류: {e}")
        # ChromaDB가 손상된 경우 복구 시도
        try:
            print("ChromaDB 복구를 시도합니다...")
            return reset_chromadb_and_retry(query, top_k, persist_dir)
        except Exception as reset_error:
            print(f"ChromaDB 복구 실패: {reset_error}")
            return []

# PDF 관련 함수는 pdf_to_vectordb.py에서 import하여 그대로 사용
# extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma

def reset_chromadb_and_retry(query, top_k, persist_dir):
    """
    ChromaDB가 손상된 경우 복구를 시도합니다.
    """
    import shutil
    import time
    
    try:
        print("⚠️ ChromaDB 손상 감지, 복구를 시도합니다...")
        
        # 백업 생성
        backup_dir = persist_dir + "_backup_" + str(int(time.time()))
        if os.path.exists(persist_dir):
            shutil.move(persist_dir, backup_dir)
            print(f"기존 데이터를 백업했습니다: {backup_dir}")
        
        # 새 디렉토리 생성
        os.makedirs(persist_dir, exist_ok=True)
        
        # 새 클라이언트로 빈 컬렉션 생성
        client = PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection("pdf_collection")
        
        print("✅ ChromaDB가 초기화되었습니다.")
        print("📁 새 PDF 파일을 업로드하여 다시 임베딩해주세요.")
        
        return []  # 빈 결과 반환
        
    except Exception as e:
        print(f"ChromaDB 복구 실패: {e}")
        return []

# 통합 검색 함수 (VOC 에이전트 전용)
def search_all_content(query, pdf_top_k=10, agent_type='voc_agent'):
    """
    PDF 내용을 검색하고 VOC 에이전트용 컨텍스트를 구성합니다.
    - VOC 에이전트: ChromaDB 사용
    
    Args:
        query: 검색할 쿼리
        pdf_top_k: PDF에서 검색할 최대 결과 수 (기본값: 10개)
        agent_type: 에이전트 타입 (현재는 'voc_agent'만 지원)
    
    Returns:
        dict: {'pdf_chunks': [], 'context_text': str}
    """
    result = {
        'pdf_chunks': [],
        'context_text': ''
    }
    
    try:
        # VOC 에이전트는 ChromaDB 사용
        pdf_chunks = search_chroma(query, top_k=pdf_top_k)
        db_type = "ChromaDB"
        
        result['pdf_chunks'] = pdf_chunks
        print(f"VOC 에이전트가 {db_type}에서 {len(pdf_chunks)}개 문서 검색")
        
        # PDF 컨텍스트 구성
        if pdf_chunks:
            context_parts = []
            context_parts.append("=== 📞 VOC 상담 관련 정보 (ChromaDB) ===")
            context_parts.append(f"고객 문의와 관련된 정보 {len(pdf_chunks)}개:")
            
            for i, chunk in enumerate(pdf_chunks, 1):
                context_parts.append(f"\n[정보 {i}]")
                # chunk가 딕셔너리인 경우 vs 문자열인 경우 처리
                if isinstance(chunk, dict):
                    context_parts.append(chunk.get('text', chunk).strip())
                else:
                    context_parts.append(chunk.strip())
            
            # VOC 상담 답변 지침
            instruction = """🔍 **VOC 상담 답변 지침** (ChromaDB 기반):
1. **고객 중심 접근**: 고객의 문제 해결에 집중하여 답변하세요.
2. **명확한 해결책**: 구체적이고 실행 가능한 해결 방안을 제시하세요.
3. **친근한 톤**: 고객 서비스 관점에서 친절하고 이해하기 쉽게 설명하세요.
4. **추가 지원**: 필요시 추가 도움이나 문의처를 안내하세요.

아래 정보를 바탕으로 고객의 문의에 최선의 답변을 제공해주세요:
"""
            
            result['context_text'] = instruction + "\n".join(context_parts)
        
        return result
        
    except Exception as e:
        print(f"PDF 검색 중 오류 (VOC 에이전트): {e}")
        return result

# 에이전트별 시스템 프롬프트 설정
def get_agent_system_prompt(agent_type, context_text=""):
    """
    VOC 에이전트용 시스템 프롬프트를 반환합니다.
    """
    
    base_context = context_text if context_text else ""
    
    # VOC 에이전트 시스템 프롬프트
    system_prompt = f"""당신은 K-ICIS 오더 VOC 전문 상담 챗봇입니다.

🎯 **주요 역할**:
- 고객의 문의사항에 대해 친절하고 정확한 답변 제공
- 오더 관련 문제 해결 지원
- VOC(Voice of Customer) 분석 및 대응

📋 **답변 지침**:
1. **친근하고 전문적인 톤**: 고객 서비스 관점에서 친절하게 응답
2. **구체적인 해결책 제시**: 문제에 대한 명확한 해결 방안 제공
3. **단계별 안내**: 복잡한 과정은 순서대로 설명
4. **추가 도움 제안**: 필요시 추가 지원 방안 안내

{base_context}

고객의 문의에 최선을 다해 도움을 드리겠습니다."""

    return system_prompt
