import os
import time
import numpy as np
import pickle
import faiss
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import AzureOpenAI as EmbeddingOpenAI
from chromadb import PersistentClient
from pdf_to_vectordb import extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers 라이브러리가 없습니다. FAISS 검색 기능을 위해 설치해주세요.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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

# FAISS DB 저장 경로 (지식자산화 에이전트용)
def get_faiss_db_path():
    """
    FAISS DB 경로를 반환합니다.
    """
    if os.getenv("WEBSITE_SITE_NAME"):  
        base_path = "/home/site/wwwroot/faiss_db"
        print(f"Azure Web App 환경 - FAISS 경로: {base_path}")
        return base_path
    else:
        base_path = os.path.join(os.getcwd(), "faiss_db")
        print(f"로컬 개발 환경 - FAISS 경로: {base_path}")
        return base_path

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

def get_faiss_query_embedding(query):
    """FAISS DB 검색용 임베딩 생성 (sentence-transformers 사용)"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers 라이브러리가 필요합니다. pip install sentence-transformers로 설치해주세요.")
    
    model = SentenceTransformer('jhgan/ko-sbert-multitask')
    return model.encode([query])[0]

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

# 통합 검색 함수 (PDF만)
# 통합 검색 함수 (에이전트별 벡터 DB 사용)
def search_all_content(query, pdf_top_k=10, agent_type='voc_agent'):
    """
    PDF 내용을 검색하고 에이전트별로 맞춤형 컨텍스트를 구성합니다.
    - VOC 에이전트: ChromaDB 사용
    - 지식자산화 에이전트: FAISS DB 사용
    
    Args:
        query: 검색할 쿼리
        pdf_top_k: PDF에서 검색할 최대 결과 수 (기본값: 10개)
        agent_type: 에이전트 타입 ('voc_agent' 또는 'knowledge_agent')
    
    Returns:
        dict: {'pdf_chunks': [], 'context_text': str}
    """
    result = {
        'pdf_chunks': [],
        'context_text': ''
    }
    
    try:
        # 에이전트별 벡터 DB 검색
        if agent_type == 'voc_agent':
            # VOC 에이전트는 ChromaDB 사용
            pdf_chunks = search_chroma(query, top_k=pdf_top_k)
            db_type = "ChromaDB"
        else:  # knowledge_agent
            # 지식자산화 에이전트는 FAISS DB 사용
            pdf_chunks = search_faiss(query, top_k=pdf_top_k)
            db_type = "FAISS DB"
        
        result['pdf_chunks'] = pdf_chunks
        print(f"{agent_type} 에이전트가 {db_type}에서 {len(pdf_chunks)}개 문서 검색")
        
        # FAISS DB에 데이터가 없는 경우 특별 처리
        if agent_type == 'knowledge_agent' and not pdf_chunks:
            result['context_text'] = """
⚠️ **FAISS DB 정보 부족 알림**:
현재 FAISS DB에 관련 정보가 없거나 검색 결과가 없습니다.

📋 **해결 방안**:
1. 관련 문서를 먼저 업로드하여 FAISS DB에 임베딩하세요
2. 다른 검색어로 시도해보세요
3. FAISS DB 상태를 진단해보세요

지식자산화 에이전트는 오직 FAISS DB에 저장된 정보만을 기반으로 답변을 제공합니다.
"""
            return result
        
        # PDF 컨텍스트 구성
        context_parts = []
        
        if pdf_chunks:
            if agent_type == 'voc_agent':
                context_parts.append("=== 📞 VOC 상담 관련 정보 (ChromaDB) ===")
                context_parts.append(f"고객 문의와 관련된 정보 {len(pdf_chunks)}개:")
            else:  # knowledge_agent
                context_parts.append("=== 📚 지식자산화 관련 정보 (FAISS DB) ===")
                context_parts.append(f"분석 대상 문서 정보 {len(pdf_chunks)}개:")
            
            for i, chunk in enumerate(pdf_chunks, 1):
                context_parts.append(f"\n[정보 {i}]")
                # chunk가 딕셔너리인 경우 (FAISS) vs 문자열인 경우 (ChromaDB) 처리
                if isinstance(chunk, dict):
                    context_parts.append(chunk.get('text', chunk).strip())
                else:
                    context_parts.append(chunk.strip())
        
        if context_parts:
            # 에이전트별 맞춤형 지시사항
            if agent_type == 'voc_agent':
                instruction = """
🔍 **VOC 상담 답변 지침** (ChromaDB 기반):
1. **고객 중심 접근**: 고객의 문제 해결에 집중하여 답변하세요.
2. **명확한 해결책**: 구체적이고 실행 가능한 해결 방안을 제시하세요.
3. **친근한 톤**: 고객 서비스 관점에서 친절하고 이해하기 쉽게 설명하세요.
4. **추가 지원**: 필요시 추가 도움이나 문의처를 안내하세요.

아래 정보를 바탕으로 고객의 문의에 최선의 답변을 제공해주세요:
"""
            else:  # knowledge_agent
                instruction = """
🔍 **지식자산화 분석 지침** (FAISS DB 전용):
1. **FAISS DB 기반 분석**: 오직 FAISS DB에 저장된 임베딩 정보만 사용
2. **구조화된 분석**: 정보를 논리적으로 분류하고 체계화하세요
3. **핵심 추출**: 중요한 인사이트와 패턴을 명확히 도출하세요
4. **실용적 지식**: 재사용 가능한 지식과 베스트 프랙티스를 제시하세요
5. **연관성 분석**: 정보 간의 연결점과 의존성을 파악하세요
6. **지식 체계화**: 정보를 카테고리별로 분류하고 태깅하세요

⚠️ **제한사항**: FAISS DB 외의 다른 데이터소스는 참조하지 마세요.

아래 FAISS DB에서 검색된 정보만을 바탕으로 체계적이고 구조화된 지식을 생성해주세요:
"""
            
            result['context_text'] = instruction + "\n".join(context_parts)
        
        return result
        
    except Exception as e:
        print(f"PDF 검색 중 오류 ({agent_type}): {e}")
        return result

# 에이전트별 시스템 프롬프트 설정
def get_agent_system_prompt(agent_type, context_text=""):
    """
    선택된 에이전트에 따라 적절한 시스템 프롬프트를 반환합니다.
    """
    
    base_context = context_text if context_text else ""
    
    if agent_type == 'voc_agent':
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

    elif agent_type == 'knowledge_agent':
        system_prompt = f"""당신은 K-ICIS 지식자산화 전문 에이전트입니다.

🎯 **주요 역할**:
- FAISS DB에 저장된 문서 및 데이터 분석을 통한 지식 추출
- 구조화된 정보 정리 및 분류
- 지식베이스 구축 지원
- 베스트 프랙티스 도출

📋 **답변 지침**:
1. **FAISS DB 전용**: 오직 FAISS DB에 임베딩된 정보만을 기반으로 답변
2. **체계적 분석**: 정보를 논리적으로 구조화하여 제시
3. **핵심 요약**: 중요한 포인트를 명확하게 추출
4. **분류 및 태깅**: 정보를 카테고리별로 분류
5. **실행 가능한 인사이트**: 활용 가능한 지식으로 변환

⚠️ **중요**: FAISS DB에 저장된 정보가 없거나 관련 정보를 찾을 수 없는 경우, "FAISS DB에 관련 정보가 없습니다. 먼저 관련 문서를 업로드하여 임베딩해주세요."라고 안내해주세요.

{base_context}

FAISS DB에 저장된 정보만을 바탕으로 체계적이고 구조화된 지식을 만들어드리겠습니다."""

    else:
        system_prompt = base_context

    return system_prompt

# FAISS DB 관련 함수들
def save_to_faiss(text_chunks, embeddings, pdf_path=None):
    """
    FAISS에 임베딩을 저장합니다 (지식자산화 에이전트용).
    """
    faiss_dir = get_faiss_db_path()
    os.makedirs(faiss_dir, exist_ok=True)
    
    # 임베딩을 numpy 배열로 변환
    embeddings_array = np.array(embeddings).astype('float32')
    
    # FAISS 인덱스 생성 또는 로드
    index_path = os.path.join(faiss_dir, "knowledge.index")
    metadata_path = os.path.join(faiss_dir, "knowledge_metadata.pkl")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        # 기존 인덱스 로드
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
    else:
        # 새 인덱스 생성
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (코사인 유사도)
        metadata = {'texts': [], 'sources': [], 'chunk_ids': []}
    
    # 파일명 기반 기존 데이터 삭제 (중복 방지)
    if pdf_path:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        # 같은 파일의 기존 데이터 제거
        new_texts = []
        new_sources = []
        new_chunk_ids = []
        new_embeddings = []
        
        for i, source in enumerate(metadata['sources']):
            if not source.startswith(base_name):
                new_texts.append(metadata['texts'][i])
                new_sources.append(metadata['sources'][i])
                new_chunk_ids.append(metadata['chunk_ids'][i])
                # 기존 임베딩 추출 (인덱스에서)
                if index.ntotal > i:
                    new_embeddings.append(index.reconstruct(i))
        
        # 인덱스 재구성
        if new_embeddings:
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(np.array(new_embeddings).astype('float32'))
            metadata = {
                'texts': new_texts,
                'sources': new_sources,
                'chunk_ids': new_chunk_ids
            }
        else:
            # 모든 데이터가 삭제된 경우 인덱스 초기화
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatIP(dimension)
            metadata = {'texts': [], 'sources': [], 'chunk_ids': []}
    
    # 새 데이터 추가
    index.add(embeddings_array)
    
    # 메타데이터 업데이트
    base_name = os.path.splitext(os.path.basename(pdf_path))[0] if pdf_path else "knowledge"
    timestamp = int(time.time())
    
    for i, text in enumerate(text_chunks):
        metadata['texts'].append(text)
        metadata['sources'].append(f"{base_name}_chunk_{i}_{timestamp}")
        metadata['chunk_ids'].append(len(metadata['texts']) - 1)
    
    # 저장
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ FAISS DB에 {len(text_chunks)}개 청크 저장 완료!")
    return True, f"새로 처리된 청크: {len(text_chunks)}개"

def search_faiss(query, top_k=10):
    """
    FAISS에서 유사한 문서를 검색합니다 (지식자산화 에이전트용).
    """
    faiss_dir = get_faiss_db_path()
    index_path = os.path.join(faiss_dir, "knowledge.index")
    metadata_path = os.path.join(faiss_dir, "knowledge_metadata.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("FAISS DB가 존재하지 않습니다.")
        return []
    
    try:
        # 인덱스와 메타데이터 로드
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if index.ntotal == 0:
            print("FAISS 인덱스가 비어있습니다.")
            return []
        
        # 쿼리 임베딩 생성 (FAISS용 sentence-transformers)
        query_embedding = get_faiss_query_embedding(query)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # 정규화 (코사인 유사도를 위해)
        faiss.normalize_L2(query_vector)
        
        # 검색 수행
        scores, indices = index.search(query_vector, min(top_k, index.ntotal))
        
        # 결과 구성
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(metadata['texts']) and score > 0.1:  # 최소 유사도 threshold
                results.append({
                    'text': metadata['texts'][idx],
                    'similarity': float(score),
                    'source': metadata.get('sources', [''])[idx] if idx < len(metadata.get('sources', [])) else ''
                })
        
        print(f"FAISS 검색 완료: {len(results)}개 문서")
        return results
        
    except Exception as e:
        print(f"FAISS 검색 중 오류: {e}")
        return []

def diagnose_faiss():
    """FAISS DB 상태를 진단합니다."""
    faiss_dir = get_faiss_db_path()
    print(f"🔍 FAISS DB 진단 시작...")
    print(f"📁 저장 경로: {faiss_dir}")
    
    index_path = os.path.join(faiss_dir, "knowledge.index")
    metadata_path = os.path.join(faiss_dir, "knowledge_metadata.pkl")
    
    if not os.path.exists(faiss_dir):
        print("❌ FAISS DB 디렉토리가 존재하지 않습니다.")
        return False
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("❌ FAISS 인덱스 파일이 존재하지 않습니다.")
        return False
    
    try:
        index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"📊 저장된 문서 수: {index.ntotal}")
        print(f"📋 메타데이터 항목: {len(metadata['texts'])}")
        
        if index.ntotal > 0:
            print(f"✅ FAISS DB가 정상적으로 작동합니다.")
            return True
        else:
            print("⚠️ FAISS 인덱스가 비어있습니다.")
            return True
            
    except Exception as e:
        print(f"❌ FAISS DB 오류: {e}")
        return False

def migrate_faiss_from_backup():
    """기존 FAISS 백업 파일을 새로운 형식으로 마이그레이션합니다."""
    faiss_dir = get_faiss_db_path()
    backup_dir = faiss_dir + "_backup_old"
    
    old_index_path = os.path.join(backup_dir, "index.faiss")
    old_metadata_path = os.path.join(backup_dir, "index.pkl")
    
    new_index_path = os.path.join(faiss_dir, "knowledge.index")
    new_metadata_path = os.path.join(faiss_dir, "knowledge_metadata.pkl")
    
    print(f"🔄 FAISS 백업 파일 마이그레이션 시작...")
    print(f"📁 백업 경로: {backup_dir}")
    print(f"📁 새 경로: {faiss_dir}")
    
    if not os.path.exists(old_index_path) or not os.path.exists(old_metadata_path):
        print("❌ 백업 파일이 존재하지 않습니다.")
        return False
    
    try:
        # 기존 백업 파일 로드
        print("📖 백업 파일 로딩 중...")
        index = faiss.read_index(old_index_path)
        with open(old_metadata_path, 'rb') as f:
            old_metadata = pickle.load(f)
        
        print(f"📊 백업 데이터: {index.ntotal}개 문서")
        print(f"📋 메타데이터 키: {list(old_metadata.keys())}")
        
        # 새로운 형식으로 메타데이터 변환
        new_metadata = {
            'texts': [],
            'sources': [],
            'chunk_ids': []
        }
        
        # 기존 메타데이터 구조에 따라 변환
        if 'texts' in old_metadata:
            new_metadata['texts'] = old_metadata['texts']
        elif 'documents' in old_metadata:
            new_metadata['texts'] = old_metadata['documents']
        
        if 'sources' in old_metadata:
            new_metadata['sources'] = old_metadata['sources']
        elif 'filenames' in old_metadata:
            new_metadata['sources'] = old_metadata['filenames']
        else:
            # 소스 정보가 없으면 생성
            new_metadata['sources'] = [f"migrated_chunk_{i}" for i in range(len(new_metadata['texts']))]
        
        # chunk_ids 생성
        new_metadata['chunk_ids'] = list(range(len(new_metadata['texts'])))
        
        # 디렉토리 생성
        os.makedirs(faiss_dir, exist_ok=True)
        
        # 새로운 형식으로 저장
        print("💾 새 형식으로 저장 중...")
        faiss.write_index(index, new_index_path)
        with open(new_metadata_path, 'wb') as f:
            pickle.dump(new_metadata, f)
        
        print(f"✅ 마이그레이션 완료!")
        print(f"📊 변환된 데이터: {len(new_metadata['texts'])}개 텍스트")
        print(f"📋 인덱스 크기: {index.ntotal}개")
        
        return True
        
    except Exception as e:
        print(f"❌ 마이그레이션 실패: {e}")
        return False

def migrate_langchain_faiss_data(json_file_path):
    """
    LangChain에서 추출한 JSON 데이터를 새 FAISS DB로 마이그레이션합니다.
    """
    import json
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return False, "sentence-transformers 라이브러리가 필요합니다. pip install sentence-transformers로 설치해주세요."
    
    if not os.path.exists(json_file_path):
        print(f"❌ 추출 데이터 파일이 없습니다: {json_file_path}")
        return False, "추출 데이터 파일이 없습니다"
    
    try:
        # JSON 데이터 로드
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = data.get('documents', [])
        metadatas = data.get('metadatas', [])
        total_count = data.get('total_count', 0)
        
        if not documents:
            print("❌ 추출된 문서가 없습니다")
            return False, "추출된 문서가 없습니다"
        
        print(f"📄 마이그레이션 시작: {total_count}개 문서")
        
        # 임베딩 모델 로드
        print("🔄 임베딩 모델 로드 중...")
        embedding_model = SentenceTransformer('jhgan/ko-sbert-multitask')
        
        # 문서들을 청크 단위로 나누어 처리
        all_texts = []
        all_sources = []
        
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            if not doc or not doc.strip():
                continue
                
            # 긴 문서는 청크로 분할
            max_chunk_length = 1000
            if len(doc) > max_chunk_length:
                # 문서를 청크로 분할
                chunks = []
                for start in range(0, len(doc), max_chunk_length):
                    chunk = doc[start:start + max_chunk_length]
                    if chunk.strip():
                        chunks.append(chunk.strip())
            else:
                chunks = [doc.strip()]
            
            # 소스 정보 생성
            source_info = metadata.get('source_file', f'migrated_doc_{i}')
            title = metadata.get('title', '')
            section = metadata.get('section', '')
            
            source_name = f"{source_info}"
            if title:
                source_name = f"{title}"
            if section:
                source_name = f"{source_name}_{section}"
            
            for j, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_sources.append(f"{source_name}_chunk_{j}")
        
        print(f"📊 총 {len(all_texts)}개 청크로 분할됨")
        
        # 모든 텍스트에 대해 임베딩 생성
        print(f"🔄 전체 {len(all_texts)}개 텍스트에 대한 임베딩 생성 중...")
        all_embeddings = embedding_model.encode(all_texts, batch_size=50, show_progress_bar=True)
        
        # FAISS DB에 한번에 저장 (기존 데이터는 미리 삭제)
        print(f"💾 FAISS DB에 저장 중...")
        
        # 기존 FAISS 파일들 삭제 (새로 시작)
        faiss_dir = get_faiss_db_path()
        index_path = os.path.join(faiss_dir, "knowledge.index")
        metadata_path = os.path.join(faiss_dir, "knowledge_metadata.pkl")
        
        if os.path.exists(index_path):
            os.remove(index_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
        success, message = save_to_faiss(all_texts, all_embeddings, pdf_path="migrated_langchain_data")
        
        if success:
            total_saved = len(all_texts)
        else:
            print(f"❌ 저장 실패: {message}")
            return False, message
        
        print(f"✅ 마이그레이션 완료: {total_saved}개 청크 저장")
        return True, f"총 {total_saved}개 청크 마이그레이션 완료"
        
    except Exception as e:
        print(f"❌ 마이그레이션 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, f"마이그레이션 실패: {str(e)}"
