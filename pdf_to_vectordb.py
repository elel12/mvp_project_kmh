import os
import pdfplumber
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
import time
import hashlib
import json
import uuid

# 환경변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI 환경변수
AZURE_OPENAI_API_KEY = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("TEXT_EMBEDDING_DEPLOYMENT_NAME")

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

# PDF에서 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 텍스트를 chunk로 분할 (임베딩 길이 제한 대비)
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Azure OpenAI 임베딩 생성 함수
from openai import AzureOpenAI

def get_azure_embeddings(text_list):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION
    )
    embeddings = []
    for text in text_list:
        response = client.embeddings.create(
            input=text,
            model=DEPLOYMENT_NAME
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

# Chroma DB에 저장 함수 (중복 처리 방지)
from chromadb import PersistentClient

def save_to_chroma(text_chunks, embeddings, pdf_path=None):
    """
    ChromaDB에 PDF 청크를 저장합니다.
    중복 처리를 방지하기 위해 파일 해시값과 수정시간을 확인합니다.
    """
    # 중복 처리 확인
    if pdf_path:
        already_processed, reason = is_file_already_processed(pdf_path)
        if already_processed:
            print(f"⚠️  파일 처리 생략: {reason}")
            print(f"📁 파일명: {os.path.basename(pdf_path)}")
            return False, reason
    
    persist_dir = get_chroma_db_path()  # 동적 경로 사용
    
    # 디렉토리가 없으면 생성
    try:
        os.makedirs(persist_dir, exist_ok=True)
        print(f"ChromaDB 저장 경로 확인/생성: {persist_dir}")
    except Exception as e:
        print(f"ChromaDB 디렉토리 생성 실패: {e}")
        raise
    
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    
    # 기존에 같은 파일명으로 저장된 데이터가 있다면 삭제
    if pdf_path:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            # 같은 파일명의 기존 청크들을 찾아서 삭제
            existing_docs = collection.get(
                where={"filename": base}
            )
            if existing_docs['ids']:
                collection.delete(ids=existing_docs['ids'])
                print(f"🗑️  기존 파일 데이터 삭제: {len(existing_docs['ids'])}개 청크")
        except Exception as e:
            print(f"기존 데이터 삭제 중 오류 (무시하고 계속): {e}")
    
    # 파일명과 타임스탬프를 사용한 안전한 ID 생성
    if pdf_path:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        # 특수문자 제거 및 안전한 ID 생성
        base = "".join(c for c in base if c.isalnum() or c in '-_').strip()
        if not base:  # 빈 문자열인 경우 기본값 사용
            base = "pdf_doc"
    else:
        base = "pdf_doc"
    
    ts = int(time.time())
    
    # UUID를 사용하여 더 안전한 ID 생성
    import uuid
    batch_id = str(uuid.uuid4())[:8]  # 8자리 UUID
    ids = [f"{base}_{ts}_{batch_id}_chunk_{i:04d}" for i in range(len(text_chunks))]
    
    for i, (text, emb, id_) in enumerate(zip(text_chunks, embeddings, ids)):
        try:
            collection.add(
                documents=[text],
                embeddings=[emb],
                ids=[id_],
                metadatas=[{
                    "type": "pdf",
                    "source": "pdf_document",
                    "filename": base,
                    "chunk_index": i,
                    "timestamp": ts,
                    "batch_id": batch_id
                }]
            )
        except Exception as add_error:
            print(f"청크 {i} 저장 중 오류 (건너뛰기): {add_error}")
            continue
    
    print(f"✅ {len(text_chunks)}개 청크 저장 완료! (저장경로: {persist_dir})")
    
    # 파일 정보 업데이트
    if pdf_path:
        update_file_info(pdf_path, len(text_chunks))
    
    # 저장된 파일 목록 출력
    if os.path.exists(persist_dir):
        print("\n[폴더 내 파일 목록]")
        for f in os.listdir(persist_dir):
            print(f"- {f}")
        show_chroma_db_status()
    else:
        print("[경고] 저장 폴더가 존재하지 않습니다.")
    
    return True, f"새로 처리된 청크: {len(text_chunks)}개"

def show_chroma_db_status(recent_n=5):
    # ChromaDB에 누적된 전체 청크/문서 개수와 최근 N개 ID, 내용을 최신순으로 출력
    persist_dir = get_chroma_db_path()  # 동적 경로 사용
    
    try:
        client = PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection("pdf_collection")
        count = collection.count()
        print(f"총 저장된 청크 개수: {count}")
        
        if count > 0:
            docs = collection.get()
            ids = docs['ids']
            documents = docs['documents']
            # 최신순(가장 마지막에 추가된 순서)으로 최근 N개 출력
            print(f"최근 저장된 문서 ID 목록 (최신순, 최대 {recent_n}개): {ids[-recent_n:][::-1]}")
            print(f"최근 저장된 문서 내용 (최신순, 최대 {recent_n}개): {documents[-recent_n:][::-1]}")
            return count, ids[-recent_n:][::-1], documents[-recent_n:][::-1]
        else:
            print("저장된 문서가 없습니다.")
            return 0, [], []
            
    except Exception as e:
        print(f"ChromaDB 상태 확인 중 오류: {e}")
        return 0, [], []

# 파일 해시값 계산 함수
def calculate_file_hash(file_path):
    """파일의 MD5 해시값을 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 파일 정보 저장/로드 함수
def get_file_info_path():
    """파일 정보를 저장할 JSON 파일 경로를 반환합니다."""
    persist_dir = get_chroma_db_path()
    return os.path.join(persist_dir, "file_info.json")

def load_file_info():
    """저장된 파일 정보를 로드합니다."""
    file_info_path = get_file_info_path()
    if os.path.exists(file_info_path):
        try:
            with open(file_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"파일 정보 로드 중 오류: {e}")
            return {}
    return {}

def save_file_info(file_info):
    """파일 정보를 저장합니다."""
    file_info_path = get_file_info_path()
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_info_path), exist_ok=True)
        with open(file_info_path, 'w', encoding='utf-8') as f:
            json.dump(file_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"파일 정보 저장 중 오류: {e}")

def is_file_already_processed(file_path):
    """파일이 이미 처리되었는지 확인합니다."""
    try:
        file_info = load_file_info()
        file_name = os.path.basename(file_path)
        
        if file_name not in file_info:
            return False, "새로운 파일"
        
        stored_info = file_info[file_name]
        current_hash = calculate_file_hash(file_path)
        current_mtime = os.path.getmtime(file_path)
        
        # 해시값과 수정시간 모두 확인
        if (stored_info.get('hash') == current_hash and 
            stored_info.get('mtime') == current_mtime):
            return True, "동일한 파일 (해시값 및 수정시간 일치)"
        else:
            return False, "파일이 수정됨 (해시값 또는 수정시간 변경)"
    
    except Exception as e:
        print(f"파일 처리 여부 확인 중 오류: {e}")
        return False, "확인 실패"

def update_file_info(file_path, chunk_count):
    """파일 정보를 업데이트합니다."""
    try:
        file_info = load_file_info()
        file_name = os.path.basename(file_path)
        
        file_info[file_name] = {
            'hash': calculate_file_hash(file_path),
            'mtime': os.path.getmtime(file_path),
            'processed_at': time.time(),
            'chunk_count': chunk_count,
            'full_path': file_path
        }
        
        save_file_info(file_info)
        print(f"파일 정보 업데이트 완료: {file_name}")
    except Exception as e:
        print(f"파일 정보 업데이트 중 오류: {e}")

def get_processed_files_info():
    """처리된 파일들의 정보를 반환합니다."""
    try:
        file_info = load_file_info()
        if not file_info:
            return "처리된 파일이 없습니다."
        
        result = "📋 **처리된 PDF 파일 목록:**\n"
        for filename, info in file_info.items():
            processed_time = time.strftime('%Y-%m-%d %H:%M:%S', 
                                         time.localtime(info.get('processed_at', 0)))
            chunk_count = info.get('chunk_count', 0)
            result += f"• **{filename}** (청크: {chunk_count}개, 처리시간: {processed_time})\n"
        
        return result
    except Exception as e:
        return f"파일 정보 조회 중 오류: {e}"

def clear_file_info():
    """저장된 파일 정보를 초기화합니다."""
    try:
        file_info_path = get_file_info_path()
        if os.path.exists(file_info_path):
            os.remove(file_info_path)
            print("파일 정보 초기화 완료")
            return True
    except Exception as e:
        print(f"파일 정보 초기화 중 오류: {e}")
        return False

def diagnose_chromadb():
    """ChromaDB 상태를 진단합니다."""
    persist_dir = get_chroma_db_path()
    
    print("🔍 ChromaDB 진단 시작...")
    print(f"📁 저장 경로: {persist_dir}")
    
    if not os.path.exists(persist_dir):
        print("❌ ChromaDB 디렉토리가 존재하지 않습니다.")
        return False
    
    try:
        client = PersistentClient(path=persist_dir)
        
        # 컬렉션 목록 확인
        collections = client.list_collections()
        print(f"📊 컬렉션 수: {len(collections)}")
        
        if not collections:
            print("❌ 컬렉션이 존재하지 않습니다.")
            return False
            
        for collection_info in collections:
            print(f"📋 컬렉션명: {collection_info.name}")
            
        # pdf_collection 상세 확인
        try:
            collection = client.get_collection("pdf_collection")
            count = collection.count()
            print(f"📄 PDF 컬렉션 문서 수: {count}")
            
            if count > 0:
                # 샘플 데이터 확인
                sample = collection.get(limit=1)
                if sample['ids']:
                    print(f"✅ 샘플 ID: {sample['ids'][0]}")
                    print("✅ ChromaDB가 정상적으로 작동합니다.")
                    return True
            else:
                print("⚠️ 컬렉션이 비어있습니다.")
                return True
                
        except Exception as e:
            print(f"❌ PDF 컬렉션 접근 오류: {e}")
            return False
            
    except Exception as e:
        print(f"❌ ChromaDB 클라이언트 오류: {e}")
        return False

def migrate_chromadb_ids():
    """기존 ChromaDB의 불안전한 ID를 새로운 안전한 형식으로 마이그레이션합니다."""
    persist_dir = get_chroma_db_path()
    
    try:
        client = PersistentClient(path=persist_dir)
        collection = client.get_collection("pdf_collection")
        
        # 모든 데이터 가져오기
        all_data = collection.get()
        
        if not all_data['ids']:
            print("마이그레이션할 데이터가 없습니다.")
            return True
            
        print(f"📊 마이그레이션 대상: {len(all_data['ids'])}개 문서")
        
        # 문제가 될 수 있는 간단한 ID 패턴 찾기
        simple_ids = [id_ for id_ in all_data['ids'] if id_.startswith('chunk_')]
        
        if simple_ids:
            print(f"⚠️  간단한 ID 패턴 발견: {len(simple_ids)}개")
            print("기존 데이터를 정리합니다...")
            
            # 기존 컬렉션 삭제
            client.delete_collection("pdf_collection")
            print("✅ 기존 컬렉션 삭제 완료")
            
            # 새 컬렉션 생성
            collection = client.create_collection("pdf_collection")
            print("✅ 새 컬렉션 생성 완료")
            
            print("📁 새 PDF 파일을 업로드하여 안전한 ID로 다시 임베딩해주세요.")
            return True
        else:
            print("✅ 모든 ID가 안전한 형식입니다.")
            return True
            
    except Exception as e:
        print(f"마이그레이션 중 오류: {e}")
        return False

if __name__ == "__main__":
    pdf_path = "/Users/minho/Desktop/MS AI/pstn_voc.pdf"  # 사용할 PDF 파일 경로
    
    # 파일 처리 여부 확인
    is_processed, message = is_file_already_processed(pdf_path)
    if is_processed:
        print(f"이미 처리된 파일입니다: {message}")
    else:
        print(f"새로운 파일로 처리 시작: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)
        chunks = split_text(text)
        embeddings = get_azure_embeddings(chunks)
        save_to_chroma(chunks, embeddings, pdf_path=pdf_path)
        # 파일 정보 업데이트
        update_file_info(pdf_path, len(chunks))
        print("PDF -> 벡터 DB 변환 완료! (Azure OpenAI)")


