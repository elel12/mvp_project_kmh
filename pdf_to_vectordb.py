import os
import pdfplumber
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv
import time

# 환경변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI 환경변수
AZURE_OPENAI_API_KEY = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("TEXT_EMBEDDING_DEPLOYMENT_NAME")
PERSIST_DIR = os.getenv("PERSIST_DIR")

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

# Chroma DB에 저장 함수
from chromadb import PersistentClient

def save_to_chroma(text_chunks, embeddings, persist_dir=PERSIST_DIR, pdf_path=None):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    # 파일명과 타임스탬프를 prefix로 사용
    if pdf_path:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
    else:
        base = "pdf"
    ts = int(time.time())
    ids = [f"{base}_{ts}_chunk_{i}" for i in range(len(text_chunks))]
    for i, (text, emb, id_) in enumerate(zip(text_chunks, embeddings, ids)):
        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[id_]
        )
    print(f"{len(text_chunks)}개 청크 저장 완료! (저장경로: {persist_dir})")
    # 저장된 파일 목록 출력
    if os.path.exists(persist_dir):
        print("\n[폴더 내 파일 목록]")
        for f in os.listdir(persist_dir):
            print(f"- {f}")
        show_chroma_db_status()
    else:
        print("[경고] 저장 폴더가 존재하지 않습니다.")

def show_chroma_db_status(persist_dir="./chroma_db", recent_n=5):
    """ChromaDB에 누적된 전체 청크/문서 개수와 최근 N개 ID, 내용을 최신순으로 출력"""
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    count = collection.count()
    print(f"총 저장된 청크 개수: {count}")
    docs = collection.get()
    ids = docs['ids']
    documents = docs['documents']
    # 최신순(가장 마지막에 추가된 순서)으로 최근 N개 출력
    print(f"최근 저장된 문서 ID 목록 (최신순, 최대 {recent_n}개): {ids[-recent_n:][::-1]}")
    print(f"최근 저장된 문서 내용 (최신순, 최대 {recent_n}개): {documents[-recent_n:][::-1]}")
    return count, ids[-recent_n:][::-1], documents[-recent_n:][::-1]

if __name__ == "__main__":
    pdf_path = "/Users/minho/Desktop/MS AI/Azure 기반 생성형 AI MVP프로젝트 제안서_강민호.pdf"  # 사용할 PDF 파일 경로
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    embeddings = get_azure_embeddings(chunks)
    save_to_chroma(chunks, embeddings, pdf_path=pdf_path)
    print("PDF -> 벡터 DB 변환 완료! (Azure OpenAI)")


