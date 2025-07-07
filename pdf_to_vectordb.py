import os
import pdfplumber
from openai import OpenAI
from chromadb import Client
from chromadb.config import Settings
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Azure OpenAI 환경변수
AZURE_OPENAI_API_KEY = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("TEXT_EMBEDDING_AZURE_OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("TEXT_EMBEDDING_DEPLOYMENT_NAME")

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

# 텍스트 임베딩 생성 함수
def get_embeddings(text_list):
    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings = []
    for text in text_list:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

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

def save_to_chroma(text_chunks, embeddings, persist_dir="/Users/minho/Desktop/mvp_project_kmh/chroma_db"):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    for i, (text, emb) in enumerate(zip(text_chunks, embeddings)):
        collection.add(
            documents=[text],
            embeddings=[emb],
            ids=[f"chunk_{i}"]
        )
    print(f"{len(text_chunks)}개 청크 저장 완료! (저장경로: {persist_dir})")
    # 저장된 파일 목록 출력
    import os
    if os.path.exists(persist_dir):
        print("\n[폴더 내 파일 목록]")
        for f in os.listdir(persist_dir):
            print(f"- {f}")
    else:
        print("[경고] 저장 폴더가 존재하지 않습니다.")

if __name__ == "__main__":
    pdf_path = "/Users/minho/Desktop/MS AI/pstn_voc.pdf"  # 사용할 PDF 파일 경로
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text(text)
    embeddings = get_azure_embeddings(chunks)
    save_to_chroma(chunks, embeddings)
    print("PDF -> 벡터 DB 변환 완료! (Azure OpenAI)")


