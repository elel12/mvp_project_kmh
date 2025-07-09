import os
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

PERSIST_DIR = os.getenv("PERSIST_DIR")

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

# ChromaDB 검색 함수
def search_chroma(query, top_k=3, persist_dir=PERSIST_DIR):
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection("pdf_collection")
    query_emb = get_query_embedding(query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        include=["documents"]
    )
    return results["documents"][0] if results["documents"] else []

# PDF 관련 함수는 pdf_to_vectordb.py에서 import하여 그대로 사용
# extract_text_from_pdf, split_text, get_azure_embeddings, save_to_chroma
