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

# ChromaDB 저장 경로 (고정)
PERSIST_DIR = "./chroma_db"

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
def search_chroma(query, top_k=3):
    """
    ChromaDB에서 유사한 문서를 검색합니다.
    저장 경로는 ./chroma_db로 고정됩니다.
    """
    persist_dir = "./chroma_db"  # 고정된 저장 경로
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

# 통합 검색 함수 (PDF + 대화 기록)
def search_all_content(query, pdf_top_k=3, conversation_top_k=2):
    """
    PDF 내용과 대화 기록을 통합하여 검색합니다.
    
    Args:
        query: 검색할 쿼리
        pdf_top_k: PDF에서 검색할 최대 결과 수
        conversation_top_k: 대화 기록에서 검색할 최대 결과 수
    
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
        
        # 3. 통합 컨텍스트 구성
        context_parts = []
        
        if pdf_chunks:
            context_parts.append("=== PDF 문서 내용 ===")
            context_parts.append("\n\n".join(pdf_chunks))
        
        if conversation_history:
            context_parts.append("=== 관련 대화 기록 ===")
            context_parts.extend(conversation_history)
        
        if context_parts:
            result['context_text'] = "\n\n".join(context_parts)
        
        return result
        
    except Exception as e:
        print(f"통합 검색 중 오류: {e}")
        return result
