"""RAG 後端核心：LlamaIndex · ChromaDB · BGE-M3 · Gemini 2.5 Flash · Chain-of-Thought"""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

# 路徑常數 
BASE_DIR = Path(__file__).resolve().parent.parent # 專案的根目錄
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

# 模型設定 
_EMBED_MODEL = "BAAI/bge-m3"
_LLM_MODEL = "models/gemini-2.5-flash"
_COLLECTION = "isrc_rag"
_TOP_K = 5
_SIMILARITY_CUTOFF = 0.3
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120

# Chain-of-Thought Prompt 
_COT_TEMPLATE = PromptTemplate(
    "你是『原資智慧服務 AI 機器人』，服務政大校園中的學生與教職員，"
    "特別關注原住民族學生的需求。\n"
    "回答必須公正、客觀、尊重，不得有任何歧視或刻板印象化的描述。\n"
    "若搜尋結果中無明確答案，請誠實說明，不得捏造。\n\n"
    "【搜尋結果】\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "請依照以下 Chain-of-Thought 思考流程回答問題：\n"
    "思考流程 1｜辨識問題類型：判斷這是行政流程、文化認識、校園支持資源還是其他類型。\n"
    "思考流程 2｜定位相關資訊：從搜尋結果中找出與問題最相關的段落。\n"
    "思考流程 3｜整合與回答：根據上述資訊，以清楚、友善且有條理的語氣提供完整回答。\n\n"
    "問題：{query_str}\n\n"
    "回答："
)


def _init_settings() -> None:
    """初始化 LlamaIndex 全域 Embedding 與 LLM 設定"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("記得去 .env 填寫 GEMINI_API_KEY")

    Settings.text_splitter = SentenceSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=_EMBED_MODEL,
        token=os.getenv("HF_TOKEN"),
        trust_remote_code=True,
    )
    Settings.llm = Gemini(
        model=_LLM_MODEL,
        api_key=api_key,
        temperature=0.2,
    )


def build_query_engine() -> RetrieverQueryEngine:
    """
    建立並回傳 RetrieverQueryEngine。
    - 若 Chroma 集合為空，自動讀取 data/ 下所有文件並建立向量索引。
    - 若已有向量，直接從 Chroma 載入，不重新計算 embedding。
    """
    _init_settings()


    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection) 

    if chroma_collection.count() == 0:
        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            recursive=True,
            required_exts=[".txt", ".pdf", ".docx"]
        ).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )
    else:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=_TOP_K)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        text_qa_template=_COT_TEMPLATE,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=_SIMILARITY_CUTOFF)
        ],
    )
    return query_engine


def query(
    engine: RetrieverQueryEngine, question: str
) -> tuple[str, list[str]]:
    """向 Query Engine 提問，回傳 (回答文字, 參考片段清單)。"""
    response = engine.query(question)
    answer = str(response)
    sources = [node.get_content() for node in response.source_nodes]
    return answer, sources
