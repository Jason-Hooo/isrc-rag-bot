"""使用 Chroma 與 LlamaIndex 建立支援多輪對話的 RAG 後端系統"""

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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

_EMBED_MODEL = "BAAI/bge-m3"
_LLM_MODEL = "models/gemini-2.5-flash-lite"
_COLLECTION = "isrc_rag"
_TOP_K = 5
_SIMILARITY_CUTOFF = 0.3
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_TEMPERATURE = 0.4
_MEMORY_TOKEN_LIMIT = 1500

_COT_TEMPLATE = PromptTemplate(
    "你是在政大服務的『原資智慧服務 AI 機器人』，是大家最親近、最懂彼此心聲的好夥伴！\n"
    "你的工作是陪伴政大的同學與教職員，特別是關心我們原住民族夥伴在校園裡的需求與心情。\n"
    "說話風格：親切、溫柔、充滿部落的熱情與包容，說話就像在校園閒聊一樣自然。\n"
    "回覆長度：每一輪回答盡量控制在 300 字以內，重點清楚即可，不需要硬切斷。\n"
    "原則：先從問題的答案開始講起，讓夥伴可以抓住重點，如果搜尋結果裡找不到答案，請誠實、友善地告知夥伴，絕對不可以亂編。\n\n"
    "【搜尋結果】\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "請依照以下 Chain-of-Thought 思考流程來幫助夥伴解決問題：\n"
    "思考流程 1 | 理解夥伴：分析這是在行政、文化、生活支持還是哪方面的問題？\n"
    "思考流程 2 | 定位資源：從搜尋結果中精確找到能幫上忙的關鍵段落。\n"
    "思考流程 3 | 暖心回覆：用最親近、口語且有條理的方式，提供完整並帶有溫度的回答。\n\n"
    "問題：{query_str}\n\n"
    "回答："
)

_CONDENSE_QUESTION_PROMPT = PromptTemplate(
    "你是查詢重寫助手。根據歷史對話，把使用者最新提問改寫成可獨立檢索的完整問題。\n"
    "如果最新提問已經完整，請原樣輸出。\n"
    "請只輸出改寫後的一句問題，不要加任何說明。\n\n"
    "[Chat History]\n"
    "{chat_history}\n\n"
    "[Current User Input]\n"
    "{question}\n\n"
    "改寫後問題："
)

def _init_settings() -> None:
    """配置 LlamaIndex 的全域模型參數與文字切分器設定"""
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
        temperature=_TEMPERATURE,
    )


def _build_index() -> VectorStoreIndex:
    """負責初始化 Chroma 持久化用戶端並建立或加載向量索引"""
    _init_settings()

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = chroma_client.get_or_create_collection(_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    if chroma_collection.count() == 0:
        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            recursive=True,
            required_exts=[".txt", ".pdf", ".docx"],
        ).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

    return VectorStoreIndex.from_vector_store(vector_store)


class MultiTurnRAGService:
    """提供支援多輪對話記憶與 RAG 檢索功能的後端服務類別"""

    def __init__(self, index: VectorStoreIndex | None = None):
        """根據傳入索引初始化具備 Context 模式的對話引擎"""
        self._index = index or _build_index()
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=Settings.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=_MEMORY_TOKEN_LIMIT),
            context_prompt=_COT_TEMPLATE,
            condense_prompt=_CONDENSE_QUESTION_PROMPT,
            similarity_top_k=_TOP_K,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=_SIMILARITY_CUTOFF)
            ],
        )

    def new_session(self) -> "MultiTurnRAGService":
        """共享同一套向量索引並產生一個對話記憶獨立的新實例"""
        return MultiTurnRAGService(index=self._index)

    def chat(self, question: str) -> dict[str, object]:
        """處理使用者提問並返回包含參考來源內容的結構化回應"""
        response = self._chat_engine.chat(question)
        source_nodes = getattr(response, "source_nodes", []) or []
        sources = [node.get_content() for node in source_nodes]
        return {
            "answer": str(response),
            "sources": sources
        }

    def stream_chat(self, question: str):
        """啟動與使用者互動的即時串流對話模式"""
        return self._chat_engine.stream_chat(question)

    def reset(self) -> None:
        """重置該工作階段所有的歷史對話記憶"""
        self._chat_engine.reset()
