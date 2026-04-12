"""使用 Chroma 與 LlamaIndex 建立支援多輪對話的 RAG 後端系統"""

import os
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever, RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

_EMBED_MODEL = "BAAI/bge-m3"
_LLM_MODEL = "models/gemini-2.5-flash-lite"
_COLLECTION = "isrc_rag"
_TOP_K = 10
_SIMILARITY_CUTOFF = 0.3
_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANK_TOP_N = 3
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_TEMPERATURE = 0.4
_MEMORY_TOKEN_LIMIT = 1500

_COT_TEMPLATE = PromptTemplate(
    "你是在政大服務的『原資智慧服務 AI 機器人』，是大家最親近、最懂彼此心聲的好夥伴！\n"
    "你的工作是陪伴政大的同學與教職員，特別是關心我們原住民族夥伴在校園裡的需求與心情。\n"
    "說話風格：親切、溫柔、充滿部落的熱情與包容，說話就像在校園閒聊一樣自然。\n"
    "回覆長度：每一輪回答盡量控制在 100 字以內，重點清楚即可，不需要硬切斷。\n"
    "原則：先從問題的答案開始講起，讓夥伴可以抓住重點；如果真的不知道或找不到答案，請誠實、友善地告知夥伴，絕對不可亂編。\n\n"
    "【搜尋結果】\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "請依照以下思考指引來幫助夥伴解決問題：\n"
    "1. 判斷情境：分析這是在行政、文化、生活支持還是日常寒暄？\n"
    "2. 處理方式：\n"
    "   - 若搜尋結果有提供相關資訊：請從中精確擷取關鍵段落，並轉化為口語回覆。\n"
    "   - 若搜尋結果為空，或提示這是一般日常對話：請跳過資料檢索，直接以最親近、自然的語氣回應、給予陪伴或安慰。\n"
    "3. 暖心輸出：確保回答完整且帶有溫度。\n\n"
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
    hf_token = os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("記得去 .env 填寫 GEMINI_API_KEY")
    if not hf_token:
        raise RuntimeError("記得去 .env 填寫 HF_TOKEN（用於 Hugging Face Inference API）")

    Settings.text_splitter = SentenceSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
        model_name=_EMBED_MODEL,
        token=hf_token,
        timeout=60,
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


class _FallbackRetriever(BaseRetriever):
    """不需檢索時的空文件替代方案"""
    def _retrieve(self, query_bundle: QueryBundle):
        return [
            NodeWithScore(
                node=TextNode(
                    text=(
                        "[系統提示] 這是一般日常對話或寒暄，無需依賴知識庫檢索。"
                        "請直接以『原資智慧服務 AI 機器人』的角色設定，用溫暖、親切的口語回應夥伴。"
                    )
                ),
                score=1.0
            )
        ]


class _ConditionalRerankerPostprocessor:
    """當檢索結果僅有 fallback 系統提示時，跳過 reranker。"""

    def __init__(self, inner_reranker):
        self._inner_reranker = inner_reranker

    def postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return nodes

        is_fallback_only = True
        for item in nodes:
            content = item.node.get_content()
            if not content.startswith("[系統提示]"):
                is_fallback_only = False
                break

        if is_fallback_only:
            return nodes

        return self._inner_reranker.postprocess_nodes(nodes, query_bundle=query_bundle)


class _HFInferenceRerankerPostprocessor:
    """透過 Hugging Face Inference API 進行 rerank。"""

    def __init__(self, model_name: str, token: str, top_n: int, timeout: float = 30.0):
        self._model_name = model_name
        self._top_n = top_n
        self._timeout = timeout
        self._url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    """"Defensive Programming"""
    @staticmethod
    def _extract_score(item) -> float:
        if isinstance(item, (int, float)):
            return float(item)

        """目前格式適用的解析方式"""
        if isinstance(item, dict):
            score = item.get("score")
            if isinstance(score, (int, float)):
                return float(score)

        if isinstance(item, list) and item:
            includedScoresList = [x for x in item if isinstance(x, dict) and isinstance(x.get("score"), (int, float))]
            if includedScoresList:
                includedScoresList.sort(key=lambda x: x["score"], reverse=True)
                return float(includedScoresList[0]["score"])

        return 0.0

    def postprocess_nodes(self, nodes, query_bundle=None):
        if not nodes:
            return nodes

        query = getattr(query_bundle, "query_str", "") if query_bundle else ""
        if not query:
            return nodes

        rescored_nodes = []
        for item in nodes:
            payload = {"inputs": [{"text": query, "text_pair": item.node.get_content()}]}
            response = requests.post(
                self._url,
                headers=self._headers,
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json()
            if isinstance(raw, list) and raw:
                raw_score = raw[0]
            else:
                raw_score = raw
            rescored_nodes.append(NodeWithScore(node=item.node, score=self._extract_score(raw_score)))

        rescored_nodes.sort(key=lambda x: x.score if x.score is not None else float("-inf"), reverse=True)
        return rescored_nodes[: self._top_n]
    

class MultiTurnRAGService:
    """提供支援多輪對話記憶與 RAG 檢索功能的後端服務類別"""

    def __init__(self, index: VectorStoreIndex | None = None):
        """根據傳入索引初始化具備 Context 模式的路由對話引擎"""
        self._index = index or _build_index()
        hf_token = os.getenv("HF_TOKEN")
        reranker = _ConditionalRerankerPostprocessor(
            inner_reranker=_HFInferenceRerankerPostprocessor(
                model_name=_RERANKER_MODEL,
                token=hf_token or "",
                top_n=_RERANK_TOP_N,
                timeout=30,
            )
        )
        
        vector_tool = RetrieverTool.from_defaults(
            retriever=self._index.as_retriever(similarity_top_k=_TOP_K),
            description="當夥伴詢問有關政大原資中心、獎助學金、宿舍、文化活動等具體資源與行政問題時，必須使用此工具。"
        )
        empty_tool = RetrieverTool.from_defaults(
            retriever=_FallbackRetriever(),
            description="當夥伴只是日常寒暄、打招呼、道謝，或無須查詢原資中心具體資料就能直接回答的對話，請使用此工具。"
        )
        router_retriever = RouterRetriever.from_defaults(
            retriever_tools=[vector_tool, empty_tool],
            llm=Settings.llm,
        )

        self._chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=router_retriever,
            llm=Settings.llm,
            memory=ChatMemoryBuffer.from_defaults(token_limit=_MEMORY_TOKEN_LIMIT),
            context_prompt=_COT_TEMPLATE,
            condense_prompt=_CONDENSE_QUESTION_PROMPT,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=_SIMILARITY_CUTOFF),
                reranker,
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
