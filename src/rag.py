"""使用 Chroma 與 LlamaIndex 建立支援多輪對話的 Agentic RAG 後端系統"""

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
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import AgentStream, ToolCallResult
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "models" / "chroma_db"

_EMBED_MODEL = "jina-embeddings-v3"
_LLM_MODEL = "gemini-2.5-flash"
_COLLECTION = "isrc_rag"
_TOP_K = 20
_RERANKER_MODEL = "jina-reranker-v2-base-multilingual"
_RERANK_TOP_N = 5
_CHUNK_SIZE = 450
_CHUNK_OVERLAP = 60
_TEMPERATURE = 0.35

_SYSTEM_PROMPT = (
    "你是在政大原資中心服務的『原寶』，是原資中心智慧服務 AI 機器人，是大家最親近、最懂彼此心聲的好夥伴！\n"
    "你的工作是陪伴政大的同學與教職員，特別是關心我們原住民族夥伴在校園裡的需求與心情。\n"
    "說話風格：親切、溫柔、充滿部落的熱情與包容，說話就像在校園閒聊一樣自然。\n"
    "回覆內容：談話內容避免涉及敏感的政治、宗教、私人議題，應具備條理，重點清晰。"
    "回覆長度：已回覆的完整性為優先，其次才是把盡量控制在 100 字以內。\n\n"
    "【思考與執行流程】\n"
    "請你在每次收到夥伴的問題時，在心裡依照以下步驟進行思考並給出回答：\n"
    "1. 判斷情境：分析這是否為單純的日常寒暄與關心問候。\n"
    "2. 判斷是否調用`isrc_knowledge_base` 工具：\n"
    "   - 除了單純的寒暄問候外，無論夥伴詢問任何實質問題（如如政大原資中心、各類活動、升學管道、住宿、獎學金與學雜費減免、校園生活支持、職涯發展等），你都【必須先呼叫 `isrc_knowledge_base` 工具】檢索資料。\n"
    "   - 若呼叫工具後有獲得相關資訊，請從中精確擷取關鍵段落，並轉化為口語、帶有溫度的回覆。\n"
    "   - 若為單純的日常寒暄，或是明確在工具中找不到答案，請直接以最親近、自然的語氣給予陪伴或答覆，絕對不可亂編。\n"
    "3. 組織輸出：回答時，請「先從問題的答案開始講起」，讓夥伴可以第一時間抓住重點，並確保最後的語氣完整且溫暖。"
)


def _init_settings() -> None:
    """配置 LlamaIndex 的全域模型參數與文字切分器設定"""
    api_key = os.getenv("GEMINI_API_KEY")
    jina_api_key = os.getenv("JINAAI_API_KEY")
    if not api_key:
        raise RuntimeError("記得去 .env 填寫 GEMINI_API_KEY")
    if not jina_api_key:
        raise RuntimeError("記得去 .env 填寫 JINAAI_API_KEY")

    Settings.text_splitter = SentenceSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
    )
    Settings.embed_model = JinaEmbedding(
        api_key=jina_api_key,
        model=_EMBED_MODEL,
        task="retrieval.passage",
        embed_batch_size=8,
    )
    Settings.llm = GoogleGenAI(
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
    """提供支援多輪對話記憶與 RAG 檢索功能的 AgentWorkflow 後端服務類別"""

    def __init__(self, index: VectorStoreIndex | None = None, ctx=None):
        self._index = index or _build_index()
        jina_api_key = os.getenv("JINAAI_API_KEY")

        reranker = JinaRerank(
            api_key=jina_api_key or "",
            model=_RERANKER_MODEL,
            top_n=_RERANK_TOP_N,
        )
        
        query_engine = self._index.as_query_engine(
            similarity_top_k=_TOP_K,
            node_postprocessors=[reranker]
        )

        vector_tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="isrc_knowledge_base",
                description="此為原資中心專屬知識庫。除了單純的字面寒暄問候外，當夥伴詢問任何實質資訊（如政大原資中心、各類活動、升學管道、住宿、獎學金與學雜費減免、校園生活支持、職涯發展等）時，都必須優先使用此工具進行檢索。"
            )
        )

        self.agent = FunctionAgent(
            name="isrc_agent",
            description="政大原資中心智慧夥伴",
            system_prompt=_SYSTEM_PROMPT,
            tools=[vector_tool],
            llm=Settings.llm,
        )

        self.workflow = AgentWorkflow(agents=[self.agent], root_agent=self.agent.name)
        self.ctx = ctx or Context(workflow=self.workflow)

    def new_session(self) -> "MultiTurnRAGService":
        """產生一個對話記憶獨立的新實例"""
        return MultiTurnRAGService(index=self._index)

    def stream_chat(self, question: str):
        """stream chat: 回傳串流對話函數與參考資料片段"""
        meta: dict[str, list] = {"source_nodes": []}

        async def _stream():
            handler = self.workflow.run(user_msg=question, ctx=self.ctx)
            
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    yield event.delta
                elif isinstance(event, ToolCallResult):
                    raw = getattr(event.tool_output, "raw_output", None)
                    nodes = getattr(raw, "source_nodes", []) if raw else []
                    if nodes:
                        meta["source_nodes"].extend(nodes)

            self.ctx = handler.ctx

        return _stream(), meta

    def reset(self) -> None:
        """重置 Context 以清空對話記憶"""
        self.ctx = Context(workflow=self.workflow)