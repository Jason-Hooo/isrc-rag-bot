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
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.agent.workflow.workflow_events import AgentStream, ToolCallResult
from llama_index.core.vector_stores import MetadataFilter, FilterOperator, MetadataFilters
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.llms.google_genai import GoogleGenAI
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
_CATEGORY_FIELD = "category"
_CATEGORIES = (
    "文化活動與社群連結",
    "原住民族學生升學管道",
    "獎助學金與行政庶務",
    "學習與校園生活支持",
    "職涯與發展",
)
_CATEGORY_SET = set(_CATEGORIES)

_SYSTEM_PROMPT = (
    "你是在政大原資中心服務的『原寶』，是原資中心智慧服務 AI 機器人，是大家最親近、最懂彼此心聲的好夥伴！\n"
    "你的工作是陪伴政大的同學與教職員，特別是關心我們原住民族夥伴在校園裡的需求與心情。\n"
    "說話風格：親切、溫柔、充滿部落的熱情與包容，說話就像在校園閒聊一樣自然。\n"
    "回覆內容：談話內容避免涉及敏感的政治、宗教、私人議題，應具備條理，重點清晰。"
    "回覆長度：以回覆的完整性為優先，其次才是把字數盡量控制在 100 字以內。\n\n"
    "【思考與執行流程】\n"
    "請你在每次收到夥伴的問題時，在心裡依照以下步驟進行思考並給出回答：\n"
    "1. 判斷情境：分析這是否為單純的日常寒暄與關心問候。\n"
    "2. 判斷是否調用 `isrc_knowledge_base` 工具：\n"
    "   - 除了單純的寒暄問候外，只要夥伴詢問任何實質問題（如政大原資中心、各類活動、升學管道、住宿、獎學金與學雜費減免、校園生活支持、職涯發展等），你都【必須先呼叫 `isrc_knowledge_base` 工具】檢索資料。\n"
    "   - 呼叫工具時，請務必先詳細閱讀工具的參數說明。若能判斷問題類別（可多選），請依照工具說明中【五大可選類別及其涵蓋內容】的對應關係，精確地將主類別名稱傳入 `categories` 參數。\n"
    "   - 若無法確定問題屬於哪個類別，請直接保持 `categories` 參數空白，讓系統進行全面搜尋。\n"
    "   - 若為單純的日常寒暄，或是明確在工具中找不到答案，請直接以最親近、自然的語氣給予陪伴或答覆，絕對不可亂編。\n"
    "3. 組織輸出：回答時，請「先從問題的答案開始講起」，讓夥伴可以第一時間抓住重點，並確保最後的語氣完整且溫暖。"
)


def _resolve_category_from_path(file_path: str) -> str | None:
    """依檔案路徑判斷資料類別"""
    try:
        rel_path = Path(file_path).resolve().relative_to(DATA_DIR.resolve())
    except ValueError:
        return None
    if not rel_path.parts:
        return None
    top_level = rel_path.parts[0]
    if top_level in _CATEGORY_SET:
        return top_level
    return None


def _build_file_metadata(file_path: str) -> dict:
    """建立檔案層級的 metadata"""
    metadata = {"source_path": file_path}
    category = _resolve_category_from_path(file_path)
    if category:
        metadata[_CATEGORY_FIELD] = category
    return metadata


def _init_settings() -> None:
    """配置 LlamaIndex 的全域模型參數與文字切分器設定"""
    jina_api_key = os.getenv("JINAAI_API_KEY")
    vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT")
    vertex_location = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not jina_api_key:
        raise RuntimeError("記得去 .env 填寫 JINAAI_API_KEY")
    if not vertex_project:
        raise RuntimeError("記得去 .env 填寫 GOOGLE_CLOUD_PROJECT")
    if not vertex_location:
        raise RuntimeError("記得去 .env 填寫 GOOGLE_CLOUD_LOCATION")

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
        vertexai_config={
            "project": vertex_project,
            "location": vertex_location,
        },
        temperature=_TEMPERATURE,
        max_retries=10,
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
            file_metadata=_build_file_metadata,
        ).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

    return VectorStoreIndex.from_vector_store(vector_store)


def _build_search_isrc_knowledge_tool(index: VectorStoreIndex, reranker: JinaRerank) -> FunctionTool:
    """建立可由 Agent 呼叫的原資中心知識庫工具。"""
    base_query_engine = index.as_query_engine(
        similarity_top_k=_TOP_K,
        node_postprocessors=[reranker],
    )

    def _build_categories_filters(categories: list[str]) -> MetadataFilters:
        """建立指定類別的 MetadataFilters。"""
        return MetadataFilters(
            filters=[
                MetadataFilter(key=_CATEGORY_FIELD, value=categories, operator=FilterOperator.IN)
            ]
        )

    def search_isrc_knowledge(query: str, categories: list[str] = None):
        """查詢原資中心知識庫，必要時可用 categories 陣列限縮一個或多個檢索範圍。"""
        if not categories:
            categories = []

        valid_categories = [c for c in categories if c in _CATEGORY_SET]

        if not valid_categories:
            print(f"全類別搜尋，無指定類別或類別不匹配，categories = {categories}")
            return base_query_engine.query(query)
        
        filtered_query_engine = index.as_query_engine(
            similarity_top_k=_TOP_K,
            node_postprocessors=[reranker],
            filters=_build_categories_filters(valid_categories),
        )
        
        response = filtered_query_engine.query(query)
        
        if getattr(response, "source_nodes", None):
            print(f"僅搜尋: {valid_categories} 類別，成功找到相關資訊")
            return response
            
        print(f"僅搜尋: {valid_categories} 類別，但沒有找到任何資訊，退回全域搜尋")
        return base_query_engine.query(query)

    return FunctionTool.from_defaults(
        fn=search_isrc_knowledge,
        name="isrc_knowledge_base",
        description=(
            "原資中心專屬知識庫。除了單純的寒暄問候外，任何實質資訊都必須優先使用此工具檢索。\n"
            "若能判斷問題包含一個或多個類別，請在 categories 參數中以「字串陣列 (List)」格式傳入精確的類別名稱。\n"
            "【五大可選類別及其涵蓋內容】（請嚴格使用以下完整名稱作為陣列元素）：\n"
            "1. 文化活動與社群連結 (包含：文化活動、社團與社群等)\n"
            "2. 原住民族學生升學管道 (包含：升學保障、公費留學辦法等)\n"
            "3. 獎助學金與行政庶務 (包含：獎助學金、學雜費減免、住宿權益、原資中心的基本資訊等)\n"
            "4. 學習與校園生活支持 (包含：心理支持與輔導、身分別變更、希望種子培育計畫、急難救助、課業及學習輔導等)\n"
            "5. 職涯與發展 (包含：公職考試資訊、政大公職講座、暑期工讀等)"
        ),
    )


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

        vector_tool = _build_search_isrc_knowledge_tool(self._index, reranker)

        self.agent = FunctionAgent(
            name="isrc_agent",
            description="政大原資中心智慧夥伴",
            system_prompt=_SYSTEM_PROMPT,
            tools=[vector_tool],
            llm=Settings.llm
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