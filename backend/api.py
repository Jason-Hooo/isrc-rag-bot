# backend/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORS
from pydantic import BaseModel
from src.rag import MultiTurnRAGService
from src.sheets_logger import log_to_sheet
from uuid import uuid4

app = FastAPI()

# 設定 CORS (非常重要！允許 React 前端呼叫這個 API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"], # React 預設的 port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入你原本的 RAG 服務 (全域變數)
rag_service = MultiTurnRAGService()

# 儲存對話 Session (簡單模擬，正式環境可放 Redis 或 Database)
sessions = {}

# 定義前端傳來的資料格式
class ChatRequest(BaseModel):
    session_id: str
    question: str

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    question = request.question
    
    # 1. 取得或建立 Session
    if session_id not in sessions:
        sessions[session_id] = {
            "service": rag_service.new_session(),
            "turn_index": 0
        }
    
    current_session = sessions[session_id]
    
    # 2. 呼叫 RAG (因為你原本是用 async generator (stream_chat)，這裡要處理一下)
    # 注意：FastAPI 也可以回傳 StreamingResponse，但為了剛開始串接簡單，我們先把它組合成完整字串回傳
    stream, meta = current_session["service"].stream_chat(question=question)
    
    answer_text = ""
    async for chunk in stream:
        answer_text += chunk
        
    source_nodes = meta.get("source_nodes", []) or []
    sources = [node.get_content() for node in source_nodes]
    
    # 3. 記錄到 Google Sheets
    current_session["turn_index"] += 1
    log_to_sheet(
        conversation_id=session_id,
        turn_index=current_session["turn_index"],
        question=question,
        answer=answer_text,
        sources=sources,
    )
    
    # 4. 回傳給前端 React
    return {
        "answer": answer_text,
        "sources": sources
    }

# 啟動伺服器的指令 (在終端機輸入)：
# uvicorn api:app --reload