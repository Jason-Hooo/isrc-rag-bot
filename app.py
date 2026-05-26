"""Chainlit 前端"""

import os
import tempfile
from uuid import uuid4

import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_creds and google_creds.strip().startswith("{"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
        temp_file.write(google_creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

from src.rag import MultiTurnRAGService
from src.sheets_logger import log_to_sheet

base_service = MultiTurnRAGService(topic="resources")


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_session", None)
    
    # 🌟 秘訣：直接把 chainlit.md 的內容寫在這裡，完美顯示！
    welcome_msg = """## 歡迎來到政大原資中心 AI 智慧服務
哈囉！我是**原寶**，你的專屬校園小幫手 🌟

在這裡，你可以隨時問我關於以下的問題：
* 💰 **獎助學金**：原民會、各縣市政府獎助學金申請資訊
* 🏠 **住宿權益**：校內住宿保障與申請辦法
* 🎉 **文化活動**：原資中心舉辦的聯誼與文化活動
* 💬 **原民議題**：一起探討與了解原住民相關議題

👉 **請點擊下方的按鈕，或者直接在對話框輸入你的問題！**"""

    actions = [
        cl.Action(name="topic_selection", payload={"value": "resources"}, label="🏫 校園資源", description="獎助學金、住宿、活動"),
        cl.Action(name="topic_selection", payload={"value": "issues"}, label="💬 原民議題", description="探討原住民相關議題")
    ]

    await cl.Message(
        content=welcome_msg,
        actions=actions,
        author="Assistant" # 👈 【已修正】統一為大寫 A，確保大頭貼完美顯示
    ).send()


@cl.action_callback("topic_selection")
async def on_action(action: cl.Action):
    topic = action.payload["value"]
    
    chat_session = base_service.new_session(topic=topic)
    cl.user_session.set("selected_topic", topic)
    cl.user_session.set("chat_session", chat_session)
    cl.user_session.set("conversation_id", str(uuid4()))
    cl.user_session.set("turn_index", 0)
    
    await action.remove()
    
    welcome_text = "想先了解校園資源、獎助學金、學雜費減免、住宿權益，還是文化活動呢？" if topic == "resources" else "這裡可以與你一起討論原住民相關的議題～你對哪方面比較有興趣呢？"
    
    await cl.Message(
        content=f"*(已選擇主題)*\n\n哈囉夥伴，{welcome_text}",
        author="Assistant" # 👈 統一為大寫 A
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    chat_session = cl.user_session.get("chat_session")
    
    if not chat_session:
        await cl.Message(content="請先點選上方的按鈕選擇你要查詢的主題喔！", author="Assistant").send()
        return

    question = message.content
    
    # 建立一個用來回覆的空白 Message 物件
    msg = cl.Message(content="", author="Assistant") # 👈 統一為大寫 A
    await msg.send()

    def update_status(status_msg: str):
        print(f"[系統狀態] {status_msg}")
    
    chat_session.status_callback = update_status
    stream_gen, meta = chat_session.stream_chat(question=question)
    
    answer_text = ""
    async for token in stream_gen:
        answer_text += token
        await msg.stream_token(token)
        
    source_nodes = meta.get("source_nodes", []) or []
    sources = [node.get_content() for node in source_nodes]
        
    # 更新最終畫面 (包含附加的參考片段)
    await msg.update()

    if sources:
        for i, src in enumerate(sources, start=1):
            async with cl.Step(name=f"🔍 參考片段 {i}") as step:
                # 這樣寫，每一個參考片段就會變成獨立的可點擊收合框！
                step.output = src
                
    # 紀錄到 Google Sheets
    turn_index = cl.user_session.get("turn_index") + 1
    cl.user_session.set("turn_index", turn_index)
    conversation_id = cl.user_session.get("conversation_id")
    
    log_to_sheet(
        conversation_id=conversation_id,
        turn_index=turn_index,
        question=question,
        answer=answer_text,
        sources=sources,
    )