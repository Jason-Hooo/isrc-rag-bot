"""streamlit 前端"""

import os
import tempfile
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if google_creds and google_creds.strip().startswith("{"):
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_file:
        temp_file.write(google_creds)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name

from src.rag import MultiTurnRAGService
from src.sheets_logger import log_to_sheet


st.set_page_config(page_title="原寶 - 原資中心智慧服務 AI 機器人")


@st.cache_resource(show_spinner="正在載入中，請稍候…") # 全域共用
def load_service():
    return MultiTurnRAGService()


st.title("原寶 - 原資中心智慧服務 AI 機器人")
st.caption(
    "協助同學查詢原住民族獎助學金、學雜費減免、住宿權益、文化活動與校園支持資源的對話式小幫手。"
)

service = load_service()

if "chat_session" not in st.session_state: # 每個使用者獨有
    st.session_state.chat_session = service.new_session()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid4())

if "turn_index" not in st.session_state:
    st.session_state.turn_index = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "哈囉夥伴，我是原寶，是原資智慧服務小幫手。想先了解獎助學金、學雜費減免、住宿權益，還是文化活動與校園支持資源呢？",
            "sources": []
        }
    ]

if st.button("重新開啟對話"):
    st.session_state.chat_session = service.new_session()
    st.session_state.conversation_id = str(uuid4())
    st.session_state.turn_index = 0
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "喔虧，我們重新開始吧！你對哪方面有疑問？",
            "sources": [] 
        }
    ]
    st.rerun()

question = st.chat_input("輸入你的問題...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        sources = list(message.get("sources") or [])
        if sources:
            with st.expander("參考資料來源"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**片段 {i}**")
                    st.write(src)
                    st.divider()

stream_slot = st.empty()

if question:
    with stream_slot.container():
        with st.chat_message("assistant"):
            status_container = st.empty()

            def update_status(msg: str):
                """更新前端狀態提示。"""
                status_container.info(msg)

            st.session_state.chat_session.status_callback = update_status

            with st.spinner("小幫手正在思考中…"):
                stream, meta = st.session_state.chat_session.stream_chat(question=question)
                answer = st.write_stream(stream)
                status_container.empty()
                source_nodes = meta.get("source_nodes", []) or []
                sources = list([node.get_content() for node in source_nodes])
                st.session_state.turn_index += 1
                log_to_sheet(
                    conversation_id=st.session_state.conversation_id,
                    turn_index=st.session_state.turn_index,
                    question=question,
                    answer=answer,
                    sources=sources,
                )

            if sources:
                with st.expander("參考來源"):
                    for i, src in enumerate(sources, start=1):
                        st.markdown(f"**片段 {i}**")
                        st.write(src)
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
