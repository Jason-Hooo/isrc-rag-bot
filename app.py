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

TOPIC_LABELS = {
    "resources": "校園資源",
    "issues": "原民議題",
}

TOPIC_WELCOME_MESSAGES = {
    "resources": "哈囉夥伴，我是原寶，是原資智慧服務小幫手。想先了解校園資源、獎助學金、學雜費減免、住宿權益，還是文化活動呢？",
    "issues": "哈囉夥伴，我是原寶，是原資智慧服務小幫手，這裡可以與你一起討論原住民相關的議題～你對哪方面比較有興趣呢？",
}

TOPIC_SELECTION_MESSAGE = (
    "哈囉夥伴，我是原寶。要開始聊天前，請先選擇你想查詢的主題。"
)


@st.cache_resource(show_spinner="正在載入中，請稍候…") # 全域共用
def load_service():
    """載入共用的 RAG 服務實例。"""
    return MultiTurnRAGService(topic="resources")


st.title("原寶 - 原資中心智慧服務 AI 機器人")
st.caption(
    "協助同學查詢原住民族獎助學金、學雜費減免、住宿權益、文化活動與校園支持資源的對話式小幫手。"
)

service = load_service()


def _start_topic_session(topic: str) -> None:
    """初始化指定主題的對話狀態。"""
    st.session_state.selected_topic = topic
    st.session_state.chat_session = service.new_session(topic=topic)
    st.session_state.conversation_id = str(uuid4())
    st.session_state.turn_index = 0
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": TOPIC_WELCOME_MESSAGES[topic],
            "sources": [],
        }
    ]


def _reset_to_topic_selection() -> None:
    """清空目前對話狀態，回到主題選擇畫面。"""
    st.session_state.selected_topic = None
    st.session_state.chat_session = None
    st.session_state.conversation_id = None
    st.session_state.turn_index = 0
    st.session_state.messages = []

if "selected_topic" not in st.session_state:
    st.session_state.selected_topic = None

if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

if "turn_index" not in st.session_state:
    st.session_state.turn_index = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.session_state.selected_topic is None:
    with st.chat_message("assistant"):
        st.write(TOPIC_SELECTION_MESSAGE)
        topic_col_1, topic_col_2 = st.columns(2)
        with topic_col_1:
            if st.button("校園資源", use_container_width=True):
                _start_topic_session("resources")
                st.rerun()
        with topic_col_2:
            if st.button("原民議題", use_container_width=True):
                _start_topic_session("issues")
                st.rerun()
    st.stop()

if "chat_session" not in st.session_state or st.session_state.chat_session is None: # 每個使用者獨有
    _start_topic_session(st.session_state.selected_topic)

st.caption(f"目前主題：{TOPIC_LABELS[st.session_state.selected_topic]}")

if st.button("重新開啟對話"):
    _reset_to_topic_selection()
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
                try:
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
                except Exception as e:
                    status_container.error(f"發生錯誤: {e}")
                    st.error(f"對話過程發生錯誤: {e}")
                    raise

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
