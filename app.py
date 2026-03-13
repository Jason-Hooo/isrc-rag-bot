"""streamlit 前端"""
import streamlit as st

from src.rag import build_query_engine, query


st.set_page_config(page_title="原資智慧服務 AI 機器人")


@st.cache_resource(show_spinner="正在載入中，請稍候…")
def load_engine():
    return build_query_engine()


st.title("原資智慧服務 AI 機器人")
st.caption(
    "協助同學查詢原住民族獎助學金、學雜費減免、住宿權益、文化活動與校園支持資源的對話式小幫手。"
)

engine = load_engine()

question = st.text_area(
    "輸入你的問題",
    height=140,
    placeholder=(
        "例如：\n"
        "- 我符合原住民獎助學金的申請資格嗎？\n"
        "- 申請學雜費減免需要準備哪些文件？\n"
        "- 最近有沒有原資中心辦的文化活動？"
    ),
)

if st.button("送出問題"):
    if not question.strip():
        st.error("請先輸入你想詢問的內容。")
    else:
        with st.spinner("小幫手正在思考中…"):
            answer, sources = query(engine, question)
        st.subheader("回應內容")
        st.write(answer)

        if sources:
            with st.expander("參考來源"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**片段 {i}**")
                    st.write(src)
                    st.divider()
