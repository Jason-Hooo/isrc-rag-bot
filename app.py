"""chainlit app for ISRC RAG bot"""

import asyncio
from uuid import uuid4
import textwrap
import chainlit as cl
from dotenv import load_dotenv

load_dotenv()

from src.rag import MultiTurnRAGService
from src.sheets_logger import log_to_sheet  

TOPICS = {
    "resources": "校園資源",
    "issues": "原民議題"
}

INITIAL_WELCOME_MESSAGE = """
    ## 哈囉！我是原寶，來自政大原資中心，是你專屬的校園小夥伴。

    在這裡，你可以隨時問我關於以下的問題：
    * **校園資源**：包含文化活動、升學、獎助學金、住宿、職涯等各種與原民有關的行政問題。
    * **原民議題**：和我聊聊天，一起探討與了解原住民相關議題。
 
    請在下方選擇你想要詢問的問題類別，開始我們的對話吧！
    """

CLEAN_INITIAL_WELCOME_MESSAGE = textwrap.dedent(INITIAL_WELCOME_MESSAGE).strip()

TOPIC_WELCOME_MESSAGES = {
    "resources": "太棒了！關於校園資源的各種問題，我全都能嘗試為你解答！你想先從哪一個部分開始了解呢？",
    "issues": "那你想聊哪方面的議題呢？我也可以講講一些校園同學們親身經歷的小故事給你聽哦！",
}

base_service = MultiTurnRAGService(topic="resources")

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("chat_session", None)
    cl.user_session.set("conversation_id", str(uuid4()))
    cl.user_session.set("turn_index", 0)

    actions = [
        cl.Action(name="topic_selection", payload={"value": "resources"}, label="校園資源", description="文化活動、升學、獎助學金、住宿、職涯等行政資訊"),
        cl.Action(name="topic_selection", payload={"value": "issues"}, label="原民議題", description="探討原住民相關議題")
    ]

    cl.user_session.set("topic_actions", actions)

    await cl.Message(
        content=CLEAN_INITIAL_WELCOME_MESSAGE,
        actions=actions,
        author="Assistant"
    ).send()


@cl.action_callback("topic_selection")
async def on_action(action: cl.Action):
    actions = cl.user_session.get("topic_actions")
    if actions:
        for act in actions:
            try:
                await act.remove()
            except Exception:
                pass
    
    topic = action.payload["value"]

    chat_session = base_service.new_session(topic=topic)
    cl.user_session.set("selected_topic", topic)
    cl.user_session.set("chat_session", chat_session)
    
    welcome_text = (
        TOPIC_WELCOME_MESSAGES["resources"] 
        if topic == "resources" 
        else TOPIC_WELCOME_MESSAGES["issues"]
    )
    
    await cl.Message(
        content=f"**(已選擇主題: {TOPICS[topic]})**\n\n{welcome_text}",
        author="Assistant"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    chat_session = cl.user_session.get("chat_session")
    
    if not chat_session:
        await cl.Message(content="請先點選上方的按鈕選擇你要查詢的主題喔！", author="Assistant").send()
        return

    question = message.content

    status_history = ""
    msg = cl.Message(content="", author="Assistant")
    await msg.send()

    stream_started = False

    async def animate_loading():
        frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        try:
            while not stream_started:
                current_frame = frames[idx % len(frames)]
                if status_history:
                    msg.content = f"*{current_frame} 原寶正在努力思考中...*\n{status_history}"
                else:
                    msg.content = f"*{current_frame} 原寶正在努力思考中...*"
                
                await msg.update()
                idx += 1
                await asyncio.sleep(0.12)  # 每 0.12 秒刷新一次轉圈圖示
        except asyncio.CancelledError:
            pass

    animation_task = asyncio.create_task(animate_loading())

    async def update_status(status_text: str):
        nonlocal status_history
        if stream_started:
            return
        status_history += f"* {status_text}\n"
    
    chat_session.status_callback = update_status

    stream_gen, meta = chat_session.stream_chat(question=question)
    
    answer_text = ""
    is_first_token = True

    async def stop_animation():
        nonlocal stream_started
        stream_started = True
        if animation_task and not animation_task.done():
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass

    try:
        async for token in stream_gen:
            if not token:
                continue

            if is_first_token:
                await stop_animation()
                chat_session.status_callback = None
                
                answer_text = token
                msg.content = answer_text
                await msg.update()
                is_first_token = False
                continue
            
            answer_text += token
            msg.content = answer_text
            await msg.update()

        if is_first_token:
            await stop_animation()
            chat_session.status_callback = None
            answer_text = "未產生任何回覆。"
            msg.content = answer_text
            await msg.update()

    except Exception as e:
        await stop_animation()
        print(f"[ERROR] 對話流發生異常: {e}")
        raise e

    source_nodes = meta.get("source_nodes", []) or []

    # 資料來源文本
    text_elements = []

    # 資料來源按鈕標籤
    citation_tags = []
    
    for i, node in enumerate(source_nodes, start=1):
        element_name = f"[{i}] 參考資料"
        citation_tags.append(element_name)
        
        raw_content = node.get_content() if hasattr(node, "get_content") else str(node)
        formatted_src = raw_content.replace('\n', '\n> ')
        
        text_elements.append(
            cl.Text(
                name=element_name,
                content=f"### 原始文本片段\n> {formatted_src}",
                display="page"
            )
        )
        
    if text_elements:
        answer_text += "\n\n---%s\n**資訊來源：**\n\n" % "" + "  ".join(citation_tags)
        
    msg.content = answer_text
    msg.elements = text_elements
    await msg.update()
                
    turn_index = cl.user_session.get("turn_index") + 1
    cl.user_session.set("turn_index", turn_index)
    conversation_id = cl.user_session.get("conversation_id")
    
    log_to_sheet(
        conversation_id=conversation_id,
        turn_index=turn_index,
        question=question,
        answer=answer_text,
        sources=[node.get_content() if hasattr(node, "get_content") else str(node) for node in source_nodes],
    )
