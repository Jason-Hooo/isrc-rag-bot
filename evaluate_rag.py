import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import shutil

import instructor
import gspread
import google.genai as genai
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from ragas.embeddings import GoogleEmbeddings
from ragas.llms.base import InstructorLLM
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from src.rag import MultiTurnRAGService

load_dotenv()

_google_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
_async_instructor = instructor.from_genai(_google_client, use_async=True)
_evaluator_llm = InstructorLLM(client=_async_instructor, model="gemini-2.5-flash", provider="google")
_evaluator_emb = GoogleEmbeddings(client=_google_client)

_m_faith  = Faithfulness(llm=_evaluator_llm)
_m_relev  = AnswerRelevancy(llm=_evaluator_llm, embeddings=_evaluator_emb)
_m_cprec  = ContextPrecision(llm=_evaluator_llm)
_m_crec   = ContextRecall(llm=_evaluator_llm)

_executor = ThreadPoolExecutor(max_workers=1)


async def _rag_answer(session, question: str) -> tuple[str, list[str]]:
    stream_gen, meta = session.stream_chat(question)
    answer = ""
    async def _collect():
        nonlocal answer
        async for delta in stream_gen:
            answer += delta
    await asyncio.wait_for(_collect(), timeout=120)
    contexts = [node.get_content() for node in meta.get("source_nodes", [])]
    return answer, contexts


def _ragas_score(question: str, answer: str, contexts: list[str], gt: str) -> tuple[float, float, float, float]:
    """回傳 (faithfulness, context_precision, context_recall, answer_relevancy)，503 時自動重試"""
    for attempt in range(4):
        try:
            f  = _m_faith.score(user_input=question, response=answer, retrieved_contexts=contexts)
            cp = _m_cprec.score(user_input=question, reference=gt, retrieved_contexts=contexts)
            cr = _m_crec.score(user_input=question, retrieved_contexts=contexts, reference=gt)
            r  = _m_relev.score(user_input=question, response=answer)
            return float(f.value), float(cp.value), float(cr.value), float(r.value)
        except Exception as e:
            if attempt < 3:
                wait = 15 * (attempt + 1)
                print(f"   ⏳ 評分失敗（{e.__class__.__name__}），{wait} 秒後重試（第 {attempt+1} 次）...")
                time.sleep(wait)
            else:
                raise


START_QUESTION = 1   # 從第幾題開始（1 = 第一題）
END_QUESTION   = None   # 到第幾題結束（None = 跑到最後一題）
FORCE_REBUILD  = True  # True = 強制刪除舊索引重建；False = 沿用現有索引


async def evaluate_task():
    print("🚀 啟動原寶 RAG 評測系統...")

    # --- Google Sheet ---
    try:
        creds_dict = json.loads(os.getenv("GOOGLE_CREDENTIALS_JSON"))
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(creds_dict).with_scopes(scopes)
        ws = gspread.authorize(creds).open("RAGAS評分").worksheet("工作表1")
    except Exception as e:
        print(f"❌ Google Sheet 連線失敗: {e}")
        return

    # --- 索引 ---
    chroma_path = Path(__file__).resolve().parent / "models" / "chroma_db"
    if FORCE_REBUILD and chroma_path.exists():
        shutil.rmtree(chroma_path)
        print("🗑️  已清除舊索引，重新建立...")
    elif chroma_path.exists():
        try:
            import chromadb as _cdb
            _cdb.PersistentClient(path=str(chroma_path)).get_or_create_collection("isrc_rag").count()
        except Exception:
            shutil.rmtree(chroma_path)
            print("⚠️  偵測到損壞的索引，自動清除重建...")
    print("🤖 正在啟動機器人大腦...")
    service = MultiTurnRAGService()

    end = END_QUESTION if END_QUESTION is None else END_QUESTION + 1
    questions = ws.col_values(1)[START_QUESTION:end]
    g_truths  = ws.col_values(2)[START_QUESTION:end]
    loop = asyncio.get_event_loop()

    for i, (q, gt) in enumerate(zip(questions, g_truths), start=START_QUESTION + 1):
        if not q or not q.strip():
            continue
        print(f"\n📝 第 {i-1} 題: {q}")

        try:
            service.reset()
            answer, contexts = await _rag_answer(service, q)

            if not answer.strip():
                print(f"⚠️  第 {i-1} 題：機器人回應為空，跳過")
                continue

            f_score, cp_score, cr_score, r_score = await loop.run_in_executor(
                _executor, _ragas_score, q, answer, contexts, gt
            )
            avg = (f_score + cp_score + cr_score + r_score) / 4

            # 欄位：C=答案 D=本題avg E=Faithfulness F=ContextPrecision G=ContextRecall H=AnswerRelevancy I=時間
            ws.update_cell(i, 3, answer)
            ws.update_cell(i, 4, round(avg, 2))
            ws.update_cell(i, 5, round(f_score, 2))
            ws.update_cell(i, 6, round(cp_score, 2))
            ws.update_cell(i, 7, round(cr_score, 2))
            ws.update_cell(i, 8, round(r_score, 2))
            ws.update_cell(i, 9, datetime.now().strftime("%H:%M:%S"))

            print(f"   Faithfulness      = {f_score:.2f}")
            print(f"   ContextPrecision  = {cp_score:.2f}")
            print(f"   ContextRecall     = {cr_score:.2f}")
            print(f"   AnswerRelevancy   = {r_score:.2f}")
            print(f"✅ 本題平均 = {avg:.2f}")
            await asyncio.sleep(8)

        except Exception as e:
            import traceback
            print(f"❌ 第 {i-1} 題發生錯誤: {e}")
            traceback.print_exc()

    print("\n✅ 評測完成")


if __name__ == "__main__":
    try:
        asyncio.run(evaluate_task())
    except KeyboardInterrupt:
        print("\n⛔ 使用者中止評測")
