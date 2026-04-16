"""Google Sheets Logger: 把多輪對話紀錄寫入 Google Sheet"""

import datetime
import json
import os

import gspread
from google.oauth2.service_account import Credentials

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets", 
    "https://www.googleapis.com/auth/drive",
]

_GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")


def _get_gspread_client() -> gspread.Client | None:
    """
    使用服務帳戶 credential 初始化並回傳 gspread client
    如果缺少 credential 或工作表名稱，則回傳 None
    """
    if not _GOOGLE_CREDENTIALS_JSON or not _SHEET_NAME:
        print("記得去 .env 填寫 GOOGLE_CREDENTIALS_JSON 或是 SHEET_NAME")
        return None

    try:
        creds_dict = json.loads(_GOOGLE_CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_dict, scopes=_SCOPES)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(e)
        return None


def log_to_sheet(
    conversation_id: str,
    turn_index: int,
    question: str,
    answer: str,
    sources: list[str],
) -> None:
    """將多輪對話中的單一輪問答寫入 Google Sheet。"""
    client = _get_gspread_client()
    if not client:
        return

    try:
        spreadsheet = client.open(_SHEET_NAME)
        worksheet = spreadsheet.worksheet("第四版 RAG")

        tz_taipei = datetime.timezone(datetime.timedelta(hours=8))
        timestamp = datetime.datetime.now(tz=tz_taipei).strftime("%Y-%m-%d %H:%M:%S")
        sources_str = "\n\n---\n\n".join(sources)

        new_row = [
            timestamp,
            conversation_id,
            turn_index,
            question,
            answer,
            sources_str,
        ]
        worksheet.append_row(new_row)
    except Exception as e:
        print(e)
