"""Google Sheets Logger: 把多輪對話紀錄寫入 Google Sheet"""

import datetime
import os

import gspread
from google.oauth2.service_account import Credentials

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets", 
    "https://www.googleapis.com/auth/drive",
]


_GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")


def _get_gspread_client() -> gspread.Client | None:
    """
    使用服務帳戶金鑰檔案路徑初始化並回傳 gspread client
    如果缺少金鑰路徑或工作表名稱，則回傳 None
    """
    if not _GOOGLE_APPLICATION_CREDENTIALS or not _SHEET_NAME:
        print("記得去 .env 填寫 GOOGLE_APPLICATION_CREDENTIALS 或是 GOOGLE_SHEET_NAME")
        return None

    try:
        creds = Credentials.from_service_account_file(
            _GOOGLE_APPLICATION_CREDENTIALS, 
            scopes=_SCOPES
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        print(f"初始化 gspread 失敗: {e}")
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
        worksheet = spreadsheet.worksheet("第六版 RAG")

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
