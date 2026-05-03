# 原寶 (isrc-rag-bot) - 政大原資中心小幫手

這是一個基於檢索增強生成 (Agentic RAG) 技術開發的 AI 問答機器人「原寶」，專為**國立政治大學原住民族學生資源中心 (ISRC)** 打造。旨在透過充滿溫度與親切的語氣，協助同學與教職員查詢校園原住民相關資源、獎助學金、學雜費減免、住宿權益以及文化活動等資訊。

## 系統架構與相關資訊

- **RAG 框架**: LlamaIndex (使用 AgentWorkflow 支援多輪對話與工具呼叫)
- **語言模型 (LLM)**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **嵌入模型 (Embedding)**: JinaAI (`jina-embeddings-v3`)
- **重排序模型 (Reranker)**: JinaAI (`jina-reranker-v2-base-multilingual`)
- **向量資料庫**: ChromaDB (本地儲存)
- **前端介面**: Streamlit
- **問答紀錄與遙測**: Google Sheets API (透過 Service Account 紀錄使用者對話)

## 專案目錄結構

```text
.
├── data/                  # 知識庫原始文件 (.txt, .pdf, .docx)
│   ├── 文化活動與社群連結/
│   ├── 原住民族學生升學管道/
│   ├── 獎助學金與行政庶務/
│   ├── 學習與校園生活支持/
│   └── 職涯與發展/
├── models/
│   └── chroma_db/         # ChromaDB 向量資料庫（由系統自動產生）
├── src/
│   ├── rag.py             # RAG 核心大腦 (讀取、切塊、建立索引與檢索)
│   └── sheets_logger.py   # 自動寫入 Google Sheets 的紀錄模組
├── app.py                 # Streamlit 應用程式前端主檔案
├── README.md              # 專案說明文件
├── requirements.txt       # Python 套件依賴清單
└── .env                   # 環境變數設定檔 (需手動建立)
```

## 部署與本機啟動設定

### 1. 安裝環境與相依套件

請確認系統已安裝 Python 3.11 或以上版本，並建議使用虛擬環境：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 環境變數設定 (`.env`)

請在專案根目錄下建立 `.env` 檔案，並填入以下必要的 API 金鑰與設定檔：
```env
GEMINI_API_KEY="你的_Google_Gemini_API_Key"
JINAAI_API_KEY="你的_JinaAI_API_Key"

# Google Sheets 紀錄用 (選用)
# 若要紀錄使用者對話，請填上 Google Service Account JSON 與試算表名稱
GOOGLE_CREDENTIALS_JSON='{"type": "service_account", ...}'
GOOGLE_SHEET_NAME="你的_工作表_名稱"
```

### 3. 資料建置規範

您可以將任何與原資中心相關的資料檔放入 `data/` 目錄中，支援的格式為 `.pdf`、`.docx`、`.txt`。
**重要注意事項**：所有的 `.txt` 檔案必須儲存為 **UTF-8** 編碼，以避免 LlamaIndex 在讀取或產出 Embedding 時出現中文字亂碼。

### 4. 啟動服務

啟動 Streamlit 伺服器：
```bash
streamlit run app.py
```
執行後，瀏覽器將會自動開啟 `http://localhost:8501`。首次啟動時，系統將會自動讀取 `data/` 下所有文件並預處理建立 ChromaDB 向量索引（需要一些時間）。

