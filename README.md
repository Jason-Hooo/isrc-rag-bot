# 原寶 (isrc-rag-bot) - 政大原資中心小幫手

這是一個基於檢索增強生成 (Agentic RAG) 技術開發的 AI 問答機器人「原寶」，專為**國立政治大學原住民族學生資源中心 (ISRC)** 打造。旨在透過充滿溫度與親切的語氣，協助同學與教職員查詢校園原住民相關資源、獎助學金、學雜費減免、住宿權益以及文化活動等資訊。

**Live Demo：** [https://jasonhoooooo-isrc-rag-bot.hf.space](https://jasonhoooooo-isrc-rag-bot.hf.space)

## 系統架構與相關資訊

- **RAG 框架**: LlamaIndex (使用 AgentWorkflow 支援多輪對話與工具呼叫)
- **語言模型 (LLM)**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
- **嵌入模型 (Embedding)**: JinaAI (`jina-embeddings-v3`)
- **重排序模型 (Reranker)**: JinaAI (`jina-reranker-v2-base-multilingual`)
- **向量資料庫**: Qdrant Cloud
- **前端介面**: Chainlit
- **問答紀錄與遙測**: Google Sheets API (透過 Service Account 紀錄使用者對話)

## 核心功能：Topic 選擇與自動 Category 判定

本系統採用**雙層檢索機制**，透過使用者選擇的主題（Topic）與 AI 自動判定的類別（Category）來精準過濾知識庫，提升檢索準確度。

### 1. 使用者主題選擇（Topic）

對話開始時，使用者需選擇兩大主題之一：

- **校園資源 (resources)**：查詢政大原資中心的實質資訊，包含文化活動、升學管道、獎助學金、住宿、職涯等行政資訊
- **原民議題 (issues)**：探討原住民族相關社會議題與小故事

### 2. 自動 Category 判定與 Metadata 過濾

當使用者選擇「校園資源」主題時，系統會根據問題內容自動判定屬於以下五大類別之一或多個：

1. **文化活動與社群連結**：文化活動、社團與社群等
2. **原住民族學生升學管道**：升學保障、公費留學辦法等
3. **獎助學金與行政庶務**：獎助學金、學雜費減免、住宿權益、原資中心基本資訊等
4. **學習與校園生活支持**：心理支持與輔導、身分別變更、希望種子培育計畫、急難救助、課業及學習輔導等
5. **職涯與發展**：公職考試資訊、政大公職講座、暑期工讀等

系統會使用 Qdrant 的 **Metadata Filters** 進行精準檢索：
- **校園資源**：filter 同時包含 `topic=resources` 與 `category`（根據 AI 判定的類別）
- **原民議題**：filter 只包含 `topic=issues`，不進行 category 過濾
- 若在指定 category 找不到足夠資料，系統會自動退回「校園資源」全域搜尋（移除 category filter，保留 topic filter）

### 3. Agent 智能工具調用判斷

系統的 AI Agent 會在回覆前先判斷問題性質：

- **寒暄與關心問候**：如「你好」、「最近好嗎」等日常對話，Agent 會直接以親切語氣回覆，不調用知識庫檢索工具
- **實質問題**：涉及政大原資中心資源、原民議題等內容時，Agent 會優先調用 `isrc_knowledge_base` 工具進行知識庫檢索
- **無答案時的處理**：若在知識庫中找不到相關資訊，Agent 會以自然語氣告知無法提供該資訊，絕不亂編內容

### 4. 檔案組織與 Metadata 自動生成

系統會根據檔案在 `data/` 目錄中的位置自動生成 metadata：

```
data/
├── 校園資源/
│   ├── 文化活動與社群連結/
│   ├── 原住民族學生升學管道/
│   ├── 獎助學金與行政庶務/
│   ├── 學習與校園生活支持/
│   └── 職涯與發展/
└── 原民議題/
```

- 第一層目錄決定 `topic` metadata（校園資源 或 原民議題）
- 第二層目錄（僅校園資源）決定 `category` metadata（五大類別之一）
- 這些 metadata 會在建立向量索引時自動寫入 Qdrant

## 專案目錄結構

```text
.
├── data/                  # 知識庫原始文件 (.txt, .pdf, .docx)
│   └── 原民議題/
├── src/
│   ├── rag.py             # RAG 核心大腦 (讀取、切塊、建立索引與檢索)
│   └── sheets_logger.py   # 自動寫入 Google Sheets 的紀錄模組
├── .chainlit/             # Chainlit 配置檔案
├── public/                # 靜態資源檔案
│   └── avatars/
├── app.py                 # Chainlit 應用程式前端主檔案
├── chainlit.md            # Chainlit 歡迎頁面設定
├── README.md              # 專案說明文件
├── requirements.txt       # Python 套件依賴清單
├── .env.example           # 環境變數範例檔案
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
# Google Cloud Service Account 金鑰 JSON
GOOGLE_APPLICATION_CREDENTIALS='{"type": "service_account", ...}'

# GCP 專案的唯一識別碼（Project ID）
GOOGLE_CLOUD_PROJECT=your-project-id

# 運行 Gemini 模型的 Google 資料中心區域
GOOGLE_CLOUD_LOCATION=us-central1

# 對話紀錄的 sheet 名稱
GOOGLE_SHEET_NAME=原資智慧服務 AI 機器人回覆蒐集表

# JinaAI API Key（用於 Embedding 與 Reranker）
JINAAI_API_KEY=your-jinaai-api-key

# Qdrant Cloud 連線設定
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 3. 資料建置規範

您可以將任何與原資中心相關的資料檔放入 `data/` 目錄中，支援的格式為 `.pdf`、`.docx`、`.txt`。
**重要注意事項**：所有的 `.txt` 檔案必須儲存為 **UTF-8** 編碼，以避免 LlamaIndex 在讀取或產出 Embedding 時出現中文字亂碼。

### 4. 啟動服務

啟動 Chainlit 伺服器：
```bash
chainlit run app.py -w
```
執行後，瀏覽器將會自動開啟 `http://localhost:8000`。首次啟動時，系統將會自動讀取 `data/` 下所有文件並預處理建立 Qdrant Cloud 向量索引（需要一些時間）。

