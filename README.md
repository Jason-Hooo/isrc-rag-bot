這是一份為你的專案 **「政大原資中心 AI 智慧服務 - 原寶 (ISRC-RAG-BOT)」** 量身打造的 README 檔案。這份檔案整合了我們剛剛完成的架構大改版、UI/UX 優化以及核心 RAG 技術的更新。

---

# 🏮 原寶 - 政大原資中心 AI 智慧服務 RAG Bot

「原寶」是專為國立政治大學原住民族學生資源中心（原資中心）開發的 AI 智慧對話機器人。旨在協助校園夥伴與原住民同學快速查詢獎助學金、學雜費減免、住宿權益、文化活動及校園支持資源，並提供具有溫度、親切且專業的互動體驗。

## 🚀 近期重大更新 (2026.05)

本專案近期完成了從單一 Streamlit 應用轉向 **現代化前後端分離架構** 的重大升級：

### 1. 架構遷移：React + FastAPI

* **前端轉型**：捨棄 Streamlit，全面改用 **React (Vite)** 構建，實現像素級的 UI 精細度與順滑的動畫效果。
* **後端強化**：”未完成“使用 **FastAPI** 重新建構 API 入口，提升響應速度與擴展性。

### 2. UI/UX 深度優化 (Dribbble 科技感風格)

* **視覺設計**：採用深色模式 (Dark Mode)，結合金黃色發光特效 (Golden Glow) 與原住民文化象徵的紅色系點綴。
* **佈局重整**：
* **頂部導覽列**：包含「原寶」Logo 與動態跑馬燈，用於展示原資中心最新活動資訊。
* **對話區 (左側)**：支援自動滾動與訊息淡入動畫。
* **功能區 (右側)**：獨立顯示「此次查詢紀錄」，並支援獨立滾動。


* **互動式快捷分類**：
* 首創「對話氣泡附屬組件」，將分類按鈕直接整合於機器人歡迎訊息下方。
* 支援層級導覽：主分類（行政類/議題類）點擊後淡出切換至細項分類。
* 內建延遲動畫，確保對話框先出、分類按鈕後出的優雅視覺順序。



### 3. RAG 檢索技術升級

* **Metadata (元資料) 應用**：為文件塊貼上年份、類別標籤，解決資訊時效性問題並實現精準過濾。
* **混合搜尋 (Hybrid Search)**：結合向量搜尋 (Vector) 與關鍵字搜尋 (BM25)，提升對專有名詞的檢索準確率。
* **重排序 (Re-ranking)**：整合 Jina Rerank 模型，大幅降低 AI 幻覺。

---

## 🛠 技術棧

* **前端**: React (Vite), CSS3 (Flexbox/Grid, Animations)
* **後端**: Python 3.12, FastAPI, Uvicorn
* **AI 框架**: LlamaIndex (Agentic Workflow)
* **模型**: Google Gemini (LLM), Jina AI (Embedding & Rerank)
* **向量資料庫**: ChromaDB
* **日誌系統**: Google Sheets API (自動紀錄對話與來源)

---

## 📂 專案架構

```text
isrc-rag-bot/
├── backend/                # Python 後端服務
│   ├── api.py              # FastAPI 進入點
│   ├── src/
│   │   ├── rag.py          # RAG 核心邏輯 (索引、檢索、代理)
│   │   └── sheets_logger.py # Google Sheets 紀錄邏輯
│   ├── data/               # 行政法規與公告原始文件 (.txt, .pdf)
│   ├── models/             # 持久化向量數據庫 (Chroma)
│   └── .venv/              # 後端虛擬環境
│
└── frontend/               # React 前端應用
    ├── src/
    │   ├── App.jsx         # 主要邏輯、動畫與 UI 結構
    │   ├── App.css         # 科技感視覺樣式、發光效果、動畫定義
    │   └── main.jsx        # 前端渲染入口
    └── index.html

```

---

## ⚙️ 快速上手

### 1. 後端設定 (Backend)

1. 進入 `backend` 資料夾：`cd backend`
2. 啟動虛擬環境：`source .venv/bin/activate`
3. 安裝依賴：`pip install -r requirements.txt`
4. 設定 `.env` 檔案，包含 `GEMINI_API_KEY` 與 `JINAAI_API_KEY`。
5. 啟動 API 伺服器：`uvicorn api:app --reload`

### 2. 前端設定 (Frontend)

1. 進入 `frontend` 資料夾：`cd frontend`
2. 安裝套件：`npm install`
3. 啟動開發伺服器：`npm run dev`
4. 點擊終端機產生的網址（預設為 `http://localhost:5173`）即可開始對話。

---

## 📌 未來展望

* 等待Jason把前後端串接FastAPI處理完成

