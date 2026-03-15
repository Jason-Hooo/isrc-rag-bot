# isrc-rag-bot

一個基於檢索增強生成 (RAG) 的 AI 問答機器人，旨在協助國立政治大學原住民族學生資源中心 (ISRC) 的學生與教職員查詢相關資源。

## 相關資訊

- **RAG 框架**: LlamaIndex
- **語言模型 (LLM)**: Google Gemini 2.5 Flash
- **嵌入模型 (Embedding)**: BAAI/bge-m3
- **向量資料庫**: ChromaDB (本地儲存)
- **前端介面**: Streamlit
- **問答紀錄**: Google Sheets API

## 專案結構

```
.
├── data/                  # 大家找的資料 (.txt, .pdf, .docx)
├── models/
│   └── chroma_db/         # ChromaDB 向量資料庫
├── src/
│   ├── rag.py             # RAG 核心邏輯 (讀取、索引、查詢)
│   └── sheets_logger.py   # Google Sheets 紀錄問答模組
├── .env                   # (本地) 環境變數 (需手動建立)
├── .env.example           # 環境變數範本
├── .gitignore             
├── app.py                 # Streamlit 應用程式主檔案
├── README.md              # 本檔案
└── requirements.txt       # Python 套件依賴清單
```
