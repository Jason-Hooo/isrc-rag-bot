import React, { useState, useRef, useEffect } from 'react';
import './App.css'; 

function App() {
  const [inputText, setInputText] = useState("");
  
  // 將第一則歡迎訊息加上 isWelcome 標記，用來定位快捷分類按鈕
  const defaultMessage = { 
      role: 'ai', 
      text: '哈囉夥伴，我是原寶！想先了解行政資源，還是校園相關的議題呢？',
      isWelcome: true 
  };
  
  const [messages, setMessages] = useState([defaultMessage]);
  const [history, setHistory] = useState([]); 
  const [selectedMainCategory, setSelectedMainCategory] = useState(null);
  
  // 自動滾動的參考點
  const chatEndRef = useRef(null);

  const subCategories = {
    "行政類": ["文化活動與社群連結", "原住民族學生升學管道", "獎助學金與行政庶務", "學習與校園生活支持", "職涯與發展"],
    "議題類": ["原住民身分認同", "校園微歧視", "傳統文化保存", "部落返鄉議題"]
  };

  // 每次訊息更新時，自動滾動到底部
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, selectedMainCategory]);

  const handleSend = async (customText = null) => {
    const textToSend = customText || inputText;
    if (!textToSend.trim()) return;
    
    setInputText(""); 
    setMessages(prev => [...prev, { role: 'user', text: textToSend }]);
    setHistory(prev => [textToSend, ...prev]);
    
    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "user-session-001", question: textToSend })
      });
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'ai', text: data.answer }]);
    } catch (error) {
      console.error("API 錯誤:", error);
      setMessages(prev => [...prev, { role: 'ai', text: "抱歉，原寶的伺服器正在打瞌睡..." }]);
    }
  };

  const handleQuickTagClick = (tag) => {
      handleSend(tag);
  }

  // 重新開啟對話功能
  const restartChat = () => {
      setMessages([defaultMessage]);
      setHistory([]);
      setSelectedMainCategory(null);
  };

  return (
    <div className="app-container">
      <header className="top-bar">
        <div className="logo-section">
          <span className="logo-icon">🏮</span>
          <h1 className="logo-text">原寶</h1>
        </div>
        <div className="marquee-container">
          <div className="marquee-text">
            🌟 最新消息：114學年度原住民學生獎助學金申請至下週五截止！ 🎉 原資中心期末聚餐開放報名中！
          </div>
        </div>
      </header>

      <main className="main-content">
        <section className="chat-section">
          
          {/* ====== 重新開啟對話按鈕 (右上角) ====== */}
          <button className="restart-btn fade-in" onClick={restartChat}>
              🔄 重新開啟對話
          </button>

          <div className="chat-history">
            {messages.map((msg, index) => (
              <div key={index} className={`message-wrapper fade-in-msg ${msg.role === 'user' ? 'wrapper-user' : 'wrapper-ai'}`}>
                
                {/* 訊息氣泡本身 */}
                <div className={`message-bubble ${msg.role === 'user' ? 'msg-user' : 'msg-ai'}`}>
                  <div className="msg-content">{msg.text}</div>
                </div>

                {/* ====== 如果是歡迎訊息，就在氣泡下方緊接著顯示快捷分類 ====== */}
                {msg.isWelcome && (
                  <div className="dynamic-categories fade-in-delayed">
                    {!selectedMainCategory ? (
                        <div className="main-categories fade-in">
                            <div className="category-buttons">
                                <button className="main-cat-btn" onClick={() => setSelectedMainCategory("行政類")}>📂 行政類</button>
                                <button className="main-cat-btn" onClick={() => setSelectedMainCategory("議題類")}>💬 議題類</button>
                            </div>
                        </div>
                    ) : (
                        <div className="sub-categories fade-in">
                            <div className="sub-header">
                                <span className="sub-title">選擇【{selectedMainCategory}】細項：</span>
                                <button className="back-btn" onClick={() => setSelectedMainCategory(null)}>↩ 返回重選</button>
                            </div>
                            <div className="category-buttons-grid">
                                {subCategories[selectedMainCategory].map((sub, idx) => (
                                    <button key={idx} className="sub-cat-btn" onClick={() => handleQuickTagClick(sub)}>
                                        {sub}
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} /> {/* 用來定位滾動到底部的隱藏元素 */}
          </div>

          <div className="input-area">
            <input 
              type="text" 
              className="glow-input"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSend()}
              placeholder="輸入你的問題..." 
            />
            <button className="send-btn" onClick={() => handleSend()}>發送</button>
          </div>
        </section>

        <section className="sidebar-section">
          <h2 className="sidebar-title">📌 此次查詢紀錄</h2>
          <div className="history-list">
            {history.length === 0 ? (
                <div className="empty-history">尚無查詢紀錄</div>
            ) : (
                history.map((item, idx) => (
                  <div key={idx} className="history-item fade-in-msg">{item}</div>
                ))
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;