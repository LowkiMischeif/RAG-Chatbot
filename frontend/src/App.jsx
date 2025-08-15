import React, { useEffect, useRef, useState } from "react";
import ChatbotIcon from "./components/ChatbotIcon";
import ChatForm from "./components/ChatForm";
import ChatMessage from "./components/ChatMessage";

const App = () => {
  const [chatHistory, setChatHistory] = useState([]);
  const [showChatbot, setshowChatbot] = useState([]);
  const chatBodyRef = useRef(null);

  const generateBotResponse = async (history) => {
    // Get the latest user message
    const updateHistory = (text) => {
      setChatHistory((prev) => [...prev.filter((msg) => msg.text !== "Thinking..."), { role: "model", text }]);
    }

  // Get the latest user message text
  const lastUserMessage = history.filter(msg => msg.role === 'user').pop()?.text || '';

  const requestOptions = {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: lastUserMessage })
  };

  try {
    const response = await fetch('http://localhost:5001/api/query', requestOptions);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Something went wrong');

    const apiResponseText = (data.answer || '').trim();
    updateHistory(apiResponseText);
  } catch (error) {
    console.log(error.message, true);
  }
  };

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTo({top: chatBodyRef.current.scrollHeight, behavior: 'smooth'});
    }
}, [chatHistory, chatBodyRef]);

  return(
    <div className = {`container ${showChatbot ? 'show-chatbot' : ""}`}>
      <button onClick={() => setshowChatbot(prev => !prev)} id="chatbot-toggler">
        <span class="material-symbols-rounded">mode_comment</span>
        <span className="material-symbols-rounded">close</span>
      </button>
      <div className="chatbot-popup">
        <div className="chat-header">
          <div className="header-info">
            <ChatbotIcon />
            <h2 className="logo-text">Chatbot</h2>
          </div>
            <button onClick={() => setshowChatbot(prev => !prev)} className="icon-btn">
              <span className="material-symbols-rounded">keyboard_arrow_down</span>
            </button>
        </div>

        {/* Chatbot body */}
        <div ref= {chatBodyRef} className="chat-body">
          <div className="message bot-message">
            <ChatbotIcon />
            <p className="message-text">
              Hey there 👋 <br /> How can I help you today?
            </p>
          </div>

          {chatHistory.map((chat, index) => (
            <ChatMessage key={index} chat={chat} />
          ))}
        </div>
        {/* Chatbot Footer */}
        <div className="chat-footer">
          <ChatForm chatHistory = {chatHistory}  setChatHistory={setChatHistory} generateBotResponse={generateBotResponse} />
        </div>      
      </div>
    </div>
  )         
}

export default App;