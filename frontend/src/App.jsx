// ----------------------------------------------------------------
// App.jsx — integrated with ChatBotWidget from "chatbot-widget-ui"
// ----------------------------------------------------------------
import React, { useState } from "react";
import { ChatBotWidget } from "chatbot-widget-ui";

// Base URL for your Flask backend
const API_BASE = "http://192.168.1.188:5001";

function App() {
  // chat history: array of { role: "user" | "assistant", content: string }
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Hi there! How can I assist you today?" },
  ]);

  // 1) API call to your backend with error handling
  const customApiCall = async (userMsg) => {
    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMsg }),
      });

      // Handle non-2xx status codes
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || `Server error: ${res.status}`);
      }

      const data = await res.json();
      // Ensure the answer is always a string for React rendering
      if (typeof data.answer === "object" && data.answer !== null) {
        // Try to extract a useful string, fallback to JSON
        return data.answer.result || JSON.stringify(data.answer);
      }
      return data.answer;
    } catch (error) {
      console.error("API call error:", error);
      throw error;
    }
  };

  // 2) handle a new outgoing message (from user)
  const handleNewMessage = (msg) => {
    setMessages((prev) => [...prev, msg]);
  };

  // 3) handle incoming bot response
  const handleBotResponse = (reply) => {
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: reply },
    ]);
  };

  return (
    <div style={{ position: "relative", height: "1200vh" }}>
      <ChatBotWidget
        callApi={customApiCall}
        onBotResponse={handleBotResponse}
        handleNewMessage={handleNewMessage}
        messages={messages}
        primaryColor="#33ff96"
        inputMsgPlaceholder="Type your message..."
        chatbotName="Quincy"
        isTypingMessage="Typing..."
        IncommingErrMsg="Oops! Something went wrong. Try again."
        chatIcon={<span>💬</span>}
        botIcon={<span>🤖</span>}
        botFontStyle={{
            fontFamily: "Arial",
            fontSize: "14px",
            color: "red",
        }}
        typingFontStyle={{
          fontFamily: "Arial",
          fontSize: "12px",
          color: "#888",
          fontStyle: "italic",
        }}
        useInnerHTML={false}
      />
    </div>
  );
}

export default App;
