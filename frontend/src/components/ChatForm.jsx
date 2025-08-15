import React, { useRef } from "react";

const ChatForm = ({chatHistory,setChatHistory, generateBotResponse}) => {
    const inputRef = useRef();

    const handleFormSubmit = (e) => {
    e.preventDefault();
    const userMessage = inputRef.current.value.trim();
    if(!userMessage) return;
    inputRef.current.value = "";
    // Add user message and 'Thinking...' placeholder
    setChatHistory((history) => [ ...history, { role: "user", text: userMessage }]);
    setTimeout(() => {
      setChatHistory((history) => [ ...history, { role: "model", text: "Thinking..." }]);
      generateBotResponse([...chatHistory, { role: "user", text: userMessage }]);
    },600);
  };

  return (
    <form action="#" className="chat-form" onSubmit={handleFormSubmit}>
      <input ref={inputRef} type="text" placeholder="Message..." className="message-input" required />
      <button className="icon-btn">
        <span className="material-symbols-rounded">keyboard_arrow_up</span>
      </button>
    </form>
  );
};

export default ChatForm;