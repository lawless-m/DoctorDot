import React from 'react';
import { Message } from '../types';
import MessageBubble from './MessageBubble';

interface MessageListProps {
  messages: Message[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  if (messages.length === 0) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <p className="text-lg mb-2">👋 Hello! I'm your HR Policy Assistant</p>
        <p className="text-sm">Ask me anything about our company policies</p>
        <div className="mt-4 text-sm text-gray-400">
          <p>Example questions:</p>
          <ul className="mt-2 space-y-1">
            <li>"What is the remote work policy?"</li>
            <li>"How many vacation days do I get?"</li>
            <li>"What's the process for requesting leave?"</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}
    </div>
  );
};

export default MessageList;
