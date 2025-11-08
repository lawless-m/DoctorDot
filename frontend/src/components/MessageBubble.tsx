import React from 'react';
import { Message } from '../types';
import CitationDisplay from './CitationDisplay';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-3xl ${isUser ? 'ml-auto' : 'mr-auto'}`}>
        <div
          className={`rounded-lg p-4 ${
            isUser
              ? 'bg-blue-600 text-white'
              : 'bg-white border border-gray-200 text-gray-800'
          }`}
        >
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>

        {!isUser && message.citations && message.citations.length > 0 && (
          <CitationDisplay citations={message.citations} />
        )}

        <div className="text-xs text-gray-400 mt-1 px-2">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
