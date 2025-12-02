import React from 'react';
import { Message } from '../types';
import CitationDisplay from './CitationDisplay';

interface MessageBubbleProps {
  message: Message;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-6`}>
      <div className={`max-w-3xl ${isUser ? 'ml-auto' : 'mr-auto'}`}>
        <div className="flex items-start gap-3">
          {!isUser && (
            <div className="w-11 h-11 bg-gradient-to-br from-blue-600 to-blue-700 rounded-xl flex-shrink-0 flex items-center justify-center shadow-md">
              <span className="text-white text-sm font-bold">HR</span>
            </div>
          )}
          <div className="flex-1">
            <div
              className={`rounded-2xl px-6 py-4 shadow-md ${
                isUser
                  ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white ml-auto'
                  : 'bg-white text-gray-900 border-2 border-gray-100'
              }`}
            >
              <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
            </div>

            {!isUser && message.citations && message.citations.length > 0 && (
              <CitationDisplay citations={message.citations} />
            )}

            <div className={`text-xs text-gray-400 mt-2 ${isUser ? 'text-right' : 'text-left'} px-2`}>
              {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>
          {isUser && (
            <div className="w-11 h-11 bg-gradient-to-br from-gray-600 to-gray-700 rounded-xl flex-shrink-0 flex items-center justify-center shadow-md">
              <span className="text-white text-xs font-bold">YOU</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;
