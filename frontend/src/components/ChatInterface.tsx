import React, { useState, useRef, useEffect } from 'react';
import { Message } from '../types';
import { chatAPI } from '../services/api';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const response = await chatAPI.query(
        content,
        'hr_policies',
        messages.map(m => ({ role: m.role, content: m.content }))
      );

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        citations: response.citations,
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get response');
      console.error('Error querying chatbot:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-700 to-blue-600 border-b border-blue-800 shadow-lg px-8 py-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 bg-white rounded-2xl flex items-center justify-center shadow-md">
              <span className="text-blue-600 text-2xl font-bold">HR</span>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white">HR Policy Assistant</h1>
              <p className="text-blue-100 text-sm">Quick answers to your policy questions</p>
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container */}
      <div className="flex-1 overflow-y-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center py-16">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-600 to-blue-700 rounded-3xl mx-auto mb-8 flex items-center justify-center shadow-xl">
                <span className="text-white text-3xl font-bold">HR</span>
              </div>
              <h2 className="text-3xl font-bold text-gray-900 mb-4">How can I help you today?</h2>
              <p className="text-gray-600 mb-8 text-lg">Ask me anything about our HR policies and procedures</p>
              <div className="flex flex-wrap gap-3 justify-center max-w-2xl mx-auto">
                {['What is the holiday policy?', 'Can I work from home?', 'How do I report sick leave?'].map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSendMessage(q)}
                    className="px-5 py-3 bg-white rounded-xl text-sm text-gray-700 hover:bg-blue-600 hover:text-white transition-all shadow-md hover:shadow-lg border border-gray-200 hover:border-blue-600 font-medium"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          <MessageList messages={messages} />
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-auto max-w-4xl w-full px-6 mb-4">
          <div className="p-4 bg-red-50 border-l-4 border-red-400 text-red-700 rounded-lg shadow-sm">
            <div className="flex items-center gap-2">
              <span className="text-xl">⚠️</span>
              <span className="font-medium">{error}</span>
            </div>
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 bg-white/95 backdrop-blur-md shadow-2xl">
        <div className="max-w-4xl mx-auto px-6 py-6">
          <MessageInput
            onSend={handleSendMessage}
            disabled={isLoading}
          />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
