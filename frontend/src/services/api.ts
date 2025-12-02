import axios from 'axios';
import { QueryResponse, HealthStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  async query(
    question: string,
    collection: string = 'hr_policies',
    conversationHistory: Array<{role: string; content: string}> = []
  ): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/query', {
      question,
      collection_name: collection,
      conversation_history: conversationHistory,
      include_citations: true,
    });
    return response.data;
  },

  async getHealth(): Promise<HealthStatus> {
    const response = await api.get<HealthStatus>('/health');
    return response.data;
  },

  async getStats(): Promise<any> {
    const response = await api.get('/stats');
    return response.data;
  },

  async listCollections(): Promise<string[]> {
    const response = await api.get<{collections: string[]}>('/collections');
    return response.data.collections;
  },
};
