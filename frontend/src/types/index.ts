export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

export interface Citation {
  document_name: string;
  page_number?: number;
  chunk_text: string;
  relevance_score: number;
  chunk_id: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  guardrail_triggered: boolean;
  rejection_reason?: string;
  processing_time_ms: number;
}

export interface HealthStatus {
  status: string;
  gpu_available: boolean;
  cuda_version?: string;
  embedding_model_loaded: boolean;
  active_collection?: string;
  duckdb_status: string;
}
