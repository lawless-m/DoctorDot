# HR Policy Chatbot

A modular chatbot system for querying document collections using semantic search and Claude AI.

## Features

- 🔍 Semantic search with GPU-accelerated embeddings
- 🛡️ Modular guardrails for domain-specific constraints
- 📚 Support for multiple document collections
- 🎯 Source citations for all responses
- ⚡ Fast vector similarity search with DuckDB
- 🤖 Powered by Claude Sonnet 4

## Architecture

```
┌──────────────┐
│   Frontend   │
│   (React)    │
└──────┬───────┘
       │
       │ HTTP
       │
┌──────▼───────────────────────────────────┐
│         FastAPI Backend                  │
│  ┌────────────┐  ┌──────────────┐       │
│  │ Embeddings │  │  Guardrails  │       │
│  │  (GPU)     │  │   (Rules)    │       │
│  └────────────┘  └──────────────┘       │
│                                          │
│  ┌────────────┐  ┌──────────────┐       │
│  │  DuckDB    │  │   Claude     │       │
│  │   (VSS)    │  │     API      │       │
│  └────────────┘  └──────────────┘       │
└──────────────────────────────────────────┘
       │                    │
       │                    │
  ┌────▼─────┐         ┌────▼──────┐
  │ PDF Docs │         │ Anthropic │
  └──────────┘         └───────────┘
```

## Quick Start

### 1. Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Configure

Add your Anthropic API key to `backend/.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Add Documents

Place PDF files in `collections/hr_policies/`

### 4. Ingest Documents

```bash
source venv/bin/activate
python backend/ingest.py --collection hr_policies --guardrail hr_policies
```

### 5. Run

Terminal 1 (Backend):
```bash
python backend/main.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Open http://localhost:5173

## Project Structure

```
DoctorDot/
├── backend/                    # FastAPI backend
│   ├── main.py                 # API entry point
│   ├── config.py               # Configuration
│   ├── models.py               # Pydantic models
│   ├── embeddings.py           # GPU embedding service
│   ├── retrieval.py            # DuckDB vector store
│   ├── claude_client.py        # Claude API wrapper
│   ├── document_processor.py   # PDF processing
│   ├── ingest.py               # Document ingestion CLI
│   └── requirements.txt        # Python dependencies
├── guardrails/                 # Modular guardrail system
│   ├── base.py                 # Abstract base class
│   ├── hr_policies.py          # HR guardrail implementation
│   └── hr_policies.yaml        # HR configuration
├── collections/                # Document storage
│   ├── hr_policies/            # HR policy PDFs
│   ├── engineering_docs/       # Engineering docs (example)
│   └── legal_contracts/        # Legal docs (example)
├── data/                       # DuckDB vector databases
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   ├── services/           # API client
│   │   └── types/              # TypeScript types
│   └── package.json
├── scripts/                    # Setup scripts
│   └── setup.sh
└── README.md
```

## Adding New Collections

1. Create directory: `collections/your_collection/`
2. Add PDF files
3. Create guardrail: `guardrails/your_collection.yaml`
4. Implement guardrail: `guardrails/your_collection.py`
5. Ingest: `python backend/ingest.py --collection your_collection --guardrail your_collection`

## GPU Requirements

- CUDA-compatible GPU (tested on NVIDIA 3070 24GB)
- CUDA Toolkit 11.8+
- ~2-3GB VRAM for embeddings

## API Endpoints

- `POST /query` - Submit a question to the chatbot
- `GET /health` - Health check and system status
- `GET /stats` - Vector store statistics
- `GET /collections` - List available collections

## Development

### Backend Development

```bash
source venv/bin/activate
cd backend
uvicorn main:app --reload
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Running Tests

```bash
pytest tests/
```

## Troubleshooting

**GPU Not Detected:**
Check CUDA installation:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

**DuckDB VSS Issues:**
Upgrade DuckDB:
```bash
pip install duckdb --upgrade
```

**Frontend Connection Issues:**
Verify backend is running on http://localhost:8000
Check CORS settings in `backend/config.py`

## License

MIT

## See Also

- Full implementation plan: `drdot.md`
- Detailed setup guide (coming soon)
- Adding collections guide (coming soon)
