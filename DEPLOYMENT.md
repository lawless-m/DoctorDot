# HR Policy Chatbot - Deployment Guide

This guide covers deploying the HR Policy Chatbot on a remote machine (CPU-only).

## Prerequisites

- Linux server with Python 3.13+
- Network access to Anthropic API
- Sufficient disk space for DuckDB database and PDF documents

## Deployment Steps

### 1. Build the Frontend

On your development machine:

```bash
cd frontend
npm run build
```

This creates optimized static files in `frontend/dist/`

### 2. Prepare Files for Transfer

Copy these files/directories to your remote machine:

```
DoctorDot/
├── backend/                    # Entire backend directory
│   ├── main.py
│   ├── requirements.txt
│   ├── config.py
│   ├── retrieval.py
│   ├── embeddings.py
│   ├── claude_client.py
│   ├── guardrails_manager.py
│   └── .env                    # With your ANTHROPIC_API_KEY
├── frontend/dist/              # Built frontend (from step 1)
├── duckdb_data/                # Contains hr_policies.duckdb
├── guardrails/                 # Contains hr_policies.yaml
└── collections/hr_policies/    # Optional: only if re-ingesting documents
```

### 3. Set Up Remote Machine

SSH into your remote machine and navigate to the project directory.

#### Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (will use CPU for inference)
pip install -r backend/requirements.txt
```

**Note:** On CPU-only machines, PyTorch will automatically use CPU. Embedding generation will be slower but fully functional.

#### Configure Environment

Edit `backend/.env` to ensure it has:

```env
ANTHROPIC_API_KEY=your_api_key_here
DUCKDB_PATH=../duckdb_data
COLLECTIONS_PATH=../collections
GUARDRAILS_PATH=../guardrails
```

### 4. Deployment Options

#### Option A: Single Process (Recommended for Simple Deployments)

Modify `backend/main.py` to serve the frontend static files from the FastAPI backend:

```python
from fastapi.staticfiles import StaticFiles

# Add after creating the app but before routes:
app.mount("/", StaticFiles(directory="../frontend/dist", html=True), name="static")
```

Then run:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

Access at: `http://your-server:8001`

#### Option B: Separate Frontend Server

Use nginx or any web server to serve `frontend/dist/` and proxy API requests to the backend.

**nginx configuration example:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Serve frontend
    location / {
        root /path/to/DoctorDot/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Proxy API requests
    location /api/ {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Run backend separately:

```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8001
```

#### Option C: Production with Gunicorn

For production deployments with multiple workers:

```bash
pip install gunicorn

cd backend
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

### 5. Running as a Service

Create a systemd service file `/etc/systemd/system/hr-chatbot.service`:

```ini
[Unit]
Description=HR Policy Chatbot
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/DoctorDot/backend
Environment="PATH=/path/to/DoctorDot/venv/bin"
Environment="PYTHONPATH=/path/to/DoctorDot"
ExecStart=/path/to/DoctorDot/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable hr-chatbot
sudo systemctl start hr-chatbot
sudo systemctl status hr-chatbot
```

## Performance Notes

- **CPU vs GPU:** The system will work on CPU-only machines. Document ingestion will be slower (~5-10x), but runtime queries are still fast (embeddings are cached in DuckDB).
- **Memory:** Expect ~2-4GB RAM usage with the sentence-transformers model loaded.
- **Concurrent Users:** Single uvicorn process handles ~10-20 concurrent users well. Use gunicorn with multiple workers for higher load.

## Troubleshooting

### Database Connection Issues

If you see "database locked" errors, ensure only one process is accessing the DuckDB file.

### Missing Documents

If queries return no results, verify:
- `duckdb_data/hr_policies.duckdb` was transferred correctly
- DUCKDB_PATH in `.env` points to the correct location
- Re-run ingestion if needed: `python backend/ingest_documents.py`

### API Key Issues

Verify ANTHROPIC_API_KEY is set correctly in `backend/.env`

### Port Conflicts

If port 8001 is in use, change it:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

Update frontend API URL in the built files if needed.

## Security Considerations

- **Firewall:** Restrict access to port 8001 (or your chosen port)
- **HTTPS:** Use nginx with SSL/TLS certificates for production
- **API Key:** Keep `.env` file secure with appropriate file permissions (chmod 600)
- **Updates:** Regularly update dependencies for security patches

## Maintenance

### Updating Documents

1. Place new PDFs in `collections/hr_policies/`
2. Run ingestion: `python backend/ingest_documents.py`
3. Restart the service

### Monitoring Logs

```bash
# If using systemd
sudo journalctl -u hr-chatbot -f

# If running directly
# Logs appear in terminal
```

### Backup

Regularly backup:
- `duckdb_data/hr_policies.duckdb` (contains all indexed documents)
- `collections/hr_policies/` (source PDFs)
- `backend/.env` (API keys and configuration)
