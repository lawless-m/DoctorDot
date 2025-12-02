#!/bin/bash

echo "Setting up HR Policy Chatbot..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p collections/hr_policies
mkdir -p collections/engineering_docs
mkdir -p collections/legal_contracts

# Copy environment template
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "Created .env file - please add your ANTHROPIC_API_KEY"
fi

# Install frontend dependencies
cd frontend
npm install
cd ..

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your ANTHROPIC_API_KEY to backend/.env"
echo "2. Place PDF files in collections/hr_policies/"
echo "3. Run: python backend/ingest.py --collection hr_policies --guardrail hr_policies"
echo "4. Start backend: python backend/main.py"
echo "5. Start frontend: cd frontend && npm run dev"
