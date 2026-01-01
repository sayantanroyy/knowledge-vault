```markdown
# KnowledgeVault Backend

FastAPI backend for document processing and semantic search.

## Setup

\`\`\`bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env and add your OpenAI API key

# Run server
uvicorn app.main:app --reload
\`\`\`

Server runs on http://localhost:8000
API docs on http://localhost:8000/docs

## Project Structure

- `models/` - Database models
- `services/` - Business logic
- `api/` - API endpoints
- `db/` - Database connections
- `tests/` - Unit tests

## Testing

\`\`\`bash
pytest tests/ -v
\`\`\`

## Environment Variables

See `.env.example` for required variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `DATABASE_URL` - PostgreSQL connection string
- `CHROMA_PERSIST_DIR` - ChromaDB storage path
```
