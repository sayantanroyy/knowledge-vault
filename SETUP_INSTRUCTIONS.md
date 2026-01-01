```markdown
# KnowledgeVault AI - Complete Setup Guide

## Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher
- Git
- PostgreSQL (or use Railway free tier)
- OpenAI API key (from https://platform.openai.com)

## Step-by-Step Setup

### 1. Clone the Repository
\`\`\`bash
git clone https://github.com/your-username/knowledge-vault.git
cd knowledge-vault
\`\`\`

### 2. Setup Backend

\`\`\`bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Edit .env and add:
# - OPENAI_API_KEY from your OpenAI account
# - DATABASE_URL for PostgreSQL

# Initialize database
python -m alembic upgrade head

# Run backend
uvicorn app.main:app --reload
\`\`\`

Backend runs on http://localhost:8000

### 3. Setup Frontend

In a new terminal:
\`\`\`bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Start development server
npm run dev
\`\`\`

Frontend runs on http://localhost:3000

### 4. Test Everything

- [ ] Upload a PDF file
- [ ] Ask a question about the PDF
- [ ] See the answer
- [ ] See the success message

## Getting OpenAI API Key

1. Go to https://platform.openai.com
2. Click "API keys" in left sidebar
3. Click "Create new secret key"
4. Copy the key
5. Paste into `.env` file

## Database Setup

### Option A: Local PostgreSQL
\`\`\`bash
# Install PostgreSQL
# Create database
createdb knowledge_vault

# Update DATABASE_URL in .env
DATABASE_URL=postgresql://your_user:your_password@localhost/knowledge_vault
\`\`\`

### Option B: Railway (Free Tier)
1. Go to https://railway.app
2. Sign up
3. Create new project
4. Add PostgreSQL
5. Copy connection string to .env

## Troubleshooting

### Port Already in Use
\`\`\`bash
# Find process using port
lsof -i :8000  # For backend
lsof -i :3000  # For frontend

# Kill process
kill -9 PID
\`\`\`

### OpenAI API Key Error
- Check `.env` file has correct key
- Try creating a new key
- Check your OpenAI account has credits

### Database Connection Error
- Verify PostgreSQL is running
- Check DATABASE_URL format
- Ensure database exists

## Next Steps

1. Read [LEARNING_GUIDE.md](./LEARNING_GUIDE.md)
2. Follow [IMPLEMENTATION_CHECKLIST.md](./IMPLEMENTATION_CHECKLIST.md)
3. Check [API_DOCUMENTATION.md](./API_DOCUMENTATION.md)
```
