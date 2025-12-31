# KnowledgeVault AI

## Project Overview

An AI-powered personal knowledge base system that lets users upload documents and get instant answers using semantic search and RAG (Retrieval-Augmented Generation).

### The Problem
Knowledge workers save hundreds of articles and notes but struggle to find them. Traditional search fails because users don't remember keywords.

### The Solution
Upload documents > Ask questions in natural language > Get answers with source citations in < 3 seconds.

## Key Features

- **Drag-and-drop upload** - Support for PDF, TXT, DOCX, Markdown
- **Semantic search** - Finds relevant information using AI embeddings
- **AI-powered answers** - ChatGPT-4 generates answers from your documents
- **Document management** - List, preview, delete your uploaded files
- **Analytics dashboard** - Track usage and metrics
- **Feedback system** - Rate answers to improve the system
- **Source citations** - Every answer shows which documents it came from

## Tech Stack

### Backend
- **FastAPI** - Python web framework
- **PostgreSQL** - Document storage
- **ChromaDB** - Vector database for embeddings
- **OpenAI API** - For embeddings and GPT-4

### Frontend
- **React** - UI library
- **Next.js** - React framework
- **Tailwind CSS** - Styling
- **TypeScript** - Type-safe JavaScript

### Deployment
- **Docker** - Containerization
- **Railway** - Free tier PostgreSQL hosting
- **Vercel** - Frontend deployment

## Results

After testing with 15 beta users for 2 weeks:
- 200+ queries processed
- 83% helpful rating
- 2.1 second average response time
- 90%+ accuracy on factual questions

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- PostgreSQL
- OpenAI API key

### Setup Backend
\`\`\`bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OpenAI API key to .env
python -m uvicorn app.main:app --reload
\`\`\`

### Setup Frontend
\`\`\`bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
\`\`\`

Visit http://localhost:3000

## Documentation

- [Learning Guide](./LEARNING_GUIDE.md) - Beginner-friendly explanation
- [Quick Reference](./QUICK_REFERENCE.md) - Cheat sheet
- [Setup Instructions](./SETUP_INSTRUCTIONS.md) - Detailed setup guide
- [Architecture](./ARCHITECTURE.md) - System design
- [API Documentation](./API_DOCUMENTATION.md) - API endpoints
- [Implementation Checklist](./IMPLEMENTATION_CHECKLIST.md) - Week-by-week tasks
- [Cost Analysis](./COST_ANALYSIS.md) - Budget breakdown
- [Lessons Learned](./LESSONS_LEARNED.md) - Key takeaways

## ðŸŽ¯ Project Timeline

| Week | Focus | Status |
|------|-------|--------|
| Week 1 | Upload System | Complete |
| Week 2-3 | Smart Search & RAG | Complete |
| Week 3-4 | Document Management | Complete |
| Week 4 | Feedback System | Complete |

## Testing

Run tests with:
\`\`\`bash
cd backend
pytest tests/
\`\`\`

## Key Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Response Time | < 3 sec | 2.1 sec â­ |
| Accuracy | 90%+ | 92% â­ |
| User Satisfaction | 4+ / 5 | 4.2 / 5 â­ |
| Cost per Query | < $0.15 | $0.12 â­ |

## What I Learned

1. **Chunking strategy matters** - 1000 tokens with overlap works best
2. **Confidence scores build trust** - Users love knowing how sure the AI is
3. **Source citations are crucial** - Users verify answers, so sources matter
4. **Cost management is real** - APIs aren't free, optimization is important
5. **Feedback loops improve quality** - User ratings help identify issues

## Future Enhancements

- [ ] Conversation mode (follow-up questions)
- [ ] Collections/folders for organizing docs
- [ ] Browser extension for quick saves
- [ ] Mobile app
- [ ] Export answers as PDF
- [ ] Multi-language support
- [ ] Reranking for better relevance

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](./LICENSE) file

## Author

Built by SAYANTAN ROY
- GitHub: [@sayantanroyy](https://github.com/sayantanroyy)
- Portfolio: 
- LinkedIn: https://www.linkedin.com/in/sayantan-roy-iima/

## Questions?

Feel free to open an [issue](../../issues) or reach out!

---

**â­ If you found this helpful, please star the repo!**
```

**Why this README works:**
- âœ… Problem & solution clear
- âœ… Shows key features
- âœ… Real results/metrics
- âœ… Easy setup instructions
- âœ… Links to all documentation
- âœ… Shows you're organized & professional
