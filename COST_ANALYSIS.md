
# KnowledgeVault AI - Cost Analysis

## Overview

This is how much the system costs to run.

## Per-User Costs (Monthly)

Assuming 50 documents and 100 queries per month:

| Component | Cost | Notes |
|-----------|------|-------|
| Embeddings | $0.002 | 50 docs Ã— 2000 tokens |
| GPT-4 Queries | $9.00 | 100 queries Ã— 2000 tokens |
| Database | Free | PostgreSQL free tier |
| Storage | Free | ChromaDB self-hosted |
| **Total** | **~$9/month** | Totally affordable! |

## Cost Breakdown

### Embeddings
- OpenAI `text-embedding-3-small`
- Cost: $0.00002 per 1K tokens
- Per document: ~$0.0002 (2000 tokens)
- Per month (50 docs): $0.01

### Queries
- OpenAI GPT-4
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens
- Per query: ~$0.09 (2000 input + 500 output)
- Per month (100 queries): $9.00

### Total Monthly Cost

**MVP Phase (100 active users):**
- Backend server: $15 (Railway)
- Database: Free
- Frontend: Free (Vercel)
- API costs: $900 (100 Ã— $9)
- **Total: ~$915/month**

**Revenue Model:**
- Free tier: 20 documents, 50 queries/month
- Pro tier: $9.99/month (unlimited)
- Business: $29.99/month + advanced features

---

Last Updated: January 2026
