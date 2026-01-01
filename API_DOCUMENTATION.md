 KnowledgeVault AI - API Documentation

Base URL: `http://localhost:8000` (development)

## Authentication

Currently no authentication (MVP). In production, use JWT tokens.

## Endpoints

### Documents

#### Upload Document
\`\`\`
POST /documents/upload
Content-Type: multipart/form-data

Parameters:
- file: File (required) - PDF, TXT, DOCX, or Markdown
- user_id: string (required) - User identifier

Response:
{
  "document_id": "uuid",
  "title": "filename",
  "size": 2300000,
  "created_at": "2025-12-20T15:30:00Z"
}
\`\`\`

**Example:**
\`\`\`bash
curl -X POST http://localhost:8000/documents/upload \\
  -F "file=@my-notes.pdf" \\
  -F "user_id=user-123"
\`\`\`

#### Add from URL
\`\`\`
POST /documents/url
Content-Type: application/json

{
  "url": "https://example.com/article",
  "user_id": "user-123"
}

Response:
{
  "document_id": "uuid",
  "title": "article-title",
  "source_url": "https://example.com/article",
  "created_at": "2025-12-20T15:30:00Z"
}
\`\`\`

#### List Documents
\`\`\`
GET /documents?user_id=user-123

Response:
[
  {
    "id": "uuid",
    "title": "My Notes",
    "size": 2300000,
    "created_at": "2025-12-20",
    "preview": "This document contains..."
  },
  ...
]
\`\`\`

#### Delete Document
\`\`\`
DELETE /documents/{document_id}?user_id=user-123
Response:
{
  "status": "deleted",
  "document_id": "uuid"
}
\`\`\`

### Queries

#### Ask Question
\`\`\`
POST /query
Content-Type: application/json

{
  "question": "What are health benefits of apples?",
  "user_id": "user-123"
}

Response:
{
  "answer": "Apples are rich in fiber and vitamin C...",
  "sources": [
    {
      "document_id": "uuid",
      "title": "Health Guide",
      "preview": "...",
      "relevance": 0.92
    }
  ],
  "confidence": 92,
  "latency_ms": 2100
}
\`\`\`

#### Submit Feedback
\`\`\`
POST /feedback
Content-Type: application/json

{
  "query_id": "uuid",
  "helpful": true,
  "user_id": "user-123"
}

Response:
{
  "status": "recorded",
  "query_id": "uuid"
}
\`\`\`

### Analytics

#### Get Statistics
\`\`\`
GET /stats?user_id=user-123

Response:
{
  "total_documents": 47,
  "total_queries": 312,
  "avg_latency_ms": 2100,
  "storage_used_mb": 25.3,
  "doc_types": {
    "pdf": 28,
    "text": 12,
    "url": 7
  }
}
\`\`\`

#### Get Feedback Analytics
\`\`\`
GET /analytics/feedback?user_id=user-123

Response:
{
  "helpful": 250,
  "not_helpful": 50,
  "satisfaction_rate": 83.3,
  "total_rated": 300
}
\`\`\`
## Error Handling

All errors return JSON with status code:

\`\`\`json
{
  "error": "Invalid file type",
  "detail": "Only PDF, TXT, DOCX accepted",
  "status_code": 400
}
\`\`\`

### Status Codes
- 200: Success
- 400: Bad request
- 401: Unauthorized
- 404: Not found
- 500: Server error

## Rate Limits

- 100 requests per minute per user
- 50 document uploads per day
- 500 queries per day

## Example: Complete Workflow

\`\`\`bash
# 1. Upload a document
curl -X POST http://localhost:8000/documents/upload \\
  -F "file=@nutrition-guide.pdf" \\
  -F "user_id=user-123"
# Response: { "document_id": "doc-abc123" }

# 2. Ask a question
curl -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What vitamins are in apples?",
    "user_id": "user-123"
  }'
# Response: { "answer": "...", "sources": [...], "confidence": 92 }

# 3. Rate the answer
curl -X POST http://localhost:8000/feedback \\
  -H "Content-Type: application/json" \\
  -d '{
    "query_id": "query-xyz789",
    "helpful": true,
    "user_id": "user-123"
  }'

# 4. Check statistics
curl http://localhost:8000/stats?user_id=user-123
# Response: { "total_documents": 1, "total_queries": 1, ... }
\`\`\`

---

**Last Updated:** January 2026
```

---
