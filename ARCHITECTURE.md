```markdown
# KnowledgeVault AI - Architecture

## System Overview

User > Frontend (React) > Backend (FastAPI)  .External Services
                            
                    PostgreSQL + ChromaDB

## Component Breakdown

### Frontend (React + Next.js)
**What it does:** User interface for uploading and querying

**Components:**
- `UploadZone` - Drag & drop file upload
- `QueryBox` - Search input
- `AnswerDisplay` - Shows results with sources
- `DocumentList` - List of uploaded documents
- `StatsCard` - Shows usage metrics

**Flow:**
1. User uploads file ’ `UploadZone`
2. File sent to backend `/documents/upload`
3. User enters query ’ `QueryBox`
4. Query sent to backend `/query`
5. Result displayed in `AnswerDisplay`

## Backend (FastAPI)
**What it does:** Process documents and answer questions

**Key Services:**

#### EmbeddingService
- Converts text to 1,536-dimensional vectors
- Uses OpenAI's `text-embedding-3-small` model
- Cost: $0.00002 per 1K tokens

#### RAGService (Retrieval-Augmented Generation)
1. **Find similar documents**
   - Convert question to embedding
   - Search ChromaDB for top 5 similar chunks
   
2. **Retrieve context**
   - Extract relevant text from documents
   - Assemble into prompt context
   
3. **Generate answer**
   - Send context + question to GPT-4
   - GPT-4 writes answer
   - Cost: $0.03 per 1K input tokens

4. **Return results**
   - Answer text
   - Source documents
   - Confidence score (0-100%)

#### TextExtractor
- PDF extraction using PyPDF2
- HTML scraping using BeautifulSoup
- Plain text handling
- Text cleaning & normalization

### Databases

#### PostgreSQL
**Purpose:** Store document metadata and queries

**Tables:**

```sql
documents:
  - id (UUID)
  - user_id (UUID)
  - title (VARCHAR)
  - source_type (PDF/TXT/DOCX/URL)
  - full_content (TEXT)
  - created_at (TIMESTAMP)

queries:
  - id (UUID)
  - user_id (UUID)
  - query_text (TEXT)
  - response_text (TEXT)
  - sources (JSONB)
  - helpful (BOOLEAN)
  - latency_ms (INTEGER)
```

#### ChromaDB
**Purpose:** Store and search document embeddings

**Structure:**
```
Collection: "knowledge_vault"
Doc #1 chunks
Chunk 1: [0.2, 0.5, 0.1, ...] (1,536 numbers)
Chunk 2: [0.3, 0.4, 0.2, ...] (1,536 numbers)
Chunk 3: [0.1, 0.6, 0.3, ...] (1,536 numbers)
Doc #2 chunks
.........
.........
```

### External Services

#### OpenAI API
- **Embeddings:** `text-embedding-3-small`
  - Converts text to vectors
  - 1,536 dimensions
  - Cost: $0.00002 per 1K tokens

- **Language Model:** `gpt-4`
  - Generates answers
  - Cost: $0.03 (input) + $0.06 (output) per 1K tokens

## Data Flow

### Upload Document Flow
```
User selects file
    
Frontend: Create FormData with file
    
POST /documents/upload
    
Backend: Save file temporarily
    
TextExtractor: Extract text from file
    
ChunkingService: Split into 1000-token chunks
    
EmbeddingService: Convert each chunk to vector
    
ChromaDB: Store vectors
    
PostgreSQL: Store metadata
    
Return: Document ID
    
Frontend: Show success message
```

### Query/Answer Flow
```
User types question
    
Frontend: Send question to POST /query
    
Backend: Receive question
    
EmbeddingService: Convert question to vector
    
ChromaDB: Find top 5 similar chunks
    
RAGService: Extract chunk context
    
Create prompt: "Here's context: [...]\nQuestion: [...]\nAnswer:"
    
OpenAI GPT-4: Generate answer
    
Extract sources from chunks
    
Calculate confidence score
    
Return: { answer, sources, confidence }
    
Frontend: Display answer with formatting
```

## API Endpoints

### Documents
- `POST /documents/upload` - Upload file
- `POST /documents/url` - Add from URL
- `GET /documents` - List user's documents
- `DELETE /documents/{id}` - Delete document

### Queries
- `POST /query` - Ask question
- `POST /feedback` - Rate answer

### Analytics
- `GET /stats` - Get usage statistics
- `GET /analytics/feedback` - Get satisfaction metrics

## Performance Optimization

### Chunking Strategy
- **Size:** 1000 tokens (â‰ˆ 750 words)
- **Overlap:** 200 tokens (prevent context loss)
- **Rationale:** Balance between context and precision

### Search Optimization
- Retrieve top 5 documents (not 1, not 50)
- Use cosine similarity (industry standard)
- Cache common queries

### Cost Optimization
- Use `text-embedding-3-small` (cheap, accurate)
- Batch embedding requests
- Cache embeddings to avoid recomputation

## Scalability Considerations

### Current Limits
- 50 documents per user (MVP)
- 100 daily queries per user
- 5GB storage per user

### Future Scaling
- Implement pagination for document list
- Add caching layer (Redis)
- Upgrade to paid tiers
- Consider fine-tuned models

## Security

### Current Implementation
- User ID isolation (users can't see others' docs)
- No sensitive data in logs
- API keys in environment variables

### Future Improvements
- Authentication with JWT
- Rate limiting per user
- Audit logging
- Data encryption at rest

---

**Last Updated:** January 2026
