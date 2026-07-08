# Team Workflow - 4 Members Task Distribution

## Branch Organization & Responsibilities

### 👤 Member 1: PDF Processing & Preprocessing
**Branch:** `member1-pdf-preprocessing`

**Tasks:**
- Read PDF files
- Preprocessing (cleaning, formatting)
- Metadata Extraction

**Files to Work On:**
- `utils/pdf_loader.py` - PDF reading functionality
- `utils/text_cleaner.py` - Text preprocessing
- `utils/data_loader.py` - Data loading logic
- `utils/metadata.py` - Metadata extraction
- `ingest.py` - Ingestion pipeline
- `backend/uploader.py` - File upload handling

**How to Work:**
```bash
git checkout member1-pdf-preprocessing
# Make your changes
git add .
git commit -m "feat: [member1] description of changes"
git push origin member1-pdf-preprocessing
```

---

### 👤 Member 2: Embedding Generation & ChromaDB Storage
**Branch:** `member2-embedding-chromadb`

**Tasks:**
- Generate embeddings for text
- Store embeddings in ChromaDB
- Vector database management

**Files to Work On:**
- `utils/embedding_model.py` - Embedding generation
- `utils/vector_store.py` - Vector store operations
- `build_vector_db.py` - Vector DB building script
- `vector_db/` - Vector database files
- `backend/database.py` - Database operations

**How to Work:**
```bash
git checkout member2-embedding-chromadb
# Make your changes
git add .
git commit -m "feat: [member2] description of changes"
git push origin member2-embedding-chromadb
```

---

### 👤 Member 3: Query Processing & Similarity Search
**Branch:** `member3-query-retrieval`

**Tasks:**
- Process user questions
- Generate query embeddings
- Perform similarity search
- Retrieve relevant resumes

**Files to Work On:**
- `utils/query_embedder.py` - Query embedding generation
- `utils/retriever.py` - Similarity search & retrieval
- `utils/reranker.py` - Result reranking
- `backend/rag_client.py` - RAG client logic
- `rag_engine.py` - RAG engine orchestration
- `utils/prompt_builder.py` - Prompt construction

**How to Work:**
```bash
git checkout member3-query-retrieval
# Make your changes
git add .
git commit -m "feat: [member3] description of changes"
git push origin member3-query-retrieval
```

---

### 👤 Member 4: Gemini Integration, Streamlit UI & Testing
**Branch:** `member4-gemini-streamlit-testing`

**Tasks:**
- Build Gemini (LLM) integration
- Build Streamlit UI
- Testing all components
- Response formatting

**Files to Work On:**
- `utils/llm_client.py` - LLM (Gemini) integration
- `utils/response_formatter.py` - Response formatting
- `app.py` - Main Streamlit app
- `pages/` - All Streamlit pages (Chat.py, History.py, Home.py, Upload.py)
- `config.py` - Configuration settings
- `assets/style.css` - UI styling
- `backend/session_manager.py` - Session management
- Testing scripts

**How to Work:**
```bash
git checkout member4-gemini-streamlit-testing
# Make your changes
git add .
git commit -m "feat: [member4] description of changes"
git push origin member4-gemini-streamlit-testing
```

---

## Workflow Instructions

### Step 1: Clone and Setup
```bash
# Clone the repository (if not already done)
git clone https://github.com/Krishna050509/Resume_Rag.git
cd Resume_Rag
```

### Step 2: Each Member Checks Out Their Branch
```bash
# Member 1
git checkout member1-pdf-preprocessing

# Member 2
git checkout member2-embedding-chromadb

# Member 3
git checkout member3-query-retrieval

# Member 4
git checkout member4-gemini-streamlit-testing
```

### Step 3: Make Changes & Commit
```bash
# Make your changes to assigned files
git add .
git commit -m "feat: [memberX] brief description of changes"
git push origin [your-branch-name]
```

### Step 4: Creating Pull Requests (Optional)
When ready to merge with master:
```bash
# Push your final changes
git push origin [your-branch-name]

# Go to GitHub and create a Pull Request from your branch to master
```

### Step 5: Final Integration
After all members complete their work:
1. All PRs should be reviewed
2. Merge to master
3. Final testing

---

## Commit Message Convention

Use this format for clear communication:
```
feat: [memberX] short description
- Detailed change 1
- Detailed change 2
```

Example:
```
feat: [member1] add PDF text extraction
- Extract text from PDF files
- Handle multi-page documents
- Clean extracted text
```

---

## Key Notes

✅ Each member works independently on their branch  
✅ No conflicts when working on different files  
✅ All changes stay on your branch until ready to merge  
✅ Push regularly to backup your work  
✅ Communicate with team about shared utilities  

---

## Emergency Commands

**Pull latest changes from master:**
```bash
git fetch origin
git merge origin/master
```

**See what changed on your branch:**
```bash
git log origin/master..HEAD
```

**Reset to last pushed version:**
```bash
git reset --hard origin/[your-branch-name]
```
