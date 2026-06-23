"""
main.py — FastAPI app serving the multi-department FAQ bot.

Endpoints:
  POST /chat  { "query": "..." }  ->  { "answer": "...", "department": "...", "sources": [...] }

CORS is enabled for all origins so faqBotTest.html can call the API
directly when opened from the local filesystem.

Start with:
  python -m uvicorn main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_service

# ---------------------------------------------------------------------------
# Lifespan — build the RAG engine once at startup
# ---------------------------------------------------------------------------

_engine_bundle = None  # Module-level handle for the engine bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the vector indices + router engine on startup."""
    global _engine_bundle
    print("[...] Building RAG engine (first run downloads the embedding model)...")
    _engine_bundle = rag_service.build_engine()
    print("[OK]  RAG engine ready.")
    yield
    # Shutdown — nothing to clean up


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FAQ Bot API",
    description="Multi-department FAQ chatbot powered by LlamaIndex + Groq",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local testing with faqBotTest.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    department: str
    mode: str
    session_id: str
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accept a user question, route it to the correct department index,
    retrieve relevant FAQ chunks, and generate an LLM answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = rag_service.query(_engine_bundle, request.query, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild():
    """Wipe ChromaDB and rebuild all indices from faq.json."""
    global _engine_bundle
    _engine_bundle = rag_service.rebuild_index(_engine_bundle)
    return {"status": "ok", "message": "Index rebuilt successfully."}
