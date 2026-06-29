from dotenv import load_dotenv
load_dotenv()  # backend/.env load karo

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ai service folder ko import path mein add karo, taake "import rag_service" chal sake

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from ai_services import rag_service  # Colleague ka RAG engine (database se FAQs leta hai ab)

# ---------------------------------------------------------------------------
# Lifespan — server start hote waqt RAG engine ek dafa build karo
# ---------------------------------------------------------------------------

_engine_bundle = None  # Module-level handle, taake har request pe rebuild na ho


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine_bundle
    print("[...] Building RAG engine (pehli baar embedding model download hoga)...")
    _engine_bundle = rag_service.build_engine()
    print("[OK]  RAG engine ready.")
    yield
    # Shutdown pe abhi kuch cleanup nahi chahiye


app = FastAPI(lifespan=lifespan)

# Frontend se connect karne k liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    User ka sawal le kar, colleague ke RAG engine ko bhejta hai
    (classify -> retrieve -> generate), aur jawab return karta hai.
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
    """FAQs table se dobara saare vector indexes rebuild karo."""
    global _engine_bundle
    _engine_bundle = rag_service.rebuild_index(_engine_bundle)
    return {"status": "ok", "message": "Index rebuilt successfully."}