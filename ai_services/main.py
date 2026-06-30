from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag_service

_engine_bundle = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[OK] FastAPI started (lazy RAG loading enabled)")
    yield


app = FastAPI(
    title="FAQ Bot API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    department: str
    text: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    department: str
    mode: str
    session_id: str
    sources: list[str] = []


@app.get("/")
def root():
    return {"status": "API is running"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global _engine_bundle

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # ✅ Lazy load engine here instead of startup
        if _engine_bundle is None:
            print("[...] Building RAG engine (first request)...")
            _engine_bundle = rag_service.build_engine()
            print("[OK] RAG engine ready.")

        result = rag_service.query(
            user_query=request.text,
            department=request.department,
            session_id=request.session_id,
            engine_bundle=_engine_bundle,
        )
        return ChatResponse(**result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild")
async def rebuild():
    global _engine_bundle
    _engine_bundle = rag_service.rebuild_index(_engine_bundle)
    return {"status": "ok", "message": "Index rebuilt successfully."}