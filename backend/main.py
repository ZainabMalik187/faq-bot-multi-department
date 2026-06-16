from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PEL Knowledge Bot API")

# Allow the frontend (running on a different port/domain) to call this API.
# Tighten allow_origins to your real frontend URL once deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Used to verify the backend is up and reachable."""
    return {"status": "ok", "service": "backend"}


@app.get("/ask")
def ask(q: str = ""):
    """
    Placeholder endpoint for the chat query flow.
    Replace this with the real LlamaIndex RAG pipeline,
    and PostgreSQL FAQ lookup.
    """
    return {
        "question": q,
        "answer": "Placeholder response - RAG pipeline not connected yet.",
        "department": "HR",
    }
