from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Router import chats

app = FastAPI()

# CORS (frontend connection)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router
app.include_router(chats.router)
