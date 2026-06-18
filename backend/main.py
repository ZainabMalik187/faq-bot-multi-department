from dotenv import load_dotenv
load_dotenv()  # backend/.env load karo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Router import chats # Import karo


app = FastAPI()

# Frontend se connect karne k liye
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)
# Router add karo
app.include_router(chats.router)
