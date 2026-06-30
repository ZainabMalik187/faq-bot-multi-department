from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from sqlalchemy import text
from sqlalchemy.orm import Session
import os

from database_connection.database import get_db
from database_connection import models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class Message(BaseModel):
    text: str
    department: str | None = None


@app.post("/chat")
async def chat(message: Message, db: Session = Depends(get_db)):
    user_message = message.text
    dept = message.department or "General"

    # ✅ STEP 1: CHECK FAQ MATCH
    exact_match = db.execute(text("""
        SELECT f.question, f.answer, d.name as department
        FROM faqs f
        LEFT JOIN departments d ON f.department_id = d.department_id
        WHERE LOWER(f.question) = LOWER(:query)
        LIMIT 1
    """), {"query": user_message}).fetchone()

    if exact_match:
        department_obj = db.query(models.Department).filter(
            models.Department.name == exact_match.department
        ).first()

        log = models.Query(
            user_query=user_message,
            department_id=department_obj.department_id if department_obj else None
        )
        db.add(log)
        db.commit()

        return {
            "response": exact_match.answer,
            "department": exact_match.department or "General"
        }

    # ✅ STEP 2: GROQ AI CALL (FIXED + SAFE)
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",  # ✅ FIXED MODEL
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a helpful FAQ bot for the {dept} department of PEL (Pak Elektron Limited), a Pakistani home appliances company.
Rules:
1. ONLY answer questions related to the {dept} department of PEL.
2. If the question is not related to {dept}, respond with: "I can only answer {dept}-related questions."
3. Always respond in English.
Always respond in this exact format:
DEPARTMENT: {dept}
ANSWER: [answer]"""
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        full_reply = response.choices[0].message.content

    except Exception as e:
        print("🔥 GROQ ERROR:", str(e))  # shows in Railway logs

        return {
            "response": f"AI error: {str(e)}",
            "department": dept
        }

    # ✅ STEP 3: PARSE RESPONSE
    department_name = dept
    answer = full_reply

    for line in full_reply.split("\n"):
        if line.startswith("DEPARTMENT:"):
            department_name = line.replace("DEPARTMENT:", "").strip()
        if line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()

    # ✅ STEP 4: SAVE QUERY
    department_obj = db.query(models.Department).filter(
        models.Department.name.ilike(department_name)
    ).first()

    log = models.Query(
        user_query=user_message,
        department_id=department_obj.department_id if department_obj else None
    )
    db.add(log)
    db.commit()

    # ✅ FINAL RESPONSE
    return {
        "response": answer,
        "department": department_name
    }