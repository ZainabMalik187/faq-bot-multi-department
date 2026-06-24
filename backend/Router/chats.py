from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from groq import Groq
import os
from database_connection.database import get_db
from database_connection import models

router = APIRouter()

class Message(BaseModel):
    text: str

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@router.post("/chat")
async def chat(message: Message, db: Session = Depends(get_db)):
    user_message = message.text

    # STEP 1: Database mein exact match dhoondein
    exact_match = db.execute(text("""
        SELECT f.question, f.answer, d.name as department
        FROM faqs f
        LEFT JOIN departments d ON f.department_id = d.department_id
        WHERE LOWER(f.question) = LOWER(:query)
        LIMIT 1
    """), {"query": user_message}).fetchone()

    if exact_match:
        # Exact match mil gaya — seedha wahi answer bhej den, Groq call nahi karni
        dept = db.query(models.Department).filter(
            models.Department.name == exact_match.department
        ).first()

        log = models.Query(
            user_query=user_message,
            department_id=dept.department_id if dept else None
        )
        db.add(log)
        db.commit()

        return {
            "response": exact_match.answer,
            "department": exact_match.department or "General"
        }

    # STEP 2: Exact match nahi mila — seedha Groq ko bhej do
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": """Tum ek helpful FAQ bot ho jo company ke employees aur customers ke sawalon ka jawab deta hai.

Rules:
1. User ke sawal ka apne general knowledge se direct aur helpful jawab do.
2. Department detect karo in se: HR, IT, Finance, Sales, Customer Support. Agar koi specific department match na ho to "General" likho.

Apna jawab hamesha is exact format mein do:
DEPARTMENT: [naam]
ANSWER: [jawab]"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
    )

    full_reply = response.choices[0].message.content
    department_name = "General"
    answer = full_reply

    for line in full_reply.split("\n"):
        if line.startswith("DEPARTMENT:"):
            department_name = line.replace("DEPARTMENT:", "").strip()
        if line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()

    dept = db.query(models.Department).filter(
        models.Department.name.ilike(department_name)
    ).first()

    log = models.Query(
        user_query=user_message,
        department_id=dept.department_id if dept else None
    )
    db.add(log)
    db.commit()

    return {"response": answer, "department": department_name}