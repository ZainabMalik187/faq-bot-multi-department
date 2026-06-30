from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, Depends, HTTPException
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


class FaqIn(BaseModel):
    question: str
    answer: str
    department_id: int


class FaqUpdate(BaseModel):
    question: str
    answer: str


@app.post("/chat")
async def chat(message: Message):
    try:
    response = client.chat.completions.create(
        model="llama3-70b-8192",
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

    if not full_reply or "ANSWER:" not in full_reply:
        full_reply = f"DEPARTMENT: {dept}\nANSWER: Sorry, I couldn't process that properly."

except Exception as e:
    print("GROQ ERROR:", str(e))
    return {
        "response": "System temporarily unavailable. Please try again later.",
        "department": dept
    }

    # STEP 3: PARSE RESPONSE
    department_name = dept
    answer = full_reply
    for line in full_reply.split("\n"):
        if line.startswith("DEPARTMENT:"):
            department_name = line.replace("DEPARTMENT:", "").strip()
        if line.startswith("ANSWER:"):
            answer = line.replace("ANSWER:", "").strip()

    # STEP 4: SAVE QUERY
    department_obj = db.query(models.Department).filter(
        models.Department.name.ilike(department_name)
    ).first()
    log = models.Query(
        user_query=user_message,
        department_id=department_obj.department_id if department_obj else None
    )
    db.add(log)
    db.commit()

    return {
        "response": answer,
        "department": department_name
    }


# ---------------------------------------------------------
# ADMIN CRUD ENDPOINTS
# ---------------------------------------------------------

@app.get("/departments")
async def get_departments(db: Session = Depends(get_db)):
    depts = db.query(models.Department).all()
    return [{"department_id": d.department_id, "name": d.name} for d in depts]


@app.get("/faqs")
async def get_faqs(db: Session = Depends(get_db)):
    rows = db.execute(text("""
        SELECT f.faq_id, f.question, f.answer, f.department_id, d.name as department
        FROM faqs f
        LEFT JOIN departments d ON f.department_id = d.department_id
        ORDER BY f.department_id, f.faq_id
    """)).fetchall()
    return [
        {
            "faq_id": r.faq_id,
            "question": r.question,
            "answer": r.answer,
            "department_id": r.department_id,
            "department": r.department
        }
        for r in rows
    ]


@app.post("/faqs")
async def create_faq(faq: FaqIn, db: Session = Depends(get_db)):
    department_obj = db.query(models.Department).filter(
        models.Department.department_id == faq.department_id
    ).first()
    if not department_obj:
        raise HTTPException(status_code=404, detail="Department not found")

    new_faq = models.FAQ(
        question=faq.question,
        answer=faq.answer,
        department_id=faq.department_id
    )
    db.add(new_faq)
    db.commit()
    db.refresh(new_faq)
    return {
        "faq_id": new_faq.faq_id,
        "question": new_faq.question,
        "answer": new_faq.answer,
        "department_id": new_faq.department_id
    }


@app.put("/faqs/{faq_id}")
async def update_faq(faq_id: int, faq: FaqUpdate, db: Session = Depends(get_db)):
    existing = db.query(models.FAQ).filter(models.FAQ.faq_id == faq_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="FAQ not found")

    existing.question = faq.question
    existing.answer = faq.answer
    db.commit()
    db.refresh(existing)
    return {
        "faq_id": existing.faq_id,
        "question": existing.question,
        "answer": existing.answer,
        "department_id": existing.department_id
    }


@app.delete("/faqs/{faq_id}")
async def delete_faq(faq_id: int, db: Session = Depends(get_db)):
    existing = db.query(models.FAQ).filter(models.FAQ.faq_id == faq_id).first()
    if not existing:
        raise HTTPException(status_code=404, detail="FAQ not found")

    db.delete(existing)
    db.commit()
    return {"detail": "FAQ deleted"}


@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_faqs = db.query(models.FAQ).count()
    total_departments = db.query(models.Department).count()
    total_queries = db.query(models.Query).count()
    return {
        "total_faqs": total_faqs,
        "total_departments": total_departments,
        "total_queries": total_queries
    }