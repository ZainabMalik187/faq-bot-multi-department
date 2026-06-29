@router.post("/chat")
async def chat(message: Message, db: Session = Depends(get_db)):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    user_message = message.text

    # STEP 1: Check exact match in DB
    exact_match = db.execute(text("""
        SELECT f.question, f.answer, d.name as department
        FROM faqs f
        LEFT JOIN departments d ON f.department_id = d.department_id
        WHERE LOWER(f.question) = LOWER(:query)
        LIMIT 1
    """), {"query": user_message}).fetchone()

    if exact_match:
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

    # STEP 2: Send to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful FAQ bot.

Rules:
1. Answer clearly using general knowledge.
2. Detect department: HR, IT, Finance, Sales, Customer Support.
3. If none match, use "General".

Always reply in this format:
DEPARTMENT: [name]
ANSWER: [response]"""
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
