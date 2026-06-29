
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from sentence_transformers import SentenceTransformer

# 1. Connect to your PostgreSQL database
# Connect using environment variables instead of hardcoded values
conn = psycopg2.connect(
    dbname=os.environ.get("DB_NAME"),
    user=os.environ.get("DB_USER"),
    password=os.environ.get("DB_PASSWORD"),
    host=os.environ.get("DB_HOST"),
    port=os.environ.get("DB_PORT")
)
cursor = conn.cursor()

# 2. Load the 384-dimension embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    # 3. Fetch all FAQs that need vectorization
    cursor.execute("SELECT faq_id, question, answer FROM FAQs;")
    rows = cursor.fetchall()
    
    for faq_id, question, answer in rows:
        # 4. Generate vectors (convert numpy arrays to standard Python lists)
        question_vector = model.encode(question).tolist()
        answer_vector = model.encode(answer).tolist()
        
        # Enriched text combines question, answer, and a helpful context tag
        enriched_text = f"Question: {question} Answer: {answer} Context: FAQ"
        enriched_vector = model.encode(enriched_text).tolist()
        
        # 5. Update the database row with the generated vectors
        cursor.execute("""
            UPDATE FAQs 
            SET question_vector = %s, answer_vector = %s, enriched_vector = %s
            WHERE faq_id = %s;
        """, (question_vector, answer_vector, enriched_vector, faq_id))
        
    conn.commit()
    print("Database successfully vectorized!")

except Exception as e:
    conn.rollback()
    print(f"An error occurred: {e}")
finally:
    cursor.close()
    conn.close()