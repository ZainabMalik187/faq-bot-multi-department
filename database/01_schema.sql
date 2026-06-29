-- Enable the pgvector extension (requires PostgreSQL 15+ and pgvector installed)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Departments table
CREATE TABLE IF NOT EXISTS Departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- 2. FAQs table with vector embedding columns
CREATE TABLE IF NOT EXISTS FAQs (
    faq_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    -- 384 dimensions for sentence-transformers/all-MiniLM-L6-v2
    question_vector vector(384), 
    answer_vector vector(384),
    -- Combined or enriched text vector (Question + Answer + Keywords)
    enriched_vector vector(384) 
);

-- 3. Historical Queries table with query vectors (useful for caching or clustering logs)
CREATE TABLE IF NOT EXISTS Queries (
    query_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE SET NULL,
    user_query TEXT NOT NULL,
    query_vector vector(384)
);

-- Create HNSW indexes for high-performance approximate nearest neighbor (ANN) searches.
-- HNSW is the recommended index type for pgvector >= 0.5.0.
CREATE INDEX IF NOT EXISTS faqs_question_vector_idx 
ON FAQs USING hnsw (question_vector vector_cosine_ops);

CREATE INDEX IF NOT EXISTS faqs_enriched_vector_idx 
ON FAQs USING hnsw (enriched_vector vector_cosine_ops);
