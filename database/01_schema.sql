-- 1. Departments table
CREATE TABLE IF NOT EXISTS Departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- 2. FAQs table
CREATE TABLE IF NOT EXISTS FAQs (
    faq_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);

-- 3. Historical Queries table
CREATE TABLE IF NOT EXISTS Queries (
    query_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE SET NULL,
    user_query TEXT NOT NULL
);