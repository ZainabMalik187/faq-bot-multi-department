
CREATE TABLE Departments (
    department_id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

CREATE TABLE FAQs (
    faq_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE CASCADE,
    question TEXT NOT NULL,
    answer TEXT NOT NULL
);

CREATE TABLE Queries (
    query_id SERIAL PRIMARY KEY,
    department_id INT REFERENCES Departments(department_id) ON DELETE SET NULL,
    user_query TEXT NOT NULL
);
