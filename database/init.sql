-- Initialize database tables
CREATE TABLE IF NOT EXISTS departments (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS faqs (
    id SERIAL PRIMARY KEY,
    department_id INTEGER REFERENCES departments(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample departments
INSERT INTO departments (name) VALUES ('Sales'), ('Support'), ('Engineering')
ON CONFLICT (name) DO NOTHING;
