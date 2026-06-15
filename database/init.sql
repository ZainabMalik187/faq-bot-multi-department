-- Placeholder schema for the FAQ bot.
--REPLACE this with the real schema design.
CREATE TABLE IF NOT EXISTS faqs (
    id SERIAL PRIMARY KEY,
    department VARCHAR(50) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    user_query TEXT NOT NULL,
    department VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Sample data so the table isn't empty during testing
INSERT INTO faqs (department, question, answer) VALUES
    ('HR', 'How many annual leave days do I get?', 'Employees get 20 annual leave days per year.'),
    ('IT', 'How do I reset my VPN password?', 'Go to the IT portal and click "Reset VPN Password".');
