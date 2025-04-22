CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document (
      id SERIAL PRIMARY KEY,
      text text,
      source text,
      embedding vector,
      created_at timestamptz DEFAULT now()
);
