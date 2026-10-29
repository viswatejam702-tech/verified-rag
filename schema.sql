-- ==============================================================================
-- VERIFIED RAG - PostgreSQL + pgvector Database Schema
-- Run this once in your Supabase SQL Editor or PostgreSQL instance
-- ==============================================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2. Documents table (Stores uploaded files, articles, raw documents)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 3. Document Chunks table (Stores chunked passages with dense vector embeddings)
-- Note: 384 dimensions matches sentence-transformers/all-MiniLM-L6-v2 or bge-small
-- (Adjust to 1536 if using text-embedding-ada-002 or text-embedding-3-small)
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB DEFAULT '{}'::jsonb,
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 4. Create indexes for fast metadata & vector lookups
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata ON document_chunks USING GIN(metadata);

-- HNSW Vector Index for sub-millisecond approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. Queries & Logs table (Auditing user sessions, retrieved context, responses)
CREATE TABLE IF NOT EXISTS query_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text TEXT NOT NULL,
    retrieved_chunk_ids UUID[],
    reranked_chunk_ids UUID[],
    generated_answer TEXT,
    faithfulness_score FLOAT,
    hallucination_rate FLOAT,
    latency_breakdown JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 6. Claim Verification Logs (Detailed breakdown of every atomic claim generated)
CREATE TABLE IF NOT EXISTS claim_verification_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id UUID REFERENCES query_logs(id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('VERIFIED', 'CONTRADICTION', 'UNSUPPORTED')),
    entailment_score FLOAT NOT NULL,
    contradiction_score FLOAT NOT NULL,
    neutral_score FLOAT NOT NULL,
    supporting_chunk_id UUID REFERENCES document_chunks(id) ON DELETE SET NULL,
    evidence_snippet TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 7. Benchmark & Evaluation Runs Table
CREATE TABLE IF NOT EXISTS eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_name TEXT NOT NULL,
    benchmark_dataset TEXT,
    num_queries INTEGER NOT NULL,
    avg_faithfulness FLOAT NOT NULL,
    avg_relevance FLOAT NOT NULL,
    hallucination_detection_rate FLOAT NOT NULL,
    avg_latency_ms FLOAT NOT NULL,
    metrics_summary JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- 8. Stored Procedure: match_document_chunks (Cosine Distance Vector Search)
CREATE OR REPLACE FUNCTION match_document_chunks (
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.2,
    match_count INT DEFAULT 10,
    filter_metadata JSONB DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id UUID,
    document_id UUID,
    chunk_index INT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE 
        (filter_metadata = '{}'::jsonb OR dc.metadata @> filter_metadata)
        AND 1 - (dc.embedding <=> query_embedding) > match_threshold
    ORDER BY dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
