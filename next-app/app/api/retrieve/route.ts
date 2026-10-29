import { NextRequest, NextResponse } from "next/server";
import { supabase, isSupabaseConfigured, localVectorStore } from "@/app/lib/supabase";
import { callEmbedService, callRerankService } from "@/app/lib/utils";

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const { query, top_k = 8, rerank_top_n = 3, filter_metadata = {} } = body;

    if (!query || typeof query !== "string") {
      return NextResponse.json(
        { error: "A non-empty query string is required." },
        { status: 400 }
      );
    }

    // 1. Generate query embedding
    const embedStart = Date.now();
    const queryEmbeddings = await callEmbedService([query]);
    const queryEmbedding = queryEmbeddings[0] || [];
    const embedLatency = Date.now() - embedStart;

    // 2. Vector similarity search
    const vectorSearchStart = Date.now();
    let initialChunks: any[] = [];

    if (isSupabaseConfigured && supabase) {
      const { data, error } = await supabase.rpc("match_document_chunks", {
        query_embedding: queryEmbedding,
        match_threshold: 0.1,
        match_count: top_k,
        filter_metadata
      });

      if (error) {
        console.warn("Supabase RPC error, falling back to local store:", error);
        initialChunks = await localVectorStore.searchChunks(query, queryEmbedding, top_k);
      } else {
        initialChunks = data || [];
      }
    } else {
      initialChunks = await localVectorStore.searchChunks(query, queryEmbedding, top_k);
    }

    const vectorLatency = Date.now() - vectorSearchStart;

    if (initialChunks.length === 0) {
      return NextResponse.json({
        query,
        results: [],
        total_retrieved: 0,
        reranked_count: 0,
        latency_breakdown: {
          embedding_ms: embedLatency,
          vector_search_ms: vectorLatency,
          rerank_ms: 0,
          total_ms: Date.now() - startTime
        }
      });
    }

    // 3. Cross-Encoder Reranking
    const rerankStart = Date.now();
    const rerankPayload = initialChunks.map((chunk, index) => ({
      id: chunk.id,
      content: chunk.content,
      metadata: chunk.metadata,
      initial_score: chunk.similarity || 0.5,
      initial_rank: index + 1
    }));

    const rerankResponse = await callRerankService(query, rerankPayload, rerank_top_n);
    const rerankLatency = Date.now() - rerankStart;

    return NextResponse.json({
      query,
      results: rerankResponse.results || [],
      total_retrieved: initialChunks.length,
      reranked_count: rerankResponse.results?.length || 0,
      model_reranker: rerankResponse.model || "cross-encoder/ms-marco-MiniLM-L-6-v2",
      latency_breakdown: {
        embedding_ms: embedLatency,
        vector_search_ms: vectorLatency,
        rerank_ms: rerankLatency,
        total_ms: Date.now() - startTime
      }
    });
  } catch (error: any) {
    console.error("Retrieval error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to retrieve and rerank passages" },
      { status: 500 }
    );
  }
}
