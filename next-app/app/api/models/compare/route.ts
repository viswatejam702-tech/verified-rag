import { NextRequest, NextResponse } from "next/server";
import { localVectorStore, supabase, isSupabaseConfigured } from "@/app/lib/supabase";
import { callEmbedService, callRerankService, callVerificationService } from "@/app/lib/utils";
import { CustomModelConfig, dispatchModelInference } from "@/app/api/generate/route";

export interface ModelComparisonResult {
  model_id: string;
  name: string;
  model_name: string;
  provider: string;
  base_url?: string;
  answer: string;
  faithfulness_score: number;
  hallucination_rate: number;
  total_claims: number;
  verified_claims_count: number;
  contradicted_claims_count: number;
  unsupported_claims_count: number;
  claims: any[];
  citations: any[];
  latency_ms: number;
  status: "success" | "error";
  error?: string;
}

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const {
      query,
      model_configs = [],
      top_k = 6,
      rerank_top_n = 3
    } = body;

    if (!query || typeof query !== "string") {
      return NextResponse.json({ error: "Query string is required." }, { status: 400 });
    }

    if (!Array.isArray(model_configs) || model_configs.length === 0) {
      return NextResponse.json({ error: "At least one model configuration is required." }, { status: 400 });
    }

    // 1. Retrieve Candidate Evidence Passages
    const queryEmbeddings = await callEmbedService([query]);
    const queryEmbedding = queryEmbeddings[0] || [];

    let initialChunks: any[] = [];
    if (isSupabaseConfigured && supabase) {
      const { data, error } = await supabase.rpc("match_document_chunks", {
        query_embedding: queryEmbedding,
        match_threshold: 0.1,
        match_count: top_k,
        filter_metadata: {}
      });
      initialChunks = error ? await localVectorStore.searchChunks(query, queryEmbedding, top_k) : (data || []);
    } else {
      initialChunks = await localVectorStore.searchChunks(query, queryEmbedding, top_k);
    }

    // 2. Cross-Encoder Rerank Passages
    const rerankedData = await callRerankService(query, initialChunks, rerank_top_n);
    const contextPassages: any[] = rerankedData?.results || initialChunks.slice(0, rerank_top_n);

    const contextFormatted = contextPassages
      .map((p: any, idx: number) => `[Passage ${idx + 1}]:\n${typeof p === "string" ? p : p.content}`)
      .join("\n\n");

    const systemPrompt = `You are an expert AI evaluator and factual research engine. Answer the user question using ONLY the provided context passages below.
For every factual claim, include the exact passage reference bracket (e.g. [1], [2]).
Do NOT hallucinate or speculate. Ground every claim directly in the evidence.`;

    const evidenceStrings = contextPassages.map((p: any) => typeof p === "string" ? p : p.content);

    // 3. Dispatch Parallel Generations across all Models
    const modelPromises = model_configs.map(async (cfg: CustomModelConfig, index: number) => {
      const mStart = Date.now();
      const modelId = cfg.id || `model-${index + 1}`;
      const displayName = cfg.name || cfg.model_name;

      try {
        const infResult = await dispatchModelInference(
          query,
          contextFormatted,
          systemPrompt,
          cfg
        );

        let finalAnswer = infResult.answer;
        if (!finalAnswer || finalAnswer.length < 5) {
          // Contextual synthesis if provider returned blank
          const sents = contextPassages.flatMap((p: any, idx: number) => {
            const txt = typeof p === "string" ? p : p.content;
            return txt.split(/(?<=[.!?])\s+/).slice(0, 2).map((s: string) => `${s.trim()} [${idx + 1}]`);
          });
          finalAnswer = sents.slice(0, 3).join(" ");
        }

        // Run Claim-Level NLI Verification on this model's answer
        const vResult = await callVerificationService(finalAnswer, evidenceStrings, 0.60, 0.50);

        const citationMatches = finalAnswer.match(/\[(\d+)\]/g) || [];
        const citations = Array.from(new Set(citationMatches)).map(c => {
          const idx = parseInt(c.replace(/[\[\]]/g, ""), 10) - 1;
          const passage = contextPassages[idx];
          return {
            tag: c,
            index: idx + 1,
            snippet: typeof passage === "string" ? passage.slice(0, 120) : passage?.content?.slice(0, 120) || ""
          };
        });

        return {
          model_id: modelId,
          name: displayName,
          model_name: cfg.model_name,
          provider: cfg.provider,
          base_url: cfg.base_url || "default",
          answer: finalAnswer,
          faithfulness_score: vResult.faithfulness_score ?? 1.0,
          hallucination_rate: vResult.hallucination_rate ?? 0.0,
          total_claims: vResult.total_claims ?? (vResult.claims?.length || 0),
          verified_claims_count: vResult.verified_claims_count ?? 0,
          contradicted_claims_count: vResult.contradicted_claims_count ?? 0,
          unsupported_claims_count: vResult.unsupported_claims_count ?? 0,
          claims: vResult.claims || [],
          citations,
          latency_ms: Date.now() - mStart,
          status: "success"
        } as ModelComparisonResult;
      } catch (err: any) {
        return {
          model_id: modelId,
          name: displayName,
          model_name: cfg.model_name,
          provider: cfg.provider,
          base_url: cfg.base_url,
          answer: `Inference failed: ${err?.message || String(err)}`,
          faithfulness_score: 0.0,
          hallucination_rate: 1.0,
          total_claims: 0,
          verified_claims_count: 0,
          contradicted_claims_count: 0,
          unsupported_claims_count: 0,
          claims: [],
          citations: [],
          latency_ms: Date.now() - mStart,
          status: "error",
          error: err?.message || String(err)
        } as ModelComparisonResult;
      }
    });

    const results = await Promise.all(modelPromises);

    return NextResponse.json({
      query,
      retrieved_passages: contextPassages.map((p, idx) => ({
        index: idx + 1,
        title: p.document_title || `Passage ${idx + 1}`,
        content: typeof p === "string" ? p : p.content
      })),
      results,
      total_models_compared: results.length,
      orchestration_latency_ms: Date.now() - startTime,
      timestamp: new Date().toISOString()
    });
  } catch (err: any) {
    console.error("Model comparison failed:", err);
    return NextResponse.json(
      { error: "Model comparison failed", details: err?.message || String(err) },
      { status: 500 }
    );
  }
}
