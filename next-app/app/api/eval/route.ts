import { NextRequest, NextResponse } from "next/server";
import { callRerankService, callVerificationService } from "@/app/lib/utils";

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const {
      query,
      answer,
      retrieved_contexts = [],
      ground_truth_answer
    } = body;

    if (!query || !answer) {
      return NextResponse.json(
        { error: "Query and answer are required." },
        { status: 400 }
      );
    }

    // 1. Evaluate Faithfulness via NLI Verification
    const verification = await callVerificationService(
      answer,
      retrieved_contexts.length > 0 ? retrieved_contexts : ["No context provided"]
    );

    // 2. Evaluate Context Relevance
    let contextRelevanceScore = 0.85;
    if (retrieved_contexts.length > 0) {
      const qTokens = query.toLowerCase().split(/\s+/).filter((w: string) => w.length > 2);
      let matchCount = 0;
      for (const ctx of retrieved_contexts) {
        const text = (typeof ctx === "string" ? ctx : ctx.content || "").toLowerCase();
        for (const qt of qTokens) {
          if (text.includes(qt)) matchCount++;
        }
      }
      contextRelevanceScore = Math.min(1.0, Math.max(0.2, matchCount / (qTokens.length * retrieved_contexts.length || 1)));
    }

    // 3. Evaluate Answer Relevance
    const qTokens = query.toLowerCase().split(/\s+/).filter((w: string) => w.length > 2);
    const aLower = answer.toLowerCase();
    const answeredTokens = qTokens.filter((qt: string) => aLower.includes(qt));
    const answerRelevanceScore = Math.min(0.98, Math.max(0.3, answeredTokens.length / (qTokens.length || 1)));

    // 4. Ground Truth Similarity (if provided)
    let groundTruthSimilarity: number | null = null;
    if (ground_truth_answer) {
      const gtTokens = ground_truth_answer.toLowerCase().split(/\s+/).filter((w: string) => w.length > 2);
      const gtMatches = gtTokens.filter((gt: string) => aLower.includes(gt));
      groundTruthSimilarity = Number((gtMatches.length / Math.max(1, gtTokens.length)).toFixed(2));
    }

    const faithfulness = verification.faithfulness_score;
    const hallucinationRate = verification.hallucination_rate;

    return NextResponse.json({
      metrics: {
        faithfulness_score: Number((faithfulness * 100).toFixed(1)),
        hallucination_rate: Number((hallucinationRate * 100).toFixed(1)),
        context_relevance: Number((contextRelevanceScore * 100).toFixed(1)),
        answer_relevance: Number((answerRelevanceScore * 100).toFixed(1)),
        ground_truth_similarity: groundTruthSimilarity ? Number((groundTruthSimilarity * 100).toFixed(1)) : null,
      },
      claim_breakdown: verification.claims,
      total_claims_analyzed: verification.total_claims,
      verified_claims_count: verification.verified_claims_count,
      unsupported_claims_count: verification.unsupported_claims_count,
      latency_ms: Date.now() - startTime
    });
  } catch (error: any) {
    console.error("Eval API error:", error);
    return NextResponse.json(
      { error: error.message || "Evaluation calculation failed" },
      { status: 500 }
    );
  }
}
