import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const EMBED_SERVICE_URL = process.env.EMBED_SERVICE_URL || 'http://localhost:8001';
const RERANK_SERVICE_URL = process.env.RERANK_SERVICE_URL || 'http://localhost:8002';
const VERIFICATION_SERVICE_URL = process.env.VERIFICATION_SERVICE_URL || 'http://localhost:8003';

/**
 * Call Python Embed Service with fallback
 */
export async function callEmbedService(texts: string[]): Promise<number[][]> {
  try {
    const res = await fetch(`${EMBED_SERVICE_URL}/embed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, normalize: true }),
      cache: 'no-store'
    });
    if (res.ok) {
      const data = await res.json();
      return data.embeddings;
    }
  } catch (err) {
    console.warn(`[Embed Service] Error or offline (${err}). Generating fallback embeddings.`);
  }

  // Deterministic 384-d pseudo embedding
  return texts.map(t => {
    const vec = new Array(384).fill(0);
    for (let i = 0; i < t.length; i++) {
      const code = t.charCodeAt(i);
      const idx = (code * (i + 1)) % 384;
      vec[idx] += 0.1;
    }
    const norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0)) || 1.0;
    return vec.map(v => v / norm);
  });
}

/**
 * Call Python Rerank Service with fallback
 */
export async function callRerankService(
  query: string,
  passages: Array<{ id?: string; content: string; metadata?: any; initial_score?: number }>,
  topN: number = 3
) {
  try {
    const res = await fetch(`${RERANK_SERVICE_URL}/rerank`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        passages: passages.map((p, idx) => ({
          id: p.id || String(idx),
          content: p.content,
          metadata: p.metadata,
          initial_rank: idx + 1,
          initial_score: p.initial_score
        })),
        top_n: topN
      }),
      cache: 'no-store'
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn(`[Rerank Service] Error or offline (${err}). Using heuristic rerank.`);
  }

  // Heuristic lexical fallback reranker
  const queryTokens = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
  const scored = passages.map((p, idx) => {
    let score = 0;
    const contentLower = p.content.toLowerCase();
    for (const token of queryTokens) {
      if (contentLower.includes(token)) score += 1.5;
    }
    if (contentLower.includes(query.toLowerCase())) score += 3.0;

    return {
      id: p.id || String(idx),
      content: p.content,
      rerank_score: score,
      confidence_score: Math.min(0.95, Math.max(0.2, score / 5.0)),
      initial_rank: idx + 1,
      new_rank: idx + 1,
      rank_delta: 0,
      metadata: p.metadata
    };
  });

  scored.sort((a, b) => b.rerank_score - a.rerank_score);
  scored.forEach((item, index) => {
    item.new_rank = index + 1;
    item.rank_delta = (item.initial_rank || 1) - item.new_rank;
  });

  return {
    query,
    results: scored.slice(0, topN),
    total_input_passages: passages.length,
    returned_passages: Math.min(topN, scored.length),
    model: 'fallback-lexical-reranker',
    latency_ms: 2.5,
    is_fallback: true
  };
}

/**
 * Call Python Verification Service with fallback
 */
export async function callVerificationService(
  answer: string,
  evidencePassages: string[],
  thresholdEntailment: number = 0.60,
  thresholdContradiction: number = 0.50
) {
  try {
    const res = await fetch(`${VERIFICATION_SERVICE_URL}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        answer,
        evidence_passages: evidencePassages,
        threshold_entailment: thresholdEntailment,
        threshold_contradiction: thresholdContradiction
      }),
      cache: 'no-store'
    });
    if (res.ok) {
      return await res.json();
    }
  } catch (err) {
    console.warn(`[Verification Service] Error or offline (${err}). Using heuristic verification.`);
  }

  // Fallback sentence-level NLI verification
  const sentences = answer
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 10);

  const claims = sentences.map((sentence, idx) => {
    let bestVerdict: 'VERIFIED' | 'CONTRADICTION' | 'UNSUPPORTED' = 'UNSUPPORTED';
    let bestConf = 0.5;
    let supportingSnippet = evidencePassages[0] || '';

    const sentWords = sentence.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    
    // Check against all evidence
    for (const ev of evidencePassages) {
      const evLower = ev.toLowerCase();
      const matchCount = sentWords.filter(w => evLower.includes(w)).length;
      const ratio = matchCount / Math.max(1, sentWords.length);

      if (ratio > 0.45) {
        bestVerdict = 'VERIFIED';
        bestConf = Math.min(0.96, 0.5 + ratio * 0.4);
        supportingSnippet = ev;
        break;
      }
    }

    return {
      claim: sentence,
      verdict: bestVerdict,
      confidence: bestConf,
      entailment_prob: bestVerdict === 'VERIFIED' ? bestConf : 0.2,
      contradiction_prob: bestVerdict === 'CONTRADICTION' ? bestConf : 0.1,
      neutral_prob: bestVerdict === 'UNSUPPORTED' ? 0.7 : 0.2,
      supporting_evidence: supportingSnippet ? supportingSnippet.slice(0, 160) + '...' : null,
      supporting_passage_index: 0
    };
  });

  const verifiedCount = claims.filter(c => c.verdict === 'VERIFIED').length;
  const faithfulness = claims.length > 0 ? Number((verifiedCount / claims.length).toFixed(2)) : 1.0;

  return {
    faithfulness_score: faithfulness,
    hallucination_rate: Number((1.0 - faithfulness).toFixed(2)),
    total_claims: claims.length,
    verified_claims_count: verifiedCount,
    contradicted_claims_count: 0,
    unsupported_claims_count: claims.length - verifiedCount,
    claims,
    model: 'fallback-heuristic-nli',
    latency_ms: 5.0,
    is_fallback: true
  };
}

export interface ScoredItem {
  id: string;
  score: number;
}

/**
 * Reciprocal Rank Fusion (RRF) algorithm for fusing BM25 keyword rankings with dense vector rankings.
 * Standard constant k = 60.
 */
export function reciprocalRankFusion(
  bm25: ScoredItem[],
  vector: ScoredItem[],
  k = 60
): ScoredItem[] {
  const rankMap = new Map<string, number>();
  const add = (list: ScoredItem[]) => {
    list.forEach((item, idx) => {
      const prev = rankMap.get(item.id) ?? 0;
      rankMap.set(item.id, prev + 1 / (k + idx + 1));
    });
  };
  add(bm25);
  add(vector);
  return Array.from(rankMap.entries())
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score);
}

