import { NextRequest, NextResponse } from "next/server";

interface BenchmarkDoc {
  chunk_id: string;
  document_title: string;
  content: string;
  domain: string;
}

const DEFAULT_LEWIS_CORPUS: BenchmarkDoc[] = [
  {
    chunk_id: "chunk-ear-01",
    document_title: "Human Ear Anatomy",
    content: "The middle ear includes the tympanic cavity and the three ossicles (malleus, incus, and stapes) which transmit sound vibrations from the tympanic membrane to the oval window.",
    domain: "anatomy"
  },
  {
    chunk_id: "chunk-obama-01",
    document_title: "Barack Obama Biography",
    content: "Barack Hussein Obama II was born on August 4, 1961, at Kapiolani Maternity & Gynecological Hospital in Honolulu, Hawaii. He served as the 44th president of the United States.",
    domain: "biography"
  },
  {
    chunk_id: "chunk-dante-01",
    document_title: "The Divine Comedy",
    content: "The Divine Comedy is an Italian narrative poem by Dante Alighieri. This 14th century work is divided into 3 major sections: Inferno, Purgatorio, and Paradiso, tracing the soul's journey toward God.",
    domain: "literature"
  },
  {
    chunk_id: "chunk-quantum-01",
    document_title: "Fault Tolerant Quantum Computing",
    content: "Superconducting transmon quantum processors operate at 15 millikelvin in dilution refrigerators and require surface codes to correct bit-flip and phase-flip errors.",
    domain: "physics"
  }
];

function hashStringToDim(str: string, dim: number): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash |= 0;
  }
  return Math.abs(hash) % dim;
}

function computeMipsEmbedding(text: string, dim = 128): number[] {
  const words = (text.toLowerCase().match(/\b\w+\b/g) || []).filter(w => w.length > 1);
  const vec = new Array(dim).fill(0);
  if (words.length === 0) return vec;

  for (const w of words) {
    const idx = hashStringToDim(w, dim);
    vec[idx] += 2.0;

    // Sub-word 3-grams
    if (w.length >= 3) {
      for (let i = 0; i < w.length - 2; i++) {
        const ngram = w.slice(i, i + 3);
        const ngIdx = hashStringToDim(ngram, dim);
        vec[ngIdx] += 0.5;
      }
    }
  }

  // L2 normalization
  const norm = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
  if (norm > 0) {
    for (let i = 0; i < dim; i++) {
      vec[i] /= norm;
    }
  }
  return vec;
}

function dotProduct(v1: number[], v2: number[]): number {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

function softmax(scores: number[], temperature = 0.5): number[] {
  if (scores.length === 0) return [];
  const maxScore = Math.max(...scores);
  const expScores = scores.map(s => Math.exp((s - maxScore) / Math.max(temperature, 1e-6)));
  const sumExp = expScores.reduce((acc, curr) => acc + curr, 0);
  return expScores.map(e => e / sumExp);
}

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const {
      query_x,
      task_type = "question_answering", // "question_answering" | "fact_verification" | "jeopardy_generation"
      model_variant = "rag_sequence", // "rag_sequence" | "rag_token"
      custom_passages = null
    } = body;

    if (!query_x || typeof query_x !== "string") {
      return NextResponse.json(
        { error: "A valid non-empty query_x is required." },
        { status: 400 }
      );
    }

    const corpus: BenchmarkDoc[] =
      Array.isArray(custom_passages) && custom_passages.length > 0
        ? custom_passages.map((p: any, idx: number) => ({
            chunk_id: p.id || `custom-chunk-${idx + 1}`,
            document_title: p.title || `Document ${idx + 1}`,
            content: typeof p === "string" ? p : p.content || "",
            domain: p.domain || "general"
          }))
        : DEFAULT_LEWIS_CORPUS;

    // 1. MIPS Retrieval: p_η(z|x)
    const qVec = computeMipsEmbedding(query_x, 128);
    const scoredDocs = corpus.map(doc => {
      const dVec = computeMipsEmbedding(doc.content, 128);
      const innerProduct = dotProduct(qVec, dVec);
      return { doc, mips_score: innerProduct };
    });

    scoredDocs.sort((a, b) => b.mips_score - a.mips_score);
    const topCandidates = scoredDocs.slice(0, 4);
    const mipsScores = topCandidates.map(c => c.mips_score);
    const priors = softmax(mipsScores, 0.5);

    const latentDocuments = topCandidates.map((c, i) => ({
      chunk_id: c.doc.chunk_id,
      document_title: c.doc.document_title,
      content: c.doc.content,
      mips_score: Number(c.mips_score.toFixed(4)),
      prior_prob: Number(priors[i].toFixed(4)),
      metadata: { domain: c.doc.domain }
    }));

    // 2. Task Head Processing
    let generatedY = "";
    let verificationLabel: string | null = null;
    let marginalProbability = 0.0;
    let perTokenAttributions: any[] | null = null;

    if (task_type === "fact_verification") {
      // Task Head B: Fact Verification (FEVER Task: SUPPORTS / REFUTES / NOT_ENOUGH_INFO)
      const factWords = (query_x.toLowerCase().match(/\b\w+\b/g) || []).filter(w => w.length > 2);
      const hasNotFact = factWords.includes("not") || factWords.includes("never") || factWords.includes("didn't");

      let bestSupport = 0.0;
      let bestRefute = 0.0;

      for (const latent of latentDocuments) {
        const docWords = new Set((latent.content.toLowerCase().match(/\b\w+\b/g) || []));
        const overlap =
          factWords.filter(w => docWords.has(w)).length / Math.max(1, factWords.length);
        const hasNotDoc = docWords.has("not") || docWords.has("never");

        if (overlap >= 0.40) {
          if (hasNotFact !== hasNotDoc) {
            const val = 0.40 * latent.prior_prob + 0.60 * overlap;
            if (val > bestRefute) bestRefute = val;
          } else {
            const val = 0.40 * latent.prior_prob + 0.60 * overlap;
            if (val > bestSupport) bestSupport = val;
          }
        }
      }

      if (bestRefute >= 0.30 && bestRefute > bestSupport) {
        verificationLabel = "refutes";
        generatedY = "refutes (y)";
        marginalProbability = Number(bestRefute.toFixed(4));
      } else if (bestSupport >= 0.30) {
        verificationLabel = "supports";
        generatedY = "supports (y)";
        marginalProbability = Number(bestSupport.toFixed(4));
      } else {
        verificationLabel = "not_enough_info";
        generatedY = "not_enough_info (y)";
        marginalProbability = Number((1.0 - Math.max(bestSupport, bestRefute)).toFixed(4));
      }
    } else if (task_type === "jeopardy_generation") {
      // Task Head C: Jeopardy Clue Generation
      const topPassage = latentDocuments[0]?.content || "";
      const clue = topPassage.replace(new RegExp(query_x, "gi"), "This subject");
      const sentences = clue.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 15);

      const clueCandidates = sentences.filter(s =>
        /divided into|sections|century|composed|written/i.test(s)
      );
      generatedY = clueCandidates[0] || sentences[0] || clue.slice(0, 150);
      marginalProbability = Number((latentDocuments[0]?.prior_prob * 0.95 || 0.85).toFixed(4));
    } else {
      // Task Head A: Open-Domain QA
      const topPassage = latentDocuments[0]?.content || "";
      const sentences = topPassage.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 15);
      const answerCandidate = sentences[0] || topPassage.slice(0, 150);

      if (model_variant === "rag_token") {
        const tokens = answerCandidate.split(/\s+/);
        perTokenAttributions = tokens.map(token => {
          const weights: Record<string, number> = {};
          let tMarginal = 0.0;
          for (const doc of latentDocuments) {
            const hasTok = doc.content.toLowerCase().includes(token.toLowerCase());
            const pTheta = hasTok ? 0.85 : 0.15;
            const w = doc.prior_prob * pTheta;
            tMarginal += w;
            weights[doc.chunk_id] = Number(w.toFixed(4));
          }
          return {
            token,
            marginal_p: Number(tMarginal.toFixed(4)),
            active_document_weights: weights
          };
        });
        generatedY = tokens.join(" ");
        marginalProbability = Number(
          (perTokenAttributions.reduce((acc, a) => acc + a.marginal_p, 0) / tokens.length).toFixed(4)
        );
      } else {
        generatedY = answerCandidate;
        marginalProbability = Number((latentDocuments[0]?.prior_prob * 0.92 || 0.88).toFixed(4));
      }
    }

    const logLikelihood = Math.log(Math.max(marginalProbability, 1e-6));

    return NextResponse.json({
      query_x,
      generated_y: generatedY,
      task_type,
      model_variant,
      marginal_probability: marginalProbability,
      sequence_log_likelihood: Number(logLikelihood.toFixed(4)),
      verification_label: verificationLabel,
      latent_documents: latentDocuments,
      per_token_attributions: perTokenAttributions,
      latency_ms: Date.now() - startTime,
      paper_reference: "Lewis et al. (NeurIPS 2020) Retrieval-Augmented Generation"
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: "Lewis RAG execution failed", details: error?.message || String(error) },
      { status: 500 }
    );
  }
}
