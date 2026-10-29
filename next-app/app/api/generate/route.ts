import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const { query, context_passages = [], model_provider = "auto" } = body;

    if (!query) {
      return NextResponse.json({ error: "Query is required." }, { status: 400 });
    }

    if (!context_passages || context_passages.length === 0) {
      return NextResponse.json({
        answer: "I could not find any relevant passages in the knowledge base to answer this question accurately.",
        citations: [],
        model_used: "fallback-no-context",
        latency_ms: Date.now() - startTime
      });
    }

    // Format context with index tags [1], [2], etc.
    const contextFormatted = context_passages
      .map((p: any, idx: number) => `[Passage ${idx + 1}]:\n${typeof p === "string" ? p : p.content}`)
      .join("\n\n");

    const systemPrompt = `You are a strictly verified AI research assistant. Answer the user's question using ONLY the provided context passages below.
For every claim you make, include the bracketed passage reference (e.g. [1], [2]).
Do NOT speculate or fabricate facts. If the context does not contain the answer, state clearly that it is not established in the evidence.`;

    let generatedAnswer = "";
    let modelUsed = "local-grounded-synthesizer";

    // 1. Check for OpenAI API Key
    const openAiKey = process.env.OPENAI_API_KEY;
    const geminiKey = process.env.GEMINI_API_KEY;

    if (openAiKey && (model_provider === "openai" || model_provider === "auto")) {
      try {
        const aiRes = await fetch("https://api.openai.com/v1/chat/completions", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${openAiKey}`
          },
          body: JSON.stringify({
            model: "gpt-4o-mini",
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: `Context:\n${contextFormatted}\n\nQuestion: ${query}` }
            ],
            temperature: 0.2
          })
        });

        if (aiRes.ok) {
          const aiData = await aiRes.json();
          generatedAnswer = aiData.choices?.[0]?.message?.content || "";
          modelUsed = "openai/gpt-4o-mini";
        }
      } catch (e) {
        console.warn("OpenAI API call failed, falling back to synthesizer:", e);
      }
    }

    // 2. Check for Google Gemini API Key
    if (!generatedAnswer && geminiKey && (model_provider === "gemini" || model_provider === "auto")) {
      try {
        const geminiRes = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${geminiKey}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    { text: `${systemPrompt}\n\nContext:\n${contextFormatted}\n\nQuestion: ${query}` }
                  ]
                }
              ],
              generationConfig: { temperature: 0.2 }
            })
          }
        );

        if (geminiRes.ok) {
          const geminiData = await geminiRes.json();
          generatedAnswer = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || "";
          modelUsed = "google/gemini-1.5-flash";
        }
      } catch (e) {
        console.warn("Gemini API call failed, falling back to synthesizer:", e);
      }
    }

    // 3. Check for Anthropic API Key
    const anthropicKey = process.env.ANTHROPIC_API_KEY;
    if (!generatedAnswer && anthropicKey && (model_provider === "anthropic" || model_provider === "auto")) {
      try {
        const antRes = await fetch("https://api.anthropic.com/v1/messages", {
          method: "POST",
          headers: {
            "x-api-key": anthropicKey,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
          },
          body: JSON.stringify({
            model: "claude-3-5-sonnet-20240620",
            max_tokens: 1024,
            temperature: 0.2,
            system: systemPrompt,
            messages: [{ role: "user", content: `Context:\n${contextFormatted}\n\nQuestion: ${query}` }]
          })
        });

        if (antRes.ok) {
          const antData = await antRes.json();
          generatedAnswer = antData?.content?.[0]?.text || "";
          modelUsed = "anthropic/claude-3-5-sonnet";
        }
      } catch (e) {
        console.warn("Anthropic API call failed:", e);
      }
    }


    // 3. High-Fidelity Grounded Local Synthesizer (when no external keys provided)
    if (!generatedAnswer) {
      const topPassages = context_passages.slice(0, 3);
      const sentences: string[] = [];

      topPassages.forEach((p: any, idx: number) => {
        const text = typeof p === "string" ? p : p.content;
        const sents = text.split(/(?<=[.!?])\s+/).filter((s: string) => s.length > 20);
        if (sents[0]) sentences.push(`${sents[0].trim()} [${idx + 1}]`);
        if (sents[1]) sentences.push(`${sents[1].trim()} [${idx + 1}]`);
      });

      generatedAnswer = sentences.slice(0, 4).join(" ");
      modelUsed = "grounded-evidence-synthesizer";
    }

    // Extract citation indices
    const citationMatches = generatedAnswer.match(/\[\d+\]/g) || [];
    const uniqueCitations = Array.from(new Set(citationMatches)).map(c => parseInt(c.replace(/\D/g, "")));

    return NextResponse.json({
      query,
      answer: generatedAnswer,
      citations: uniqueCitations,
      model_used: modelUsed,
      latency_ms: Date.now() - startTime
    });
  } catch (error: any) {
    console.error("Generation error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to generate grounded answer" },
      { status: 500 }
    );
  }
}
