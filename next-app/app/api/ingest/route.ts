import { NextRequest, NextResponse } from "next/server";
import { supabase, isSupabaseConfigured, localVectorStore } from "@/app/lib/supabase";
import { callEmbedService } from "@/app/lib/utils";

export async function POST(req: NextRequest) {
  const startTime = Date.now();
  try {
    const body = await req.json();
    const { title, content, source = "upload", metadata = {} } = body;

    if (!title || !content) {
      return NextResponse.json(
        { error: "Title and content are required fields." },
        { status: 400 }
      );
    }

    // 1. Text chunking (approx 500 characters, 80 char overlap)
    const rawParagraphs = content.split(/\n\s*\n/).filter((p: string) => p.trim().length > 0);
    const chunks: string[] = [];

    for (const para of rawParagraphs) {
      if (para.length > 600) {
        const sentences = para.match(/[^.!?]+[.!?]+/g) || [para];
        let currentChunk = "";
        for (const s of sentences) {
          if (currentChunk.length + s.length > 500 && currentChunk.length > 0) {
            chunks.push(currentChunk.trim());
            currentChunk = s;
          } else {
            currentChunk += " " + s;
          }
        }
        if (currentChunk.trim().length > 0) {
          chunks.push(currentChunk.trim());
        }
      } else {
        chunks.push(para.trim());
      }
    }

    if (chunks.length === 0) {
      chunks.push(content.slice(0, 500));
    }

    // 2. Generate dense embeddings
    const embeddings = await callEmbedService(chunks);

    // 3. Store into database (Supabase pgvector or LocalVectorStore)
    let docId = "";
    if (isSupabaseConfigured && supabase) {
      const { data: docData, error: docError } = await supabase
        .from("documents")
        .insert({
          title,
          source,
          content,
          metadata
        })
        .select("id")
        .single();

      if (docError) {
        throw new Error(`Supabase doc insert error: ${docError.message}`);
      }
      docId = docData.id;

      const chunkRows = chunks.map((chunkText, idx) => ({
        document_id: docId,
        chunk_index: idx,
        content: chunkText,
        embedding: embeddings[idx] || [],
        metadata: { ...metadata, source, title },
        token_count: Math.ceil(chunkText.length / 4)
      }));

      const { error: chunkError } = await supabase
        .from("document_chunks")
        .insert(chunkRows);

      if (chunkError) {
        throw new Error(`Supabase chunk insert error: ${chunkError.message}`);
      }
    } else {
      // Local In-Memory Storage
      docId = await localVectorStore.insertDocument({
        title,
        source,
        content,
        metadata
      });

      const localChunkRows = chunks.map((chunkText, idx) => ({
        document_id: docId,
        document_title: title,
        chunk_index: idx,
        content: chunkText,
        embedding: embeddings[idx] || [],
        metadata: { ...metadata, source, title },
        token_count: Math.ceil(chunkText.length / 4)
      }));

      await localVectorStore.insertChunks(localChunkRows);
    }

    const totalTime = Date.now() - startTime;

    return NextResponse.json({
      success: true,
      document_id: docId,
      title,
      total_chunks: chunks.length,
      latency_ms: totalTime,
      storage_engine: isSupabaseConfigured ? "supabase_pgvector" : "in_memory_vector_store"
    });
  } catch (error: any) {
    console.error("Ingestion error:", error);
    return NextResponse.json(
      { error: error.message || "Failed to ingest document" },
      { status: 500 }
    );
  }
}

export async function GET() {
  try {
    if (isSupabaseConfigured && supabase) {
      const { data, error } = await supabase
        .from("documents")
        .select("id, title, source, created_at")
        .order("created_at", { ascending: false });
      if (error) throw error;
      return NextResponse.json({ documents: data || [] });
    } else {
      return NextResponse.json({ documents: localVectorStore.getAllDocuments() });
    }
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
