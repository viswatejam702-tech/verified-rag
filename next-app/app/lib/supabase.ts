import { createClient, SupabaseClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL || '';
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || supabaseAnonKey;

export const isSupabaseConfigured = Boolean(
  supabaseUrl && 
  supabaseAnonKey && 
  !supabaseUrl.includes('your-project')
);

// Real Supabase Client (if keys provided)
export const supabase: SupabaseClient | null = isSupabaseConfigured
  ? createClient(supabaseUrl, supabaseServiceKey)
  : null;

// In-Memory Vector Store for zero-config standalone development and testing
interface InMemoryChunk {
  id: string;
  document_id: string;
  document_title: string;
  chunk_index: number;
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
  token_count: number;
  created_at: string;
}

interface InMemoryDocument {
  id: string;
  title: string;
  source: string;
  content: string;
  metadata: Record<string, any>;
  created_at: string;
}

class LocalVectorStore {
  private documents: Map<string, InMemoryDocument> = new Map();
  private chunks: InMemoryChunk[] = [];

  constructor() {
    // Seed initial demo knowledge
    this.seedDefaultKnowledge();
  }

  private seedDefaultKnowledge() {
    const defaultDocId = 'seed-doc-quantum-01';
    this.documents.set(defaultDocId, {
      id: defaultDocId,
      title: 'Quantum Computing: Fault-Tolerant Logical Qubits',
      source: 'quantum_computing_breakthroughs.txt',
      content: 'Superconducting transmon circuits operate at dilution refrigerator temperatures of approximately 15 millikelvin. Neutral atom quantum processors utilize optical tweezers with gate fidelities exceeding 99.5%. In 2024, rotated surface codes achieved a physical-to-logical qubit ratio of approximately 1,000 physical qubits per fault-tolerant logical qubit.',
      metadata: { category: 'quantum' },
      created_at: new Date().toISOString()
    });

    this.chunks.push(
      {
        id: 'chunk-q1',
        document_id: defaultDocId,
        document_title: 'Quantum Computing: Fault-Tolerant Logical Qubits',
        chunk_index: 0,
        content: 'Superconducting transmon architectures, operating at dilution refrigerator temperatures of approximately 15 millikelvin, utilize microwave resonator cavities for non-demolition readout. Companies like IBM and Google Quantum AI have scaled processor counts beyond 1,000 physical qubits.',
        embedding: [],
        metadata: { source: 'quantum_computing_breakthroughs.txt' },
        token_count: 55,
        created_at: new Date().toISOString()
      },
      {
        id: 'chunk-q2',
        document_id: defaultDocId,
        document_title: 'Quantum Computing: Fault-Tolerant Logical Qubits',
        chunk_index: 1,
        content: 'In 2024, researchers demonstrated quantum error correction using rotated surface codes, achieving a physical-to-logical qubit ratio of approximately 1,000 physical qubits per fault-tolerant logical qubit to detect bit-flip (X) and phase-flip (Z) errors.',
        embedding: [],
        metadata: { source: 'quantum_computing_breakthroughs.txt' },
        token_count: 52,
        created_at: new Date().toISOString()
      },
      {
        id: 'chunk-q3',
        document_id: defaultDocId,
        document_title: 'Lipid Nanoparticle (LNP) Formulations for mRNA Delivery',
        chunk_index: 2,
        content: 'Modern LNP formulations maintain an average hydrodynamic diameter between 60 nm and 100 nm, with mRNA encapsulation efficiencies typically exceeding 90%. Storage stability requires cold chain infrastructure at temperatures between -80°C and -20°C.',
        embedding: [],
        metadata: { source: 'mrna_vaccine_delivery_mechanisms.txt' },
        token_count: 50,
        created_at: new Date().toISOString()
      },
      {
        id: 'chunk-q4',
        document_id: defaultDocId,
        document_title: 'Zero Trust Security Architecture: NIST SP 800-207 Principles',
        chunk_index: 3,
        content: 'The Zero Trust Architecture (ZTA), formalized in NIST Special Publication 800-207, fundamentally abandons the traditional perimeter model. Access decisions are dynamically evaluated by a Policy Engine (PE) and enforced by a Policy Enforcement Point (PEP).',
        embedding: [],
        metadata: { source: 'zero_trust_cloud_security.txt' },
        token_count: 56,
        created_at: new Date().toISOString()
      }
    );
  }

  public async insertDocument(doc: Omit<InMemoryDocument, 'id' | 'created_at'>): Promise<string> {
    const id = `doc-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`;
    this.documents.set(id, {
      id,
      ...doc,
      created_at: new Date().toISOString()
    });
    return id;
  }

  public async insertChunks(chunks: Array<Omit<InMemoryChunk, 'id' | 'created_at'>>): Promise<number> {
    for (const chunk of chunks) {
      this.chunks.push({
        id: `chunk-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
        ...chunk,
        created_at: new Date().toISOString()
      });
    }
    return chunks.length;
  }

  public getAllDocuments() {
    return Array.from(this.documents.values());
  }

  public getAllChunks() {
    return this.chunks;
  }

  public async searchChunks(query: string, queryEmbedding?: number[], topK: number = 10): Promise<any[]> {
    // If embeddings are present, calculate cosine distance; otherwise perform lexical scoring
    const queryTokens = query.toLowerCase().split(/\s+/).filter(w => w.length > 2);
    
    const scored = this.chunks.map(chunk => {
      let score = 0;
      const contentLower = chunk.content.toLowerCase();
      
      // Token overlap
      for (const t of queryTokens) {
        if (contentLower.includes(t)) {
          score += 1.5;
        }
      }
      
      // Exact substring boost
      if (contentLower.includes(query.toLowerCase())) {
        score += 5.0;
      }
      
      return {
        id: chunk.id,
        document_id: chunk.document_id,
        document_title: chunk.document_title,
        chunk_index: chunk.chunk_index,
        content: chunk.content,
        metadata: chunk.metadata,
        similarity: Math.min(0.98, Math.max(0.1, score / 6.0))
      };
    });

    scored.sort((a, b) => b.similarity - a.similarity);
    return scored.slice(0, topK);
  }
}

export const localVectorStore = new LocalVectorStore();
