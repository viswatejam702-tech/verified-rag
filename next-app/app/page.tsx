"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  ShieldCheck,
  Search,
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sparkles,
  Send,
  RefreshCw,
  FileText,
  Sliders,
  ChevronRight,
  BarChart3,
  ExternalLink,
  BookOpen,
  ArrowRight,
  Info,
  Terminal,
  Zap,
  Key,
  Server,
  Globe,
  Plus,
  Trash2,
  Check,
  Activity
} from "lucide-react";

interface ClaimResult {
  claim: string;
  verdict: "VERIFIED" | "CONTRADICTION" | "UNSUPPORTED";
  confidence: number;
  entailment_prob: number;
  contradiction_prob: number;
  neutral_prob: number;
  supporting_evidence?: string;
  supporting_passage_index?: number;
}

interface PassageItem {
  id: string;
  content: string;
  rerank_score?: number;
  confidence_score?: number;
  new_rank?: number;
  initial_rank?: number;
  rank_delta?: number;
  metadata?: any;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  claims?: ClaimResult[];
  passages?: PassageItem[];
  faithfulness_score?: number;
  hallucination_rate?: number;
  latency_breakdown?: {
    embedding_ms?: number;
    vector_search_ms?: number;
    rerank_ms?: number;
    generation_ms?: number;
    verification_ms?: number;
    total_ms?: number;
  };
  model_used?: string;
  timestamp: string;
}

interface GatewayModel {
  id: string;
  name: string;
  model_name: string;
  provider: "openai_compatible" | "openai" | "anthropic" | "gemini" | "ollama" | "deepseek" | "groq" | "custom";
  base_url: string;
  api_key: string;
  temperature: number;
  max_tokens: number;
  is_preset?: boolean;
}

const DEFAULT_GATEWAY_MODELS: GatewayModel[] = [
  {
    id: "model-deepseek-chat",
    name: "DeepSeek V3 (Chat)",
    model_name: "deepseek-chat",
    provider: "openai_compatible",
    base_url: "https://api.deepseek.com/v1",
    api_key: "",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  },
  {
    id: "model-deepseek-r1",
    name: "DeepSeek R1 (Reasoner)",
    model_name: "deepseek-reasoner",
    provider: "openai_compatible",
    base_url: "https://api.deepseek.com/v1",
    api_key: "",
    temperature: 0.6,
    max_tokens: 2048,
    is_preset: true
  },
  {
    id: "model-gpt4o",
    name: "OpenAI GPT-4o",
    model_name: "gpt-4o",
    provider: "openai",
    base_url: "https://api.openai.com/v1",
    api_key: "",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  },
  {
    id: "model-claude35",
    name: "Claude 3.5 Sonnet",
    model_name: "claude-3-5-sonnet-20241022",
    provider: "anthropic",
    base_url: "https://api.anthropic.com/v1",
    api_key: "",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  },
  {
    id: "model-gemini20",
    name: "Gemini 2.0 Flash",
    model_name: "gemini-2.0-flash",
    provider: "gemini",
    base_url: "https://generativelanguage.googleapis.com",
    api_key: "",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  },
  {
    id: "model-groq-llama",
    name: "Groq LLaMA 3.3 70B",
    model_name: "llama-3.3-70b-versatile",
    provider: "openai_compatible",
    base_url: "https://api.groq.com/openai/v1",
    api_key: "",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  },
  {
    id: "model-ollama-local",
    name: "Local Ollama / vLLM",
    model_name: "llama3.2",
    provider: "ollama",
    base_url: "http://localhost:11434/v1",
    api_key: "ollama",
    temperature: 0.2,
    max_tokens: 1024,
    is_preset: true
  }
];

export default function VerifiedRagApp() {
  const [activeTab, setActiveTab] = useState<"chat" | "pipeline" | "knowledge" | "eval" | "lewis-rag" | "gateway">("chat");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome-msg",
      role: "assistant",
      content: "Welcome to Verified RAG. Ask any question regarding the indexed scientific & technical knowledge base. Every generated claim will be verified via cross-encoder NLI and color-coded in real time.",
      faithfulness_score: 1.0,
      hallucination_rate: 0.0,
      timestamp: "Just now"
    }
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState<string>("");
  const [selectedClaim, setSelectedClaim] = useState<ClaimResult | null>(null);
  const [selectedPassage, setSelectedPassage] = useState<PassageItem | null>(null);

  // Service health states
  const [serviceHealth, setServiceHealth] = useState({
    embed: false,
    rerank: false,
    verify: false
  });

  // Settings State
  const [settings, setSettings] = useState({
    topK: 8,
    rerankTopN: 3,
    entailmentThreshold: 0.60,
    modelProvider: "auto"
  });
  const [showSettings, setShowSettings] = useState(false);

  // Knowledge base document list
  const [documents, setDocuments] = useState<any[]>([]);
  const [newDocTitle, setNewDocTitle] = useState("");
  const [newDocContent, setNewDocContent] = useState("");
  const [isIngesting, setIsIngesting] = useState(false);

  // Eval State
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalResults, setEvalResults] = useState<any>(null);

  // Lewis et al. (NeurIPS 2020) RAG State
  const [lewisQuery, setLewisQuery] = useState('Define "middle ear"');
  const [lewisTaskType, setLewisTaskType] = useState<"question_answering" | "fact_verification" | "jeopardy_generation">("question_answering");
  const [lewisModelVariant, setLewisModelVariant] = useState<"rag_sequence" | "rag_token">("rag_sequence");
  const [lewisLoading, setLewisLoading] = useState(false);
  const [lewisResult, setLewisResult] = useState<any>(null);

  const runLewisRAG = async (queryOverride?: string, taskOverride?: "question_answering" | "fact_verification" | "jeopardy_generation") => {
    setLewisLoading(true);
    const q = queryOverride !== undefined ? queryOverride : lewisQuery;
    const task = taskOverride !== undefined ? taskOverride : lewisTaskType;
    try {
      const res = await fetch("/api/lewis-rag", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_x: q,
          task_type: task,
          model_variant: lewisModelVariant,
        })
      });
      const data = await res.json();
      setLewisResult(data);
    } catch (err) {
      console.error("Lewis RAG run failed", err);
    } finally {
      setLewisLoading(false);
    }
  };

  // Model Gateway & Multi-Model Comparison State
  const [gatewayModels, setGatewayModels] = useState<GatewayModel[]>(DEFAULT_GATEWAY_MODELS);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(["model-deepseek-chat", "model-gpt4o"]);
  const [compareQuery, setCompareQuery] = useState("Explain fault-tolerant logical qubits in superconducting quantum computing");
  const [compareLoading, setCompareLoading] = useState(false);
  const [comparisonResults, setComparisonResults] = useState<any>(null);
  const [showAddModelModal, setShowAddModelModal] = useState(false);

  // New Model Form State
  const [newModelName, setNewModelName] = useState("");
  const [newModelLabel, setNewModelLabel] = useState("");
  const [newModelProvider, setNewModelProvider] = useState<"openai_compatible" | "openai" | "anthropic" | "gemini" | "ollama">("openai_compatible");
  const [newModelBaseUrl, setNewModelBaseUrl] = useState("https://api.openai.com/v1");
  const [newModelApiKey, setNewModelApiKey] = useState("");
  const [newModelTemperature, setNewModelTemperature] = useState(0.2);
  const [newModelMaxTokens, setNewModelMaxTokens] = useState(1024);

  const handleAddModel = () => {
    if (!newModelName.trim() || !newModelBaseUrl.trim()) return;
    const newEntry: GatewayModel = {
      id: `custom-model-${Date.now()}`,
      name: newModelLabel.trim() || newModelName.trim(),
      model_name: newModelName.trim(),
      provider: newModelProvider,
      base_url: newModelBaseUrl.trim(),
      api_key: newModelApiKey.trim(),
      temperature: newModelTemperature,
      max_tokens: newModelMaxTokens,
      is_preset: false
    };
    setGatewayModels(prev => [...prev, newEntry]);
    setSelectedModelIds(prev => [...prev, newEntry.id]);
    setShowAddModelModal(false);
    setNewModelName("");
    setNewModelLabel("");
    setNewModelApiKey("");
  };

  const handleDeleteModel = (id: string) => {
    setGatewayModels(prev => prev.filter(m => m.id !== id));
    setSelectedModelIds(prev => prev.filter(mid => mid !== id));
  };

  const toggleModelSelection = (id: string) => {
    if (selectedModelIds.includes(id)) {
      if (selectedModelIds.length > 1) {
        setSelectedModelIds(selectedModelIds.filter(m => m !== id));
      }
    } else {
      if (selectedModelIds.length < 4) {
        setSelectedModelIds([...selectedModelIds, id]);
      }
    }
  };

  const runModelComparison = async (queryOverride?: string) => {
    setCompareLoading(true);
    const q = queryOverride || compareQuery;
    const activeModels = gatewayModels.filter(m => selectedModelIds.includes(m.id));
    try {
      const res = await fetch("/api/models/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          model_configs: activeModels.map(m => ({
            id: m.id,
            name: m.name,
            model_name: m.model_name,
            provider: m.provider,
            base_url: m.base_url,
            api_key: m.api_key,
            temperature: m.temperature,
            max_tokens: m.max_tokens
          }))
        })
      });
      const data = await res.json();
      setComparisonResults(data);
    } catch (err) {
      console.error("Comparison execution failed", err);
    } finally {
      setCompareLoading(false);
    }
  };

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Check health of microservices
  useEffect(() => {
    checkServices();
    fetchDocuments();
  }, []);

  const checkServices = async () => {
    try {
      const eRes = await fetch("http://localhost:8001/health").catch(() => null);
      const rRes = await fetch("http://localhost:8002/health").catch(() => null);
      const vRes = await fetch("http://localhost:8003/health").catch(() => null);

      setServiceHealth({
        embed: Boolean(eRes && eRes.ok),
        rerank: Boolean(rRes && rRes.ok),
        verify: Boolean(vRes && vRes.ok)
      });
    } catch {
      // Fallback handlers active
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await fetch("/api/ingest");
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (e) {
      console.warn("Could not fetch documents", e);
    }
  };

  const handleSendMessage = async (queryText?: string) => {
    const query = queryText || inputQuery;
    if (!query.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `usr-${Date.now()}`,
      role: "user",
      content: query,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    if (!queryText) setInputQuery("");
    setIsLoading(true);

    const timing: any = {};
    const pipelineStart = Date.now();

    try {
      // STEP 1 & 2: Dense Retrieval & Cross-Encoder Rerank
      setCurrentStep("Retrieving & Reranking Passages (MiniLM + Cross-Encoder)...");
      const retStart = Date.now();
      const retRes = await fetch("/api/retrieve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: settings.topK,
          rerank_top_n: settings.rerankTopN
        })
      });
      const retData = await retRes.json();
      const passages = retData.results || [];
      timing.retrieve_ms = Date.now() - retStart;

      // STEP 3: Grounded Answer Generation
      setCurrentStep("Synthesizing Grounded Response with Source Citations...");
      const genStart = Date.now();
      const genRes = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          context_passages: passages,
          model_provider: settings.modelProvider
        })
      });
      const genData = await genRes.json();
      const answer = genData.answer || "No response generated.";
      timing.generation_ms = Date.now() - genStart;

      // STEP 4: Claim Extraction & NLI Verification
      setCurrentStep("Extracting Atomic Claims & Computing NLI Entailment Logits...");
      const verStart = Date.now();
      const verRes = await fetch("/api/verify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer,
          evidence_passages: passages.map((p: any) => p.content),
          threshold_entailment: settings.entailmentThreshold
        })
      });
      const verData = await verRes.json();
      timing.verification_ms = Date.now() - verStart;
      timing.total_ms = Date.now() - pipelineStart;

      // Create Assistant Message with Verified Claim Annotations
      const assistantMessage: ChatMessage = {
        id: `asst-${Date.now()}`,
        role: "assistant",
        content: answer,
        claims: verData.claims || [],
        passages: passages,
        faithfulness_score: verData.faithfulness_score ?? 1.0,
        hallucination_rate: verData.hallucination_rate ?? 0.0,
        latency_breakdown: timing,
        model_used: genData.model_used || "grounded-synthesizer",
        timestamp: new Date().toLocaleTimeString()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error("Pipeline error:", err);
      const errorMsg: ChatMessage = {
        id: `err-${Date.now()}`,
        role: "assistant",
        content: `Error during RAG execution: ${err.message || "Failed to process query."}`,
        timestamp: new Date().toLocaleTimeString()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
      setCurrentStep("");
    }
  };

  const handleIngestDocument = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newDocTitle.trim() || !newDocContent.trim()) return;

    setIsIngesting(true);
    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: newDocTitle,
          content: newDocContent,
          source: "user_upload"
        })
      });
      if (res.ok) {
        setNewDocTitle("");
        setNewDocContent("");
        await fetchDocuments();
        alert("Document successfully chunked, vectorized, and indexed!");
      }
    } catch (err) {
      alert("Failed to ingest document.");
    } finally {
      setIsIngesting(false);
    }
  };

  const runLiveEvaluation = async () => {
    setEvalLoading(true);
    try {
      const testQueries = [
        {
          query: "What operating temperature do superconducting transmons require?",
          expected: "15 millikelvin in dilution refrigerators."
        },
        {
          query: "What is the hydrodynamic diameter of modern Lipid Nanoparticles?",
          expected: "60 nm to 100 nm."
        },
        {
          query: "What standard formalizes Zero Trust Architecture?",
          expected: "NIST SP 800-207."
        }
      ];

      const runResults = [];
      for (const t of testQueries) {
        const res = await fetch("/api/retrieve", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: t.query, top_k: 4, rerank_top_n: 2 })
        });
        const rData = await res.json();
        const genRes = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: t.query, context_passages: rData.results || [] })
        });
        const gData = await genRes.json();
        const evalRes = await fetch("/api/eval", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: t.query,
            answer: gData.answer,
            retrieved_contexts: (rData.results || []).map((p: any) => p.content),
            ground_truth_answer: t.expected
          })
        });
        const evalData = await evalRes.json();
        runResults.push({
          query: t.query,
          metrics: evalData.metrics,
          claims: evalData.claim_breakdown
        });
      }

      setEvalResults({
        timestamp: new Date().toLocaleString(),
        cases: runResults,
        avg_faithfulness: (
          runResults.reduce((acc, r) => acc + r.metrics.faithfulness_score, 0) / runResults.length
        ).toFixed(1),
        avg_relevance: (
          runResults.reduce((acc, r) => acc + r.metrics.context_relevance, 0) / runResults.length
        ).toFixed(1),
        hallucination_rate: (
          runResults.reduce((acc, r) => acc + r.metrics.hallucination_rate, 0) / runResults.length
        ).toFixed(1)
      });
    } catch (e) {
      console.error(e);
    } finally {
      setEvalLoading(false);
    }
  };

  // Render color-coded claims inside assistant message
  const renderVerifiedText = (msg: ChatMessage) => {
    if (!msg.claims || msg.claims.length === 0) {
      return <p className="leading-relaxed">{msg.content}</p>;
    }

    return (
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5 leading-relaxed">
          {msg.claims.map((claimObj, idx) => {
            let badgeClass = "claim-verified";
            let badgeText = "VERIFIED";

            if (claimObj.verdict === "CONTRADICTION") {
              badgeClass = "claim-contradiction";
              badgeText = "CONTRADICTION";
            } else if (claimObj.verdict === "UNSUPPORTED") {
              badgeClass = "claim-unsupported";
              badgeText = "UNSUPPORTED";
            }

            return (
              <span
                key={idx}
                onClick={() => setSelectedClaim(claimObj)}
                className={`${badgeClass} inline-flex items-center gap-1.5 text-sm transition-all hover:scale-[1.01]`}
                title={`Click to inspect claim evidence (${claimObj.verdict} - ${(claimObj.confidence * 100).toFixed(0)}% confidence)`}
              >
                {claimObj.claim}
                <span className="text-[10px] px-1 py-0.5 rounded uppercase font-mono font-bold bg-black/30 opacity-80">
                  {badgeText}
                </span>
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 border-b border-white/10 glass-panel">
        <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-indigo-600 to-emerald-500 text-white shadow-lg shadow-indigo-500/20">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-bold text-lg tracking-tight text-white">VERIFIED RAG</span>
                <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded-full font-mono font-semibold">
                  CLAIM-LEVEL NLI
                </span>
              </div>
              <p className="text-xs text-slate-400">Zero-Hallucination Retrieval & Cross-Encoder Fact Verification</p>
            </div>
          </div>

          {/* Center Tabs */}
          <nav className="hidden md:flex items-center space-x-1 p-1 bg-slate-900/60 border border-white/5 rounded-xl">
            <button
              onClick={() => setActiveTab("chat")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "chat"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Live Chat
            </button>
            <button
              onClick={() => setActiveTab("pipeline")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "pipeline"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Pipeline Visualizer
            </button>
            <button
              onClick={() => setActiveTab("knowledge")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "knowledge"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Knowledge Base ({documents.length})
            </button>
            <button
              onClick={() => setActiveTab("eval")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
                activeTab === "eval"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Benchmark Hub
            </button>
            <button
              onClick={() => {
                setActiveTab("lewis-rag");
                if (!lewisResult) {
                  runLewisRAG();
                }
              }}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeTab === "lewis-rag"
                  ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-lg shadow-purple-600/30"
                  : "text-purple-300 hover:text-white hover:bg-purple-900/30"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>NeurIPS 2020 RAG</span>
            </button>
            <button
              onClick={() => setActiveTab("gateway")}
              className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center gap-1.5 ${
                activeTab === "gateway"
                  ? "bg-gradient-to-r from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-600/30"
                  : "text-emerald-300 hover:text-white hover:bg-emerald-900/30"
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              <span>Model Gateway & Compare</span>
            </button>
          </nav>

          {/* Right Service Health & Settings */}
          <div className="flex items-center space-x-3">
            <div className="hidden lg:flex items-center space-x-2 text-[11px] font-mono bg-slate-900/80 px-3 py-1.5 rounded-lg border border-white/5">
              <span className="text-slate-400">Services:</span>
              <span className="flex items-center gap-1" title="Embed Service :8001">
                <span className={`w-2 h-2 rounded-full ${serviceHealth.embed ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-amber-400"}`} />
                <span>Embed</span>
              </span>
              <span className="text-slate-600">|</span>
              <span className="flex items-center gap-1" title="Rerank Service :8002">
                <span className={`w-2 h-2 rounded-full ${serviceHealth.rerank ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-amber-400"}`} />
                <span>Rerank</span>
              </span>
              <span className="text-slate-600">|</span>
              <span className="flex items-center gap-1" title="Verification Service :8003">
                <span className={`w-2 h-2 rounded-full ${serviceHealth.verify ? "bg-emerald-400 shadow-[0_0_8px_#34d399]" : "bg-amber-400"}`} />
                <span>NLI</span>
              </span>
            </div>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-slate-300 transition-colors border border-white/10"
              title="Pipeline Settings"
            >
              <Sliders className="w-4 h-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Drawer Modal */}
      {showSettings && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex justify-end">
          <div className="w-full max-w-md bg-slate-900 border-l border-white/10 p-6 flex flex-col justify-between overflow-y-auto">
            <div className="space-y-6">
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-indigo-400" /> Pipeline Configuration
                </h3>
                <button
                  onClick={() => setShowSettings(false)}
                  className="text-slate-400 hover:text-white text-sm px-2 py-1 bg-slate-800 rounded"
                >
                  ✕ Close
                </button>
              </div>

              <div className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-300 font-medium mb-1">
                    Retrieval Top-K Candidates: <span className="text-indigo-400">{settings.topK}</span>
                  </label>
                  <input
                    type="range"
                    min="3"
                    max="20"
                    value={settings.topK}
                    onChange={e => setSettings({ ...settings, topK: Number(e.target.value) })}
                    className="w-full accent-indigo-500"
                  />
                  <p className="text-slate-500 text-[11px] mt-1">Dense embeddings retrieved from pgvector / vector index.</p>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">
                    Cross-Encoder Rerank Top-N: <span className="text-indigo-400">{settings.rerankTopN}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="8"
                    value={settings.rerankTopN}
                    onChange={e => setSettings({ ...settings, rerankTopN: Number(e.target.value) })}
                    className="w-full accent-indigo-500"
                  />
                  <p className="text-slate-500 text-[11px] mt-1">Deep cross-attention passages forwarded into generation context.</p>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">
                    NLI Entailment Threshold: <span className="text-emerald-400">{(settings.entailmentThreshold * 100).toFixed(0)}%</span>
                  </label>
                  <input
                    type="range"
                    min="0.40"
                    max="0.90"
                    step="0.05"
                    value={settings.entailmentThreshold}
                    onChange={e => setSettings({ ...settings, entailmentThreshold: Number(e.target.value) })}
                    className="w-full accent-emerald-500"
                  />
                  <p className="text-slate-500 text-[11px] mt-1">Minimum DeBERTa NLI probability required to mark claim as VERIFIED.</p>
                </div>

                <div>
                  <label className="block text-slate-300 font-medium mb-1">Generation Engine</label>
                  <select
                    value={settings.modelProvider}
                    onChange={e => setSettings({ ...settings, modelProvider: e.target.value })}
                    className="w-full bg-slate-800 border border-white/10 rounded-lg p-2 text-slate-200"
                  >
                    <option value="auto">Auto (OpenAI / Gemini / Grounded Synthesizer)</option>
                    <option value="openai">OpenAI GPT-4o-mini (via OPENAI_API_KEY)</option>
                    <option value="gemini">Google Gemini 1.5 Flash (via GEMINI_API_KEY)</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="pt-6 border-t border-white/10 text-center">
              <button
                onClick={() => setShowSettings(false)}
                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium text-xs shadow-lg shadow-indigo-500/20"
              >
                Apply & Save Parameters
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main App Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left / Center: Interactive Chat or Selected View */}
        <div className="lg:col-span-8 flex flex-col space-y-4">
          {activeTab === "chat" && (
            <div className="flex-1 flex flex-col h-[78vh] glass-panel rounded-2xl border border-white/10 overflow-hidden shadow-2xl">
              {/* Chat Header */}
              <div className="px-5 py-3 border-b border-white/10 bg-slate-900/80 flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-semibold text-slate-200">Real-Time Claim Verification Channel</span>
                </div>
                <div className="flex items-center space-x-3 text-[11px] text-slate-400">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded bg-emerald-400" /> Verified
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded bg-amber-400" /> Unsupported
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded bg-rose-400" /> Contradiction
                  </span>
                </div>
              </div>

              {/* Message Feed */}
              <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map(msg => (
                  <div
                    key={msg.id}
                    className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}
                  >
                    <div
                      className={`max-w-[88%] rounded-2xl px-4 py-3.5 shadow-md ${
                        msg.role === "user"
                          ? "bg-indigo-600 text-white rounded-br-none"
                          : "bg-slate-900/90 text-slate-200 border border-white/10 rounded-bl-none"
                      }`}
                    >
                      {msg.role === "assistant" ? renderVerifiedText(msg) : <p>{msg.content}</p>}

                      {/* Assistant Verification Badges & Telemetry Footer */}
                      {msg.role === "assistant" && msg.faithfulness_score !== undefined && (
                        <div className="mt-3 pt-2.5 border-t border-white/10 flex flex-wrap items-center justify-between text-[11px] text-slate-400 gap-2">
                          <div className="flex items-center space-x-3">
                            <span className="flex items-center gap-1 text-emerald-400 font-medium">
                              <CheckCircle2 className="w-3.5 h-3.5" />
                              Faithfulness: {(msg.faithfulness_score * 100).toFixed(0)}%
                            </span>
                            {msg.hallucination_rate! > 0 && (
                              <span className="flex items-center gap-1 text-rose-400 font-medium">
                                <AlertTriangle className="w-3.5 h-3.5" />
                                Hallucination Risk: {(msg.hallucination_rate! * 100).toFixed(0)}%
                              </span>
                            )}
                          </div>
                          {msg.latency_breakdown?.total_ms && (
                            <span className="text-[10px] font-mono text-slate-500">
                              ⚡ {msg.latency_breakdown.total_ms}ms (E2E)
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    <span suppressHydrationWarning className="text-[10px] text-slate-500 mt-1 px-2">{msg.timestamp}</span>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex items-center space-x-3 p-4 bg-slate-900/80 rounded-2xl border border-white/10 max-w-md animate-pulse">
                    <RefreshCw className="w-4 h-4 text-indigo-400 animate-spin" />
                    <span className="text-xs text-slate-300 font-medium">{currentStep}</span>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Sample Benchmark Quick Query Buttons */}
              <div className="px-4 py-2 bg-slate-950/60 border-t border-white/5 flex items-center space-x-2 overflow-x-auto text-[11px]">
                <span className="text-slate-500 flex items-center gap-1 shrink-0">
                  <Sparkles className="w-3 h-3 text-indigo-400" /> Prompts:
                </span>
                <button
                  onClick={() => handleSendMessage("What operating temperature do superconducting transmon circuits require in dilution refrigerators?")}
                  className="px-2.5 py-1 bg-slate-800/70 hover:bg-slate-700 text-slate-300 rounded-full shrink-0 border border-white/5"
                >
                  Quantum Dilution Temp
                </button>
                <button
                  onClick={() => handleSendMessage("What is the average hydrodynamic diameter and encapsulation efficiency of modern Lipid Nanoparticles?")}
                  className="px-2.5 py-1 bg-slate-800/70 hover:bg-slate-700 text-slate-300 rounded-full shrink-0 border border-white/5"
                >
                  mRNA LNP Diameter
                </button>
                <button
                  onClick={() => handleSendMessage("Which NIST Special Publication formalizes Zero Trust Architecture?")}
                  className="px-2.5 py-1 bg-slate-800/70 hover:bg-slate-700 text-slate-300 rounded-full shrink-0 border border-white/5"
                >
                  Zero Trust NIST
                </button>
                <button
                  onClick={() => handleSendMessage("Are superconducting transmon quantum computers able to operate at 300 Kelvin without vacuum chambers?")}
                  className="px-2.5 py-1 bg-rose-950/50 hover:bg-rose-900/50 text-rose-300 rounded-full shrink-0 border border-rose-800/30"
                >
                  🚨 Test Hallucination Trap
                </button>
              </div>

              {/* Input Area */}
              <div className="p-3 bg-slate-900 border-t border-white/10 flex items-center space-x-2">
                <input
                  type="text"
                  value={inputQuery}
                  onChange={e => setInputQuery(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && handleSendMessage()}
                  placeholder="Ask a scientific/technical question..."
                  className="flex-1 bg-slate-800/90 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-indigo-500 transition-colors"
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={isLoading || !inputQuery.trim()}
                  className="p-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white shadow-lg shadow-indigo-600/30 transition-all"
                >
                  <Send className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {activeTab === "pipeline" && (
            <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-6">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Layers className="w-5 h-5 text-indigo-400" /> Multi-Stage RAG Pipeline Architecture
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-400">1. Ingest & Embedding</span>
                    <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-300 font-mono">:8001</span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Text is segmented into 500-char chunks with 80-char overlap. 384-dimensional dense vectors are generated via MiniLM.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-indigo-400">2. Cross-Encoder Rerank</span>
                    <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-300 font-mono">:8002</span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Joint cross-attention evaluates top candidates, filtering vector noise and reordering with sigmoid confidence logits.
                  </p>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-emerald-400">3. NLI Claim Verification</span>
                    <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded text-slate-300 font-mono">:8003</span>
                  </div>
                  <p className="text-xs text-slate-300">
                    Deconstructs response into atomic propositions. Evaluates Premise vs Hypothesis to flag hallucinations.
                  </p>
                </div>
              </div>

              {/* Interactive Telemetry Log of Latest Message */}
              {messages.length > 1 && messages[messages.length - 1].role === "assistant" && (
                <div className="space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Latest Query Telemetry</h3>
                  <div className="p-4 rounded-xl bg-slate-950 border border-white/10 font-mono text-xs space-y-3">
                    <div className="flex justify-between border-b border-white/10 pb-2">
                      <span className="text-slate-400">Retrieved Passages:</span>
                      <span className="text-indigo-400">{messages[messages.length - 1].passages?.length || 0} chunks</span>
                    </div>
                    <div className="flex justify-between border-b border-white/10 pb-2">
                      <span className="text-slate-400">Claims Verified:</span>
                      <span className="text-emerald-400">{messages[messages.length - 1].claims?.length || 0} claims</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">Faithfulness Index:</span>
                      <span className="text-emerald-400">
                        {((messages[messages.length - 1].faithfulness_score || 1) * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "knowledge" && (
            <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-6">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-indigo-400" /> Indexed Knowledge Base
                </h2>
                <span className="text-xs text-slate-400">Active Documents: {documents.length}</span>
              </div>

              {/* Upload New Document Form */}
              <form onSubmit={handleIngestDocument} className="p-4 rounded-xl bg-slate-900/80 border border-white/10 space-y-3">
                <h3 className="text-xs font-bold text-slate-200">Ingest New Research Document</h3>
                <input
                  type="text"
                  placeholder="Document Title (e.g., Quantum Error Correction 2024)"
                  value={newDocTitle}
                  onChange={e => setNewDocTitle(e.target.value)}
                  className="w-full bg-slate-800 border border-white/10 rounded-lg p-2.5 text-xs text-white placeholder-slate-500"
                  required
                />
                <textarea
                  placeholder="Paste document text or research paper excerpt here..."
                  value={newDocContent}
                  onChange={e => setNewDocContent(e.target.value)}
                  rows={4}
                  className="w-full bg-slate-800 border border-white/10 rounded-lg p-2.5 text-xs text-white placeholder-slate-500"
                  required
                />
                <button
                  type="submit"
                  disabled={isIngesting}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-medium shadow-md transition-all flex items-center gap-2"
                >
                  {isIngesting ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <FileText className="w-3.5 h-3.5" />}
                  {isIngesting ? "Chunking & Vectorizing..." : "Ingest Document"}
                </button>
              </form>

              {/* Document List Table */}
              <div className="space-y-2">
                <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider">Available Seed Documents</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-64 overflow-y-auto">
                  {documents.map((doc, i) => (
                    <div key={i} className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                      <div className="text-xs font-semibold text-slate-200">{doc.title}</div>
                      <div className="text-[10px] text-slate-400 font-mono">Source: {doc.source}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {activeTab === "eval" && (
            <div className="glass-panel p-6 rounded-2xl border border-white/10 space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-bold text-white flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-indigo-400" /> Evaluation & Benchmark Suite
                  </h2>
                  <p className="text-xs text-slate-400">Automated hallucination scoring & retrieval accuracy</p>
                </div>
                <button
                  onClick={runLiveEvaluation}
                  disabled={evalLoading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-xl text-xs font-medium shadow-lg shadow-emerald-600/20 flex items-center gap-2 transition-all"
                >
                  {evalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                  {evalLoading ? "Running Benchmark..." : "Run Evaluation Suite"}
                </button>
              </div>

              {/* Gauge Cards */}
              {evalResults && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-emerald-500/30 text-center">
                    <div className="text-2xl font-black text-emerald-400">{evalResults.avg_faithfulness}%</div>
                    <div className="text-[11px] text-slate-400 mt-1 uppercase font-semibold">Average Faithfulness</div>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-indigo-500/30 text-center">
                    <div className="text-2xl font-black text-indigo-400">{evalResults.avg_relevance}%</div>
                    <div className="text-[11px] text-slate-400 mt-1 uppercase font-semibold">Context Relevance</div>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-900/80 border border-rose-500/30 text-center">
                    <div className="text-2xl font-black text-rose-400">{evalResults.hallucination_rate}%</div>
                    <div className="text-[11px] text-slate-400 mt-1 uppercase font-semibold">Hallucination Rate</div>
                  </div>
                </div>
              )}

              {/* Benchmark Questions */}
              {evalResults && (
                <div className="space-y-3">
                  <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Detailed Test Runs</h3>
                  <div className="space-y-2">
                    {evalResults.cases.map((c: any, idx: number) => (
                      <div key={idx} className="p-3 rounded-xl bg-slate-900/60 border border-white/5 space-y-1">
                        <div className="text-xs font-semibold text-slate-200">{c.query}</div>
                        <div className="flex gap-3 text-[11px] text-slate-400 font-mono">
                          <span>Faithfulness: {c.metrics.faithfulness_score}%</span>
                          <span>Context Relevance: {c.metrics.context_relevance}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === "gateway" && (
            <div className="glass-panel p-6 rounded-2xl border border-emerald-500/20 space-y-6 animate-in fade-in duration-200">
              {/* Header */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-gradient-to-tr from-emerald-600 to-teal-600 text-white shadow-lg shadow-emerald-500/20">
                      <Cpu className="w-5 h-5" />
                    </div>
                    <h2 className="text-base font-bold text-white tracking-tight">
                      Enterprise Model Gateway & Multi-Model Arena
                    </h2>
                    <span className="text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full font-mono font-semibold">
                      MULTI-GATEWAY ROUTING
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Connect any OpenAI-compatible proxy, Anthropic, Gemini, DeepSeek, Groq, or Local Ollama/vLLM gateway with model name, custom Base URL, and API key.
                  </p>
                </div>

                <button
                  onClick={() => setShowAddModelModal(true)}
                  className="px-3.5 py-2 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-semibold shadow-lg shadow-emerald-600/20 flex items-center gap-1.5 transition-all self-start md:self-auto"
                >
                  <Plus className="w-4 h-4" />
                  <span>Add New Model / Gateway</span>
                </button>
              </div>

              {/* Gateway Models Grid */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                    <Server className="w-3.5 h-3.5 text-emerald-400" /> Registered Model Endpoints ({gatewayModels.length})
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Selected for comparison: <strong className="text-emerald-400">{selectedModelIds.length}</strong> (select 2–4 models)
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                  {gatewayModels.map(model => {
                    const isSelected = selectedModelIds.includes(model.id);
                    return (
                      <div
                        key={model.id}
                        onClick={() => toggleModelSelection(model.id)}
                        className={`p-3.5 rounded-xl border transition-all cursor-pointer relative ${
                          isSelected
                            ? "bg-emerald-950/40 border-emerald-500/50 shadow-lg shadow-emerald-950/40"
                            : "bg-slate-900/60 border-white/5 hover:border-white/20"
                        }`}
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="text-xs font-bold text-white flex items-center gap-1.5">
                              <span>{model.name}</span>
                            </div>
                            <div className="text-[11px] font-mono text-emerald-400 mt-0.5 truncate max-w-[170px]">
                              {model.model_name}
                            </div>
                          </div>
                          <div className="flex items-center gap-1.5">
                            <span
                              className={`w-4 h-4 rounded flex items-center justify-center text-[10px] ${
                                isSelected ? "bg-emerald-500 text-black font-bold" : "border border-slate-600"
                              }`}
                            >
                              {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                            </span>
                            {!model.is_preset && (
                              <button
                                onClick={e => {
                                  e.stopPropagation();
                                  handleDeleteModel(model.id);
                                }}
                                className="p-1 text-slate-500 hover:text-rose-400 transition-colors"
                                title="Remove model"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            )}
                          </div>
                        </div>

                        <div className="mt-3 pt-2 border-t border-white/5 flex items-center justify-between text-[10px] text-slate-400 font-mono">
                          <span className="flex items-center gap-1">
                            <Globe className="w-3 h-3 text-slate-500" />
                            <span className="truncate max-w-[130px]">{model.base_url.replace(/^https?:\/\//, "")}</span>
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 uppercase text-[9px]">
                            {model.provider.replace("_compatible", "")}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Multi-Model Comparison Arena Section */}
              <div className="pt-2 border-t border-white/10 space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-indigo-400" /> Side-by-Side Grounded Comparison Arena
                  </span>
                  <div className="flex flex-wrap gap-1.5 text-[11px]">
                    <button
                      onClick={() => {
                        const q = "Explain fault-tolerant logical qubits in superconducting quantum computing";
                        setCompareQuery(q);
                        runModelComparison(q);
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-white/5 transition-colors font-mono text-[10px]"
                    >
                      #1 Quantum Qubits
                    </button>
                    <button
                      onClick={() => {
                        const q = "Define the anatomical structure and acoustic function of the human middle ear";
                        setCompareQuery(q);
                        runModelComparison(q);
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-white/5 transition-colors font-mono text-[10px]"
                    >
                      #2 Middle Ear Anatomy
                    </button>
                    <button
                      onClick={() => {
                        const q = "How does differential privacy with Rényi divergence bound re-identification risk?";
                        setCompareQuery(q);
                        runModelComparison(q);
                      }}
                      className="px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-white/5 transition-colors font-mono text-[10px]"
                    >
                      #3 Rényi Privacy
                    </button>
                  </div>
                </div>

                <div className="p-3 bg-slate-900/90 rounded-xl border border-white/10 flex items-center gap-2">
                  <input
                    type="text"
                    value={compareQuery}
                    onChange={e => setCompareQuery(e.target.value)}
                    onKeyDown={e => e.key === "Enter" && runModelComparison()}
                    placeholder="Enter benchmark prompt to compare all selected models side-by-side..."
                    className="flex-1 bg-slate-800/90 border border-white/10 rounded-lg px-4 py-2.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-emerald-500 font-mono transition-colors"
                  />
                  <button
                    onClick={() => runModelComparison()}
                    disabled={compareLoading || !compareQuery.trim() || selectedModelIds.length < 1}
                    className="px-4 py-2.5 rounded-lg bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-emerald-600/30 flex items-center gap-2 transition-all whitespace-nowrap"
                  >
                    {compareLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                    <span>{compareLoading ? "Running Models..." : `Compare (${selectedModelIds.length})`}</span>
                  </button>
                </div>

                {/* Comparison Results Grid */}
                {comparisonResults && comparisonResults.results && (
                  <div className="space-y-4 animate-in fade-in duration-200">
                    <div className="flex items-center justify-between text-xs text-slate-400 font-mono border-b border-white/10 pb-2">
                      <span>Query: "{comparisonResults.query}"</span>
                      <span>Total Time: ⚡ {comparisonResults.orchestration_latency_ms}ms</span>
                    </div>

                    <div className={`grid grid-cols-1 ${comparisonResults.results.length === 2 ? "md:grid-cols-2" : comparisonResults.results.length === 3 ? "md:grid-cols-3" : "md:grid-cols-2 lg:grid-cols-4"} gap-4`}>
                      {comparisonResults.results.map((res: any, idx: number) => (
                        <div
                          key={idx}
                          className="p-4 rounded-xl bg-slate-900/90 border border-white/10 space-y-3.5 flex flex-col justify-between shadow-xl relative overflow-hidden"
                        >
                          <div className="space-y-3">
                            {/* Column Header */}
                            <div className="flex items-start justify-between border-b border-white/10 pb-2.5">
                              <div>
                                <span className="text-xs font-bold text-white block">{res.name}</span>
                                <span className="text-[10px] font-mono text-emerald-400">{res.model_name}</span>
                              </div>
                              <span className="text-[10px] font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
                                ⚡ {res.latency_ms}ms
                              </span>
                            </div>

                            {/* Fidelity & Hallucination Gauges */}
                            <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                              <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-center">
                                <div className="text-xs font-bold text-emerald-400">
                                  {(res.faithfulness_score * 100).toFixed(0)}%
                                </div>
                                <div className="text-[9px] text-slate-400 uppercase mt-0.5">Faithfulness</div>
                              </div>
                              <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/20 text-center">
                                <div className="text-xs font-bold text-rose-400">
                                  {(res.hallucination_rate * 100).toFixed(0)}%
                                </div>
                                <div className="text-[9px] text-slate-400 uppercase mt-0.5">Hallucination</div>
                              </div>
                            </div>

                            {/* Generated Answer with Claims Color-coding */}
                            <div className="space-y-1.5">
                              <span className="text-[10px] text-slate-400 uppercase font-mono font-bold">Answer Output:</span>
                              <div className="p-3 rounded-lg bg-slate-950/80 border border-white/5 text-xs text-slate-200 leading-relaxed max-h-56 overflow-y-auto">
                                {res.claims && res.claims.length > 0 ? (
                                  <div className="space-y-1">
                                    {res.claims.map((c: any, cIdx: number) => {
                                      const isVerified = c.verdict === "VERIFIED";
                                      const isContra = c.verdict === "CONTRADICTION";
                                      return (
                                        <span
                                          key={cIdx}
                                          title={`Claim verdict: ${c.verdict} (${(c.confidence * 100).toFixed(0)}%)`}
                                          className={`inline mr-1 px-1 py-0.5 rounded text-xs leading-relaxed ${
                                            isVerified
                                              ? "bg-emerald-500/15 text-emerald-200 border-b border-emerald-500/40"
                                              : isContra
                                              ? "bg-rose-500/20 text-rose-200 border-b border-rose-500/40"
                                              : "bg-amber-500/15 text-amber-200 border-b border-amber-500/40"
                                          }`}
                                        >
                                          {c.claim}{" "}
                                        </span>
                                      );
                                    })}
                                  </div>
                                ) : (
                                  <p>{res.answer}</p>
                                )}
                              </div>
                            </div>

                            {/* Claims Breakdown Summary */}
                            <div className="text-[10px] font-mono text-slate-400 flex items-center justify-between border-t border-white/5 pt-2">
                              <span>Verified: <strong className="text-emerald-400">{res.verified_claims_count}</strong></span>
                              <span>Unsupported: <strong className="text-amber-400">{res.unsupported_claims_count}</strong></span>
                              <span>Contradicted: <strong className="text-rose-400">{res.contradicted_claims_count}</strong></span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Shared Evidence Context Passages */}
                    {comparisonResults.retrieved_passages && (
                      <div className="p-4 rounded-xl bg-slate-950 border border-white/10 space-y-2">
                        <span className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5 font-mono">
                          <Database className="w-3.5 h-3.5 text-indigo-400" /> Retrieved Context Evidence Shared Across Models:
                        </span>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 text-xs">
                          {comparisonResults.retrieved_passages.map((p: any, idx: number) => (
                            <div key={idx} className="p-2.5 rounded-lg bg-slate-900 border border-white/5 space-y-1">
                              <span className="text-[10px] font-mono text-indigo-400 font-bold block">[Passage {p.index}] {p.title}</span>
                              <p className="text-slate-300 text-[11px] line-clamp-3 italic">"{p.content}"</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Add Custom Model Modal */}
          {showAddModelModal && (
            <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
              <div className="w-full max-w-lg bg-slate-900 border border-white/10 rounded-2xl p-6 space-y-5 shadow-2xl animate-in zoom-in-95 duration-150">
                <div className="flex items-center justify-between border-b border-white/10 pb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-emerald-600/30 text-emerald-400 border border-emerald-500/40">
                      <Cpu className="w-4 h-4" />
                    </div>
                    <div>
                      <h3 className="text-sm font-bold text-white">Add Model Gateway Endpoint</h3>
                      <p className="text-[11px] text-slate-400">Connect any model with custom API key & base URL</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setShowAddModelModal(false)}
                    className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                  >
                    ✕
                  </button>
                </div>

                {/* Quick Autofill Presets */}
                <div className="space-y-1.5">
                  <span className="text-[10px] font-bold uppercase text-slate-400 font-mono">Quick Endpoint Presets:</span>
                  <div className="flex flex-wrap gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("DeepSeek V3");
                        setNewModelName("deepseek-chat");
                        setNewModelProvider("openai_compatible");
                        setNewModelBaseUrl("https://api.deepseek.com/v1");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      DeepSeek
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("OpenAI GPT-4o");
                        setNewModelName("gpt-4o");
                        setNewModelProvider("openai");
                        setNewModelBaseUrl("https://api.openai.com/v1");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      OpenAI
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("Groq LLaMA 3.3");
                        setNewModelName("llama-3.3-70b-versatile");
                        setNewModelProvider("openai_compatible");
                        setNewModelBaseUrl("https://api.groq.com/openai/v1");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      Groq
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("Anthropic Claude 3.5");
                        setNewModelName("claude-3-5-sonnet-20241022");
                        setNewModelProvider("anthropic");
                        setNewModelBaseUrl("https://api.anthropic.com/v1");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      Anthropic
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("Local Ollama");
                        setNewModelName("llama3.2");
                        setNewModelProvider("ollama");
                        setNewModelBaseUrl("http://localhost:11434/v1");
                        setNewModelApiKey("ollama");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      Local Ollama
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setNewModelLabel("Local vLLM");
                        setNewModelName("mistralai/Mistral-7B-Instruct-v0.3");
                        setNewModelProvider("openai_compatible");
                        setNewModelBaseUrl("http://localhost:8000/v1");
                      }}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono transition-colors border border-white/5"
                    >
                      vLLM / LMStudio
                    </button>
                  </div>
                </div>

                {/* Form Fields */}
                <div className="space-y-3.5 text-xs">
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-slate-300 font-medium mb-1">Display Label</label>
                      <input
                        type="text"
                        value={newModelLabel}
                        onChange={e => setNewModelLabel(e.target.value)}
                        placeholder="e.g. DeepSeek V3 Production"
                        className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 text-xs"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-300 font-medium mb-1">Model Name / ID *</label>
                      <input
                        type="text"
                        value={newModelName}
                        onChange={e => setNewModelName(e.target.value)}
                        placeholder="e.g. deepseek-chat"
                        className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono text-xs"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1">Provider Protocol</label>
                    <select
                      value={newModelProvider}
                      onChange={e => setNewModelProvider(e.target.value as any)}
                      className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500 text-xs"
                    >
                      <option value="openai_compatible">OpenAI-Compatible Gateway (DeepSeek, Groq, Together, vLLM, OpenRouter)</option>
                      <option value="openai">OpenAI Native API</option>
                      <option value="anthropic">Anthropic Native API</option>
                      <option value="gemini">Google Gemini Native API</option>
                      <option value="ollama">Local Ollama API</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1 flex items-center justify-between">
                      <span className="flex items-center gap-1">
                        <Globe className="w-3 h-3 text-slate-400" /> Gateway Base URL *
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono">e.g. https://api.deepseek.com/v1</span>
                    </label>
                    <input
                      type="text"
                      value={newModelBaseUrl}
                      onChange={e => setNewModelBaseUrl(e.target.value)}
                      placeholder="https://api.openai.com/v1"
                      className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono text-xs"
                    />
                  </div>

                  <div>
                    <label className="block text-slate-300 font-medium mb-1 flex items-center justify-between">
                      <span className="flex items-center gap-1">
                        <Key className="w-3 h-3 text-slate-400" /> API Key (Optional / Gateway Token)
                      </span>
                      <span className="text-[10px] text-slate-500">Stored client-side only</span>
                    </label>
                    <input
                      type="password"
                      value={newModelApiKey}
                      onChange={e => setNewModelApiKey(e.target.value)}
                      placeholder="sk-..."
                      className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500 font-mono text-xs"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-3 pt-1">
                    <div>
                      <label className="block text-slate-300 font-medium mb-1">
                        Temperature: <span className="text-emerald-400 font-mono">{newModelTemperature}</span>
                      </label>
                      <input
                        type="range"
                        min="0.0"
                        max="1.0"
                        step="0.05"
                        value={newModelTemperature}
                        onChange={e => setNewModelTemperature(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                    <div>
                      <label className="block text-slate-300 font-medium mb-1">
                        Max Output Tokens: <span className="text-emerald-400 font-mono">{newModelMaxTokens}</span>
                      </label>
                      <input
                        type="range"
                        min="256"
                        max="4096"
                        step="256"
                        value={newModelMaxTokens}
                        onChange={e => setNewModelMaxTokens(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-white/10 flex items-center justify-end gap-2.5">
                  <button
                    type="button"
                    onClick={() => setShowAddModelModal(false)}
                    className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleAddModel}
                    disabled={!newModelName.trim() || !newModelBaseUrl.trim()}
                    className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-emerald-600/20 flex items-center gap-1.5 transition-all"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Save & Register Model</span>
                  </button>
                </div>
              </div>
            </div>
          )}
            <div className="glass-panel p-6 rounded-2xl border border-purple-500/20 space-y-6 animate-in fade-in duration-200">
              {/* Header */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-white/10 pb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-gradient-to-tr from-purple-600 to-indigo-600 text-white">
                      <Sparkles className="w-5 h-5" />
                    </div>
                    <h2 className="text-base font-bold text-white tracking-tight">
                      Lewis et al. (NeurIPS 2020) RAG Architecture
                    </h2>
                    <span className="text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded-full font-mono">
                      FIGURE 1 REPRODUCTION
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Non-parametric MIPS retriever p_η(z|x) coupled with marginalized parametric generator p_θ(y|x, z).
                  </p>
                </div>

                <div className="flex items-center gap-2 text-xs font-mono">
                  <span className="text-slate-400 text-[11px]">Variant:</span>
                  <div className="inline-flex rounded-lg p-0.5 bg-slate-900 border border-white/10">
                    <button
                      onClick={() => setLewisModelVariant("rag_sequence")}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                        lewisModelVariant === "rag_sequence"
                          ? "bg-purple-600 text-white shadow"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      RAG-Sequence
                    </button>
                    <button
                      onClick={() => setLewisModelVariant("rag_token")}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                        lewisModelVariant === "rag_token"
                          ? "bg-purple-600 text-white shadow"
                          : "text-slate-400 hover:text-white"
                      }`}
                    >
                      RAG-Token
                    </button>
                  </div>
                </div>
              </div>

              {/* Task Head Selector & Benchmark Presets */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
                    Figure 1 Benchmark Presets:
                  </span>
                  <span className="text-[11px] text-purple-400 font-mono">
                    Task Head: {lewisTaskType.replace("_", " ").toUpperCase()}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <button
                    onClick={() => {
                      setLewisTaskType("question_answering");
                      setLewisQuery('Define "middle ear"');
                      runLewisRAG('Define "middle ear"', "question_answering");
                    }}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      lewisTaskType === "question_answering"
                        ? "bg-purple-950/40 border-purple-500/50 shadow-lg shadow-purple-950/50"
                        : "bg-slate-900/60 border-white/5 hover:border-white/20"
                    }`}
                  >
                    <div className="text-[11px] font-bold text-indigo-400 flex items-center gap-1.5 mb-1">
                      <BookOpen className="w-3.5 h-3.5" /> Head A: Question Answering
                    </div>
                    <div className="text-xs text-white font-medium">Define "middle ear" (x)</div>
                    <div className="text-[10px] text-slate-400 mt-1 font-mono">Output: Answer Generation (y)</div>
                  </button>

                  <button
                    onClick={() => {
                      setLewisTaskType("fact_verification");
                      setLewisQuery("Barack Obama was born in Hawaii.");
                      runLewisRAG("Barack Obama was born in Hawaii.", "fact_verification");
                    }}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      lewisTaskType === "fact_verification"
                        ? "bg-purple-950/40 border-purple-500/50 shadow-lg shadow-purple-950/50"
                        : "bg-slate-900/60 border-white/5 hover:border-white/20"
                    }`}
                  >
                    <div className="text-[11px] font-bold text-emerald-400 flex items-center gap-1.5 mb-1">
                      <ShieldCheck className="w-3.5 h-3.5" /> Head B: Fact Verification
                    </div>
                    <div className="text-xs text-white font-medium">Barack Obama was born in Hawaii. (x)</div>
                    <div className="text-[10px] text-slate-400 mt-1 font-mono">Output: FEVER Label supports (y)</div>
                  </button>

                  <button
                    onClick={() => {
                      setLewisTaskType("jeopardy_generation");
                      setLewisQuery("The Divine Comedy");
                      runLewisRAG("The Divine Comedy", "jeopardy_generation");
                    }}
                    className={`p-3 rounded-xl border text-left transition-all ${
                      lewisTaskType === "jeopardy_generation"
                        ? "bg-purple-950/40 border-purple-500/50 shadow-lg shadow-purple-950/50"
                        : "bg-slate-900/60 border-white/5 hover:border-white/20"
                    }`}
                  >
                    <div className="text-[11px] font-bold text-amber-400 flex items-center gap-1.5 mb-1">
                      <Zap className="w-3.5 h-3.5" /> Head C: Jeopardy Clue
                    </div>
                    <div className="text-xs text-white font-medium">The Divine Comedy (x)</div>
                    <div className="text-[10px] text-slate-400 mt-1 font-mono">Output: Clue Question (y)</div>
                  </button>
                </div>
              </div>

              {/* Execution Form */}
              <div className="p-3 bg-slate-900/90 rounded-xl border border-white/10 flex items-center gap-2">
                <input
                  type="text"
                  value={lewisQuery}
                  onChange={e => setLewisQuery(e.target.value)}
                  onKeyDown={e => e.key === "Enter" && runLewisRAG()}
                  placeholder="Enter query x (question, fact statement, or entity)..."
                  className="flex-1 bg-slate-800/90 border border-white/10 rounded-lg px-4 py-2.5 text-xs text-white placeholder-slate-400 focus:outline-none focus:border-purple-500 font-mono transition-colors"
                />
                <button
                  onClick={() => runLewisRAG()}
                  disabled={lewisLoading || !lewisQuery.trim()}
                  className="px-4 py-2.5 rounded-lg bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 disabled:opacity-50 text-white text-xs font-semibold shadow-lg shadow-purple-600/30 flex items-center gap-2 transition-all"
                >
                  {lewisLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
                  <span>{lewisLoading ? "Marginalizing..." : "Run Marginalization"}</span>
                </button>
              </div>

              {/* Live Output Section */}
              {lewisResult && (
                <div className="space-y-5 animate-in fade-in duration-200">
                  {/* Generator Output Card */}
                  <div className="p-4 rounded-xl bg-slate-900/90 border border-purple-500/30 space-y-3 shadow-xl">
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 pb-2.5">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase font-bold text-slate-400">Parametric Generator p_θ</span>
                        <span className="text-[10px] font-mono bg-purple-500/20 text-purple-300 px-2 py-0.5 rounded border border-purple-500/30">
                          {lewisResult.model_variant}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 text-xs font-mono">
                        <span className="text-emerald-400 font-bold">
                          p(y|x): {(lewisResult.marginal_probability * 100).toFixed(1)}%
                        </span>
                        <span className="text-slate-500">
                          log p: {lewisResult.sequence_log_likelihood}
                        </span>
                        <span className="text-slate-500">
                          ⚡ {lewisResult.latency_ms}ms
                        </span>
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="text-[11px] text-slate-400 font-mono">Generated Output (y):</div>
                      <div className="p-3 rounded-lg bg-black/40 border border-white/10 text-sm font-medium text-slate-100 leading-relaxed flex items-center justify-between">
                        <span>{lewisResult.generated_y}</span>
                        {lewisResult.verification_label && (
                          <span
                            className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold uppercase tracking-wider ${
                              lewisResult.verification_label === "supports"
                                ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                                : lewisResult.verification_label === "refutes"
                                ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                                : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                            }`}
                          >
                            {lewisResult.verification_label}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Per-token Attributions (for RAG-Token) */}
                    {lewisResult.per_token_attributions && (
                      <div className="pt-2 border-t border-white/10 space-y-1.5">
                        <div className="text-[10px] text-slate-400 uppercase font-mono font-bold">
                          RAG-Token Step-Wise Marginalization & Attributions:
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {lewisResult.per_token_attributions.map((t: any, idx: number) => (
                            <span
                              key={idx}
                              title={`Token: "${t.token}" | Marginal P: ${(t.marginal_p * 100).toFixed(1)}%`}
                              className="px-1.5 py-0.5 rounded text-[11px] font-mono bg-purple-900/40 border border-purple-500/30 text-purple-200"
                            >
                              {t.token}
                              <span className="ml-1 text-[9px] text-purple-400 font-bold">
                                {(t.marginal_p * 100).toFixed(0)}%
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Latent Retrieved Documents p_η(z|x) */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                        <Database className="w-3.5 h-3.5 text-purple-400" />
                        Retriever p_η(z|x) MIPS Latent Passages (Top Candidates):
                      </span>
                      <span className="text-[11px] text-slate-500 font-mono">
                        Formula: p_η(z|x) = softmax( q(x)^T d(z) / τ )
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {lewisResult.latent_documents.map((doc: any, idx: number) => (
                        <div
                          key={idx}
                          className="p-3.5 rounded-xl bg-slate-900/70 border border-white/10 space-y-2 hover:border-purple-500/40 transition-colors"
                        >
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                              <span className="w-5 h-5 rounded-full bg-purple-600/30 border border-purple-500/50 flex items-center justify-center text-[10px] font-mono text-purple-300">
                                z{idx + 1}
                              </span>
                              {doc.document_title}
                            </span>
                            <span className="font-mono text-xs font-bold text-purple-400">
                              p_η: {(doc.prior_prob * 100).toFixed(1)}%
                            </span>
                          </div>

                          {/* Prior Probability Progress Bar */}
                          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div
                              className="bg-gradient-to-r from-purple-500 to-indigo-400 h-full rounded-full transition-all duration-300"
                              style={{ width: `${Math.min(doc.prior_prob * 100, 100)}%` }}
                            />
                          </div>

                          <p className="text-xs text-slate-300 line-clamp-3 leading-relaxed italic">
                            "{doc.content}"
                          </p>

                          <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-white/5">
                            <span>MIPS Score: {doc.mips_score}</span>
                            <span>ID: {doc.chunk_id}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right Column: Claim Inspector & Evidence Drawer */}
        <div className="lg:col-span-4 flex flex-col space-y-4">
          <div className="glass-panel p-5 rounded-2xl border border-white/10 space-y-4 flex-1">
            <h3 className="text-sm font-bold text-white flex items-center gap-2 border-b border-white/10 pb-3">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              Claim-Level Fact Inspector
            </h3>

            {selectedClaim ? (
              <div className="space-y-4 text-xs animate-in fade-in duration-200">
                <div className="p-3 rounded-xl bg-slate-900 border border-white/10 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 uppercase text-[10px] font-bold">Selected Claim</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase font-mono ${
                        selectedClaim.verdict === "VERIFIED"
                          ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                          : selectedClaim.verdict === "CONTRADICTION"
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          : "bg-amber-500/20 text-amber-300 border border-amber-500/40"
                      }`}
                    >
                      {selectedClaim.verdict}
                    </span>
                  </div>
                  <p className="text-slate-200 font-medium leading-relaxed">{selectedClaim.claim}</p>
                </div>

                {/* Probability Breakdown */}
                <div className="p-3 rounded-xl bg-slate-900 border border-white/10 space-y-2">
                  <span className="text-slate-400 uppercase text-[10px] font-bold">NLI Model Probabilities</span>
                  <div className="space-y-1.5 font-mono text-[11px]">
                    <div className="flex justify-between items-center">
                      <span className="text-emerald-400">Entailment (Support):</span>
                      <span>{(selectedClaim.entailment_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-emerald-400 h-full rounded-full"
                        style={{ width: `${selectedClaim.entailment_prob * 100}%` }}
                      />
                    </div>

                    <div className="flex justify-between items-center mt-2">
                      <span className="text-amber-400">Neutral (Unsubstantiated):</span>
                      <span>{(selectedClaim.neutral_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-amber-400 h-full rounded-full"
                        style={{ width: `${selectedClaim.neutral_prob * 100}%` }}
                      />
                    </div>

                    <div className="flex justify-between items-center mt-2">
                      <span className="text-rose-400">Contradiction (Falsehood):</span>
                      <span>{(selectedClaim.contradiction_prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-rose-400 h-full rounded-full"
                        style={{ width: `${selectedClaim.contradiction_prob * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Grounding Source Snippet */}
                {selectedClaim.supporting_evidence && (
                  <div className="p-3 rounded-xl bg-slate-900 border border-white/10 space-y-1.5">
                    <span className="text-slate-400 uppercase text-[10px] font-bold">Source Context Snippet</span>
                    <p className="text-slate-300 italic text-[11px] leading-relaxed">
                      "{selectedClaim.supporting_evidence}"
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 space-y-2">
                <Info className="w-8 h-8 mx-auto text-slate-600" />
                <p className="text-xs">Click on any highlighted sentence in the chat to inspect its atomic NLI evidence and logits.</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
