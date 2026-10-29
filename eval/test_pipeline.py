"""
Unit & Pipeline Test Suite for Verified RAG Microservices
Tests embedding generation, text chunking, cross-encoder reranking, claim extraction, and NLI verification.
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
embed_utils = load_module_from_file("embed_utils", os.path.join(root_dir, "embed-service", "app", "utils.py"))
rerank_main = load_module_from_file("rerank_main", os.path.join(root_dir, "rerank-service", "app", "main.py"))
verify_main = load_module_from_file("verify_main", os.path.join(root_dir, "verification-service", "app", "main.py"))

chunk_text = embed_utils.chunk_text
generate_fallback_embedding = embed_utils.generate_fallback_embedding
compute_cosine_similarity = embed_utils.compute_cosine_similarity
fallback_bm25_rerank = rerank_main.fallback_bm25_rerank
extract_claims_from_text = verify_main.extract_claims_from_text
fallback_heuristic_nli = verify_main.fallback_heuristic_nli



def test_chunking():
    print("Testing Text Chunking...")
    sample_text = """
    Quantum computing relies on qubits. Unlike classical bits that represent 0 or 1, qubits exist in superpositions.
    
    Fault-tolerant quantum computing requires quantum error correction using surface codes. In 2024, researchers achieved 1000 physical qubits per logical qubit.
    
    Cryogenic systems maintain temperatures around 15 millikelvin in dilution refrigerators to prevent thermal decoherence.
    """
    chunks = chunk_text(sample_text, chunk_size=150, chunk_overlap=30)
    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"
    print(f"✅ Chunking passed: generated {len(chunks)} chunks.")


def test_embeddings():
    print("\nTesting Deterministic Embeddings & Cosine Similarity...")
    t1 = "Superconducting qubits require 15 millikelvin temperature."
    t2 = "Cryogenic dilution refrigerators cool quantum processors to 15 mK."
    t3 = "Making pizza dough requires flour, yeast, and water."

    v1 = generate_fallback_embedding(t1, 384)
    v2 = generate_fallback_embedding(t2, 384)
    v3 = generate_fallback_embedding(t3, 384)

    assert len(v1) == 384, f"Expected 384 dims, got {len(v1)}"
    sim_related = compute_cosine_similarity(v1, v2)
    sim_unrelated = compute_cosine_similarity(v1, v3)

    print(f"  Similarity (Quantum vs Cryo): {sim_related:.4f}")
    print(f"  Similarity (Quantum vs Pizza): {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, "Related texts should have higher similarity than unrelated texts"
    print("✅ Embedding cosine similarity test passed.")


def test_reranking():
    print("\nTesting Cross-Encoder Lexical Fallback Reranking...")
    query = "What temperature do superconducting transmons operate at?"
    passages = [
        "Pizza dough should ferment in the refrigerator for 24 hours.",
        "Superconducting transmons operate in dilution refrigerators at temperatures of 15 millikelvin.",
        "Zero trust security requires continuous dynamic least privilege."
    ]

    scores = fallback_bm25_rerank(query, passages)
    top_idx = scores.index(max(scores))
    assert top_idx == 1, f"Expected passage 1 (transmons) to rank top, got {top_idx}"
    print(f"  Rerank Scores: {scores}")
    print(f"  Top Ranked Passage: '{passages[top_idx][:60]}...' (Score: {scores[top_idx]:.4f})")
    print("✅ Reranking test passed.")


def test_claim_extraction():
    print("\nTesting Atomic Claim Extraction...")
    answer = """
    Certainly! Here is the summary:
    Superconducting transmon circuits operate at 15 millikelvin [1].
    Modern lipid nanoparticles have an average diameter of 80 nm [2].
    Let me know if you need more details.
    """
    claims = extract_claims_from_text(answer)
    assert len(claims) == 2, f"Expected 2 atomic claims, got {len(claims)}: {claims}"
    print(f"  Extracted {len(claims)} claims: {claims}")
    print("✅ Claim extraction test passed.")


def test_nli_verification():
    print("\nTesting NLI Claim Verification & Hallucination Flagging...")
    premise = "Superconducting transmon circuits operate at dilution refrigerator temperatures of approximately 15 millikelvin."
    
    # 1. Grounded True Claim (Entailment)
    true_claim = "Superconducting transmon circuits operate at 15 millikelvin."
    res_true = fallback_heuristic_nli(premise, true_claim)
    print(f"  True Claim: Entailment={res_true['entailment']:.2f}, Contradiction={res_true['contradiction']:.2f}")
    assert res_true['entailment'] > res_true['contradiction'], "True claim should have high entailment"

    # 2. Contradictory Claim (Hallucination)
    false_claim = "Superconducting circuits operate at 500 Kelvin without cooling."
    res_false = fallback_heuristic_nli(premise, false_claim)
    print(f"  False Claim: Entailment={res_false['entailment']:.2f}, Contradiction={res_false['contradiction']:.2f}")
    assert res_false['contradiction'] > res_false['entailment'], "Contradictory claim should be flagged as contradiction"

    # 3. Unsupported / Out-of-Domain Claim
    neutral_claim = "French chefs prefer high-gluten flour for baguettes."
    res_neutral = fallback_heuristic_nli(premise, neutral_claim)
    print(f"  Neutral Claim: Neutral={res_neutral['neutral']:.2f}")
    assert res_neutral['neutral'] > res_neutral['entailment'], "Unsupported claim should be marked neutral"

    print("✅ NLI Claim Verification test passed.")


def main():
    print("=" * 60)
    print(" 🧪 RUNNING VERIFIED RAG UNIT & PIPELINE TESTS")
    print("=" * 60)
    test_chunking()
    test_embeddings()
    test_reranking()
    test_claim_extraction()
    test_nli_verification()
    print("\n" + "=" * 60)
    print(" 🎉 ALL PIPELINE TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
