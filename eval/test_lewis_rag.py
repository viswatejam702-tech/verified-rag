"""
Unit and Integration Tests for Lewis et al. (NeurIPS 2020) RAG Architecture.

Tests:
1. MIPS Retriever vector normalization and inner product scoring.
2. Softmax prior probability distribution p_η(z|x) summing to 1.0.
3. RAG-Sequence marginalization over latent passages: p(y|x) = sum_z p_η(z|x) * prod_i p_θ.
4. RAG-Token step-wise marginalization: p(y|x) = prod_i [ sum_z p_η(z|x) * p_θ ].
5. Task Head A: Open-Domain Question Answering (Question Query -> Answer Generation).
6. Task Head B: Fact Verification (FEVER Task -> SUPPORTS / REFUTES / NOT_ENOUGH_INFO).
7. Task Head C: Jeopardy Question Generation (Answer Entity -> Clue Generation).
"""

import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.ingestion import DocumentChunk
from core.lewis_rag import (
    MIPSRetriever,
    LewisRAGGenerator,
    LewisRAGSystem,
    RAGModelVariant,
    VerificationLabel,
    compute_mips_embedding,
    dot_product,
    softmax,
)


class TestLewisRAGArchitecture(unittest.TestCase):

    def setUp(self):
        # Build canonical benchmark corpus corresponding to Lewis et al. Figure 1
        self.chunks = [
            DocumentChunk(
                chunk_id="chunk-ear-01",
                document_id="doc-anatomy-ear",
                document_title="Human Ear Anatomy",
                source_uri="corpus://medical/ear_anatomy.txt",
                content="The middle ear includes the tympanic cavity and the three ossicles (malleus, incus, and stapes) which transmit sound vibrations from the tympanic membrane to the oval window.",
                token_count=35,
                chunk_index=0,
                metadata={"domain": "anatomy"}
            ),
            DocumentChunk(
                chunk_id="chunk-obama-01",
                document_id="doc-us-presidents",
                document_title="Barack Obama Biography",
                source_uri="corpus://biography/barack_obama.txt",
                content="Barack Hussein Obama II was born on August 4, 1961, at Kapiolani Maternity & Gynecological Hospital in Honolulu, Hawaii. He served as the 44th president of the United States.",
                token_count=36,
                chunk_index=0,
                metadata={"domain": "biography"}
            ),
            DocumentChunk(
                chunk_id="chunk-dante-01",
                document_id="doc-world-literature",
                document_title="The Divine Comedy",
                source_uri="corpus://literature/divine_comedy.txt",
                content="The Divine Comedy is an Italian narrative poem by Dante Alighieri. This 14th century work is divided into 3 major sections: Inferno, Purgatorio, and Paradiso, tracing the soul's journey toward God.",
                token_count=38,
                chunk_index=0,
                metadata={"domain": "literature"}
            ),
            DocumentChunk(
                chunk_id="chunk-quantum-01",
                document_id="doc-quantum-physics",
                document_title="Fault Tolerant Quantum Computing",
                source_uri="corpus://physics/quantum_computing.txt",
                content="Superconducting transmon quantum processors operate at 15 millikelvin in dilution refrigerators and require surface codes to correct bit-flip and phase-flip errors.",
                token_count=28,
                chunk_index=0,
                metadata={"domain": "physics"}
            )
        ]

        self.retriever = MIPSRetriever(dim=128, temperature=0.5)
        self.retriever.index_documents(self.chunks)
        self.system = LewisRAGSystem(self.retriever)

    # ── Test 1: MIPS Embedding & Math Primitives ──────────────────────────────
    def test_mips_embedding_unit_norm(self):
        vec = compute_mips_embedding("Define middle ear anatomy", dim=128)
        self.assertEqual(len(vec), 128)
        norm = math.sqrt(sum(x * x for x in vec))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_dot_product_and_softmax(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(dot_product(v1, v2), 1.0)

        scores = [2.0, 1.0, 0.5]
        probs = softmax(scores, temperature=1.0)
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    # ── Test 2: MIPS Retrieval Prior p_η(z|x) ─────────────────────────────────
    def test_mips_retrieval_and_prior_distribution(self):
        query = 'Define "middle ear"'
        latent_docs = self.retriever.retrieve_mips(query, top_k=3)
        self.assertGreaterEqual(len(latent_docs), 1)

        # Top document must be the ear anatomy chunk
        top_doc = latent_docs[0]
        self.assertEqual(top_doc.chunk_id, "chunk-ear-01")
        self.assertGreater(top_doc.mips_score, 0.0)

        # Softmax priors must sum to approximately 1.0 across top candidates
        p_eta_sum = sum(doc.prior_prob for doc in latent_docs)
        self.assertAlmostEqual(p_eta_sum, 1.0, delta=0.02)

    # ── Test 3: RAG-Sequence Marginalization ──────────────────────────────────
    def test_rag_sequence_marginalization(self):
        generator = LewisRAGGenerator(self.retriever)
        query = "Define middle ear"
        candidates = [
            "The middle ear includes the tympanic cavity and the three ossicles.",
            "Superconducting transmon quantum processors operate at 15 millikelvin."
        ]

        best_y, marg_prob, latent_docs = generator.marginalize_rag_sequence(query, candidates, top_k=4)
        self.assertEqual(best_y, candidates[0])
        self.assertGreater(marg_prob, 0.0)
        self.assertEqual(latent_docs[0].chunk_id, "chunk-ear-01")

    # ── Test 4: RAG-Token Marginalization ─────────────────────────────────────
    def test_rag_token_marginalization(self):
        generator = LewisRAGGenerator(self.retriever)
        query = "Define middle ear"
        tokens = ["The", "middle", "ear", "includes", "the", "tympanic", "cavity"]

        seq, marg_prob, attributions, latent_docs = generator.marginalize_rag_token(query, tokens, top_k=3)
        self.assertIn("middle", seq)
        self.assertGreater(marg_prob, 0.0)
        self.assertEqual(len(attributions), len(tokens))

        # Check attribution for "tympanic" points predominantly to chunk-ear-01
        tympanic_attr = [a for a in attributions if a["token"] == "tympanic"][0]
        active_weights = tympanic_attr["active_document_weights"]
        self.assertIn("chunk-ear-01", active_weights)
        self.assertGreater(active_weights["chunk-ear-01"], 0.0)

    # ── Test 5: Head A — Question Answering ────────────────────────────────────
    def test_task_head_qa(self):
        # Exact prompt from Figure 1: Define "middle ear" (x)
        output = self.system.answer_question('Define "middle ear"', model_variant=RAGModelVariant.RAG_SEQUENCE)
        self.assertEqual(output.task_type, "question_answering")
        self.assertIn("tympanic", output.generated_y.lower())
        self.assertIn("ossicles", output.generated_y.lower())
        self.assertGreater(output.marginal_probability, 0.0)
        self.assertEqual(output.latent_documents[0].chunk_id, "chunk-ear-01")

    # ── Test 6: Head B — Fact Verification (FEVER) ────────────────────────────
    def test_task_head_fact_verification_supports(self):
        # Exact prompt from Figure 1: Barack Obama was born in Hawaii. (x)
        output = self.system.verify_fact("Barack Obama was born in Hawaii.")
        self.assertEqual(output.task_type, "fact_verification")
        self.assertEqual(output.verification_label, VerificationLabel.SUPPORTS)
        self.assertIn("supports", output.generated_y.lower())
        self.assertGreaterEqual(output.marginal_probability, 0.50)

    def test_task_head_fact_verification_refutes(self):
        # Negated claim
        output = self.system.verify_fact("Barack Obama was not born in Hawaii.")
        self.assertEqual(output.task_type, "fact_verification")
        self.assertEqual(output.verification_label, VerificationLabel.REFUTES)
        self.assertIn("refutes", output.generated_y.lower())

    def test_task_head_fact_verification_not_enough_info(self):
        # Unindexed entity claim
        output = self.system.verify_fact("Alexander the Great founded 70 cities across Asia.")
        self.assertEqual(output.task_type, "fact_verification")
        self.assertEqual(output.verification_label, VerificationLabel.NOT_ENOUGH_INFO)
        self.assertIn("not_enough_info", output.generated_y.lower())

    # ── Test 7: Head C — Jeopardy Question Generation ─────────────────────────
    def test_task_head_jeopardy_generation(self):
        # Exact prompt from Figure 1: The Divine Comedy (x)
        output = self.system.generate_jeopardy_clue("The Divine Comedy")
        self.assertEqual(output.task_type, "jeopardy_generation")
        # Output clue must mention the sections or Dante without answering the title itself directly
        self.assertTrue(
            "14th century" in output.generated_y or "inferno" in output.generated_y.lower()
        )
        self.assertGreater(output.marginal_probability, 0.0)
        self.assertEqual(output.latent_documents[0].chunk_id, "chunk-dante-01")


if __name__ == "__main__":
    unittest.main()
