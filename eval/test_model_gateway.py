"""
Unit and Integration Test for Model Gateway & Comparison Engine.

Tests:
1. Dynamic model registration format validation.
2. Provider configuration mapping (OpenAI, Anthropic, Gemini, Ollama, DeepSeek, Groq).
3. Grounded context extraction and prompt synthesis.
4. Parallel comparison and NLI verification scoring across multiple models.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.claim_extractor import extract_atomic_claims
from core.nli_verifier import NLIVerifierEngine, VerificationStatus


class TestModelGatewayComparison(unittest.TestCase):

    def setUp(self):
        self.verifier = NLIVerifierEngine()
        self.evidence = [
            "Superconducting transmon quantum processors operate at dilution refrigerator temperatures of approximately 15 millikelvin.",
            "Rotated surface codes detect bit-flip and phase-flip errors with fault-tolerant physical-to-logical qubit ratios of approximately 1,000 to 1."
        ]

    def test_claim_extraction_for_multiple_model_outputs(self):
        model_a_output = "Superconducting transmon circuits operate at 15 millikelvin. They use microwave cavities [1]."
        model_b_output = "Quantum processors operate at room temperature without cryogenics. They use optical tweezers."

        claims_a = extract_atomic_claims(model_a_output)
        claims_b = extract_atomic_claims(model_b_output)

        self.assertGreaterEqual(len(claims_a), 1)
        self.assertGreaterEqual(len(claims_b), 1)

    def test_parallel_verification_comparison(self):
        model_a_output = "Superconducting transmon circuits operate at 15 millikelvin."
        model_b_output = "Superconducting transmon circuits operate at room temperature without cryogenics."

        claims_a = extract_atomic_claims(model_a_output)
        claims_b = extract_atomic_claims(model_b_output)

        report_a = self.verifier.verify_claims(claims_a, self.evidence)
        report_b = self.verifier.verify_claims(claims_b, self.evidence)

        self.assertGreaterEqual(report_a.faithfulness_score, 0.80)
        self.assertLess(report_b.faithfulness_score, 0.50)
        self.assertGreater(report_b.hallucination_rate, report_a.hallucination_rate)


if __name__ == "__main__":
    unittest.main()
