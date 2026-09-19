"""
Unit Tests for Open SAIR Linguistics, Orthography, and Morphology.
"""

import unittest

from open_sair.evaluation.benchmark_suite import OpenSAIRBenchmarkSuite
from open_sair.evaluation.metrics import EvaluationMetrics
from open_sair.linguistics.grammar_transformer import CrossLingualGrammarTransformer
from open_sair.linguistics.morphology import SantaliMorphologyEngine
from open_sair.linguistics.normalizer import OlChikiNormalizer
from open_sair.linguistics.ol_chiki_unicode import (
    CHECKED_CONSONANTS,
    VOWELS,
    contains_ol_chiki,
    is_ol_chiki,
)
from open_sair.linguistics.pronoun_engine import (
    Clusivity,
    GrammaticalNumber,
    GrammaticalPerson,
    SantaliPronounMatrixEngine,
)


class TestOlChikiLinguistics(unittest.TestCase):
    def setUp(self):
        self.normalizer = OlChikiNormalizer()
        self.pronoun_engine = SantaliPronounMatrixEngine()
        self.morph_engine = SantaliMorphologyEngine()
        self.grammar = CrossLingualGrammarTransformer()
        self.metrics = EvaluationMetrics()

    def test_unicode_checks(self):
        self.assertTrue(is_ol_chiki("ᱚ"))
        self.assertTrue(is_ol_chiki("᱿"))
        self.assertFalse(is_ol_chiki("A"))
        self.assertFalse(is_ol_chiki("ক"))
        self.assertTrue(contains_ol_chiki("Hello ᱥᱟᱱᱛᱟᱲᱤ"))
        self.assertFalse(contains_ol_chiki("Hello World"))

    def test_normalizer(self):
        # Test punctuation conversion
        raw = "ᱱᱚᱣᱟ ᱫᱚ ᱯᱩᱛᱷᱤ ᱠᱟᱱᱟ."
        normalized = self.normalizer.normalize(raw)
        self.assertTrue(normalized.endswith("᱾"))

        # Test deduplication
        raw_dup = "ᱟᱸᱸ"
        normalized_dup = self.normalizer.normalize(raw_dup)
        self.assertEqual(normalized_dup, "ᱟᱸ")

    def test_pronoun_matrix(self):
        # 1st Dual Inclusive: alaṅ (ᱟᱞᱟᱝ)
        p_incl = self.pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=True)
        self.assertEqual(p_incl.ol_chiki, "ᱟᱞᱟᱝ")
        self.assertEqual(p_incl.clusivity, Clusivity.INCLUSIVE)
        self.assertEqual(p_incl.number, GrammaticalNumber.DUAL)

        # 1st Dual Exclusive: aliń (ᱟᱞᱤᱧ)
        p_excl = self.pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=False)
        self.assertEqual(p_excl.ol_chiki, "ᱟᱞᱤᱧ")
        self.assertEqual(p_excl.clusivity, Clusivity.EXCLUSIVE)
        self.assertEqual(p_excl.number, GrammaticalNumber.DUAL)

        # 1st Plural Inclusive: abo (ᱟᱵᱚ)
        p_pl_incl = self.pronoun_engine.resolve_we_disambiguation(count=5, includes_listener=True)
        self.assertEqual(p_pl_incl.ol_chiki, "ᱟᱵᱚ")
        self.assertEqual(p_pl_incl.clusivity, Clusivity.INCLUSIVE)

        # 1st Plural Exclusive: ale (ᱟᱞᱮ)
        p_pl_excl = self.pronoun_engine.resolve_we_disambiguation(count=5, includes_listener=False)
        self.assertEqual(p_pl_excl.ol_chiki, "ᱟᱞᱮ")
        self.assertEqual(p_pl_excl.clusivity, Clusivity.EXCLUSIVE)

    def test_reciprocal_infix(self):
        # dal -> dapal
        root = "ᱫᱟᱞ"
        recip = self.morph_engine.apply_reciprocal_infix(root)
        self.assertEqual(recip, "ᱫᱟᱯᱟᱞ")

        is_recip, unrecip = self.morph_engine.detect_reciprocal("ᱫᱟᱯᱟᱞ")
        self.assertTrue(is_recip)
        self.assertEqual(unrecip, "ᱫᱟᱞ")

    def test_word_order_transformation(self):
        # English: [We, read, books] (SVO: s=0, v=1, o=2)
        # SOV: [We, books, read]
        tokens = ["We", "read", "books"]
        sov = self.grammar.svo_to_sov(tokens, s_idx=0, v_idx=1, o_idx=2)
        self.assertEqual(sov, ["We", "books", "read"])

    def test_metrics(self):
        hyp = "ᱥᱟᱱᱛᱟᱲᱤ"
        ref = "ᱥᱟᱱᱛᱟᱲᱤ"
        self.assertEqual(self.metrics.compute_cer(hyp, ref), 0.0)
        self.assertEqual(self.metrics.compute_wer(hyp, ref), 0.0)
        self.assertAlmostEqual(self.metrics.compute_chrf(hyp, ref), 100.0, places=1)

    def test_benchmark_suite(self):
        suite = OpenSAIRBenchmarkSuite()
        results = suite.run_all_benchmarks()
        for r in results:
            self.assertTrue(r.passed, f"Benchmark {r.test_name} failed: {r.details}")


if __name__ == "__main__":
    unittest.main()
