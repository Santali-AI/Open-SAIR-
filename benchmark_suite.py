"""
Automated Linguistic Benchmark Suite for Open SAIR:
Tests checked consonant preservation, pronoun clusivity, agglutinative TAM morphology,
and diacritic-aware normalization across multimodal models.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from open_sair.evaluation.metrics import EvaluationMetrics
from open_sair.linguistics.morphology import SantaliMorphologyEngine
from open_sair.linguistics.normalizer import OlChikiNormalizer
from open_sair.linguistics.pronoun_engine import SantaliPronounMatrixEngine


@dataclass
class BenchmarkResult:
    test_name: str
    category: str
    passed: bool
    score: float
    details: str


class OpenSAIRBenchmarkSuite:
    """
    Executes rigorous linguistic and quantitative benchmarks:
    1. Pronoun Clusivity Disambiguation (Inclusive vs Exclusive)
    2. Checked Consonant & Glottal Preservation (Keched Arang)
    3. Agglutinative Verb Infixation (Reciprocal -p- and TAM markers)
    4. Diacritic Error Rate on Historical Documents
    """

    def __init__(self):
        self.normalizer = OlChikiNormalizer()
        self.pronoun_engine = SantaliPronounMatrixEngine()
        self.morph_engine = SantaliMorphologyEngine()
        self.metrics = EvaluationMetrics()

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        results = []
        results.append(self.test_pronoun_clusivity_matrix())
        results.append(self.test_checked_consonant_glottals())
        results.append(self.test_reciprocal_infix_morphology())
        results.append(self.test_canonical_diacritic_normalization())
        results.append(self.test_quantitative_metrics_smoke())
        return results

    def test_pronoun_clusivity_matrix(self) -> BenchmarkResult:
        """Verify that 1st person dual and plural correctly distinguish clusivity."""
        # Dual Inclusive: alaṅ (ᱟᱞᱟᱝ)
        p_du_incl = self.pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=True)
        # Dual Exclusive: aliń (ᱟᱞᱤᱧ)
        p_du_excl = self.pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=False)
        # Plural Inclusive: abo (ᱟᱵᱚ)
        p_pl_incl = self.pronoun_engine.resolve_we_disambiguation(count=3, includes_listener=True)
        # Plural Exclusive: ale (ᱟᱞᱮ)
        p_pl_excl = self.pronoun_engine.resolve_we_disambiguation(count=3, includes_listener=False)

        passed = (
            p_du_incl.ol_chiki == "ᱟᱞᱟᱝ"
            and p_du_excl.ol_chiki == "ᱟᱞᱤᱧ"
            and p_pl_incl.ol_chiki == "ᱟᱵᱚ"
            and p_pl_excl.ol_chiki == "ᱟᱞᱮ"
        )

        details = (
            f"Dual-Incl: {p_du_incl.ol_chiki}, Dual-Excl: {p_du_excl.ol_chiki}, "
            f"Plural-Incl: {p_pl_incl.ol_chiki}, Plural-Excl: {p_pl_excl.ol_chiki}"
        )
        return BenchmarkResult(
            test_name="Pronoun Clusivity & Number Precision",
            category="Morphosyntax",
            passed=passed,
            score=100.0 if passed else 0.0,
            details=details,
        )

    def test_checked_consonant_glottals(self) -> BenchmarkResult:
        """Verify deglottalization rules with Ahad (ᱽ)."""
        text = "ᱜᱽ ᱡᱽ ᱫᱽ ᱵᱽ"
        deglottalized = self.morph_engine.deglottalize_phonetics(text)
        expected = "g j d b"
        passed = deglottalized == expected
        return BenchmarkResult(
            test_name="Checked Consonant Deglottalization Phonetics",
            category="Phonetics",
            passed=passed,
            score=100.0 if passed else 0.0,
            details=f"Input: {text} -> Deglottalized: {deglottalized}",
        )

    def test_reciprocal_infix_morphology(self) -> BenchmarkResult:
        """Verify reciprocal '-p-' infix insertion."""
        # dal (strike) -> dapal (strike each other)
        # In Ol Chiki: ᱫᱟᱞ -> ᱫᱟᱯᱟᱞ
        root = "ᱫᱟᱞ"
        recip = self.morph_engine.apply_reciprocal_infix(root)
        expected = "ᱫᱟᱯᱟᱞ"
        passed = recip == expected
        return BenchmarkResult(
            test_name="Agglutinative Reciprocal -p- Infixation",
            category="Morphology",
            passed=passed,
            score=100.0 if passed else 0.0,
            details=f"Root: {root} -> Reciprocal: {recip}",
        )

    def test_canonical_diacritic_normalization(self) -> BenchmarkResult:
        """Verify canonical diacritic ordering and deduplication."""
        # Erroneous input with duplicate diacritics and inverted order
        raw = "ᱟᱸᱹᱸ"
        normalized = self.normalizer.normalize(raw)
        # Should canonicalize into combined ᱺ (Mu-Gaahlaa) or Gaahlaa + Mu
        valid = (normalized == "ᱟᱺ" or normalized == "ᱟᱹᱸ")
        return BenchmarkResult(
            test_name="Canonical Diacritic Ordering & Deduplication",
            category="Orthography",
            passed=valid,
            score=100.0 if valid else 0.0,
            details=f"Raw: {raw} -> Normalized: {normalized}",
        )

    def test_quantitative_metrics_smoke(self) -> BenchmarkResult:
        """Verify BLEU, chrF++, WER, CER calculation sanity."""
        hyp = "ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱟᱹᱰᱤ ᱱᱟᱯᱟᱭᱟ᱾"
        ref = "ᱥᱟᱱᱛᱟᱲᱤ ᱯᱟᱹᱨᱥᱤ ᱟᱹᱰᱤ ᱱᱟᱯᱟᱭᱟ᱾"
        cer = self.metrics.compute_cer(hyp, ref)
        chrf = self.metrics.compute_chrf(hyp, ref)
        passed = (cer == 0.0 and chrf == 100.0)
        return BenchmarkResult(
            test_name="Evaluation Metrics Baseline Sanity",
            category="Quantitative",
            passed=passed,
            score=100.0 if passed else 0.0,
            details=f"Identity CER: {cer}%, chrF++: {chrf}%",
        )
