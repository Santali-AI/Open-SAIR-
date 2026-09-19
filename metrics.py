"""
Quantitative Evaluation Metrics Framework for Open SAIR:
BLEU, chrF++, Word Error Rate (WER), Character Error Rate (CER),
Glottal Error Rate (GER), Diacritic Error Rate (DER), and F0 RMSE.
"""

from collections import Counter
import math
from typing import Dict, List, Set, Tuple

from open_sair.linguistics.ol_chiki_unicode import (
    CHECKED_CONSONANTS,
    MODIFIERS,
    is_ol_chiki,
)


class EvaluationMetrics:
    """
    Mathematical implementations of multimodal evaluation metrics:
    - chrF++ / BLEU for NMT
    - Levenshtein-based WER & CER for ASR & OCR
    - Specialized Glottal Error Rate (GER)
    - Specialized Diacritic Error Rate (DER)
    """

    @staticmethod
    def compute_bleu(
        hypothesis: List[str],
        reference: List[str],
        max_n: int = 4,
    ) -> float:
        """
        Computes sentence-level BLEU score with modified n-gram precision and brevity penalty.
        BP = exp(1 - r / c) if c <= r else 1.0
        """
        c = len(hypothesis)
        r = len(reference)
        if c == 0:
            return 0.0

        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = [tuple(hypothesis[i : i + n]) for i in range(len(hypothesis) - n + 1)]
            ref_ngrams = [tuple(reference[i : i + n]) for i in range(len(reference) - n + 1)]

            if not hyp_ngrams:
                precisions.append(1e-7)
                continue

            hyp_counts = Counter(hyp_ngrams)
            ref_counts = Counter(ref_ngrams)

            clipped = sum(min(count, ref_counts[ng]) for ng, count in hyp_counts.items())
            prec = clipped / float(len(hyp_ngrams))
            precisions.append(max(prec, 1e-7))

        log_avg = sum(math.log(p) for p in precisions) / float(max_n)
        bp = math.exp(1.0 - float(r) / float(c)) if c <= r else 1.0
        return bp * math.exp(log_avg) * 100.0

    @staticmethod
    def compute_chrf(
        hypothesis: str,
        reference: str,
        n: int = 6,
        beta: float = 2.0,
    ) -> float:
        """
        Computes chrF++ (character n-gram F-score) particularly critical for
        morphologically rich and agglutinative languages like Santali.
        """
        hyp_chars = list(hypothesis.replace(" ", ""))
        ref_chars = list(reference.replace(" ", ""))

        if not hyp_chars or not ref_chars:
            return 0.0

        n_scores = []
        for order in range(1, n + 1):
            hyp_ngrams = [tuple(hyp_chars[i : i + order]) for i in range(len(hyp_chars) - order + 1)]
            ref_ngrams = [tuple(ref_chars[i : i + order]) for i in range(len(ref_chars) - order + 1)]

            if not hyp_ngrams or not ref_ngrams:
                continue

            hyp_c = Counter(hyp_ngrams)
            ref_c = Counter(ref_ngrams)

            overlap = sum(min(count, ref_c[ng]) for ng, count in hyp_c.items())
            p = overlap / float(len(hyp_ngrams)) if hyp_ngrams else 0.0
            r = overlap / float(len(ref_ngrams)) if ref_ngrams else 0.0

            if p + r > 0:
                f = (1.0 + beta ** 2) * (p * r) / ((beta ** 2 * p) + r)
                n_scores.append(f)

        return (sum(n_scores) / float(len(n_scores)) * 100.0) if n_scores else 0.0

    @staticmethod
    def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
        """Standard Levenshtein dynamic programming edit distance."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1],  # Substitution
                    )

        return dp[m][n]

    @classmethod
    def compute_wer(cls, hypothesis: str, reference: str) -> float:
        """Word Error Rate (WER) = Distance / len(reference_words)."""
        hyp_words = hypothesis.split()
        ref_words = reference.split()
        if not ref_words:
            return 0.0 if not hyp_words else 100.0
        dist = cls.levenshtein_distance(hyp_words, ref_words)
        return (dist / float(len(ref_words))) * 100.0

    @classmethod
    def compute_cer(cls, hypothesis: str, reference: str) -> float:
        """Character Error Rate (CER) = Distance / len(reference_chars)."""
        hyp_chars = list(hypothesis)
        ref_chars = list(reference)
        if not ref_chars:
            return 0.0 if not hyp_chars else 100.0
        dist = cls.levenshtein_distance(hyp_chars, ref_chars)
        return (dist / float(len(ref_chars))) * 100.0

    @classmethod
    def compute_diacritic_error_rate(cls, hypothesis: str, reference: str) -> float:
        """
        Diacritic Error Rate (DER): Specifically tracks error rate on Ol Chiki modifiers:
        Mu Ttuddag (ᱸ), Gaahlaa Ttuddag (ᱹ), Mu-Gaahlaa (ᱺ), Relaa (ᱻ), Phaarkaa (ᱼ), Ahad (ᱽ).
        """
        hyp_diacritics = [c for c in hypothesis if c in MODIFIERS]
        ref_diacritics = [c for c in reference if c in MODIFIERS]
        if not ref_diacritics:
            return 0.0 if not hyp_diacritics else 100.0
        dist = cls.levenshtein_distance(hyp_diacritics, ref_diacritics)
        return (dist / float(len(ref_diacritics))) * 100.0

    @classmethod
    def compute_glottal_error_rate(cls, hypothesis: str, reference: str) -> float:
        """
        Glottal Error Rate (GER): Measures accuracy of checked consonants (ᱜ, ᱡ, ᱫ, ᱵ).
        Crucial for linguistic validity of Santali ASR and OCR.
        """
        hyp_glottals = [c for c in hypothesis if c in CHECKED_CONSONANTS]
        ref_glottals = [c for c in reference if c in CHECKED_CONSONANTS]
        if not ref_glottals:
            return 0.0 if not hyp_glottals else 100.0
        dist = cls.levenshtein_distance(hyp_glottals, ref_glottals)
        return (dist / float(len(ref_glottals))) * 100.0
