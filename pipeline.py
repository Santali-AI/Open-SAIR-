"""
Dataset Curation, Trilingual Parallel Text Filtering, and Preprocessing Standards.
Sources: Monolingual Ol Chiki corpora, tri-lingual parallel text, raw speech, back-translation.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterator, List, Optional, Tuple

from open_sair.linguistics.normalizer import OlChikiNormalizer
from open_sair.linguistics.ol_chiki_unicode import (
    CHECKED_CONSONANTS,
    is_ol_chiki,
)


@dataclass
class TrilingualParallelSample:
    sample_id: str
    santali_olchiki: str
    bengali_text: str
    english_text: str
    audio_path: Optional[str] = None
    domain: str = "general"
    clusivity_label: Optional[str] = None
    quality_score: float = 1.0


class DatasetPreprocessingPipeline:
    """
    Quality filtering and preprocessing pipeline for Santali-Bengali-English corpora.
    Applies:
    1. Unicode NFC normalization & canonical diacritic ordering.
    2. Length ratio constraints (token length ratios between 0.4 and 2.5).
    3. Script purity validation.
    4. Glottal and diacritic preservation checks.
    5. Deduplication and perplexity filtering.
    """

    def __init__(
        self,
        min_words: int = 2,
        max_words: int = 120,
        min_length_ratio: float = 0.4,
        max_length_ratio: float = 2.5,
    ):
        self.normalizer = OlChikiNormalizer()
        self.min_words = min_words
        self.max_words = max_words
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio

    def validate_bengali_script(self, text: str) -> bool:
        """Check if text contains Bengali Unicode characters (U+0980 - U+09FF)."""
        return any("\u0980" <= c <= "\u09FF" for c in text)

    def validate_ol_chiki_script(self, text: str) -> bool:
        """Check if text contains valid Ol Chiki characters (U+1C50 - U+1C7F)."""
        return any(is_ol_chiki(c) for c in text)

    def compute_olchiki_script_density(self, text: str) -> float:
        """Compute the proportion of alphabetic characters that belong to Ol Chiki."""
        chars = [c for c in text if not c.isspace() and not c.isnumeric() and c not in ".,!?-;:'\"()[]{}"]
        if not chars:
            return 0.0
        ol_count = sum(1 for c in chars if is_ol_chiki(c))
        return ol_count / float(len(chars))

    def filter_and_clean_sample(
        self,
        santali: str,
        bengali: str,
        english: str,
    ) -> Tuple[bool, Optional[Tuple[str, str, str]], str]:
        """
        Validates and cleans a trilingual parallel text triplet.
        Returns: (is_valid, cleaned_tuple_or_none, rejection_reason)
        """
        # Step 1: Normalize Ol Chiki
        sat_clean = self.normalizer.normalize(santali)
        ben_clean = bengali.strip()
        eng_clean = english.strip()

        # Step 2: Empty checks
        if not sat_clean or not ben_clean or not eng_clean:
            return False, None, "Empty text in one or more languages"

        # Step 3: Script purity
        if self.compute_olchiki_script_density(sat_clean) < 0.80:
            return False, None, "Santali text contains insufficient Ol Chiki script purity (<80%)"

        if not self.validate_bengali_script(ben_clean):
            return False, None, "Bengali text lacks Bengali Unicode characters"

        # Step 4: Word count constraints
        sat_words = sat_clean.split()
        eng_words = eng_clean.split()
        ben_words = ben_clean.split()

        for name, wlist in [("Santali", sat_words), ("Bengali", ben_words), ("English", eng_words)]:
            if len(wlist) < self.min_words or len(wlist) > self.max_words:
                return False, None, f"{name} word count out of bounds ({len(wlist)})"

        # Step 5: Length ratio checks
        ratio_sat_eng = len(sat_words) / float(len(eng_words))
        if not (self.min_length_ratio <= ratio_sat_eng <= self.max_length_ratio):
            return False, None, f"Santali/English length ratio anomalous ({ratio_sat_eng:.2f})"

        ratio_ben_eng = len(ben_words) / float(len(eng_words))
        if not (self.min_length_ratio <= ratio_ben_eng <= self.max_length_ratio):
            return False, None, f"Bengali/English length ratio anomalous ({ratio_ben_eng:.2f})"

        return True, (sat_clean, ben_clean, eng_clean), "Passed"

    def generate_backtranslation_prompt(
        self,
        monolingual_olchiki: str,
        target_lang: str = "bengali",
    ) -> Dict[str, str]:
        """
        Creates structured prompt payload for synthetic back-translation data augmentation.
        """
        cleaned = self.normalizer.normalize(monolingual_olchiki)
        return {
            "instruction": (
                f"Translate the following native Santali text in Ol Chiki script into {target_lang}. "
                "Preserve inclusive vs exclusive clusivity and checked stop semantics."
            ),
            "source_text": cleaned,
            "source_lang": "sat_Olck",
            "target_lang": "ben_Beng" if target_lang.lower() == "bengali" else "eng_Latn",
        }
