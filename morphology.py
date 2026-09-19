"""
Santali Agglutinative Morphology, TAM Infix/Suffix Parser, and Glottal Mutation Engine.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from open_sair.linguistics.ol_chiki_unicode import (
    CHECKED_CONSONANTS,
    CONSONANT_AG,
    CONSONANT_AAJ,
    CONSONANT_UD,
    CONSONANT_OB,
    MODIFIER_AHAD,
    VOWELS,
)


@dataclass
class MorphemeAnalysis:
    raw_word: str
    root: str
    reciprocal: bool
    tam_markers: List[str]
    object_marker: Optional[str]
    finite_marker: bool
    subject_marker: Optional[str]
    gloss: str


class SantaliMorphologyEngine:
    """
    Morphological parser and glottal assimilation handler for Santali in Ol Chiki.
    Handles:
    - Reciprocal '-p-' infixation
    - Tense-Aspect-Mood (TAM) infixes / suffixes
    - Checked consonant phonology (glottal vs deglottalized via Ahad)
    - Pronominal enclitics and transitive object agreement
    """

    # TAM markers in Ol Chiki
    TAM_PATTERNS = {
        "ᱠᱟᱱ": ("kan", "PRES.CONT"),
        "ᱠᱮᱫ": ("ked", "PAST.TRANS"),
        "ᱠᱮᱛ": ("ket", "PAST.TRANS.CHECKED"),
        "ᱮᱱ": ("en", "PAST.INTRANS"),
        "ᱟᱠᱟᱫ": ("akad", "PERF.TRANS"),
        "ᱟᱠᱟᱱ": ("akan", "PERF.INTRANS"),
        "ᱮᱫ": ("ed", "PRES.TRANS"),
        "ᱞᱮᱫ": ("led", "PLUPERF.TRANS"),
        "ᱞᱮᱱ": ("len", "PLUPERF.INTRANS"),
    }

    # Subject/Object pronominal clitics
    CLITIC_PATTERNS = {
        "ᱧ": ("-ń", "1SG"),
        "ᱢ": ("-m", "2SG.SUBJ"),
        "ᱢᱮ": ("-me", "2SG.OBJ"),
        "ᱮ": ("-e", "3SG.ANIM"),
        "ᱭ": ("-y", "3SG.ANIM.GLIDE"),
        "ᱞᱟᱝ": ("-laṅ", "1DU.INCL"),
        "ᱞᱤᱧ": ("-liń", "1DU.EXCL"),
        "ᱵᱮᱱ": ("-ben", "2DU"),
        "ᱠᱤᱱ": ("-kin", "3DU"),
        "ᱵᱚ": ("-bo", "1PL.INCL"),
        "ᱵᱚᱱ": ("-bon", "1PL.INCL"),
        "ᱞᱮ": ("-le", "1PL.EXCL"),
        "ᱯᱮ": ("-pe", "2PL"),
        "ᱠᱚ": ("-ko", "3PL"),
    }

    # Deglottalization pairs: checked consonant followed by Ahad (ᱽ)
    DEGLOTTALIZATION_RULES = {
        CONSONANT_AG + MODIFIER_AHAD: ("ᱜᱽ", "g", "Voiced Velar Plosive"),
        CONSONANT_AAJ + MODIFIER_AHAD: ("ᱡᱽ", "j", "Voiced Palatal Plosive"),
        CONSONANT_UD + MODIFIER_AHAD: ("ᱫᱽ", "d", "Voiced Dental Plosive"),
        CONSONANT_OB + MODIFIER_AHAD: ("ᱵᱽ", "b", "Voiced Bilabial Plosive"),
    }

    # Intervocalic unreleased-to-voiced consonant softening (Sandhi)
    INTERVOCALIC_SOFTENING = {
        "ᱛ": "ᱫ",  # /t/ -> /d/ intervocalic
        "ᱠ": "ᱜ",  # /k/ -> /g/ intervocalic
        "ᱯ": "ᱵ",  # /p/ -> /b/ intervocalic
        "ᱪ": "ᱡ",  # /c/ -> /j/ intervocalic
    }

    def apply_reciprocal_infix(self, root: str) -> str:
        """
        Applies reciprocal '-p-' (ᱯ) infixation after the first vowel of a verb root.
        Example:
            dal (ᱫᱟᱞ) -> d-ap-al (ᱫᱟᱯᱟᱞ) [hit -> fight each other]
            ñel (ᱧᱮᱞ) -> ñ-ep-el (ᱧᱮᱯᱮᱞ) [see -> meet each other]
            goc' (ᱜᱚᱪ) -> g-op-oc' (ᱜᱚᱯᱚᱪ) [kill -> slay each other]
        """
        chars = list(root)
        first_vowel_idx = -1
        first_vowel = ""

        for idx, ch in enumerate(chars):
            if ch in VOWELS:
                first_vowel_idx = idx
                first_vowel = ch
                break

        if first_vowel_idx == -1:
            return root  # No vowel found, return unaltered

        # Insert ᱯ followed by harmony echo of the first vowel
        infix = "ᱯ" + first_vowel
        return "".join(chars[: first_vowel_idx + 1]) + infix + "".join(chars[first_vowel_idx + 1 :])

    def detect_reciprocal(self, word: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if a word contains a reciprocal '-p-' infix and reconstructs candidate root.
        """
        # Look for [Consonant][Vowel1] ᱯ [Vowel1] pattern
        for i in range(len(word) - 2):
            if word[i] in VOWELS and word[i + 1] == "ᱯ" and i + 2 < len(word) and word[i + 2] == word[i]:
                # Found reciprocal pattern
                reconstructed = word[: i + 1] + word[i + 3 :]
                return True, reconstructed
        return False, None

    def deglottalize_phonetics(self, text: str) -> str:
        """
        Transcribes checked consonants and deglottalized pairs into IPA / phonetic notation.
        """
        result = text
        for cluster, (ol, latin, desc) in self.DEGLOTTALIZATION_RULES.items():
            result = result.replace(cluster, latin)
        return result

    def analyze_verb(self, verb_form: str) -> MorphemeAnalysis:
        """
        Heuristic morphological analyzer for complex agglutinative Santali verbs.
        Example: dal-ed-e-kan-a-e (ᱫᱟᱞᱮᱫᱮᱠᱟᱱᱟᱭ)
        """
        word = verb_form
        is_recip, unrecip_root = self.detect_reciprocal(word)

        # Check finite marker 'a' (ᱟ)
        has_finite = False
        subject = None
        if "ᱟ" in word:
            parts = word.split("ᱟ")
            stem_part = parts[0]
            tail_part = parts[1] if len(parts) > 1 else ""
            has_finite = True

            # Subject clitic attaches after finite marker
            for clitic, (lat, gloss_code) in self.CLITIC_PATTERNS.items():
                if tail_part.startswith(clitic) or tail_part == clitic:
                    subject = gloss_code
                    break
        else:
            stem_part = word

        # Detect TAM in stem
        detected_tam = []
        for tam_ol, (tam_lat, tam_gloss) in self.TAM_PATTERNS.items():
            if tam_ol in stem_part:
                detected_tam.append(tam_gloss)

        # Build gloss string
        gloss_parts = [unrecip_root if is_recip else stem_part[:3]]
        if is_recip:
            gloss_parts.append("RECIP")
        gloss_parts.extend(detected_tam)
        if has_finite:
            gloss_parts.append("FIN")
        if subject:
            gloss_parts.append(f"SUBJ.{subject}")

        return MorphemeAnalysis(
            raw_word=verb_form,
            root=unrecip_root if is_recip else stem_part[:3],
            reciprocal=is_recip,
            tam_markers=detected_tam,
            object_marker=None,
            finite_marker=has_finite,
            subject_marker=subject,
            gloss="-".join(gloss_parts),
        )
