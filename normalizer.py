"""
Unicode NFC Normalization, Canonical Diacritic Ordering, and Script Cleaning for Ol Chiki.
"""

import re
import unicodedata
from typing import Optional, Tuple

from open_sair.linguistics.ol_chiki_unicode import (
    CHECKED_CONSONANTS,
    CONSONANT_OH,
    MODIFIER_AHAD,
    MODIFIER_GAAHLAA_TTUDDAAG,
    MODIFIER_MU_GAAHLAA,
    MODIFIER_MU_TTUDDAG,
    MODIFIER_PHAARKAA,
    MODIFIER_RELAA,
    MODIFIERS,
    PUNCTUATION_DOUBLE_MUCAAD,
    PUNCTUATION_MUCAAD,
    is_ol_chiki,
)


class OlChikiNormalizer:
    """
    Normalizes Ol Chiki text to strict canonical Unicode standards:
    1. Unicode NFC normalization.
    2. Strips extraneous Zero-Width Joiners (ZWJ/ZWNJ) which are inapplicable in Ol Chiki.
    3. Normalizes punctuation to native Mucaad (᱾) and Double Mucaad (᱿).
    4. Canonical diacritic reordering: Base -> Gaahlaa (ᱹ) -> Mu (ᱸ) -> Ahad/Relaa/Phaarkaa.
    5. Deduplication of erroneous duplicate diacritics.
    6. Validation of checked consonant modifier attachments.
    """

    def __init__(self, convert_punctuation: bool = True, canonicalize_combined_diacritic: bool = True):
        self.convert_punctuation = convert_punctuation
        self.canonicalize_combined_diacritic = canonicalize_combined_diacritic

        # Regex patterns
        self._zw_pattern = re.compile(r"[\u200B-\u200D\uFEFF]")
        self._multi_space = re.compile(r"[ \t\u00A0]+")
        self._duplicate_mu = re.compile(f"{MODIFIER_MU_TTUDDAG}+")
        self._duplicate_gaahlaa = re.compile(f"{MODIFIER_GAAHLAA_TTUDDAAG}+")
        self._duplicate_ahad = re.compile(f"{MODIFIER_AHAD}+")

    def normalize(self, text: str) -> str:
        """Execute full normalization pipeline on input text."""
        if not text:
            return ""

        # Step 1: Unicode NFC Normalization
        text = unicodedata.normalize("NFC", text)

        # Step 2: Strip zero-width characters (ZWJ, ZWNJ, BOM)
        text = self._zw_pattern.sub("", text)

        # Step 3: Normalize whitespace
        text = self._multi_space.sub(" ", text)

        # Step 4: Canonicalize diacritic ordering and deduplicate
        text = self._normalize_diacritics(text)

        # Step 5: Convert punctuation if enabled
        if self.convert_punctuation:
            text = self._normalize_punctuation(text)

        return text.strip()

    def _normalize_diacritics(self, text: str) -> str:
        """
        Orders diacritics in canonical sequence:
        Base character -> Gaahlaa Ttuddag (ᱹ) -> Mu Ttuddag (ᱸ) -> Ahad (ᱽ) / Relaa (ᱻ) / Phaarkaa (ᱼ).
        """
        # Deduplicate consecutive identical diacritics
        text = self._duplicate_mu.sub(MODIFIER_MU_TTUDDAG, text)
        text = self._duplicate_gaahlaa.sub(MODIFIER_GAAHLAA_TTUDDAAG, text)
        text = self._duplicate_ahad.sub(MODIFIER_AHAD, text)

        # Inverted diacritic ordering: Mu (ᱸ) followed by Gaahlaa (ᱹ) -> reorder to Gaahlaa + Mu
        inverted_pattern = f"{MODIFIER_MU_TTUDDAG}{MODIFIER_GAAHLAA_TTUDDAAG}"
        if self.canonicalize_combined_diacritic:
            # Replace separate Gaahlaa + Mu or Mu + Gaahlaa with combined Mu-Gaahlaa Ttuddag (ᱺ)
            # or standard canonical pair
            text = text.replace(inverted_pattern, MODIFIER_MU_GAAHLAA)
            standard_pair = f"{MODIFIER_GAAHLAA_TTUDDAAG}{MODIFIER_MU_TTUDDAG}"
            text = text.replace(standard_pair, MODIFIER_MU_GAAHLAA)
            # Strip redundant modifiers adjacent to combined Mu-Gaahlaa
            text = text.replace(f"{MODIFIER_MU_GAAHLAA}{MODIFIER_MU_TTUDDAG}", MODIFIER_MU_GAAHLAA)
            text = text.replace(f"{MODIFIER_MU_GAAHLAA}{MODIFIER_GAAHLAA_TTUDDAAG}", MODIFIER_MU_GAAHLAA)
            text = text.replace(f"{MODIFIER_MU_TTUDDAG}{MODIFIER_MU_GAAHLAA}", MODIFIER_MU_GAAHLAA)
            text = text.replace(f"{MODIFIER_GAAHLAA_TTUDDAAG}{MODIFIER_MU_GAAHLAA}", MODIFIER_MU_GAAHLAA)
        else:
            canonical_pair = f"{MODIFIER_GAAHLAA_TTUDDAAG}{MODIFIER_MU_TTUDDAG}"
            text = text.replace(inverted_pattern, canonical_pair)

        return text

    def _normalize_punctuation(self, text: str) -> str:
        """Map Latin and Bengali punctuation to native Ol Chiki danda characters."""
        # Double punctuation
        text = text.replace("||", PUNCTUATION_DOUBLE_MUCAAD)
        text = text.replace("।।", PUNCTUATION_DOUBLE_MUCAAD)
        text = text.replace("..", PUNCTUATION_DOUBLE_MUCAAD)

        # Single punctuation (in Ol Chiki context)
        # Avoid replacing decimal points in digits: only replace if preceded by Ol Chiki
        result = []
        for i, ch in enumerate(text):
            if ch in ("|", "।"):
                result.append(PUNCTUATION_MUCAAD)
            elif ch == ".":
                # If preceded by an Ol Chiki character and not surrounded by digits
                prev_char = text[i - 1] if i > 0 else ""
                next_char = text[i + 1] if i + 1 < len(text) else ""
                if is_ol_chiki(prev_char) and not (prev_char.isdigit() and next_char.isdigit()):
                    result.append(PUNCTUATION_MUCAAD)
                else:
                    result.append(ch)
            else:
                result.append(ch)

        return "".join(result)

    def validate_orthography(self, text: str) -> Tuple[bool, list]:
        """
        Validates orthographic constraints:
        - Ahad (ᱽ) must only follow Checked Consonants (ᱜ, ᱡ, ᱫ, ᱵ) or Aspiration (ᱷ).
        - Diacritics must attach to valid base characters.
        """
        errors = []
        chars = list(text)
        for i, ch in enumerate(chars):
            if ch == MODIFIER_AHAD:
                if i == 0:
                    errors.append(f"Ahad ({MODIFIER_AHAD}) cannot appear at start of word.")
                else:
                    prev = chars[i - 1]
                    allowed_bases = CHECKED_CONSONANTS | {CONSONANT_OH}
                    if prev not in allowed_bases:
                        errors.append(
                            f"Illegal Ahad attachment: '{prev}' ({ord(prev):04X}) + '{ch}'. "
                            f"Ahad only follows checked consonants: {sorted(list(CHECKED_CONSONANTS))}."
                        )

            elif ch in (MODIFIER_MU_TTUDDAG, MODIFIER_GAAHLAA_TTUDDAAG, MODIFIER_MU_GAAHLAA):
                if i == 0:
                    errors.append(f"Diacritic ({ch}) cannot appear without base character.")

        return len(errors) == 0, errors
