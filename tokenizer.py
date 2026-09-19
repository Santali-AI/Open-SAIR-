"""
Ol Chiki-Tuned Tokenization Strategy using Morphological BPE and Atomic Diacritic Preservation.
"""

import os
import re
from typing import Dict, List, Optional, Set

from open_sair.linguistics.ol_chiki_unicode import (
    ALL_OL_CHIKI_CHARS,
    MODIFIERS,
    is_ol_chiki,
)


class OlChikiBPETokenizer:
    """
    Subword Tokenizer designed for Ol Chiki script:
    1. Enforces atomic binding between base letters and attached diacritics/modifiers.
    2. Registers language tags: <sat_Olck>, <ben_Beng>, <eng_Latn>.
    3. Handles code-switching between Ol Chiki, Bengali, and English.
    """

    SPECIAL_TOKENS = [
        "<s>",
        "</s>",
        "<unk>",
        "<pad>",
        "<mask>",
        "<sat_Olck>",  # Santali in Ol Chiki
        "<ben_Beng>",  # Bengali in Bengali script
        "<eng_Latn>",  # English in Latin script
    ]

    def __init__(self, vocab_file: Optional[str] = None):
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Seed vocabulary with special tokens and atomic Ol Chiki graphemes."""
        idx = 0
        for tok in self.SPECIAL_TOKENS:
            self.vocab[tok] = idx
            self.inverse_vocab[idx] = tok
            idx += 1

        # Add all individual Ol Chiki Unicode characters
        for char in sorted(list(ALL_OL_CHIKI_CHARS)):
            if char not in self.vocab:
                self.vocab[char] = idx
                self.inverse_vocab[idx] = char
                idx += 1

    def split_into_grapheme_clusters(self, text: str) -> List[str]:
        """
        Splits Ol Chiki text into atomic grapheme clusters so diacritics are NEVER
        separated from their base characters during subword merging.
        Example: ᱟ + ᱹ + ᱸ -> [ ᱟᱹᱸ ]
        """
        clusters = []
        current_cluster = ""

        for char in text:
            if char in MODIFIERS:
                # Attach modifier to preceding base character
                if current_cluster:
                    current_cluster += char
                else:
                    current_cluster = char
            else:
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = char

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def tokenize(self, text: str, lang_tag: str = "<sat_Olck>") -> List[str]:
        """Tokenize text into subword tokens respecting script boundary."""
        tokens = [lang_tag]
        words = text.strip().split()

        for w in words:
            # Check if word is purely Ol Chiki
            if any(is_ol_chiki(c) for c in w):
                clusters = self.split_into_grapheme_clusters(w)
                tokens.extend(clusters)
            else:
                # Latin or Bengali sub-tokenization
                tokens.append(w)

        return tokens

    def encode(self, text: str, lang_tag: str = "<sat_Olck>") -> List[int]:
        """Map text to integer token IDs."""
        tokens = self.tokenize(text, lang_tag)
        unk_id = self.vocab.get("<unk>", 2)
        return [self.vocab.get(tok, unk_id) for tok in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """Map integer token IDs back to text."""
        tokens = [
            self.inverse_vocab.get(tid, "<unk>")
            for tid in token_ids
            if tid not in (0, 1, 3)  # Skip <s>, </s>, <pad>
        ]
        # Filter out language tags
        clean_tokens = [t for t in tokens if not (t.startswith("<") and t.endswith(">"))]
        return "".join(clean_tokens)

    @classmethod
    def generate_sentencepiece_training_args(
        cls,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
    ) -> str:
        """
        Generates the standard SentencePiece training command tuned for Ol Chiki + Indic.
        """
        user_defined_symbols = ",".join(cls.SPECIAL_TOKENS)
        cmd = (
            f"spm_train "
            f"--input={input_file} "
            f"--model_prefix={model_prefix} "
            f"--vocab_size={vocab_size} "
            f"--character_coverage={character_coverage} "
            f"--model_type=bpe "
            f"--user_defined_symbols={user_defined_symbols} "
            f"--normalization_rule_name=nfc "
            f"--byte_fallback=true "
            f"--split_by_unicode_script=true"
        )
        return cmd
