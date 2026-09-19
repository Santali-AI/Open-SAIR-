"""
Cross-Lingual Structural Mapping and Word Order Transformations:
SVO (English) <-> SOV (Bengali) <-> Agglutinative SOV (Santali).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ConstituentSentence:
    subject: str
    direct_object: Optional[str] = None
    indirect_object: Optional[str] = None
    verb_root: str = ""
    tense_aspect: str = "PRESENT"
    adverbs_locative: Optional[str] = None
    postposition_or_prep: Optional[str] = None


class CrossLingualGrammarTransformer:
    """
    Syntactic reordering and structural transformation engine across:
    - English (SVO, Prepositional, Analytic)
    - Bengali (SOV, Postpositional, Indo-Aryan inflecting)
    - Santali (SOV, Agglutinative, Polysynthetic verb complexes, Clitic incorporation)
    """

    # Postposition mappings
    POSTPOSITIONS = {
        "LOCATIVE": {"english": "in / at", "bengali": "-এ / -তে", "santali": "-re (ᱨᱮ)"},
        "INSTRUMENTAL_ALLATIVE": {"english": "by / with / to", "bengali": "-দিয়ে / -দ্বারা", "santali": "-te (ᱛᱮ)"},
        "ABLATIVE": {"english": "from", "bengali": "-থেকে / -হতে", "santali": "-khon (ᱠᱷᱚᱱ)"},
        "GENITIVE_ANIMATE": {"english": "of (animate)", "bengali": "-এর", "santali": "-ren (ᱨᱮᱱ)"},
        "GENITIVE_INANIMATE": {"english": "of (inanimate)", "bengali": "-এর", "santali": "-ak' (ᱟᱜ)"},
        "ASSOCIATIVE": {"english": "along with", "bengali": "-এর সাথে", "santali": "-são (ᱥᱟᱶ)"},
    }

    def svo_to_sov(self, tokens: List[str], s_idx: int, v_idx: int, o_idx: int) -> List[str]:
        """
        Reorders English SVO constituent sequence into SOV order.
        [Subject, Verb, Object] -> [Subject, Object, Verb]
        """
        reordered = list(tokens)
        subject = tokens[s_idx]
        verb = tokens[v_idx]
        obj = tokens[o_idx]

        # In SOV: S comes first, then O, then V
        prefix = [t for i, t in enumerate(tokens) if i not in (s_idx, v_idx, o_idx)]
        return [subject, obj, verb] + prefix

    def sov_to_svo(self, tokens: List[str], s_idx: int, o_idx: int, v_idx: int) -> List[str]:
        """
        Reorders Bengali/Santali SOV constituent sequence into English SVO order.
        [Subject, Object, Verb] -> [Subject, Verb, Object]
        """
        subject = tokens[s_idx]
        obj = tokens[o_idx]
        verb = tokens[v_idx]
        prefix = [t for i, t in enumerate(tokens) if i not in (s_idx, v_idx, o_idx)]
        return [subject, verb, obj] + prefix

    def format_structural_comparison(
        self,
        english_ex: str,
        bengali_ex: str,
        santali_ex: str,
        santali_gloss: str,
    ) -> Dict[str, str]:
        """Format tri-lingual structural comparison for documentation and benchmarks."""
        return {
            "English (SVO)": english_ex,
            "Bengali (SOV)": bengali_ex,
            "Santali (Agglutinative SOV)": santali_ex,
            "Santali Morphological Gloss": santali_gloss,
        }
