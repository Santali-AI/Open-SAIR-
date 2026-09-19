"""
Santali Pronoun Matrix Engine: Morphological Disambiguation across Number and Clusivity.
Enforces strict distinction between Singular, Dual, and Plural; and Inclusive vs Exclusive.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class GrammaticalPerson(Enum):
    FIRST = "1"
    SECOND = "2"
    THIRD = "3"


class GrammaticalNumber(Enum):
    SINGULAR = "SG"
    DUAL = "DU"
    PLURAL = "PL"


class Clusivity(Enum):
    INCLUSIVE = "INCL"  # Includes listener
    EXCLUSIVE = "EXCL"  # Excludes listener
    NOT_APPLICABLE = "NA"


class Animacy(Enum):
    ANIMATE = "ANIM"
    INANIMATE = "INANIM"


class Deixis(Enum):
    PROXIMATE = "PROX"  # Near speaker ("this")
    DISTANT = "DIST"    # Away from speaker ("that")


@dataclass(frozen=True)
class PronounRecord:
    ol_chiki: str
    latin_translit: str
    person: GrammaticalPerson
    number: GrammaticalNumber
    clusivity: Clusivity
    animacy: Animacy
    deixis: Optional[Deixis]
    bengali_equivalents: Tuple[str, ...]
    english_equivalents: Tuple[str, ...]
    subject_clitic: str
    object_clitic: str
    gloss: str


class SantaliPronounMatrixEngine:
    """
    Exhaustive Pronoun Resolution and Disambiguation Engine for Santali.
    Provides matrix lookup, contextual clusivity resolution, and cross-lingual alignment.
    """

    def __init__(self):
        self._pronoun_table: List[PronounRecord] = self._build_matrix()
        self._ol_chiki_index: Dict[str, PronounRecord] = {
            p.ol_chiki: p for p in self._pronoun_table
        }

    def _build_matrix(self) -> List[PronounRecord]:
        return [
            # 1st Person Singular
            PronounRecord(
                ol_chiki="ᱤᱧ",
                latin_translit="iń",
                person=GrammaticalPerson.FIRST,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("আমি", "আমাকে"),
                english_equivalents=("I", "me"),
                subject_clitic="ᱧ",
                object_clitic="ᱧ",
                gloss="1SG",
            ),
            # 1st Person Dual Inclusive (You and I)
            PronounRecord(
                ol_chiki="ᱟᱞᱟᱝ",
                latin_translit="alaṅ",
                person=GrammaticalPerson.FIRST,
                number=GrammaticalNumber.DUAL,
                clusivity=Clusivity.INCLUSIVE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("আমরা দুজন (তুমি ও আমি)",),
                english_equivalents=("we two (you and I)", "us two (incl)"),
                subject_clitic="ᱞᱟᱝ",
                object_clitic="ᱞᱟᱝ",
                gloss="1DU.INCL",
            ),
            # 1st Person Dual Exclusive (He/She and I, not you)
            PronounRecord(
                ol_chiki="ᱟᱞᱤᱧ",
                latin_translit="aliń",
                person=GrammaticalPerson.FIRST,
                number=GrammaticalNumber.DUAL,
                clusivity=Clusivity.EXCLUSIVE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("আমরা দুজন (সে ও আমি, তুমি নও)",),
                english_equivalents=("we two (he/she and I, not you)", "us two (excl)"),
                subject_clitic="ᱞᱤᱧ",
                object_clitic="ᱞᱤᱧ",
                gloss="1DU.EXCL",
            ),
            # 1st Person Plural Inclusive (All of us, including you)
            PronounRecord(
                ol_chiki="ᱟᱵᱚ",
                latin_translit="abo",
                person=GrammaticalPerson.FIRST,
                number=GrammaticalNumber.PLURAL,
                clusivity=Clusivity.INCLUSIVE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("আমরা সবাই (তোমারা সহ)", "আমরা"),
                english_equivalents=("we all (incl)", "us (incl)"),
                subject_clitic="ᱵᱚ",
                object_clitic="ᱵᱚ",
                gloss="1PL.INCL",
            ),
            # 1st Person Plural Exclusive (All of us, excluding you)
            PronounRecord(
                ol_chiki="ᱟᱞᱮ",
                latin_translit="ale",
                person=GrammaticalPerson.FIRST,
                number=GrammaticalNumber.PLURAL,
                clusivity=Clusivity.EXCLUSIVE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("আমরা (তোমরা ছাড়া)", "আমরা"),
                english_equivalents=("we (excl)", "us (excl)"),
                subject_clitic="ᱞᱮ",
                object_clitic="ᱞᱮ",
                gloss="1PL.EXCL",
            ),
            # 2nd Person Singular
            PronounRecord(
                ol_chiki="ᱟᱢ",
                latin_translit="am",
                person=GrammaticalPerson.SECOND,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("তুমি", "তুই", "আপনি"),
                english_equivalents=("you (sg)",),
                subject_clitic="ᱢ",
                object_clitic="ᱢᱮ",
                gloss="2SG",
            ),
            # 2nd Person Dual (You two)
            PronounRecord(
                ol_chiki="ᱟᱵᱮᱱ",
                latin_translit="aben",
                person=GrammaticalPerson.SECOND,
                number=GrammaticalNumber.DUAL,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("তোমরা দুজন", "আপনারা দুজন"),
                english_equivalents=("you two",),
                subject_clitic="ᱵᱮᱱ",
                object_clitic="ᱵᱮᱱ",
                gloss="2DU",
            ),
            # 2nd Person Plural (You all)
            PronounRecord(
                ol_chiki="ᱟᱯᱮ",
                latin_translit="ape",
                person=GrammaticalPerson.SECOND,
                number=GrammaticalNumber.PLURAL,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=None,
                bengali_equivalents=("তোমরা", "আপনারা", "তোরা"),
                english_equivalents=("you all", "you guys"),
                subject_clitic="ᱯᱮ",
                object_clitic="ᱯᱮ",
                gloss="2PL",
            ),
            # 3rd Person Singular Animate Distant
            PronounRecord(
                ol_chiki="ᱩᱱᱤ",
                latin_translit="uni",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=Deixis.DISTANT,
                bengali_equivalents=("সে", "তিনি", "তাকে"),
                english_equivalents=("he", "she", "him", "her"),
                subject_clitic="ᱭ",
                object_clitic="ᱮ",
                gloss="3SG.ANIM.DIST",
            ),
            # 3rd Person Singular Animate Proximate
            PronounRecord(
                ol_chiki="ᱱᱩᱭ",
                latin_translit="nui",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=Deixis.PROXIMATE,
                bengali_equivalents=("এ", "ইনি", "একে"),
                english_equivalents=("this person", "he/she (here)"),
                subject_clitic="ᱭ",
                object_clitic="ᱮ",
                gloss="3SG.ANIM.PROX",
            ),
            # 3rd Person Dual Animate Distant
            PronounRecord(
                ol_chiki="ᱩᱱᱠᱤᱱ",
                latin_translit="unkin",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.DUAL,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=Deixis.DISTANT,
                bengali_equivalents=("তারা দুজন", "ওঁরা দুজন"),
                english_equivalents=("they two", "those two"),
                subject_clitic="ᱠᱤᱱ",
                object_clitic="ᱠᱤᱱ",
                gloss="3DU.ANIM.DIST",
            ),
            # 3rd Person Plural Animate Distant
            PronounRecord(
                ol_chiki="ᱩᱱᱠᱩ",
                latin_translit="unku",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.PLURAL,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.ANIMATE,
                deixis=Deixis.DISTANT,
                bengali_equivalents=("তারা", "ওঁরা", "তাদের"),
                english_equivalents=("they", "them"),
                subject_clitic="ᱠᱚ",
                object_clitic="ᱠᱚ",
                gloss="3PL.ANIM.DIST",
            ),
            # 3rd Person Inanimate Singular Distant
            PronounRecord(
                ol_chiki="ᱚᱱᱟ",
                latin_translit="ona",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.INANIMATE,
                deixis=Deixis.DISTANT,
                bengali_equivalents=("সেটি", "ওটা"),
                english_equivalents=("that", "it"),
                subject_clitic="",
                object_clitic="",
                gloss="3SG.INANIM.DIST",
            ),
            # 3rd Person Inanimate Singular Proximate
            PronounRecord(
                ol_chiki="ᱱᱚᱣᱟ",
                latin_translit="nowa",
                person=GrammaticalPerson.THIRD,
                number=GrammaticalNumber.SINGULAR,
                clusivity=Clusivity.NOT_APPLICABLE,
                animacy=Animacy.INANIMATE,
                deixis=Deixis.PROXIMATE,
                bengali_equivalents=("এটি", "এটা"),
                english_equivalents=("this", "it"),
                subject_clitic="",
                object_clitic="",
                gloss="3SG.INANIM.PROX",
            ),
        ]

    def get_by_ol_chiki(self, token: str) -> Optional[PronounRecord]:
        """Lookup pronoun record by Ol Chiki script token."""
        return self._ol_chiki_index.get(token)

    def resolve_we_disambiguation(
        self,
        count: int,
        includes_listener: bool,
    ) -> PronounRecord:
        """
        Disambiguates English 'we/us' or Bengali 'amra' into the exact Ol Chiki pronoun.
        
        Args:
            count: Number of entities (2 for dual, >=3 for plural)
            includes_listener: True if the addressed person is included in the action.
        """
        if count == 2:
            if includes_listener:
                return self._ol_chiki_index["ᱟᱞᱟᱝ"]  # alaṅ (1DU.INCL)
            else:
                return self._ol_chiki_index["ᱟᱞᱤᱧ"]  # aliń (1DU.EXCL)
        else:
            if includes_listener:
                return self._ol_chiki_index["ᱟᱵᱚ"]   # abo (1PL.INCL)
            else:
                return self._ol_chiki_index["ᱟᱞᱮ"]   # ale (1PL.EXCL)

    def get_matrix_table(self) -> List[Dict[str, str]]:
        """Return structured dictionary table of the entire pronoun matrix."""
        return [
            {
                "Ol Chiki": p.ol_chiki,
                "Translit": p.latin_translit,
                "Person": p.person.value,
                "Number": p.number.value,
                "Clusivity": p.clusivity.value,
                "Animacy": p.animacy.value,
                "Gloss": p.gloss,
                "Subject Clitic": p.subject_clitic,
                "English": ", ".join(p.english_equivalents),
                "Bengali": ", ".join(p.bengali_equivalents),
            }
            for p in self._pronoun_table
        ]
