"""
Ol Chiki Unicode Character Definitions, Phonetic Classifications, and Orthographic Rules.
Standard Unicode Range: U+1C50 to U+1C7F.
"""

from typing import Dict, List, Set, Tuple

# Unicode Range Limits
OL_CHIKI_START = 0x1C50
OL_CHIKI_END = 0x1C7F

# 1. Digits (U+1C50 - U+1C59)
DIGIT_ZERO = "\u1C50"  # ᱐
DIGIT_ONE = "\u1C51"   # ᱑
DIGIT_TWO = "\u1C52"   # ᱒
DIGIT_THREE = "\u1C53" # ᱓
DIGIT_FOUR = "\u1C54"  # ᱔
DIGIT_FIVE = "\u1C55"  # ᱕
DIGIT_SIX = "\u1C56"   # ᱖
DIGIT_SEVEN = "\u1C57" # ᱗
DIGIT_EIGHT = "\u1C58" # ᱘
DIGIT_NINE = "\u1C59"  # ᱙

DIGITS: List[str] = [chr(cp) for cp in range(0x1C50, 0x1C5A)]

# 2. Vowels (Basic 6 Cardinal Vowels)
VOWEL_LA = "\u1C5A"   # ᱚ /ɔ/ (Open-mid back rounded)
VOWEL_LAA = "\u1C5F"  # ᱟ /a/ (Open central unrounded)
VOWEL_LI = "\u1C64"   # ᱤ /i/ (Close front unrounded)
VOWEL_LU = "\u1C69"   # ᱩ /u/ (Close back rounded)
VOWEL_LE = "\u1C6E"   # ᱮ /e/ (Close-mid front unrounded)
VOWEL_LO = "\u1C73"   # ᱳ /o/ (Close-mid back rounded)

VOWELS: Set[str] = {
    VOWEL_LA, VOWEL_LAA, VOWEL_LI, VOWEL_LU, VOWEL_LE, VOWEL_LO
}

# 3. Consonants (Base 24 consonants)
# Grouped according to Pt. Raghunath Murmu's 30-letter structural matrix
CONSONANT_AT = "\u1C5B"   # ᱛ /t/ (Voiceless dental plosive)
CONSONANT_AG = "\u1C5C"   # ᱜ /k'/ or /g'/ (Checked velar / pre-stopped glottal)
CONSONANT_ANG = "\u1C5D"  # ᱝ /ŋ/ (Velar nasal)
CONSONANT_AL = "\u1C5E"   # ᱞ /l/ (Alveolar lateral approximant)
CONSONANT_AAK = "\u1C60"  # ᱠ /k/ (Voiceless velar plosive)
CONSONANT_AAJ = "\u1C61"  # ᱡ /c'/ or /ɟ'/ (Checked palatal)
CONSONANT_AAM = "\u1C62"  # ᱢ /m/ (Bilabial nasal)
CONSONANT_AAW = "\u1C63"  # ᱣ /w/ (Labio-velar approximant)
CONSONANT_IS = "\u1C65"   # ᱥ /s/ (Voiceless alveolar fricative)
CONSONANT_IH = "\u1C66"   # ᱦ /h/ (Voiceless glottal fricative)
CONSONANT_INY = "\u1C67"  # ᱧ /ɲ/ (Palatal nasal)
CONSONANT_IR = "\u1C68"   # ᱨ /r/ (Alveolar trill)
CONSONANT_UC = "\u1C6A"   # ᱪ /c/ (Voiceless palatal plosive)
CONSONANT_UD = "\u1C6B"   # ᱫ /t'/ or /d'/ (Checked dental/alveolar)
CONSONANT_UNN = "\u1C6C"  # ᱬ /ɳ/ (Retroflex nasal)
CONSONANT_UY = "\u1C6D"   # ᱭ /j/ (Palatal approximant)
CONSONANT_EP = "\u1C6F"   # ᱯ /p/ (Voiceless bilabial plosive)
CONSONANT_EDD = "\u1C70"  # ᱰ /ɖ/ (Voiced retroflex plosive)
CONSONANT_EN = "\u1C71"   # ᱱ /n/ (Alveolar nasal)
CONSONANT_ERR = "\u1C72"  # ᱲ /ɽ/ (Retroflex flap)
CONSONANT_OTT = "\u1C74"  # ᱴ /ʈ/ (Voiceless retroflex plosive)
CONSONANT_OB = "\u1C75"   # ᱵ /p'/ or /b'/ (Checked bilabial)
CONSONANT_OV = "\u1C76"   # ᱶ /w̃/ (Nasalized bilabial glide)
CONSONANT_OH = "\u1C77"   # ᱷ /ʰ/ (Aspiration sign)

BASE_CONSONANTS: Set[str] = {
    CONSONANT_AT, CONSONANT_AG, CONSONANT_ANG, CONSONANT_AL,
    CONSONANT_AAK, CONSONANT_AAJ, CONSONANT_AAM, CONSONANT_AAW,
    CONSONANT_IS, CONSONANT_IH, CONSONANT_INY, CONSONANT_IR,
    CONSONANT_UC, CONSONANT_UD, CONSONANT_UNN, CONSONANT_UY,
    CONSONANT_EP, CONSONANT_EDD, CONSONANT_EN, CONSONANT_ERR,
    CONSONANT_OTT, CONSONANT_OB, CONSONANT_OV, CONSONANT_OH
}

# 4. Checked Consonants (Keched Arang - Unreleased Glottalized Stops)
CHECKED_CONSONANTS: Set[str] = {
    CONSONANT_AG,  # ᱜ /k'/ (velar checked stop)
    CONSONANT_AAJ, # ᱡ /c'/ (palatal checked stop)
    CONSONANT_UD,  # ᱫ /t'/ (dental checked stop)
    CONSONANT_OB,  # ᱵ /p'/ (bilabial checked stop)
}

# 5. Modifiers and Diacritics (U+1C78 - U+1C7D)
MODIFIER_MU_TTUDDAG = "\u1C78"      # ᱸ (Nasalization dot - Mu Ttuddag)
MODIFIER_GAAHLAA_TTUDDAAG = "\u1C79" # ᱹ (Baseline dot / low vowel - Gaahlaa Ttuddag)
MODIFIER_MU_GAAHLAA = "\u1C7A"       # ᱺ (Combined nasal + baseline - Mu-Gaahlaa Ttuddag)
MODIFIER_RELAA = "\u1C7B"            # ᱻ (Vowel lengthener / prolongation - Relaa)
MODIFIER_PHAARKAA = "\u1C7C"         # ᱼ (Glottal protector / separator - Phaarkaa)
MODIFIER_AHAD = "\u1C7D"             # ᱽ (Deglottalizer - Ahad: softens checked consonants)

MODIFIERS: Set[str] = {
    MODIFIER_MU_TTUDDAG,
    MODIFIER_GAAHLAA_TTUDDAAG,
    MODIFIER_MU_GAAHLAA,
    MODIFIER_RELAA,
    MODIFIER_PHAARKAA,
    MODIFIER_AHAD
}

# 6. Punctuation (U+1C7E, U+1C7F)
PUNCTUATION_MUCAAD = "\u1C7E"         # ᱾ (Single danda / full stop)
PUNCTUATION_DOUBLE_MUCAAD = "\u1C7F"  # ᱿ (Double danda / stanza end)

PUNCTUATIONS: Set[str] = {
    PUNCTUATION_MUCAAD,
    PUNCTUATION_DOUBLE_MUCAAD
}

ALL_OL_CHIKI_CHARS: Set[str] = (
    set(DIGITS) | VOWELS | BASE_CONSONANTS | MODIFIERS | PUNCTUATIONS
)

# Deglottalization mappings: Checked consonant + Ahad (ᱽ) -> Voiced released continuant
DEGLOTTALIZED_MAPPING: Dict[str, str] = {
    CONSONANT_AG + MODIFIER_AHAD: "g",   # ᱜᱽ -> /g/
    CONSONANT_AAJ + MODIFIER_AHAD: "j",  # ᱡᱽ -> /ɟ/ or /dʒ/
    CONSONANT_UD + MODIFIER_AHAD: "d",   # ᱫᱽ -> /d/
    CONSONANT_OB + MODIFIER_AHAD: "b",   # ᱵᱽ -> /b/
}

# Phoneme / IPA Representation Mapping
IPA_MAP: Dict[str, str] = {
    VOWEL_LA: "ɔ",
    VOWEL_LAA: "a",
    VOWEL_LI: "i",
    VOWEL_LU: "u",
    VOWEL_LE: "e",
    VOWEL_LO: "o",
    CONSONANT_AT: "t",
    CONSONANT_AG: "k̚ˀ", # unreleased velar with glottal closure
    CONSONANT_ANG: "ŋ",
    CONSONANT_AL: "l",
    CONSONANT_AAK: "k",
    CONSONANT_AAJ: "c̚ˀ", # unreleased palatal with glottal closure
    CONSONANT_AAM: "m",
    CONSONANT_AAW: "w",
    CONSONANT_IS: "s",
    CONSONANT_IH: "h",
    CONSONANT_INY: "ɲ",
    CONSONANT_IR: "r",
    CONSONANT_UC: "c",
    CONSONANT_UD: "t̚ˀ", # unreleased dental with glottal closure
    CONSONANT_UNN: "ɳ",
    CONSONANT_UY: "j",
    CONSONANT_EP: "p",
    CONSONANT_EDD: "ɖ",
    CONSONANT_EN: "n",
    CONSONANT_ERR: "ɽ",
    CONSONANT_OTT: "ʈ",
    CONSONANT_OB: "p̚ˀ", # unreleased bilabial with glottal closure
    CONSONANT_OV: "w̃",
    CONSONANT_OH: "ʰ",
    MODIFIER_MU_TTUDDAG: "̃",       # nasalization
    MODIFIER_GAAHLAA_TTUDDAAG: "˕", # lowering
    MODIFIER_MU_GAAHLAA: "̃˕",
    MODIFIER_RELAA: "ː",           # length
    MODIFIER_PHAARKAA: "ʔ",        # glottal separator
    MODIFIER_AHAD: "",             # triggers release
}


def is_ol_chiki(char: str) -> bool:
    """Check if a character is in the Ol Chiki Unicode block."""
    if not char:
        return False
    cp = ord(char[0])
    return OL_CHIKI_START <= cp <= OL_CHIKI_END


def contains_ol_chiki(text: str) -> bool:
    """Check if the text string contains any Ol Chiki characters."""
    return any(is_ol_chiki(c) for c in text)
