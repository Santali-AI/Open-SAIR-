"""
Pydantic Request and Response Schemas for the Open SAIR Microservices Gateway.
"""

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        def dict(self) -> Dict[str, Any]:
            return {
                k: v for k, v in self.__dict__.items()
                if not k.startswith("_")
            }

    def Field(default=..., **kwargs):
        return default


# 1. Translation Schemas
class TranslationRequest(BaseModel):
    text: str = Field(..., description="Input text to be translated")
    src_lang: str = Field(default="eng", description="Source language: 'eng', 'ben', or 'sat'")
    tgt_lang: str = Field(default="sat", description="Target language: 'eng', 'ben', or 'sat'")
    num_beams: int = Field(default=5, ge=1, le=10, description="Beam search width")
    max_length: int = Field(default=128, ge=8, le=512, description="Maximum generated sequence length")
    enforce_pronoun_disambiguation: bool = Field(default=True, description="Enforce clusivity checks")
    async_execution: bool = Field(default=False, description="Queue request asynchronously to Celery/Redis")


class TranslationResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "completed"
    source_text: str
    translated_text: str
    src_lang: str
    tgt_lang: str
    canonical_normalized: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


# 2. ASR Schemas
class ASRRequest(BaseModel):
    audio_base64: Optional[str] = Field(None, description="Base64-encoded 16kHz PCM audio")
    audio_url: Optional[str] = Field(None, description="URL pointing to remote WAV/MP3 file")
    detect_glottal_stops: bool = Field(default=True, description="Enable <50ms glottal closure analysis")
    async_execution: bool = Field(default=False, description="Queue request asynchronously")


class ASRResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "completed"
    transcription: str
    confidence: float
    duration_seconds: float
    glottal_stops_detected: int
    language: str = "sat_Olck"


# 3. TTS Schemas
class TTSRequest(BaseModel):
    text: str = Field(..., description="Input text in Ol Chiki script")
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed scaling factor")
    pitch_scale: float = Field(default=1.0, ge=0.5, le=2.0, description="F0 fundamental frequency multiplier")
    enable_nasalization_control: bool = Field(default=True, description="Modulate velopharyngeal opening via Mu Ttuddag")
    async_execution: bool = Field(default=False)


class TTSResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "completed"
    audio_base64: Optional[str] = None
    sample_rate: int = 22050
    duration_seconds: float
    nasalized_phonemes_count: int
    phoneme_count: int


# 4. OCR Schemas
class OCRRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded manuscript image")
    image_url: Optional[str] = Field(None, description="URL pointing to document scan")
    expand_diacritic_bounds: bool = Field(default=True, description="Expand bounding boxes for upper/lower diacritics")
    async_execution: bool = Field(default=False)


class OCRBox(BaseModel):
    box: List[int]
    text: str
    confidence: float


class OCRResponse(BaseModel):
    task_id: Optional[str] = None
    status: str = "completed"
    full_text: str
    total_lines: int
    diacritics_preserved: bool
    boxes: List[OCRBox]


# 5. Morphology & Pronoun Schemas
class PronounDisambiguationRequest(BaseModel):
    count: int = Field(default=2, ge=1, description="Number of participants (2 for dual, >=3 for plural)")
    includes_listener: bool = Field(default=True, description="True if addressee is included")


class PronounDisambiguationResponse(BaseModel):
    ol_chiki: str
    latin_translit: str
    person: str
    number: str
    clusivity: str
    gloss: str
    english_explanation: str
    bengali_explanation: str


# 6. Morphology Analyze Schemas
class MorphologyAnalyzeRequest(BaseModel):
    word: str = Field(..., description="Santali verb or word in Ol Chiki to analyze")


class MorphologyAnalyzeResponse(BaseModel):
    raw_word: str
    root: str
    reciprocal: bool
    reciprocal_form: Optional[str] = None
    tam_markers: List[str]
    finite_marker: bool
    subject_marker: Optional[str]
    gloss: str
    deglottalized_phonetics: str


# 7. Health & Status Schemas
class HealthResponse(BaseModel):
    status: str
    version: str
    backends: Dict[str, str]
