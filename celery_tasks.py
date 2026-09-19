"""
Celery Task Definitions for Heavy Deep Learning Workers:
Routes NMT, Whisper ASR, VITS TTS, and ResNet OCR to dedicated GPU task queues.
"""

import os
from typing import Any, Dict

try:
    from celery import Celery
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if HAS_CELERY:
    celery_app = Celery(
        "open_sair_workers",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_routes={
            "open_sair.tasks.translate": {"queue": "nmt_queue"},
            "open_sair.tasks.asr": {"queue": "asr_queue"},
            "open_sair.tasks.tts": {"queue": "tts_queue"},
            "open_sair.tasks.ocr": {"queue": "ocr_queue"},
        },
    )

    @celery_app.task(name="open_sair.tasks.translate", bind=True)
    def task_translate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Worker task executing Neural Machine Translation."""
        from open_sair.nmt.inference import OpenSAIRTranslator
        translator = OpenSAIRTranslator()
        return translator.translate(
            text=payload["text"],
            src_lang=payload.get("src_lang", "eng"),
            tgt_lang=payload.get("tgt_lang", "sat"),
            num_beams=payload.get("num_beams", 5),
        )

    @celery_app.task(name="open_sair.tasks.asr", bind=True)
    def task_asr(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Worker task executing Whisper / Wav2Vec2 ASR."""
        from open_sair.asr.wav2vec2_whisper_engine import SantaliASREngine
        engine = SantaliASREngine()
        # Mock sample buffer
        samples = [0.0] * 16000
        res = engine.transcribe(samples)
        return {
            "text": res.text,
            "confidence": res.confidence,
            "glottal_stops_detected": res.glottal_stops_detected,
            "duration": res.duration_seconds,
        }

    @celery_app.task(name="open_sair.tasks.tts", bind=True)
    def task_tts(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Worker task executing VITS speech synthesis."""
        from open_sair.tts.vits_engine import SantaliTTSEngine
        engine = SantaliTTSEngine()
        return engine.synthesize(
            text=payload["text"],
            speaking_rate=payload.get("speaking_rate", 1.0),
            pitch_scale=payload.get("pitch_scale", 1.0),
        )

    @celery_app.task(name="open_sair.tasks.ocr", bind=True)
    def task_ocr(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Worker task executing manuscript OCR."""
        from open_sair.ocr.pipeline import OlChikiOCREngine
        engine = OlChikiOCREngine()
        return engine.process_document(
            image_bytes=b"",
            filename=payload.get("filename", "manuscript.png"),
        )
else:
    celery_app = None
