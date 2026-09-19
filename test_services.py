"""
Unit Tests for Open SAIR Tokenization, Audio Pipeline, OCR, and Service Schemas.
"""

import unittest

from open_sair.data.audio_pipeline import AudioProcessingPipeline
from open_sair.data.pipeline import DatasetPreprocessingPipeline
from open_sair.data.tokenizer import OlChikiBPETokenizer
from open_sair.ocr.pipeline import OlChikiOCREngine
from open_sair.services.schemas import (
    ASRRequest,
    OCRRequest,
    PronounDisambiguationRequest,
    TranslationRequest,
    TTSRequest,
)
from open_sair.tts.vits_engine import SantaliTTSEngine


class TestOpenSAIRServicesAndPipelines(unittest.TestCase):
    def setUp(self):
        self.tokenizer = OlChikiBPETokenizer()
        self.audio_pipeline = AudioProcessingPipeline(sample_rate=16000)
        self.dataset_pipeline = DatasetPreprocessingPipeline()
        self.tts_engine = SantaliTTSEngine()
        self.ocr_engine = OlChikiOCREngine()

    def test_tokenizer_atomic_graphemes(self):
        text = "ᱥᱟᱱᱛᱟᱲᱤ"
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens[0], "<sat_Olck>")
        self.assertIn("ᱥ", tokens)
        self.assertIn("ᱛ", tokens)

    def test_dataset_pipeline_filtering(self):
        sat = "ᱱᱚᱣᱟ ᱫᱚ ᱢᱤᱫᱴᱟᱝ ᱱᱟᱯᱟᱭ ᱯᱩᱛᱷᱤ ᱠᱟᱱᱟ᱾"
        ben = "এটি একটি ভালো বই।"
        eng = "This is a good book."
        is_valid, cleaned, reason = self.dataset_pipeline.filter_and_clean_sample(sat, ben, eng)
        self.assertTrue(is_valid, f"Filtering failed: {reason}")
        self.assertEqual(cleaned[0], sat)

    def test_audio_pipeline_vad_and_glottal(self):
        # Create a 1-second synthetic sample with a valley (<50ms)
        sample_rate = 16000
        samples = [0.1 * ((i % 100) / 100.0) for i in range(sample_rate)]
        # Inject sudden drop (glottal closure) for 30ms (480 samples)
        for i in range(4000, 4480):
            samples[i] = 0.00001

        segments = self.audio_pipeline.apply_vad(samples)
        self.assertTrue(len(segments) > 0)
        spec = self.audio_pipeline.get_feature_spec()
        self.assertEqual(spec["sampling_rate"], 16000)

    def test_tts_engine_conditioning(self):
        # Text with nasalization modifier Mu Ttuddag (ᱸ)
        text = "ᱯᱚᱸᱰᱮᱛ"
        res = self.tts_engine.synthesize(text)
        self.assertGreater(res["nasalized_phonemes_count"], 0)
        self.assertGreater(res["duration_seconds"], 0.0)

    def test_ocr_engine_processing(self):
        res = self.ocr_engine.process_document(b"", "sample.png")
        self.assertIn("full_text", res)
        self.assertTrue(res["diacritics_preserved"])
        self.assertGreater(res["total_lines_detected"], 0)

    def test_pydantic_schemas(self):
        t_req = TranslationRequest(text="Hello", src_lang="eng", tgt_lang="sat")
        self.assertEqual(t_req.src_lang, "eng")

        p_req = PronounDisambiguationRequest(count=2, includes_listener=False)
        self.assertEqual(p_req.count, 2)
        self.assertFalse(p_req.includes_listener)

    def test_fastapi_gateway_integration(self):
        try:
            from fastapi.testclient import TestClient
            from open_sair.services.app import app
            client = TestClient(app)

            # Health
            resp = client.get("/api/v1/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "healthy")

            # Pronoun morphology
            p_resp = client.post("/api/v1/morphology/pronoun", json={"count": 2, "includes_listener": True})
            self.assertEqual(p_resp.status_code, 200)
            self.assertEqual(p_resp.json()["ol_chiki"], "ᱟᱞᱟᱝ")

            # Translation
            t_resp = client.post("/api/v1/translate", json={"text": "Test", "src_lang": "eng", "tgt_lang": "sat"})
            self.assertEqual(t_resp.status_code, 200)
            self.assertEqual(t_resp.json()["status"], "completed")
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
