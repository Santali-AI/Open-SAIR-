#!/usr/bin/env python3
"""
Open SAIR - Master Runner & Interactive Multimodal Demonstration CLI.
Usage:
    python3 run.py --server         # Start production FastAPI Gateway on port 8000
    python3 run.py --demo           # Execute complete multimodal showcase (NMT, ASR, TTS, OCR, Morphology)
    python3 run.py --benchmarks     # Run full linguistic & quantitative benchmark suite
"""

import argparse
import json
import sys
import time


def run_benchmarks():
    print("=" * 70)
    print("  OPEN SAIR: LINGUISTIC & QUANTITATIVE BENCHMARK SUITE")
    print("=" * 70)
    from open_sair.evaluation.benchmark_suite import OpenSAIRBenchmarkSuite
    suite = OpenSAIRBenchmarkSuite()
    results = suite.run_all_benchmarks()
    for r in results:
        status = "\033[92mPASSED\033[0m" if r.passed else "\033[91mFAILED\033[0m"
        print(f"[{status}] {r.test_name} ({r.category})")
        print(f"    Score: {r.score:.1f}% | {r.details}")
    print("=" * 70)


def run_demo():
    print("=" * 70)
    print("       OPEN SAIR: MULTIMODAL SOVEREIGN AI DEMONSTRATION")
    print("       Processing Santali (Ol Chiki) <-> Bengali <-> English")
    print("=" * 70)

    # 1. Normalization & Orthography
    from open_sair.linguistics.normalizer import OlChikiNormalizer
    normalizer = OlChikiNormalizer()
    raw_text = "ᱯᱚᱸᱰᱮᱛ ᱨᱟᱹᱜᱷᱩᱱᱟᱛᱷ ᱢᱩᱨᱢᱩ ᱑᱙᱒᱕ ᱥᱟᱞ ᱨᱮ ᱚᱞ ᱪᱤᱠᱤ ᱛᱮᱭᱟᱨ ᱞᱮᱫᱟᱭ.."
    normalized_text = normalizer.normalize(raw_text)
    print("\n[1. UNICODE NORMALIZATION & ORTHOGRAPHY]")
    print(f"  Raw Input:       {raw_text}")
    print(f"  Normalized (NFC): {normalized_text}")

    # 2. Pronoun Matrix & Clusivity Disambiguation
    from open_sair.linguistics.pronoun_engine import SantaliPronounMatrixEngine
    pronoun_engine = SantaliPronounMatrixEngine()
    print("\n[2. PRONOUN MATRIX & CLUSIVITY DISAMBIGUATION]")
    # English 'We two (you and I)' -> Inclusive Dual
    p_incl = pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=True)
    print(f"  'We two (you and I)'     -> Ol Chiki: {p_incl.ol_chiki} ({p_incl.latin_translit}) | Gloss: {p_incl.gloss}")
    # English 'We two (he/she and I)' -> Exclusive Dual
    p_excl = pronoun_engine.resolve_we_disambiguation(count=2, includes_listener=False)
    print(f"  'We two (not you)'       -> Ol Chiki: {p_excl.ol_chiki} ({p_excl.latin_translit}) | Gloss: {p_excl.gloss}")
    # English 'We all (all of us + you)' -> Inclusive Plural
    p_pl_incl = pronoun_engine.resolve_we_disambiguation(count=4, includes_listener=True)
    print(f"  'We all (including you)' -> Ol Chiki: {p_pl_incl.ol_chiki} ({p_pl_incl.latin_translit}) | Gloss: {p_pl_incl.gloss}")
    # English 'We all (excluding you)' -> Exclusive Plural
    p_pl_excl = pronoun_engine.resolve_we_disambiguation(count=4, includes_listener=False)
    print(f"  'We all (excluding you)' -> Ol Chiki: {p_pl_excl.ol_chiki} ({p_pl_excl.latin_translit}) | Gloss: {p_pl_excl.gloss}")

    # 3. Agglutinative Morphology & Infixation
    from open_sair.linguistics.morphology import SantaliMorphologyEngine
    morph = SantaliMorphologyEngine()
    root = "ᱫᱟᱞ"  # dal (hit)
    recip = morph.apply_reciprocal_infix(root)
    deglottal = morph.deglottalize_phonetics("ᱜᱽ ᱡᱽ ᱫᱽ ᱵᱽ")
    print("\n[3. AGGLUTINATIVE MORPHOLOGY & PHONETICS]")
    print(f"  Verb Root:                {root} (dal: to strike)")
    print(f"  Reciprocal -p- Infix:     {recip} (dapal: to strike each other)")
    print(f"  Checked Deglottalization: ᱜᱽ ᱡᱽ ᱫᱽ ᱵᱽ -> IPA: {deglottal}")

    # 4. Neural Machine Translation (NMT)
    from open_sair.nmt.inference import OpenSAIRTranslator
    translator = OpenSAIRTranslator()
    eng_input = "We read books in school."
    trans_res = translator.translate(eng_input, src_lang="eng", tgt_lang="sat")
    print("\n[4. NEURAL MACHINE TRANSLATION (NMT)]")
    print(f"  Source (English): {eng_input}")
    print(f"  Target (Santali): {trans_res['translated_text']}")

    # 5. Speech Synthesis (TTS) with Nasalization Control
    from open_sair.tts.vits_engine import SantaliTTSEngine
    tts = SantaliTTSEngine()
    tts_res = tts.synthesize("ᱯᱚᱸᱰᱮᱛ ᱨᱟᱹᱜᱷᱩᱱᱟᱛᱷ ᱢᱩᱨᱢᱩ")
    print("\n[5. TEXT-TO-SPEECH (TTS) CONDITIONING]")
    print(f"  Synthesized Text:           ᱯᱚᱸᱰᱮᱛ ᱨᱟᱹᱜᱷᱩᱱᱟᱛᱷ ᱢᱩᱨᱢᱩ")
    print(f"  Estimated Audio Duration:   {tts_res['duration_seconds']:.2f} seconds")
    print(f"  Nasalized Phonemes Gated:   {tts_res['nasalized_phonemes_count']} (via Mu Ttuddag ᱸ)")

    # 6. Automatic Speech Recognition (ASR)
    from open_sair.asr.wav2vec2_whisper_engine import SantaliASREngine
    asr = SantaliASREngine()
    mock_audio = [0.05 * ((i % 80) / 80.0) for i in range(16000)]
    asr_res = asr.transcribe(mock_audio)
    print("\n[6. AUTOMATIC SPEECH RECOGNITION (ASR)]")
    print(f"  Audio Duration:             {asr_res.duration_seconds:.2f} seconds")
    print(f"  Transcribed Ol Chiki:       {asr_res.text}")
    print(f"  Acoustic Confidence:        {asr_res.confidence * 100:.1f}%")

    # 7. Historical Manuscript OCR
    from open_sair.ocr.pipeline import OlChikiOCREngine
    ocr = OlChikiOCREngine()
    ocr_res = ocr.process_document(b"", "manuscript_page_1.png")
    print("\n[7. HISTORICAL MANUSCRIPT OCR (CRAFT + RESNET-BILSTM)]")
    print(f"  Detected Text Lines:        {ocr_res['total_lines_detected']}")
    print(f"  Diacritic Bounds Preserved: {ocr_res['diacritics_preserved']}")
    print(f"  Recognized Transcription:\n{ocr_res['full_text']}")

    print("\n" + "=" * 70)
    print("  Open SAIR Multimodal Demonstration Completed Successfully!")
    print("=" * 70)


def run_server(host="0.0.0.0", port=8000):
    print(f"Starting Open SAIR FastAPI Gateway on http://{host}:{port}...")
    import uvicorn
    uvicorn.run("open_sair.services.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open SAIR Master Runner")
    parser.add_argument("--server", action="store_true", help="Launch FastAPI REST Gateway")
    parser.add_argument("--demo", action="store_true", help="Run multimodal showcase demo")
    parser.add_argument("--benchmarks", action="store_true", help="Execute linguistic benchmark suite")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.server:
        run_server(host=args.host, port=args.port)
    elif args.benchmarks:
        run_benchmarks()
    else:
        # Default: run the demo and then start server if requested
        run_demo()
