# Open SAIR: Open Santali AI Research
**Unified Multimodal Sovereign AI Framework for Santali (Ol Chiki), Bengali, and English**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)](https://www.python.org/)
[![Unicode: Ol Chiki U+1C50-U+1C7F](https://img.shields.io/badge/Unicode-Ol%20Chiki-orange.svg)](https://unicode.org/charts/PDF/U1C50.pdf)

---

## 1. Project Overview

**Open SAIR (Open Santali AI Research)** is the premier open-source, sovereign multimodal artificial intelligence research framework engineered to process, translate, transcribe, synthesize, and digitize the Santali language in its native **Ol Chiki script** (`U+1C50`–`U+1C7F`), alongside **Bengali** and **English**.

### Key Architectural Pillars
- **Multimodal Engine:** Neural Machine Translation (NLLB-200/mBART-50), Automatic Speech Recognition (Wav2Vec2/Whisper), Speech Synthesis (VITS with F0 and Nasalization gating), and Historical Manuscript OCR (Sauvola + CRAFT + ResNet-BiLSTM-CTC).
- **Linguistic Precision:** Rigorous modeling of checked consonants (*Keched Arang*), deglottalizer (*Ahad*), agglutinative TAM infixes (e.g. reciprocal `-p-`), and explicit clusivity disambiguation for dual/plural pronouns.
- **Sovereign & Edge-Ready:** INT8 quantized mobile pipelines capable of running offline on budget smartphones in rural regions of Jharkhand, Odisha, West Bengal, and Assam.
- **Production Microservice Stack:** FastAPI asynchronous REST gateway backed by Redis and Celery distributed GPU workers.

---

## 2. Repository Structure

```
/workspace/open-sair/
├── docs/
│   ├── system_architecture.md     # Complete end-to-end technical blueprint & mathematical specs
│   ├── linguistic_matrix.md       # Orthography, phonetics, pronoun clusivity, & TAM matrix tables
│   └── deployment_roadmap.md      # Sovereign AI roadmap, edge INT8 optimization, & open governance
├── open_sair/
│   ├── alignment/
│   │   └── cross_lingual_embedder.py # InfoNCE contrastive alignment & Procrustes transformations
│   ├── linguistics/
│   │   ├── ol_chiki_unicode.py    # Unicode constants (U+1C50-U+1C7F), IPA mappings, character classes
│   │   ├── normalizer.py          # NFC normalization, canonical diacritic ordering, & deduplication
│   │   ├── pronoun_engine.py      # Dual/Plural, Inclusive vs Exclusive pronoun resolution engine
│   │   ├── morphology.py          # Agglutinative TAM infixes, reciprocal -p-, & glottal phonetics
│   │   └── grammar_transformer.py # SVO (English) <-> SOV (Bengali) <-> Agglutinative SOV (Santali)
│   ├── data/
│   │   ├── tokenizer.py           # Morphological BPE preserving atomic Ol Chiki diacritic clusters
│   │   ├── audio_pipeline.py      # 16kHz PCM preprocessing, VAD, & <50ms glottal closure analyzer
│   │   └── pipeline.py            # Trilingual parallel dataset cleaning & length-ratio filtering
│   ├── nmt/
│   │   ├── train_nllb.py          # HuggingFace NLLB-200 / mBART-50 fine-tuning pipeline
│   │   └── inference.py           # Beam search decoding with pronoun consistency enforcement
│   ├── asr/
│   │   └── wav2vec2_whisper_engine.py # ASR pipeline with sub-50ms glottal acoustic CTC decoder
│   ├── tts/
│   │   └── vits_engine.py         # VITS neural vocoder with Mu Ttuddag (ᱸ) nasalization control
│   ├── ocr/
│   │   └── pipeline.py            # Sauvola binarization, CRAFT detection, & ResNet-BiLSTM-CTC
│   ├── services/
│   │   ├── app.py                 # Production FastAPI Gateway routes (/translate, /asr, /tts, /ocr)
│   │   ├── schemas.py             # Pydantic request & response models
│   │   └── celery_tasks.py        # Asynchronous task queue workers for GPU pools
│   └── evaluation/
│       ├── metrics.py             # SacreBLEU, chrF++, WER, CER, Glottal & Diacritic Error Rates
│       └── benchmark_suite.py     # Automated linguistic benchmark suite
├── docker/
│   ├── Dockerfile.gateway         # FastAPI REST service container
│   ├── Dockerfile.worker          # GPU Celery worker container
│   └── docker-compose.yml         # Multi-service stack (Redis, Gateway, Workers)
├── tests/
│   ├── test_linguistics.py        # Unit tests for orthography, morphology, and benchmarks
│   └── test_services.py           # Unit tests for tokenization, audio, OCR, and schemas
├── requirements.txt               # Complete Python dependencies
└── setup.py                       # Packaging setup script
```

---

## 3. Quickstart Guide

### 3.1. Installation
```bash
cd /workspace/open-sair
pip install -e .
```

### 3.2. Running the Test Suite & Linguistic Benchmarks
Run the automated test suite covering all linguistic engines, phonetic rules, and evaluation metrics:
```bash
python3 -m unittest discover tests -v
```

Run the standalone linguistic benchmark suite directly:
```bash
python3 -c "
from open_sair.evaluation.benchmark_suite import OpenSAIRBenchmarkSuite
suite = OpenSAIRBenchmarkSuite()
for res in suite.run_all_benchmarks():
    status = 'PASSED' if res.passed else 'FAILED'
    print(f'[{status}] {res.test_name} ({res.category}): {res.details}')
"
```

### 3.3. Launching the Microservice Gateway
Start the local FastAPI gateway on port 8000:
```bash
uvicorn open_sair.services.app:app --host 0.0.0.0 --port 8000
```
Interactive OpenAPI documentation will be accessible at `http://localhost:8000/docs`.

### 3.4. Orchestrating with Docker Compose
To deploy the complete Redis broker, FastAPI gateway, and GPU model workers:
```bash
docker-compose -f docker/docker-compose.yml up --build -d
```

---

## 4. API Endpoints Quick Reference

| Endpoint | Method | Input Modality | Description |
| :--- | :--- | :--- | :--- |
| `/api/v1/translate` | POST | Text (JSON) | Cross-lingual NMT (`sat_Olck` $\leftrightarrow$ `ben_Beng` $\leftrightarrow$ `eng_Latn`) |
| `/api/v1/asr` | POST | Audio (Base64/URL) | 16kHz speech transcription with sub-50ms glottal stop detection |
| `/api/v1/tts` | POST | Text (JSON) | Speech synthesis with explicit Mu Ttuddag (`ᱸ`) nasalization control |
| `/api/v1/ocr` | POST | Image (Base64/URL) | Manuscript scan transcription with diacritic boundary preservation |
| `/api/v1/morphology/pronoun` | POST | JSON Parameters | Dual/Plural and Inclusive/Exclusive clusivity disambiguation |
| `/api/v1/health` | GET | None | Microservice health, backend availability, and worker pool status |

---

## 5. Architectural & Research Documentation
- **[System Architecture & Blueprint](docs/system_architecture.md):** Mathematical formulations, Mermaid dataflow diagrams, cross-attention layer mappings, and pipeline specifications.
- **[Linguistic & Orthographic Matrix](docs/linguistic_matrix.md):** Detailed tables of the 30-letter Ol Chiki matrix, diacritic modifiers, checked stop phonetics, pronoun clusivity, and agglutinative TAM markers.
- **[Sovereign AI Deployment Roadmap](docs/deployment_roadmap.md):** 4-phase rollout plan, ONNX/TensorRT INT8 edge optimization, mobile offline execution, and indigenous data governance protocols (CARE / OCAP).

---

## 6. Community & Citation

If using Open SAIR in academic research or sovereign language revitalization initiatives, please cite:

```bibtex
@software{open_sair_2026,
  author = {Open SAIR Research Consortium},
  title = {Open SAIR: Unified Multimodal Sovereign AI Framework for Santali and Ol Chiki Script},
  year = {2026},
  publisher = {Open Santali AI Research},
  url = {https://github.com/open-sair/open-sair}
}
```

*Dedicated to the enduring legacy of **Pandit Raghunath Murmu** (1905–1982), creator of the Ol Chiki script.*
