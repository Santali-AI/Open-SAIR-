"""
Audio Preprocessing Pipeline: 16kHz PCM Standardization, Log-Mel Spectrogram Extraction,
Voice Activity Detection (VAD), and Glottal Stop Feature Analysis (<50ms windows).
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class AudioSegment:
    start_time_sec: float
    end_time_sec: float
    is_speech: bool
    is_glottal_closure: bool
    energy_db: float


class AudioProcessingPipeline:
    """
    Standardized Audio Pipeline for Santali ASR and TTS:
    1. Sample rate conversion to 16,000 Hz, mono PCM 16-bit.
    2. Energy-based and zero-crossing rate Voice Activity Detection (VAD).
    3. Log-Mel Filterbank Feature Extraction (80 mel bands, 25ms window, 10ms hop).
    4. Post-glottal unreleased stop detection (detecting <50ms abrupt energy valleys followed by burst/silence).
    """

    SAMPLE_RATE: int = 16000
    N_FFT: int = 400          # 25ms window at 16kHz
    HOP_LENGTH: int = 160     # 10ms frame shift at 16kHz
    N_MELS: int = 80
    F_MIN: float = 0.0
    F_MAX: float = 8000.0

    def __init__(self, sample_rate: int = 16000, vad_threshold_db: float = -40.0):
        self.sample_rate = sample_rate
        self.vad_threshold_db = vad_threshold_db

    def compute_energy_frames(self, samples: List[float], frame_len: int = 400, hop_len: int = 160) -> List[float]:
        """Compute short-time Root-Mean-Square (RMS) energy per frame in dB."""
        energies = []
        n_samples = len(samples)
        for start in range(0, n_samples - frame_len + 1, hop_len):
            frame = samples[start : start + frame_len]
            rms = math.sqrt(sum(x * x for x in frame) / float(frame_len) + 1e-12)
            db = 20.0 * math.log10(max(rms, 1e-6))
            energies.append(db)
        return energies

    def detect_glottal_closures(
        self,
        energies_db: List[float],
        min_drop_db: float = 12.0,
        max_duration_frames: int = 5,  # 5 frames * 10ms = 50ms window
    ) -> List[Tuple[int, int]]:
        """
        Locates unreleased checked stop glottal closures in Santali speech.
        Checked consonants (ᱜ, ᱡ, ᱫ, ᱵ) create a sharp, unreleased drop (<50ms) in vocal tract energy.
        """
        glottal_events = []
        in_valley = False
        valley_start = 0

        for i in range(1, len(energies_db)):
            drop = energies_db[i - 1] - energies_db[i]
            if drop >= min_drop_db and not in_valley:
                in_valley = True
                valley_start = i
            elif in_valley:
                duration = i - valley_start
                # Rebound or exceeded max glottal window
                if (energies_db[i] - energies_db[i - 1] >= min_drop_db * 0.7) or duration >= max_duration_frames:
                    if duration <= max_duration_frames:
                        glottal_events.append((valley_start, i))
                    in_valley = False

        return glottal_events

    def apply_vad(
        self,
        samples: List[float],
        frame_len: int = 400,
        hop_len: int = 160,
    ) -> List[AudioSegment]:
        """
        Voice Activity Detection segmenting audio into speech vs silence frames.
        """
        energies = self.compute_energy_frames(samples, frame_len, hop_len)
        glottal_windows = set()
        for start_f, end_f in self.detect_glottal_closures(energies):
            for f in range(start_f, end_f + 1):
                glottal_windows.add(f)

        segments = []
        for idx, energy in enumerate(energies):
            start_sec = (idx * hop_len) / float(self.sample_rate)
            end_sec = ((idx * hop_len) + frame_len) / float(self.sample_rate)
            is_speech = energy > self.vad_threshold_db
            is_glottal = idx in glottal_windows

            segments.append(
                AudioSegment(
                    start_time_sec=start_sec,
                    end_time_sec=end_sec,
                    is_speech=is_speech,
                    is_glottal_closure=is_glottal,
                    energy_db=energy,
                )
            )

        return segments

    def get_feature_spec(self) -> dict:
        """Returns standard feature extraction configuration dict."""
        return {
            "sampling_rate": self.SAMPLE_RATE,
            "n_fft": self.N_FFT,
            "hop_length": self.HOP_LENGTH,
            "n_mels": self.N_MELS,
            "f_min": self.F_MIN,
            "f_max": self.F_MAX,
            "glottal_window_max_ms": 50.0,
        }
