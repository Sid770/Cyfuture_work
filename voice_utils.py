# voice_utils.py
import os
import wave
import numpy as np
import webrtcvad
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine

# Where to store enrolled embeddings:
ENROLLED_EMB_PATH = os.path.join("data", "enrolled_embedding.npy")
# We’ll also track a “dissimilar voice counter” in memory per session
# (for simplicity, store it here; in production you’d tie it to a user session ID)
VOICE_MISMATCH_DURATION = 0.0  # in seconds
MISMATCH_THRESHOLD = 10.0      # trigger alert after 10 seconds of mismatch

class VoiceUtils:
    def _init_(self):
        # Initialize Resemblyzer’s pre‐built speaker encoder
        self.encoder = VoiceEncoder()

        # Initialize a simple frame‐based VAD (mode=1: low aggressiveness)
        self.vad = webrtcvad.Vad(1)

    def _read_wav(self, wav_path: str) -> np.ndarray:
        """
        Load a wav file, resample to 16 kHz mono if needed, and return waveform array.
        """
        wav = preprocess_wav(wav_path)  # Resemblyzer helper: returns 16 kHz np.float32
        return wav

    def extract_embedding(self, wav_np: np.ndarray) -> np.ndarray:
        """
        Given a preprocessed waveform (np.ndarray, 16 kHz), return a 256‐dim speaker embedding.
        """
        return self.encoder.embed_utterance(wav_np)

    def enroll_user(self, wav_path: str) -> np.ndarray:
        """
        Read a 10–15 sec WAV (16 kHz mono), extract a single averaged embedding,
        and save it to disk so future chunks can be compared.
        """
        wav = self._read_wav(wav_path)
        # Resemblyzer’s embed_utterance() will average across the whole file:
        emb = self.extract_embedding(wav)
        # Save to disk:
        np.save(ENROLLED_EMB_PATH, emb)
        return emb

    def load_enrolled_embedding(self) -> np.ndarray:
        """
        Load previously‐saved embedding. Raise if not found.
        """
        if not os.path.exists(ENROLLED_EMB_PATH):
            raise FileNotFoundError("No enrolled voice embedding found. Register first.")
        return np.load(ENROLLED_EMB_PATH)

    def is_speech(self, pcm_bytes: bytes, sample_rate: int = 16000) -> bool:
        """
        Run WebRTC VAD on raw PCM bytes. Return True if speech is detected in this chunk.
        We expect pcm_bytes to be 16-bit mono PCM at 16 kHz.
        """
        # We’ll split the bytes into 20 ms frames (frame_length = sample_rate * 0.02 * 2 bytes/sample)
        frame_duration = 20  # milliseconds
        bytes_per_frame = int(sample_rate * (frame_duration / 1000.0) * 2)
        if len(pcm_bytes) < bytes_per_frame:
            return False

        # Check each frame; if any frame is “speech” → return True
        for i in range(0, len(pcm_bytes) - bytes_per_frame + 1, bytes_per_frame):
            frame = pcm_bytes[i:i + bytes_per_frame]
            if self.vad.is_speech(frame, sample_rate):
                return True
        return False

    def chunk_to_embedding(self, wav_path: str) -> np.ndarray:
        """
        Given a short WAV file path (2–5 sec), return its speaker embedding.
        Internally uses preprocess_wav to load and downsample.
        """
        wav = self._read_wav(wav_path)
        return self.extract_embedding(wav)

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Return cosine similarity between two embeddings in [–1, 1].
        We’ll clamp at [0,1] (if negative, treat as 0).
        """
        sim = 1.0 - cosine(emb1, emb2)
        return max(0.0, sim)

    def process_chunk(
        self,
        chunk_wav_path: str,
        last_aggregated: float,
        sim_threshold: float = 0.75
    ) -> (float, bool):
        """
        Process one short chunk:
         1. Check if speech is present using VAD.
         2. If speech → extract embedding and compare to enrolled.
         3. If similarity < sim_threshold → accumulate mismatch time.
         4. Otherwise reset mismatch accumulation.
        Returns: (new_mismatch_accumulated_seconds, trigger_alert_flag)
        """
        rms_speech = False
        # Read raw PCM to feed VAD:
        with wave.open(chunk_wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            pcm_bytes = wf.readframes(wf.getnframes())

        # Check if this chunk contains speech:
        if self.is_speech(pcm_bytes, sample_rate=sample_rate):
            rms_speech = True

        if not rms_speech:
            # No speech → do not change mismatch timer; no alert
            return last_aggregated, False

        # Extract embeddings:
        enrolled_emb = self.load_enrolled_embedding()
        chunk_emb = self.chunk_to_embedding(chunk_wav_path)
        sim = self.compare_embeddings(enrolled_emb, chunk_emb)

        if sim < sim_threshold:
            # Mismatch. Add chunk length (in seconds) to accumulated mismatch:
            duration_secs = wav_duration(chunk_wav_path)
            new_acc = last_aggregated + duration_secs
            trigger = new_acc >= MISMATCH_THRESHOLD
            return new_acc, trigger
        else:
            # Reset on good match
            return 0.0, False

def wav_duration(wav_path: str) -> float:
    """
    Return duration in seconds of a WAV file (assuming PCM).
    """
    with wave.open(wav_path, 'rb') as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)
