import hashlib
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable, List

import numpy
import soundfile
import torch
import webrtcvad
from speechbrain.inference import EncoderClassifier

# ===============================
# Tunables
# ===============================

# --- Preprocess (ffmpeg) ---
# Telephony speech is concentrated ≈ 300–3400 Hz. We use a slightly broader phone-band:
# - highpass 120 Hz: removes rumble/handling noise below fundamental; keeps male fundamentals (>~85 Hz) thanks to harmonics.
# - lowpass 3800 Hz: cuts hiss; keeps key consonant energy ~>2 kHz while respecting phone bandwidth.
# Denoise strength nr=12 (on 0–100-ish scale for afftdn): mild noise reduction that usually preserves fricatives.
FFMPEG_AF = "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=120,lowpass=f=3800,afftdn=nr=12"

# --- Speaker embedding windows ---

# --- Clustering ---

# --- ASR ---
# accurate for English telephony; change with --asr-model if VRAM tight.
ASR_MODEL = "large-v3"
# modest beam for good quality without big latency.
ASR_BEAM = 5
# or "int8_float16" to reduce VRAM with small WER hit.
ASR_COMPUTE_TYPE = "float16"
# lets Faster-Whisper downweight non-speech internally for stability.
ASR_VAD_FILTER = True

# --- Profiles (optional) ---
# Persist speaker prototypes to keep "A/B" stable across calls:
# set via --profiles path to enable (e.g., profiles.json)
PROFILE_DB = None
# EMA factor when updating centroids: new = (1-a)*old + a*new (a=0.3 reacts but stays stable).
PROFILE_SMOOTH = 0.3

# --- Constants ---
MAX_16BIT_INT_VAL = 32767.0
INPUT_SAMPLE_RATE_KHZ = 16000


# ===============================
# Data structures
# ===============================
@dataclass
class Args:
    input: Path
    output: Path
    profiles: None | Path
    # Minimum speech segment length (ms) to keep tiny back-channels (~ “oh”, “ok”) without keeping clicks.
    min_voice: int
    # Minimum non-speech gap (ms) to terminate a segment (short to preserve rapid turn-taking).
    min_silence: int
    # Aggressiveness 0..3: higher = more speech rejections. 2 balances false accepts/rejects on phone noise.
    vad_aggression: int
    # WebRTC VAD accepts 10/20/30 ms frames. 30 ms gives steadier decisions in noise.
    frame_ms: int
    # Spectral clustering on cosine affinities is robust to blob shapes and common in diarization literature.
    num_speakers: int
    # 1.5 s windows: long enough for stable ECAPA embedding; short for frequent turns.
    emb_window_s: float
    # 50% overlap improves robustness without exploding compute.
    emb_hop_s: float
    # empirically needed for ECAPA stability
    ecapa_stability_floor: float


@dataclass
class VADSegment:
    start: float
    end: float
    voiced: bool


# ===============================
# Utilities
# ===============================
def frange(start=0.0, end=0.0, jump=1.0):
    while start < end:
        yield start
        start += jump


# ===============================
# Main
# ===============================


def parse_args() -> Args:
    parser = ArgumentParser(description="Transcribe and diarize audio files.")
    parser.add_argument("input", help="audio filepath", type=Path)
    parser.add_argument("--output", default=os.getcwd(), type=Path)
    parser.add_argument("--profiles", default=None, help="speaker profiles.json path")
    parser.add_argument("--min_voice", default=120, type=int, help="min voice ms")
    parser.add_argument("--min_silence", default=120, type=int, help="min silence ms")
    parser.add_argument("--vad_aggression", default=2, type=int, help="0..3")
    parser.add_argument("--frame_ms", default=30, type=int, help="10/20/30 ms")
    parser.add_argument("--num_speakers", default=2, type=int)
    parser.add_argument("--emb_window_s", default=1.5, type=float)
    parser.add_argument("--emb_hop_s", default=0.75, type=float)
    parser.add_argument("--ecapa_stability_floor", default=0.6, type=float)

    parsed = parser.parse_args()
    args = Args(**vars(parsed))
    assert args.input.is_file()
    assert args.output.is_dir()
    assert args.vad_aggression > 0 and args.vad_aggression <= 3
    assert args.frame_ms == 10 or args.frame_ms == 20 or args.frame_ms == 30

    return args


def preprocess_audio(input: Path) -> Path:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found in PATH. Install ffmpeg (sudo apt install ffmpeg).")

    digest = hashlib.sha1(str(input).encode()).hexdigest()
    output = Path(f"/tmp/{digest}.wav")

    if not output.exists():
        cmd = f"""
ffmpeg -y \
    -i "{input}" \
    -ac 1 \
    -ar {INPUT_SAMPLE_RATE_KHZ} \
    -af "{FFMPEG_AF}" \
    "{output}"
"""
        subprocess.run(cmd, shell=True, check=True)

    return output


def voice_activation_detection(
    audio: numpy.typing.NDArray, sample_rate: int, args: Args
) -> List[VADSegment]:
    if audio.ndim > 1:
        audio = audio[:, 0]  # ensure mono
    audio = numpy.ascontiguousarray(audio)
    pcm16 = (numpy.clip(audio, -1.0, 1.0) * MAX_16BIT_INT_VAL).astype(numpy.int16)
    frame_length = int(sample_rate * args.frame_ms / 1000)
    max_length = len(pcm16) // frame_length
    segments: List[VADSegment] = []

    to_ms: Callable[[float], float] = lambda n: n / frame_length * args.frame_ms

    vad = webrtcvad.Vad(args.vad_aggression)
    segment_start = 0
    segment_stop = 0
    segment_voice = False
    for i in range(0, max_length):
        start = i * frame_length
        stop = start + frame_length
        frame = pcm16[start:stop]
        voiced: bool = vad.is_speech(frame.tobytes(), sample_rate)

        if segment_voice != voiced:
            segments.append(
                VADSegment(
                    to_ms(segment_start),
                    to_ms(segment_stop),
                    segment_voice,
                )
            )
            segment_voice = voiced
            segment_start = start

        segment_stop = stop

    return segments


def embed_windows(
    audio: numpy.typing.NDArray, sr: int, vad: List[VADSegment], args: Args
) -> tuple[numpy.typing.NDArray, List[tuple[float, float]]]:
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"}
    )
    assert isinstance(classifier, EncoderClassifier)

    embeddings: list = []
    windows: List[tuple[float, float]] = []
    for seg in vad:
        for start in frange(seg.start, seg.end, args.emb_hop_s):
            end = min(seg.end, start + args.emb_window_s)
            if end - start >= args.ecapa_stability_floor:
                windows.append((start, end))
                start = max(0, int(start * sr))
                end = min(len(audio), int(end * sr))
                clip = audio[start:end].astype(numpy.float32)
                if clip.size == 0 or numpy.max(numpy.abs(clip)) == 0:
                    continue
                with torch.no_grad():
                    t = torch.from_numpy(clip).float().unsqueeze(0)
                e = classifier.encode_batch(t)
                embedding = e.squeeze().cpu().numpy().astype(numpy.float32)
                embeddings.append(embedding)

    if embeddings:
        return numpy.stack(embeddings), windows

    with torch.no_grad():
        null_tensor = torch.zeros(1, INPUT_SAMPLE_RATE_KHZ)
    null_embedding = classifier.encode_batch(null_tensor)
    embed_dimension_length = int(null_embedding.squeeze().shape[-1])
    null_window = numpy.zeros((0, embed_dimension_length), dtype=numpy.float32)
    return null_window, []


if __name__ == "__main__":
    assert torch.cuda.is_available()
    assert torch.version.cuda == "12.8"  # type: ignore

    args = parse_args()
    wav = preprocess_audio(args.input)
    audio, sample_rate = soundfile.read(wav)
    assert sample_rate == INPUT_SAMPLE_RATE_KHZ

    vad = voice_activation_detection(audio, sample_rate, args)
    embeddings, windows = embed_windows(audio, sample_rate, vad, args)
    print("=== EMBEDDINGS ===\n")
    pprint(embeddings)
    print("=== WINDOWS ===\n")
    pprint(windows)
