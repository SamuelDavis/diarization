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
from uuid import uuid4

import numpy
import soundfile
import webrtcvad

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
# Short windows help capture quick timbre cues during fast exchanges.
EMB_WINDOW_S = 1.5  # 1.5 s windows: long enough for stable ECAPA embedding; short for frequent turns.
EMB_HOP_S = 0.75  # 50% overlap improves robustness without exploding compute.

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


@dataclass
class VADSegment:
    start: float
    end: float
    voiced: bool


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


def voice_activation_detection(input: Path, args: Args) -> List[VADSegment]:
    audio, sample_rate = soundfile.read(input)
    assert sample_rate == INPUT_SAMPLE_RATE_KHZ
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


if __name__ == "__main__":
    args = parse_args()
    wav = preprocess_audio(args.input)
    vad = voice_activation_detection(wav, args)
    pprint(vad)
