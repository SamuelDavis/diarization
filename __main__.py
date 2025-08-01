import argparse
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from os import environ
from pathlib import Path
from uuid import uuid4

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
from whisper import load_model


@dataclass
class Args:
    input: Path
    output: Path
    speakers: int
    token: str


@dataclass
class Transcript:
    start: float
    end: float
    text: str
    speaker: str


def cmd_exists(cmd: str) -> None:
    subprocess.run(
        f"command -v {cmd}", shell=True, stdout=subprocess.DEVNULL, check=True
    )


def validate_input(value: str) -> Path:
    path = Path(value)
    assert path.exists(), f"{path} does not exist"
    assert path.suffix == ".mp3", f"{path} is not an .mp3"
    return path


def validate_output(value: str) -> Path:
    path = Path(value)
    for suffix in ".csv", ".rttm", ".wav":
        filepath = path.with_suffix(suffix)
        assert not filepath.exists(), f"{filepath} already exists"
    return path


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=validate_input, help="the .mp3 file to be parsed")
    parser.add_argument(
        "output",
        type=validate_output,
        help="where to place the parsed .csv, .rttm, and .wav files",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=environ.get("HUGGING_FACE_AUTH_TOKEN"),
        help="hugging face auth token",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=2,
        help="The number of speakers expected in the parsed audio file.",
    )

    args = parser.parse_args()
    return Args(
        input=args.input, output=args.output, speakers=args.speakers, token=args.token
    )


def convert(input: Path) -> Path:
    output = Path(f"/tmp/{uuid4()}.wav")
    cmd = f"""
ffmpeg -y -i '{input}' \
    -acodec pcm_s16le \
    -ac 1 \
    -ar 48000 \
    {output}
"""
    subprocess.run(cmd, shell=True, check=True)
    return output


def denoise(input: Path) -> Path:
    output = Path(f"/tmp/{uuid4()}/htdemucs")
    cmd = f"""
demucs '{input.as_posix()}' \
    --two-stems vocals \
    --name '{output.name}' \
    --out '{output.parent}' \
    --filename '{{stem}}.{{ext}}'
"""
    subprocess.run(cmd, shell=True, check=True)
    return Path(f"{output}/vocals.wav")


def normalize(input: Path) -> Path:
    output = Path(f"/tmp/{uuid4()}.wav")
    cmd = f"""
ffmpeg -i '{input}' -af loudnorm=print_format=json -f null -
"""
    result = subprocess.run(
        cmd, shell=True, stderr=subprocess.PIPE, text=True, check=True
    )

    match = re.search(r"\{[\s\S]*?\}", result.stderr)
    assert match
    measurements = json.loads(match.group(0))

    apply_filter = (
        f"loudnorm=I=-16:TP=-1.5:LRA=11"
        f":measured_I={measurements['input_i']}"
        f":measured_TP={measurements['input_tp']}"
        f":measured_LRA={measurements['input_lra']}"
        f":measured_thresh={measurements['input_thresh']}"
        f":offset={measurements['target_offset']}"
        f":linear=true:print_format=summary"
    )
    cmd = f"""
ffmpeg -y -i '{input}' -af {apply_filter} {output}
"""
    subprocess.run(cmd, shell=True, check=True)

    return output


def transcribe(input: Path) -> list[dict]:
    transcription = load_model("large").transcribe(str(input), word_timestamps=True)
    segments = transcription["segments"]
    assert isinstance(segments, list)

    words: list[dict] = []
    for segment in segments:
        assert isinstance(segment, dict)
        for word in segment["words"]:
            assert isinstance(word, dict)
            words.append(word)
    return words


def annotate(token: str, speakers: int, input: Path) -> Annotation:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    ).to(torch.device("cuda"))

    annotation = pipeline(input, num_speakers=speakers)
    assert isinstance(annotation, Annotation)

    return annotation


def align(transcription: list[dict], annotation: Annotation) -> list[Transcript]:
    alignment: list[Transcript] = []

    _start: float = 0.0
    _end: float = 0.0
    _text: str = ""
    _speaker: str = ""

    for word in transcription:
        start, end, word = word["start"], word["end"], word["word"]
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(word, str)
        segment = Segment(start, end)
        speaker = str(annotation.crop(segment).argmax() or _speaker)

        if speaker == _speaker:
            _text = _text + word
            _end = end
        else:
            if _speaker:
                alignment.append(
                    Transcript(start=_start, end=_end, text=_text, speaker=_speaker)
                )

            _speaker = speaker
            _text = word
            _start = start
            _end = end

    if _speaker:
        alignment.append(
            Transcript(start=_start, end=_end, text=_text, speaker=_speaker)
        )

    return alignment


if __name__ == "__main__":
    assert torch.cuda.is_available()
    assert torch.version.cuda == "12.8"  # type: ignore
    token = environ.get("HUGGING_FACE_AUTH_TOKEN")
    assert token

    cmd_exists("ffmpeg")
    cmd_exists("demucs")

    args = parse_args()
    converted = convert(args.input)
    denoised = denoise(converted)
    normalized = normalize(denoised)
    annotation = annotate(args.token, args.speakers, normalized)
    transcription = transcribe(normalized)
    alignment = align(transcription, annotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(normalized, args.output.with_suffix(".wav"))
    with open(args.output.with_suffix(".rttm"), "w") as f:
        annotation.write_rttm(f)
    with open(args.output.with_suffix(".csv"), "w") as f:
        writer = csv.writer(f)
        for t in alignment:
            row = [t.start, t.end, t.speaker, t.text]
            writer.writerow(row)
