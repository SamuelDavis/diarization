import csv
import subprocess
from json import dumps
from pathlib import Path

import numpy
import torch
from numpy import ndarray
from pyannote.audio import Audio, Model, Pipeline
from pyannote.core import Segment
from pyannote.core.annotation import Annotation, Label
from pyannote.core.segment import Segment
from whisper import load_model


def cmd_exists(cmd: str):
    subprocess.check_call(f"command -v {cmd}", shell=True, stdout=subprocess.DEVNULL)


def parse_args(args: list[str]) -> list[Path]:
    assert len(args) == 3
    _, input, output = args

    input = Path(input).resolve()
    assert input.is_file()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    return [input, output]


def preprocess(input: Path, output: Path) -> Path:
    cmd_exists("ffmpeg")
    cmd_exists("ffprobe")

    model = "htdemucs"
    output = output.with_suffix(".wav")

    cmd = " ".join(
        [
            f"demucs {input}",
            f"--name {model}",
            "--out /tmp",
            "--filename '{stem}.{ext}'",
        ]
    )
    subprocess.check_call(cmd, shell=True)
    tmp = f"/tmp/{model}/vocals.wav"

    cmd = " ".join(
        [
            "ffprobe",
            f"-i {tmp}",
            "-show_entries format=duration",
            "-v quiet",
            '-of csv="p=0"',
        ]
    )
    buffer = 2.55
    seconds = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
    seconds = float(seconds) - 2.55

    cmd = " ".join(
        [
            "ffmpeg",
            "-y",
            f"-i {tmp}",
            f"-ss {buffer}",
            f"-t {seconds}",
            "-ar 16000",
            "-ac 1",
            "-af loudnorm=I=-16:TP=-1.5:LRA=11,silenceremove=1:0:-45dB",
            "-rf64",
            "auto",
            str(output),
        ]
    )

    subprocess.check_call(cmd, shell=True)

    return output


def transcribe(input: Path) -> dict:
    output = input.with_suffix(".json")
    model = load_model("large")
    transcription = model.transcribe(str(input), fp16=True)
    assert isinstance(transcription, dict)

    with open(output, "w") as f:
        f.write(dumps(transcription))

    return transcription


def annotate(token: str, input: Path) -> Annotation:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=token
    ).to(torch.device("cuda"))

    annotation = pipeline({"audio": input})
    assert isinstance(annotation, Annotation)

    with open(input.with_suffix(".rttm"), "w") as f:
        annotation.write_rttm(f)

    return annotation


def align(
    input: Path, transcription: dict, annotation: Annotation
) -> list[list[float | str]]:
    result = []
    _speaker: str = ""
    _start: float = 0.0
    _end: float = 0.0
    _text: list[str] = []

    for segment in transcription["segments"]:
        assert isinstance(segment, dict)
        start, end, text = segment["start"], segment["end"], segment["text"]
        assert isinstance(start, float)
        assert isinstance(end, float)
        assert isinstance(text, str)
        speaker = annotation.crop(Segment(start, end)).argmax()
        assert isinstance(speaker, Label)
        speaker = str(speaker)

        if not _speaker:
            _speaker = speaker
            _start = start
            _text = []

        if speaker is _speaker:
            _text.append(text.strip())
        else:
            result.append([_start, _end, _speaker, " ".join(_text).strip()])
            _speaker = speaker
            _start = start
            _text = [text]
        _end = end

    with open(input.with_suffix(".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(result)

    return result


def enroll(
    speaker00: list[Path], speaker01: list[Path]
) -> dict[str, dict[str, list[ndarray]]]:
    speaker00_arr: list[ndarray] = list(map(numpy.load, speaker00))
    speaker01_arr: list[ndarray] = list(map(numpy.load, speaker01))

    return {
        "SPEAKER_00": {"embeddings": speaker00_arr},
        "SPEAKER_01": {"embeddings": speaker01_arr},
    }
