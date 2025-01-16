import re
import subprocess

from pyannote.core.annotation import Annotation
from pyannote.core.segment import Segment


def read_lines(filename: str) -> list[float]:
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        filename,
        "-af",
        "silencedetect=d=0.5",
        "-f",
        "null",
        "-",
    ]
    ffmpeg_result = subprocess.run(ffmpeg_cmd, capture_output=True)

    awk_cmd = ["awk", "/silencedetect/ {{print $4,$5}}"]
    awk_result = subprocess.run(
        awk_cmd, capture_output=True, input=ffmpeg_result.stderr
    )

    regex = re.compile("silence_(?:start|end): ([\\d.]+)")
    lines = awk_result.stdout.splitlines()
    lines = map(lambda line: line.decode(), lines)
    lines = map(lambda line: regex.findall(line), lines)
    lines = filter(lambda line: len(line) > 0, lines)
    lines = map(lambda line: float(line[0]), lines)
    lines = list(lines)

    if lines:
        regex = re.compile("Duration: (\\d+):(\\d+):([\\d\\.]+)")
        result = regex.findall(ffmpeg_result.stderr.decode())
        if result:
            [result] = result
            (hrs, mins, secs) = tuple(map(float, result))
            total = secs + mins * 60 + hrs * 60 * 60
            lines = [total]

    lines = map(lambda line: line * 1000, lines)

    return list(lines)


def get_chunks(lines: list[float], size=30) -> list[list[float]]:
    start = 0
    chunks: list[list[float]] = []

    for end in lines:
        delta = end - start
        if delta > size:
            chunks.append([start, end])
            start = end

    end = lines[-1]
    if end > start:
        chunks.append([start, end])

    return chunks


def print_results(transcription: dict, diarization: Annotation) -> None:
    for tuple in diarization.itertracks(yield_label=True):
        (segment, _, speaker) = (tuple + (None, None, None))[:3]
        assert isinstance(segment, Segment)
        assert isinstance(speaker, str)
        print(f"{segment.start}, {segment.end}: speaker {speaker}")

    segments = transcription.get("segments")
    assert isinstance(segments, list)
    for segment in segments:
        assert isinstance(segment, dict)
        start = segment.get("start")
        assert isinstance(start, float)
        end = segment.get("end")
        assert isinstance(end, float)
        text = segment.get("text")
        assert isinstance(text, str)
        print(f'{start}, {end}: "{text}"')
