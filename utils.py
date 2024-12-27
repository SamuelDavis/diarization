import re
import subprocess
from pprint import pprint


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

    match = re.compile("silence_(?:start|end): ([\\d.]+)")
    lines = awk_result.stdout.splitlines()
    lines = map(lambda line: line.decode(), lines)
    lines = map(lambda line: match.findall(line), lines)
    lines = filter(lambda line: len(line) > 0, lines)
    lines = map(lambda line: float(line[0]) * 1000, lines)

    return list(lines)


def get_chunks(lines: list[float], size=30) -> list[list[float]]:
    start = 0
    chunks: list[list[float]] = []

    for i in range(1, len(lines), 2):
        end = lines[i]
        delta = end - start
        if delta > size:
            chunks.append([start, end])
            start = end

    return chunks
