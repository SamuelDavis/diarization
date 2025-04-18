import subprocess
from os.path import exists
from pathlib import Path
from sys import argv


def cmd_exists(cmd):
    subprocess.check_call(f"command -v {cmd}", shell=True, stdout=subprocess.DEVNULL)


cmd_exists("ffmpeg")
cmd_exists("ffprobe")


assert len(argv) == 3
input = str(Path(argv[1]).resolve())
assert isinstance(input, str)
assert exists(input)
output = str(Path(argv[2]).resolve())
assert isinstance(output, str)

model = "htdemucs"
cmd = " ".join(
    [
        f"demucs {input}",
        "--out /tmp",
        "--filename '{stem}.{ext}'",
    ]
)
subprocess.check_call(cmd, shell=True)
input = f"/tmp/{model}/vocals.wav"

cmd = " ".join(
    [
        "ffprobe",
        f"-i {input}",
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
        f"-i {input}",
        f"-ss {buffer}",
        f"-t {seconds}",
        "-ar 16000",
        "-ac 1",
        "-af loudnorm=I=-16:TP=-1.5:LRA=11",
        "-rf64 auto",
        output,
    ]
)

subprocess.check_call(cmd, shell=True)
