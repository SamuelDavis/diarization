import re
import subprocess
from os import error


def cmd_exists(cmd):
    subprocess.check_call(f"command -v {cmd}", shell=True, stdout=subprocess.DEVNULL)


cmd_exists("ffmpeg")
cmd_exists("ffprobe")

cmd = " ".join(
    [
        "ffprobe",
        "-i input.mp3",
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
        "-i input.mp3",
        f"-ss {buffer}",
        f"-t {seconds}",
        "-ar 16000",
        "-ac 1",
        "-af loudnorm=I=-16:TP=-1.5:LRA=11",
        "-rf64 auto",
        "output.wav",
    ]
)

subprocess.check_call(cmd, shell=True)
