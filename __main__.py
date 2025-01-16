import sys
from os import environ
from os.path import exists

import torch
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from pydub import AudioSegment
from whisper import load_model

from utils import print_results

assert torch.cuda.is_available()

filename = sys.argv[1] if len(sys.argv) > 0 else None
assert filename and exists(filename)

token = environ.get("HUGGING_FACE_AUTH_TOKEN")
assert token

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
).to(torch.device("cuda"))


model = load_model("medium")
audio = AudioSegment.from_file(filename)
assert isinstance(audio, AudioSegment)
audio.export("_tmp/full.mp3", format="mp3")

transcription = model.transcribe(filename)
assert isinstance(transcription, dict)
diarization = pipeline(filename, num_speakers=2)
assert isinstance(diarization, Annotation)

print_results(transcription, diarization)
