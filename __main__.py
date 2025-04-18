import warnings
from os import environ
from os.path import exists
from pathlib import Path
from pprint import pp
from sys import argv

import torch
from pyannote.audio import Pipeline
from pyannote.audio.utils.reproducibility import ReproducibilityWarning
from pyannote.core.annotation import Annotation
from whisper import load_model

warnings.filterwarnings("ignore", category=ReproducibilityWarning)
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0.*")

assert torch.cuda.is_available()
token = environ.get("HUGGING_FACE_AUTH_TOKEN")
assert token

assert len(argv) == 3
input = str(Path(argv[1]).resolve())
assert isinstance(input, str)
assert exists(input)
output = str(Path(argv[2]).resolve())
assert isinstance(output, str)

model = load_model("large")
result = model.transcribe(input, fp16=True)
assert isinstance(result, dict)
pp(result)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=token,
).to(torch.device("cuda"))

diarization = pipeline(input, num_speakers=2)
assert isinstance(diarization, Annotation)

with open(output, "w") as f:
    diarization.write_rttm(f)
