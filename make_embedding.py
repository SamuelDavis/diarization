from os import environ
from pathlib import Path
from sys import argv

import numpy
import torch
from pyannote.audio import Audio, Model

token = environ.get("HUGGING_FACE_AUTH_TOKEN")
assert token

assert len(argv) == 2
_, input = argv

input = Path(input).resolve()
assert input.is_file()

model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
audio = Audio(sample_rate=16000)

waveform = audio(input)
embedding = model(waveform)
assert isinstance(embedding, torch.Tensor)

enrollment = embedding.detach().numpy().mean(axis=0)
assert isinstance(enrollment, numpy.ndarray)

numpy.save(input.with_suffix(".npy"), enrollment)
