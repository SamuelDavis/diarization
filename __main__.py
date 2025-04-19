import warnings
from os import environ
from sys import argv

import torch
from pyannote.audio.utils.reproducibility import ReproducibilityWarning

from utils import align, annotate, parse_args, preprocess, transcribe

assert torch.cuda.is_available()
token = environ.get("HUGGING_FACE_AUTH_TOKEN")
assert token

input, output = parse_args(argv)
input = preprocess(input, output)
transcription = transcribe(input)
annotation = annotate(input, token)
alignment = align(input, transcription, annotation)
