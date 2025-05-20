from os import environ
from sys import argv

import torch

from utils import align, annotate, parse_args, preprocess, transcribe

assert torch.cuda.is_available()
assert torch.version.cuda == "12.8"
token = environ.get("HUGGING_FACE_AUTH_TOKEN")
assert token

input, output = parse_args(argv)
input = preprocess(input, output)
transcription = transcribe(input)
annotation = annotate(token, input)
alignment = align(input, transcription, annotation)
