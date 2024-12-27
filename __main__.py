import sys
from os.path import exists
from pprint import pprint
from typing import Generator

import torch
from numpy import bool_
from pyannote.audio import Pipeline
from pyannote.core.annotation import Annotation
from pyannote.core.segment import Segment
from pydub import AudioSegment
from pydub.audio_segment import AudioSegment as Foobar
from whisper import load_model

from utils import get_chunks, read_lines

assert torch.cuda.is_available()
assert len(sys.argv) >= 2

filename = sys.argv[1]
lines = read_lines(filename)
chunks = get_chunks(lines)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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",
).to(torch.device("cuda"))

model = load_model("medium")
audio = AudioSegment.from_file(filename)
assert isinstance(audio, AudioSegment)
audio.export("_tmp/full.mp3", format="mp3")

print("\n\n")

for [start, end] in chunks:
    slice = audio[start:end]
    assert isinstance(slice, AudioSegment)

    file = f"_tmp/chunk_{start}.mp3"
    slice.export(file, format="mp3")
    assert exists(file)

    print(f"{start}, {end}: {file}")
    print("==============\n")

    transcription = model.transcribe(file)
    assert isinstance(transcription, dict)
    diarization = pipeline(file)
    assert isinstance(diarization, Annotation)

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
exit(0)
