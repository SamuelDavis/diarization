#!/bin/bash
set -e

if [ ! -d "$PWD/.venv/bin" ]; then
	python3 -m venv .venv
fi

source $PWD/.venv/bin/activate

if ! command -v deactivate >/dev/null 2>&1; then
  echo "Error: 'deactivate' command not found." >&2
  exit 1
fi

pip3 install \
  --index-url https://download.pytorch.org/whl/nightly/cu128 \
  --pre torch torchaudio 

pip3 install \
  git+https://github.com/openai/whisper.git \
  pyannote.audio \
  demucs

pip3 install faster-whisper webrtcvad soundfile pydub numpy scipy scikit-learn tqdm speechbrain
