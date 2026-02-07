
from pyannote.audio import Pipeline
import librosa
import torch
import os

HF_TOKEN = os.getenv("HF_TOKEN")



waveform, sample_rate = librosa.load(
    "challenging_audio.wav", sr=16000, mono=True
)
waveform = torch.tensor(waveform).unsqueeze(0)

audio = {
    "waveform": waveform,
    "sample_rate": sample_rate
}

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

diarization = pipeline(audio)

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Speaker {speaker}: {turn.start:.2f}s -> {turn.end:.2f}s")