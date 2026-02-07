import whisper
from pyannote.audio import Pipeline
import librosa
import torch
import os
# =========================
# CONFIG
# =========================
AUDIO_PATH = "challenging_audio.wav"
WHISPER_MODEL = "tiny"
HF_TOKEN = os.getenv("HF_TOKEN")

# =========================
# 1. WHISPER TRANSCRIPTION (word-level timestamps)
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")

print("Running Whisper transcription...")
whisper_result = whisper_model.transcribe(
    AUDIO_PATH,
    fp16=False,
    word_timestamps=True
)

# Collect all words
words = []
for segment in whisper_result["segments"]:
    if "words" in segment:
        for w in segment["words"]:
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"]
            })

# =========================
# 2. PYANNOTE DIARIZATION
# =========================
print("Loading audio for diarization...")
waveform, sample_rate = librosa.load(AUDIO_PATH, sr=16000, mono=True)
waveform = torch.tensor(waveform).unsqueeze(0)

audio = {
    "waveform": waveform,
    "sample_rate": sample_rate
}

print("Loading diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)

print("Running speaker diarization...")
diarization = pipeline(audio)

speaker_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    speaker_segments.append({
        "speaker": speaker,
        "start": turn.start,
        "end": turn.end
    })

# =========================
# 3. ALIGNMENT FUNCTION
# =========================
def assign_speaker(word_start, word_end, segments):
    for seg in segments:
        if word_start >= seg["start"] and word_end <= seg["end"]:
            return seg["speaker"]
    return "UNKNOWN"

# =========================
# 4. MERGE WORDS + SPEAKERS
# =========================
aligned_words = []

for w in words:
    speaker = assign_speaker(w["start"], w["end"], speaker_segments)
    aligned_words.append({
        "word": w["word"],
        "start": w["start"],
        "end": w["end"],
        "speaker": speaker
    })

# =========================
# 5. OUTPUT
# =========================
print("\n--- Speaker-Aligned Transcript ---\n")

for w in aligned_words:
    print(
        f"[{w['start']:.2f}-{w['end']:.2f}] "
        f"{w['speaker']}: {w['word']}"
    )

# Save to file
with open("speaker_aligned_transcript.txt", "w", encoding="utf-8") as f:
    for w in aligned_words:
        f.write(
            f"[{w['start']:.2f}-{w['end']:.2f}] "
            f"{w['speaker']}: {w['word']}\n"
        )

print("\nSaved to speaker_aligned_transcript.txt")
