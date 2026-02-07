import whisper
from pyannote.audio import Pipeline
import librosa
import torch
from collections import defaultdict
import os

# =========================
# CONFIG (edit once)
# =========================
WHISPER_MODEL = "tiny"


HF_TOKEN = os.getenv("HF_TOKEN")

# Load models once (important for reuse)
_whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")

_diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HF_TOKEN
)


def transcribe_and_diarize(audio_path: str) -> str:
    """
    Transcribe an audio file and return a speaker-labeled transcript.

    Args:
        audio_path (str): path to audio file

    Returns:
        str: speaker-labeled transcript
    """

    # -------------------------
    # 1. Whisper transcription
    # -------------------------
    whisper_result = _whisper_model.transcribe(
        audio_path,
        fp16=False,
        word_timestamps=True
    )

    words = []
    for segment in whisper_result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"].strip(),
                "start": w["start"],
                "end": w["end"]
            })

    # -------------------------
    # 2. Speaker diarization
    # -------------------------
    waveform, sample_rate = librosa.load(
        audio_path, sr=16000, mono=True
    )
    waveform = torch.tensor(waveform).unsqueeze(0)

    audio = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    diarization = _diarization_pipeline(audio)

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end
        })

    # -------------------------
    # 3. Align words to speakers
    # -------------------------
    def get_speaker(word_start, word_end):
        for seg in speaker_segments:
            if word_start >= seg["start"] and word_end <= seg["end"]:
                return seg["speaker"]
        return "UNKNOWN"

    aligned = []
    for w in words:
        aligned.append({
            "speaker": get_speaker(w["start"], w["end"]),
            "word": w["word"]
        })

    # -------------------------
    # 4. Group words by speaker
    # -------------------------
    speaker_text = defaultdict(list)

    for item in aligned:
        speaker_text[item["speaker"]].append(item["word"])

    # -------------------------
    # 5. Build clean output
    # -------------------------
    output_lines = []
    speaker_map = {}

    speaker_counter = 0
    for speaker, words in speaker_text.items():
        if speaker not in speaker_map:
            speaker_map[speaker] = chr(ord("A") + speaker_counter)
            speaker_counter += 1

        label = speaker_map[speaker]
        sentence = " ".join(words)
        output_lines.append(f"Speaker {label}: {sentence}")

    return "\n".join(output_lines)
