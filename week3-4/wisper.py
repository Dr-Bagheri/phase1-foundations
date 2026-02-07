import whisper

audio_path = "challenging_audio.mp3"

model = whisper.load_model("tiny", device="cpu")

result = model.transcribe(
    audio_path,
    fp16=False
)

print(result["text"])