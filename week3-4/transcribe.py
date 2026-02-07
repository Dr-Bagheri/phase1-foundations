import argparse
import time
import whisper

def main():
    parser = argparse.ArgumentParser(description="Whisper CLI Transcription Tool")
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size"
    )

    args = parser.parse_args()

    audio_path = "challenging_audio.mp3"

    model = whisper.load_model(args.model, device="cpu")

    start_time = time.time()
    result = model.transcribe(audio_path, fp16=False)
    end_time = time.time()

    print("\n--- Transcription ---\n")
    print(result["text"])
    print("\n---------------------")
    print(f"Model used: {args.model}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()