# Phase 1 – Hello World Deployment

## Tools Used
- Git & GitHub
- Docker
- Flask
- Google Cloud VM

## Steps
1. Built Flask app
2. Dockerized application
3. Pushed image to Docker Hub
4. Deployed to GPU-enabled VM
5. Exposed via port 80

## Commands
```bash
docker build -t hello-flask .
docker run -p 5000:5000 hello-flask




# Whisper Paper – Summary

The Whisper paper presents an automatic speech recognition system that focuses on being reliable in real-world conditions instead of only performing well on clean, curated datasets. The main idea behind Whisper is that scale and data diversity matter more than complex, speech-specific modeling tricks.

Whisper is trained on around **680,000 hours of audio** collected from the internet. The data covers many languages, accents, and recording environments, and a lot of it is **weakly supervised**, meaning the transcripts are noisy (for example, subtitles that don’t perfectly match the audio). Instead of trying to clean this data, the model is trained directly on it. This forces Whisper to learn how to handle background noise, overlapping speech, pronunciation differences, and imperfect labels, which makes it much more robust in practice.

The model itself is fairly straightforward. Whisper uses a **Transformer encoder–decoder architecture**. The encoder takes log-Mel spectrograms from the audio, and the decoder generates text one token at a time. Special tokens are added to tell the model what to do, such as which language the audio is in, whether it should transcribe or translate, and whether to include timestamps. Because of this setup, the same model can handle transcription, translation, and language detection without needing separate systems.

One of the biggest reasons Whisper works so well is its **multilingual and multitask training**. By training on many languages and tasks at the same time, the model learns more general audio-text representations instead of overfitting to one language or dataset. The decoder also acts as a strong language model, which helps it recover text even when the audio is unclear, although this sometimes causes hallucinations when the model becomes overconfident.

The results in the paper show that Whisper is especially strong in noisy and low-resource settings. While it may not always beat specialized models on clean benchmarks, it performs consistently well across different accents, recording qualities, and languages.

Overall, the key takeaway from the paper is that **large-scale, diverse, weakly supervised data combined with a simple Transformer architecture can produce a very robust speech recognition system**, and that this approach generalizes better to real-world audio than traditional, heavily engineered ASR pipelines.

