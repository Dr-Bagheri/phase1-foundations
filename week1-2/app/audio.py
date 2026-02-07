import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt

# Load audio (mp3 or wav)
y, sr = librosa.load("sample.wav", sr=None)

print("Sample rate:", sr)
print("Audio length (seconds):", len(y) / sr)

# Compute MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

# Display MFCCs
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis="time")
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()

# Save first 3 seconds
clip = y[:3 * sr]
sf.write("clip_3_seconds.wav", clip, sr)
