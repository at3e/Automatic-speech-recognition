import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pretrained model and processor
model_name = "facebook/wav2vec2-base-960h"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

total_params = sum(p.numel() for p in model.parameters())
memory_bytes = total_params * 4
memory_gb = memory_bytes / (1024 ** 3)
print(f"Model size: {memory_gb:.2f} GB")


# Load and preprocess audio
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze()

# Run inference
def transcribe(file_path):
    audio = load_audio(file_path)[0, :]
    print(audio.shape)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
   
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Example usage
audio_path = "subjects/Digjoy_/1.wav"  # Replace with your local audio file path
text = transcribe(audio_path)
print("Transcription:", text)

