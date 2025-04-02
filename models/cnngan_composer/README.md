# CNNGAN Composer for RagaVani

This model uses a Convolutional Neural Network Generative Adversarial Network (CNNGAN) to generate audio samples in the style of specific ragas.

## Model Architecture

The CNNGAN combines:
- A **Generator** network that creates new audio samples
- A **Discriminator** network that evaluates the authenticity of samples

This architecture is particularly good for capturing timbral qualities and micro-tonal nuances in Indian classical music.

## Features

- Generates realistic audio samples in the style of specific ragas
- Captures the timbral qualities of Indian classical instruments
- Preserves micro-tonal inflections and ornamentations
- Can generate samples of variable duration

## Usage

```python
from modules.music_composition import CNNGANComposer

# Initialize the composer
composer = CNNGANComposer()

# Generate audio for Raga Yaman
audio, sr = composer.generate_audio(
    raga="Yaman",
    duration=5.0  # Duration in seconds
)

# Evaluate the audio
metrics = composer.evaluate_audio(audio, sr, "Yaman")
print(f"Overall quality: {metrics['overall_quality']:.2f}")

# Play the audio
import soundfile as sf
sf.write("yaman_sample.wav", audio, sr)
```

## Evaluation Metrics

The model provides several metrics to evaluate generated audio:

- **Authenticity**: How well the audio matches the style of the raga (from discriminator)
- **Timbral Quality**: Richness and quality of the sound
- **Spectral Richness**: Presence of harmonics and spectral content
- **Overall Quality**: Combined score of all metrics

## Supported Ragas

The model currently supports the following ragas:
- Yaman
- Bhairav
- Bhimpalasi
- Darbari
- Khamaj
- Malkauns
- Bageshri
- Todi