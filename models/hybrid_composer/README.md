# Hybrid Composer for RagaVani

This model combines the strengths of both Bi-LSTM and CNNGAN architectures to create complete compositions in Indian classical music.

## Model Architecture

The Hybrid Composer leverages:
- **Bi-LSTM** for melodic structure and raga grammar
- **CNNGAN** for audio synthesis and timbral qualities

This combination allows for generating both the symbolic representation (notes) and the audio rendering of compositions.

## Features

- Generates complete compositions in specific ragas
- Produces both symbolic notation and audio output
- Respects the rules and grammar of ragas
- Captures the timbral qualities of Indian classical music
- Creates coherent musical phrases and structures

## Usage

```python
from modules.music_composition import HybridComposer

# Initialize the composer
composer = HybridComposer()

# Generate a composition for Raga Yaman
audio, sr, sequence = composer.generate_composition(
    raga="Yaman",
    duration=30.0,  # Duration in seconds
    seed="SRGM"     # Optional seed phrase
)

# Evaluate the composition
metrics = composer.evaluate_composition(audio, sr, sequence, "Yaman")
print(f"Overall quality: {metrics['overall_quality']:.2f}")

# Save the audio
import soundfile as sf
sf.write("yaman_composition.wav", audio, sr)

# Save the sequence
with open("yaman_composition.txt", "w") as f:
    f.write(sequence)
```

## Evaluation Metrics

The model provides comprehensive metrics to evaluate compositions:

- **Melodic Structure**: Quality of the note sequence
- **Audio Quality**: Quality of the synthesized audio
- **Raga Adherence**: How well the composition follows raga rules
- **Timbral Quality**: Richness and quality of the sound
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