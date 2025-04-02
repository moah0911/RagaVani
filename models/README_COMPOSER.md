# Music Composition Models for RagaVani

This directory contains models for generating Indian classical music compositions using deep learning techniques.

## Model Types

### 1. Bi-LSTM Composer
- **Purpose**: Generate melodic sequences following the grammar and structure of specific ragas
- **Architecture**: Bidirectional LSTM network
- **Input**: Seed sequence and raga parameters
- **Output**: Melodic sequence in Indian classical notation

### 2. CNNGAN Composer
- **Purpose**: Generate audio samples in the style of specific ragas
- **Architecture**: Convolutional Neural Network Generative Adversarial Network
- **Input**: Latent vector and raga parameters
- **Output**: Audio waveform

### 3. Hybrid Composer
- **Purpose**: Combine the strengths of both models for complete compositions
- **Architecture**: Bi-LSTM for melodic structure + CNNGAN for audio synthesis
- **Input**: Raga parameters and optional seed sequence
- **Output**: Both symbolic notation and audio rendering

## Setup Instructions

1. Ensure you have the required dependencies:
   ```
   pip install tensorflow librosa numpy scipy matplotlib
   ```

2. The models are automatically loaded by the music_composition module when needed.

3. If the models are not available, the system will fall back to rule-based generation.

## Usage

The models can be accessed through the Music Composer page in the RagaVani application, or programmatically:

```python
from modules.music_composition import BiLSTMComposer, CNNGANComposer, HybridComposer

# Generate a melodic sequence
bilstm = BiLSTMComposer()
sequence = bilstm.generate_sequence(raga="Yaman", length=128)

# Generate audio
cnngan = CNNGANComposer()
audio, sr = cnngan.generate_audio(raga="Yaman", duration=30.0)

# Generate a complete composition
hybrid = HybridComposer()
audio, sr, sequence = hybrid.generate_composition(raga="Yaman", duration=30.0)
```

## Evaluation

Each model provides evaluation metrics for the generated content:

- **Bi-LSTM**: Raga adherence, melodic complexity, phrase coherence
- **CNNGAN**: Authenticity, timbral quality, spectral richness
- **Hybrid**: Combined metrics from both models

## Supported Ragas

The models currently support the following ragas:
- Yaman
- Bhairav
- Bhimpalasi
- Darbari
- Khamaj
- Malkauns
- Bageshri
- Todi

## Future Improvements

- Fine-tuning on larger datasets of Indian classical music
- Adding more ragas and instruments
- Improving the audio quality of the CNNGAN model
- Incorporating more sophisticated evaluation metrics
- Adding user feedback for model improvement