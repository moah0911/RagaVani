# RagaVani Neural Models

This directory will contain trained neural models for Indian classical music analysis.

## Model Types

### 1. Raga Classifier
- **Purpose**: Identify ragas from audio recordings
- **Architecture**: Convolutional Neural Network (CNN) with attention mechanism
- **Input**: Mel spectrogram and chroma features
- **Output**: Raga classification with confidence scores

### 2. Tala Detector
- **Purpose**: Detect tala patterns and tempo from audio recordings
- **Architecture**: Recurrent Neural Network (RNN) with LSTM layers
- **Input**: Onset strength envelope and beat features
- **Output**: Tala classification, tempo, and beat positions

### 3. Ornament Recognizer
- **Purpose**: Identify and classify ornaments (gamaka, meend, etc.)
- **Architecture**: CNN-LSTM hybrid
- **Input**: Pitch contour and spectral features
- **Output**: Ornament type, position, and duration

### 4. Pitch Contour Analyzer
- **Purpose**: Analyze pitch movements and microtonal inflections
- **Architecture**: Transformer-based model
- **Input**: Raw pitch data
- **Output**: Refined pitch contour with ornament annotations

## Model Format

Models are saved in TensorFlow SavedModel format and can be loaded using:

```python
import tensorflow as tf
model = tf.keras.models.load_model('models/raga_classifier')
```

## Training Data

Models are trained on a combination of:
- Publicly available Indian classical music recordings
- Synthetic data generated using symbolic processing
- Expert-annotated recordings for supervised learning

## Usage

These models are automatically loaded by the neural processing module when analyzing audio. If models are not present, the system will fall back to traditional signal processing methods.

## Future Improvements

- Fine-tuning on user-provided recordings
- Transfer learning from larger music understanding models
- Multi-task learning across different aspects of Indian classical music
- Incorporation of cultural and contextual knowledge