# Bi-LSTM Composer for RagaVani

This model uses a Bidirectional LSTM architecture to generate melodic sequences following the grammar and structure of specific ragas.

## Model Architecture

The Bi-LSTM model processes sequence data in both forward and backward directions, making it excellent for capturing melodic patterns and long-term dependencies in music. It's particularly well-suited for learning the grammar and structure of ragas.

## Features

- Generates melodic sequences in Indian classical notation (S, R, G, M, P, D, N)
- Respects the rules and grammar of specific ragas
- Maintains melodic coherence and phrase structure
- Can be seeded with initial phrases to guide composition

## Usage

```python
from modules.music_composition import BiLSTMComposer

# Initialize the composer
composer = BiLSTMComposer()

# Generate a sequence for Raga Yaman
sequence = composer.generate_sequence(
    raga="Yaman",
    seed="SRGMPDNS",  # Optional seed phrase
    length=128,       # Length of the sequence
    temperature=1.0   # Controls randomness (higher = more random)
)

# Evaluate the sequence
metrics = composer.evaluate_sequence(sequence, "Yaman")
print(f"Overall quality: {metrics['overall_quality']:.2f}")
```

## Evaluation Metrics

The model provides several metrics to evaluate generated sequences:

- **Raga Adherence**: How well the sequence follows the rules of the raga
- **Melodic Complexity**: Variety and richness of the melodic content
- **Phrase Coherence**: Presence of meaningful musical phrases
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