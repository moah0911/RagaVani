# RagaVani: Indian Classical Music Analysis & Synthesis

RagaVani is a comprehensive application for the analysis and synthesis of Indian Classical Music, addressing the unique challenges of this rich musical tradition through a hybrid approach combining neural, symbolic, and traditional signal processing techniques.

![RagaVani Logo](assets/logo.png)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Music Composer](#music-composer)
- [Deployment](#deployment)
- [Recent Improvements](#recent-improvements)
- [Future Directions](#future-directions)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

### Problem Statement

Indian Classical Music presents unique challenges for computational analysis and synthesis due to its:

- **Complex melodic structures** (ragas) with microtonal inflections and ornamentations
- **Intricate rhythmic patterns** (talas) with variable tempo and complex cycles
- **Emphasis on improvisation** rather than fixed compositions
- **Rich cultural and theoretical frameworks** that influence performance
- **Subtle ornamentations** (gamaka, meend, etc.) that are essential to the music's character

Existing music analysis tools often fail to capture these nuances, creating a need for specialized solutions that understand the unique characteristics of Indian Classical Music.

### Solution Approach

RagaVani addresses these challenges through a hybrid approach:

1. **Traditional Signal Processing**
   - Advanced pitch detection for microtonal variations
   - Rhythm analysis adapted for Indian talas
   - Specialized audio feature extraction

2. **Symbolic Processing**
   - Formal grammar rules for analyzing and generating melodic patterns
   - Symbolic representation of rhythmic cycles and variations
   - Structure analysis based on traditional forms

3. **Neural Models**
   - CNN models with attention mechanisms for raga identification
   - RNN/LSTM networks for complex rhythm pattern recognition
   - Specialized models for detecting and classifying ornaments
   - Transformer and latent diffusion models for generating authentic performances

## Features

### 🎵 Audio Analysis
- Raga identification from audio recordings
- Tala and tempo detection
- Ornament recognition and classification
- Pitch contour analysis with microtonal precision
- Performance assessment based on traditional rules

### 🎼 Music Theory
- Comprehensive database of ragas and talas
- Detailed information on raga characteristics and rules
- Visualization of melodic and rhythmic patterns
- Comparison between similar ragas and talas

### 🎹 Synthesis & Generation
- Generate melodic phrases following raga grammar
- Synthesize authentic-sounding performances
- Create tala patterns with appropriate stress patterns
- Compose new material in traditional styles

### 📊 Visualization
- Interactive pitch contour displays
- Raga scale and pattern visualizations
- Tala cycle representations
- Ornament and phrase highlighting

## Technical Architecture

### Core Modules

- **Audio Analysis**: Traditional DSP techniques for audio feature extraction
  - `audio_analysis.py`: Comprehensive audio analysis
  - `audio_analysis_simplified.py`: Lightweight analysis for faster processing
  - `audio_analysis_hybrid.py`: Combined traditional and neural approaches

- **Neural Processing**: Deep learning models for advanced analysis
  - `neural_processing.py`: Core neural network functionality
  - `neural_synthesis.py`: AI-based audio synthesis

- **Symbolic Processing**: Rule-based systems for music theory
  - `symbolic_processing.py`: Implementation of raga grammar and tala patterns
  - `raga_knowledge.py`: Database and rules for ragas
  - `tala_knowledge.py`: Database and rules for talas

- **Synthesis**: Audio generation capabilities
  - `audio_synthesis.py`: Techniques for generating audio based on symbolic representations
  - `music_composition.py`: Advanced composition models

- **Visualization**: Rich visual representations
  - `visualization.py`: Plotting and interactive visualization tools

### Data Resources

- `data/raga_grammar.json`: Formal grammar rules for ragas
- `data/tala_patterns.json`: Definitions of tala patterns and variations

### Neural Models

The application includes several neural models for analysis and synthesis:

#### Analysis Models
- Raga classifier (CNN with attention)
- Tala detector (RNN/LSTM)
- Ornament recognizer (CNN-LSTM hybrid)
- Pitch contour analyzer (Transformer-based)

#### Composition Models
- **Bi-LSTM Composer**: Generates melodic sequences following raga grammar
- **CNNGAN Composer**: Generates audio samples in the style of specific ragas
- **Hybrid Composer**: Combines both models for complete compositions

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/moah0911/RagaVani.git
cd RagaVani
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

### Analyzing an Audio Recording

```python
from modules.audio_analysis import analyze_audio
import librosa

# Load audio file
y, sr = librosa.load('your_recording.wav', sr=None)

# Analyze the audio
results = analyze_audio(y, sr)

# Access the results
print(f"Detected Raga: {results['raga']['name']}")
print(f"Confidence: {results['raga']['confidence']:.2f}")
print(f"Detected Tala: {results['tala']['name']}")
```

### Generating Melodic Phrases

```python
from modules.symbolic_processing import RagaGrammar
from modules.audio_synthesis import synthesize_phrase

# Initialize raga grammar
raga_grammar = RagaGrammar()

# Generate a phrase in Raga Yaman
phrase = raga_grammar.generate_phrase("Yaman", length=8)
print(f"Generated phrase: {phrase}")

# Synthesize the phrase
audio = synthesize_phrase(phrase, raga="Yaman")
```

## Music Composer

RagaVani includes a sophisticated Music Composer module that allows users to generate authentic Indian classical music compositions using advanced deep learning models.

### Composition Models

1. **Melodic Composer (Bi-LSTM)**
   - Generates melodic sequences that follow the grammar and structure of specific ragas
   - Captures characteristic phrases, note relationships, and traditional patterns
   - Provides evaluation metrics for the generated melodies

2. **Audio Composer (CNNGAN)**
   - Generates high-quality audio renditions with authentic timbral qualities
   - Captures micro-tonal nuances and ornamentations
   - Creates natural performance dynamics and expressions

3. **Hybrid Composer**
   - Combines the strengths of both models for complete compositions
   - Generates an authentic melodic structure using Bi-LSTM
   - Synthesizes high-quality audio with proper timbral qualities using CNNGAN
   - Applies appropriate ornamentations and micro-tonal nuances

### Using the Music Composer

The Music Composer can be accessed through the web interface or programmatically:

```python
from modules.music_composition import BiLSTMComposer, CNNGANComposer, HybridComposer

# Generate a melodic sequence
bilstm = BiLSTMComposer()
sequence = bilstm.generate_sequence(raga="Yaman", length=128, temperature=1.0)

# Generate audio
cnngan = CNNGANComposer()
audio, sr = cnngan.generate_audio(raga="Yaman", duration=30.0)

# Generate a complete composition
hybrid = HybridComposer()
audio, sr, sequence = hybrid.generate_composition(
    raga="Yaman",
    duration=30.0,
    temperature=1.0,
    melodic_complexity=0.7,
    ornamentation_level=0.6
)
```

## Deployment

RagaVani can be deployed on various platforms:

### Render.com Deployment

RagaVani is configured for deployment on Render.com with persistent data storage:

1. Fork or clone this repository
2. Connect your GitHub account to Render
3. Create a new Web Service pointing to your repository
4. Render will automatically use the configuration settings

#### Environment Variables for Render

- `PYTHON_VERSION`: 3.9.0
- `DEFAULT_USER_EMAIL`: (optional) Custom admin email
- `DEFAULT_USER_PASSWORD`: (optional) Custom admin password

#### User Data Persistence on Render

The application uses Render's persistent disk feature to store user data:

- User registrations are stored in `/data/user_database.json`
- This file persists between application restarts
- User data will be preserved even when the application is redeployed

### Snowflake Deployment

RagaVani can also be deployed on Snowflake using Snowpark Container Services:

1. Set up the Snowflake environment using the provided SQL script
2. Build and push the Docker image to Snowflake's registry
3. Deploy the service using Snowpark Container Services
4. Access the application through the provided service URL

For detailed instructions, see the [Snowflake Deployment Guide](SNOWFLAKE_DEPLOYMENT_GUIDE.md).

#### Requirements for Snowflake Deployment

- Snowflake account with access to Snowpark Container Services
- Snowflake account with ACCOUNTADMIN role or equivalent privileges
- Docker installed on your local machine
- Snowflake CLI installed on your local machine

#### Data Persistence on Snowflake

The application is configured to store data in a mounted volume:

- User data is stored in a persistent volume mounted at `/data`
- Data persists between service restarts and updates

## Recent Improvements

### Raga Detection Improvements

- Pre-filtering of candidate ragas based on dominant notes
- Improved note detection with proper Indian classical music note ranges
- Added microtone support for komal and tivra swaras
- Implemented weighted matching with frequency-based note importance
- Added caching mechanism for intermediate results
- Implemented dynamic thresholding based on audio length
- Added pakad (characteristic phrase) pattern matching

### Tala Detection Improvements

- Implemented multi-method approach for tala detection
- Added percussion enhancement with harmonic separation
- Improved onset detection with adaptive thresholding
- Implemented pattern consistency analysis
- Added tala-specific weighting for common patterns
- Optimized performance with reduced hop length
- Added compatibility improvements for different librosa versions

### Audio Segmentation for Multiple Sounds

- Added automatic detection of distinct sound segments
- Implemented per-segment analysis for more accurate results
- Added result aggregation with confidence-based selection
- Included segment metadata in results for reference

## Future Directions

- **Expanded Raga Database**: Include rare and regional ragas
- **Multi-Instrument Support**: Extend analysis to various Indian instruments
- **Performance Feedback**: Provide detailed feedback for students
- **Collaborative Features**: Allow musicians to share and analyze recordings
- **Mobile Application**: Develop companion mobile apps for on-the-go analysis
- **Machine Learning Integration**: Train models on larger labeled datasets
- **Parallel Processing**: Implement parallel processing for faster analysis
- **GPU Acceleration**: Move computationally intensive operations to GPU
- **Adaptive Algorithm Selection**: Dynamically choose the best algorithm based on audio characteristics
- **User Feedback Integration**: Allow users to provide feedback to improve future detections

## License

© MIT License | All Rights Reserved

## Acknowledgments

- Indian Classical Music experts who provided domain knowledge
- Open-source audio analysis libraries that formed the foundation
- Research papers on computational ethnomusicology that inspired approaches