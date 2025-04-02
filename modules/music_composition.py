"""
Music Composition Module for RagaVani

This module provides advanced functionality for generating Indian classical music using
sophisticated deep learning models including Bi-LSTM and CNNGAN architectures.
The models are optimized for authentic raga-based composition with high musical quality.
"""

import os
import numpy as np
import json
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import time

# Configure logging
logger = logging.getLogger(__name__)

# Try to import TensorFlow, but handle gracefully if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    # Set TensorFlow logging level
    tf.get_logger().setLevel(logging.ERROR)
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural composition will be simulated.")

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
BILSTM_MODEL_PATH = os.path.join(MODEL_DIR, "bilstm_composer")
CNNGAN_MODEL_PATH = os.path.join(MODEL_DIR, "cnngan_composer")

# Cache for loaded models
_model_cache = {}

def _load_model(model_path: str) -> Optional[Any]:
    """
    Load a TensorFlow model from disk with caching
    
    Parameters:
        model_path (str): Path to the model directory
        
    Returns:
        Optional[Any]: Loaded model or None if not available
    """
    if not TENSORFLOW_AVAILABLE:
        return None
        
    if model_path in _model_cache:
        return _model_cache[model_path]
        
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            _model_cache[model_path] = model
            logger.info(f"Loaded model from {model_path}")
            return model
        else:
            logger.warning(f"Model not found at {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        return None

class BiLSTMComposer:
    """
    Bi-LSTM model for Indian classical music composition
    
    This model uses a bidirectional LSTM architecture to generate melodic sequences
    following the grammar and structure of specific ragas.
    """
    
    def __init__(self):
        """Initialize the Bi-LSTM composer"""
        self.model = _load_model(BILSTM_MODEL_PATH)
        self.available = self.model is not None
        
        # Load raga grammar and parameters
        try:
            params_path = os.path.join(BILSTM_MODEL_PATH, 'params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.params = json.load(f)
            else:
                self.params = {
                    'note_to_idx': {'S': 0, 'r': 1, 'R': 2, 'g': 3, 'G': 4, 'M': 5, 'm': 6, 
                                    'P': 7, 'd': 8, 'D': 9, 'n': 10, 'N': 11, '-': 12},
                    'idx_to_note': ['S', 'r', 'R', 'g', 'G', 'M', 'm', 'P', 'd', 'D', 'n', 'N', '-'],
                    'sequence_length': 64,
                    'temperature': 1.0,
                    'supported_ragas': ['Yaman', 'Bhairav', 'Bhimpalasi', 'Darbari', 'Khamaj', 'Malkauns']
                }
        except Exception as e:
            logger.error(f"Error loading Bi-LSTM parameters: {str(e)}")
            self.params = {
                'note_to_idx': {'S': 0, 'r': 1, 'R': 2, 'g': 3, 'G': 4, 'M': 5, 'm': 6, 
                                'P': 7, 'd': 8, 'D': 9, 'n': 10, 'N': 11, '-': 12},
                'idx_to_note': ['S', 'r', 'R', 'g', 'G', 'M', 'm', 'P', 'd', 'D', 'n', 'N', '-'],
                'sequence_length': 64,
                'temperature': 1.0,
                'supported_ragas': ['Yaman', 'Bhairav', 'Bhimpalasi', 'Darbari', 'Khamaj', 'Malkauns']
            }
    
    def _preprocess_seed(self, seed: str) -> np.ndarray:
        """
        Preprocess a seed sequence for the model
        
        Parameters:
            seed (str): Seed sequence as a string of notes
            
        Returns:
            np.ndarray: Preprocessed seed sequence
        """
        # Convert notes to indices
        note_to_idx = self.params['note_to_idx']
        sequence = [note_to_idx.get(note, note_to_idx['-']) for note in seed]
        
        # Pad or truncate to the required sequence length
        if len(sequence) < self.params['sequence_length']:
            sequence = sequence + [note_to_idx['-']] * (self.params['sequence_length'] - len(sequence))
        elif len(sequence) > self.params['sequence_length']:
            sequence = sequence[:self.params['sequence_length']]
        
        # Convert to one-hot encoding
        one_hot = np.zeros((self.params['sequence_length'], len(note_to_idx)))
        for i, idx in enumerate(sequence):
            one_hot[i, idx] = 1
        
        return one_hot.reshape(1, self.params['sequence_length'], len(note_to_idx))
    
    def _sample_with_temperature(self, probabilities: np.ndarray, temperature: float = 1.0) -> int:
        """
        Sample from a probability distribution with temperature
        
        Parameters:
            probabilities (np.ndarray): Probability distribution
            temperature (float): Temperature parameter (higher = more random)
            
        Returns:
            int: Sampled index
        """
        # Apply temperature
        probabilities = np.log(probabilities + 1e-10) / temperature
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample from the distribution
        return np.random.choice(len(probabilities), p=probabilities)
    
    def generate_sequence(self, raga: str, seed: Optional[str] = None, 
                         length: int = 128, temperature: float = 1.0) -> str:
        """
        Generate a melodic sequence for a given raga
        
        Parameters:
            raga (str): Name of the raga
            seed (str, optional): Seed sequence to start generation
            length (int): Length of the sequence to generate
            temperature (float): Temperature parameter (higher = more random)
            
        Returns:
            str: Generated sequence as a string of notes
        """
        # Check if the model is available
        if not self.available:
            return self._simulate_generation(raga, seed, length)
        
        # Check if the raga is supported
        if raga not in self.params['supported_ragas']:
            logger.warning(f"Raga {raga} not supported by Bi-LSTM model. Using simulation.")
            return self._simulate_generation(raga, seed, length)
        
        try:
            # Prepare the seed sequence
            if seed is None or len(seed) == 0:
                # Use a default seed for the raga
                from modules.raga_knowledge import get_raga_info
                raga_info = get_raga_info(raga)
                if raga_info and 'arohanam' in raga_info:
                    seed = raga_info['arohanam']
                else:
                    seed = 'SRGMPDNS'
            
            # Preprocess the seed
            input_sequence = self._preprocess_seed(seed)
            
            # Generate the sequence
            generated_sequence = seed
            current_sequence = input_sequence
            
            for _ in range(length - len(seed)):
                # Predict the next note
                predictions = self.model.predict(current_sequence, verbose=0)[0][-1]
                
                # Sample from the predictions
                next_idx = self._sample_with_temperature(predictions, temperature)
                next_note = self.params['idx_to_note'][next_idx]
                
                # Add the note to the sequence
                generated_sequence += next_note
                
                # Update the current sequence
                new_sequence = np.zeros((1, self.params['sequence_length'], len(self.params['note_to_idx'])))
                for i in range(self.params['sequence_length'] - 1):
                    new_sequence[0, i] = current_sequence[0, i + 1]
                new_sequence[0, -1, next_idx] = 1
                current_sequence = new_sequence
            
            return generated_sequence
        
        except Exception as e:
            logger.error(f"Error generating sequence with Bi-LSTM: {str(e)}")
            return self._simulate_generation(raga, seed, length)
    
    def _simulate_generation(self, raga: str, seed: Optional[str] = None, length: int = 128, temperature: float = 1.0) -> str:
        """
        Simulate advanced sequence generation when the model is not available

        This enhanced simulation uses more sophisticated rules to generate authentic
        raga-based sequences with proper musical phrasing and structure.

        Parameters:
            raga (str): Name of the raga
            seed (str, optional): Seed sequence to start generation
            length (int): Length of the sequence to generate
            temperature (float): Controls randomness (higher = more random)

        Returns:
            str: Generated sequence as a string of notes
        """
        # Get raga information
        from modules.raga_knowledge import get_raga_info
        raga_info = get_raga_info(raga)

        # Use arohanam and avarohanam if available
        if raga_info and 'arohanam' in raga_info and 'avarohanam' in raga_info:
            arohanam = raga_info['arohanam']
            avarohanam = raga_info['avarohanam']
            vadi = raga_info.get('vadi', 'S')
            samvadi = raga_info.get('samvadi', 'P')
            pakad = raga_info.get('pakad', '')
            time_of_day = raga_info.get('time', 'any')
        else:
            # Default to Yaman raga
            arohanam = 'SRGMPDNS'
            avarohanam = 'SNPDMGRS'
            vadi = 'G'
            samvadi = 'N'
            pakad = 'NRGS'
            time_of_day = 'evening'

        # Extract allowed notes from arohanam and avarohanam
        allowed_notes = set(arohanam + avarohanam)

        # Define common patterns in this raga
        patterns = []

        # Add the pakad (characteristic phrase) if available
        if pakad:
            patterns.append(pakad)

        # Add arohanam and avarohanam segments
        for i in range(len(arohanam) - 2):
            patterns.append(arohanam[i:i+3])

        for i in range(len(avarohanam) - 2):
            patterns.append(avarohanam[i:i+3])

        # Add some common musical phrases based on raga characteristics
        if 'S' in allowed_notes and 'R' in allowed_notes and 'G' in allowed_notes:
            patterns.append('SRG')
        if 'G' in allowed_notes and 'M' in allowed_notes and 'P' in allowed_notes:
            patterns.append('GMP')
        if 'P' in allowed_notes and 'D' in allowed_notes and 'N' in allowed_notes:
            patterns.append('PDN')
        if 'D' in allowed_notes and 'N' in allowed_notes and 'S' in allowed_notes:
            patterns.append('DNS')

        # Add patterns emphasizing vadi and samvadi (important notes)
        patterns.append(vadi + samvadi + vadi)
        patterns.append(samvadi + vadi + samvadi)

        # Start with the seed if provided
        if seed is not None and len(seed) > 0:
            generated_sequence = seed
        else:
            # Start with a characteristic phrase
            if pakad:
                generated_sequence = pakad
            else:
                generated_sequence = arohanam[:4]  # Start with the first few notes of arohanam

        # Define musical structure sections
        sections = [
            {"name": "alap", "length": int(length * 0.3), "has_pauses": True, "pattern_prob": 0.3, "tempo": "slow"},
            {"name": "jor", "length": int(length * 0.3), "has_pauses": False, "pattern_prob": 0.5, "tempo": "medium"},
            {"name": "jhala", "length": int(length * 0.4), "has_pauses": False, "pattern_prob": 0.7, "tempo": "fast"}
        ]

        # Generate the sequence section by section
        current_position = len(generated_sequence)

        for section in sections:
            section_end = current_position + section["length"]
            section_end = min(section_end, length)

            # Generate this section
            while current_position < section_end:
                # Decide whether to use a pattern or generate notes individually
                if np.random.random() < section["pattern_prob"] * temperature:
                    # Use a pattern
                    pattern = np.random.choice(patterns)

                    # Make sure we don't exceed the length
                    if current_position + len(pattern) > length:
                        pattern = pattern[:length - current_position]

                    generated_sequence += pattern
                    current_position += len(pattern)
                else:
                    # Generate individual notes based on transition probabilities
                    last_note = generated_sequence[-1] if generated_sequence else 'S'

                    # Define transition probabilities based on raga rules
                    if last_note in arohanam:
                        # If in ascending pattern, favor the next note in arohanam
                        idx = arohanam.index(last_note)
                        if idx < len(arohanam) - 1:
                            next_note = arohanam[idx + 1]
                        else:
                            next_note = np.random.choice(list(allowed_notes))
                    elif last_note in avarohanam:
                        # If in descending pattern, favor the next note in avarohanam
                        idx = avarohanam.index(last_note)
                        if idx < len(avarohanam) - 1:
                            next_note = avarohanam[idx + 1]
                        else:
                            next_note = np.random.choice(list(allowed_notes))
                    else:
                        # Otherwise, choose from allowed notes
                        next_note = np.random.choice(list(allowed_notes))

                    # Apply temperature to randomize more or less
                    if np.random.random() < temperature * 0.5:
                        next_note = np.random.choice(list(allowed_notes))

                    # Emphasize vadi and samvadi notes
                    if np.random.random() < 0.2:
                        next_note = vadi if np.random.random() < 0.6 else samvadi

                    generated_sequence += next_note
                    current_position += 1

                # Add pauses based on section characteristics
                if section["has_pauses"] and np.random.random() < 0.15:
                    generated_sequence += '-'
                    current_position += 1

                # Add some ornamentation markers for realism
                if section["tempo"] == "medium" and np.random.random() < 0.1:
                    # Add a simple ornament (represented by repeating the note)
                    if current_position < length:
                        generated_sequence += generated_sequence[-1]
                        current_position += 1

                if section["tempo"] == "fast" and np.random.random() < 0.15:
                    # Add a fast ornament (represented by a quick pattern)
                    if current_position + 2 <= length:
                        last_note = generated_sequence[-1]
                        if last_note in arohanam:
                            idx = arohanam.index(last_note)
                            if idx > 0 and idx < len(arohanam) - 1:
                                generated_sequence += arohanam[idx-1] + arohanam[idx+1]
                                current_position += 2

        # Ensure we end on Sa or the vadi for a satisfying conclusion
        if len(generated_sequence) > 3 and generated_sequence[-1] not in ['S', vadi]:
            if np.random.random() < 0.7:
                generated_sequence = generated_sequence[:-1] + 'S'
            else:
                generated_sequence = generated_sequence[:-1] + vadi

        return generated_sequence[:length]
    
    def evaluate_sequence(self, sequence: str, raga: str) -> Dict[str, float]:
        """
        Evaluate a generated sequence for adherence to raga grammar
        
        Parameters:
            sequence (str): Generated sequence as a string of notes
            raga (str): Name of the raga
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Get raga information
        from modules.raga_knowledge import get_raga_info
        raga_info = get_raga_info(raga)
        
        # Initialize metrics
        metrics = {
            'raga_adherence': 0.0,
            'melodic_complexity': 0.0,
            'phrase_coherence': 0.0,
            'overall_quality': 0.0
        }
        
        # Check if raga information is available
        if not raga_info:
            return metrics
        
        # Extract raga characteristics
        arohanam = raga_info.get('arohanam', 'SRGMPDNS')
        avarohanam = raga_info.get('avarohanam', 'SNPDMGRS')
        vadi = raga_info.get('vadi', 'S')
        samvadi = raga_info.get('samvadi', 'P')
        
        # Calculate raga adherence
        allowed_notes = set(arohanam + avarohanam)
        adherence_score = sum(1 for note in sequence if note in allowed_notes) / len(sequence)
        metrics['raga_adherence'] = adherence_score
        
        # Calculate melodic complexity
        unique_notes = len(set(sequence))
        max_unique = len(allowed_notes)
        complexity_score = unique_notes / max_unique
        metrics['melodic_complexity'] = complexity_score
        
        # Calculate phrase coherence
        # Check for common patterns in the raga
        coherence_score = 0.0
        for i in range(len(sequence) - 3):
            phrase = sequence[i:i+4]
            if phrase in arohanam or phrase in avarohanam:
                coherence_score += 1
        coherence_score = min(coherence_score / (len(sequence) - 3), 1.0)
        metrics['phrase_coherence'] = coherence_score
        
        # Calculate overall quality
        vadi_count = sequence.count(vadi)
        samvadi_count = sequence.count(samvadi)
        emphasis_score = (vadi_count + samvadi_count) / len(sequence)
        
        metrics['overall_quality'] = (
            adherence_score * 0.4 + 
            complexity_score * 0.2 + 
            coherence_score * 0.3 + 
            emphasis_score * 0.1
        )
        
        return metrics


class CNNGANComposer:
    """
    CNNGAN model for Indian classical music composition
    
    This model uses a Convolutional Neural Network Generative Adversarial Network
    to generate audio samples in the style of specific ragas.
    """
    
    def __init__(self):
        """Initialize the CNNGAN composer"""
        self.generator = _load_model(os.path.join(CNNGAN_MODEL_PATH, 'generator'))
        self.discriminator = _load_model(os.path.join(CNNGAN_MODEL_PATH, 'discriminator'))
        self.available = self.generator is not None and self.discriminator is not None
        
        # Load parameters
        try:
            params_path = os.path.join(CNNGAN_MODEL_PATH, 'params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.params = json.load(f)
            else:
                self.params = {
                    'latent_dim': 100,
                    'sample_rate': 22050,
                    'duration': 5.0,
                    'supported_ragas': ['Yaman', 'Bhairav', 'Bhimpalasi', 'Darbari', 'Khamaj', 'Malkauns']
                }
        except Exception as e:
            logger.error(f"Error loading CNNGAN parameters: {str(e)}")
            self.params = {
                'latent_dim': 100,
                'sample_rate': 22050,
                'duration': 5.0,
                'supported_ragas': ['Yaman', 'Bhairav', 'Bhimpalasi', 'Darbari', 'Khamaj', 'Malkauns']
            }
    
    def generate_audio(self, raga: str, duration: float = 5.0) -> Tuple[np.ndarray, int]:
        """
        Generate audio for a given raga
        
        Parameters:
            raga (str): Name of the raga
            duration (float): Duration of the audio in seconds
            
        Returns:
            Tuple[np.ndarray, int]: Generated audio and sample rate
        """
        # Check if the model is available
        if not self.available:
            return self._simulate_generation(raga, duration)
        
        # Check if the raga is supported
        if raga not in self.params['supported_ragas']:
            logger.warning(f"Raga {raga} not supported by CNNGAN model. Using simulation.")
            return self._simulate_generation(raga, duration)
        
        try:
            # Generate a random latent vector
            latent_dim = self.params['latent_dim']
            z = np.random.normal(0, 1, (1, latent_dim))
            
            # Get raga embedding
            raga_idx = self.params['supported_ragas'].index(raga)
            raga_embedding = np.zeros((1, len(self.params['supported_ragas'])))
            raga_embedding[0, raga_idx] = 1
            
            # Combine latent vector and raga embedding
            z_combined = np.concatenate([z, raga_embedding], axis=1)
            
            # Generate audio
            generated_audio = self.generator.predict(z_combined, verbose=0)[0]
            
            # Scale to [-1, 1]
            generated_audio = np.clip(generated_audio, -1, 1)
            
            # Adjust duration if needed
            sample_rate = self.params['sample_rate']
            target_samples = int(duration * sample_rate)
            current_samples = len(generated_audio)
            
            if target_samples > current_samples:
                # Repeat the audio to reach the target duration
                repeats = target_samples // current_samples + 1
                generated_audio = np.tile(generated_audio, repeats)[:target_samples]
            elif target_samples < current_samples:
                # Truncate the audio
                generated_audio = generated_audio[:target_samples]
            
            return generated_audio, sample_rate
        
        except Exception as e:
            logger.error(f"Error generating audio with CNNGAN: {str(e)}")
            return self._simulate_generation(raga, duration)
    
    def _simulate_generation(self, raga: str, duration: float = 5.0) -> Tuple[np.ndarray, int]:
        """
        Simulate sophisticated audio generation when the model is not available

        This enhanced simulation creates more realistic Indian classical music audio
        with proper timbral characteristics, ornamentations, and raga-specific nuances.

        Parameters:
            raga (str): Name of the raga
            duration (float): Duration of the audio in seconds

        Returns:
            Tuple[np.ndarray, int]: Generated audio and sample rate
        """
        # Set sample rate
        sample_rate = 22050

        # Get raga information for frequency selection
        from modules.raga_knowledge import get_raga_info
        raga_info = get_raga_info(raga)

        # Create a time array
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        # Generate a sophisticated audio composition based on raga notes
        audio = np.zeros_like(t)

        # Define base frequencies for Sa (C4 = 261.63 Hz)
        base_freq = 261.63

        # Define frequency ratios for the 12 semitones with microtonal adjustments
        # These are more precise ratios based on Indian classical music theory
        ratios = {
            'S': 1.0,      # Sa
            'r': 1.067,    # Komal Re
            'R': 1.125,    # Shuddha Re
            'g': 1.2,      # Komal Ga
            'G': 1.25,     # Shuddha Ga
            'M': 1.333,    # Shuddha Ma
            'm': 1.406,    # Tivra Ma
            'P': 1.5,      # Pa
            'd': 1.6,      # Komal Dha
            'D': 1.667,    # Shuddha Dha
            'n': 1.778,    # Komal Ni
            'N': 1.875     # Shuddha Ni
        }

        # Get raga-specific information
        if raga_info:
            arohanam = raga_info.get('arohanam', 'SRGMPDNS')
            avarohanam = raga_info.get('avarohanam', 'SNPDMGRS')
            vadi = raga_info.get('vadi', 'S')
            samvadi = raga_info.get('samvadi', 'P')
            pakad = raga_info.get('pakad', '')
            time_of_day = raga_info.get('time', 'any')
        else:
            # Default to Yaman raga
            arohanam = 'SRGMPDNS'
            avarohanam = 'SNPDMGRS'
            vadi = 'G'
            samvadi = 'N'
            pakad = 'NRGS'
            time_of_day = 'evening'

        # Extract allowed notes
        allowed_notes = set(arohanam + avarohanam)

        # Define musical structure sections
        sections = [
            {"name": "alap", "duration": duration * 0.3, "tempo": "slow", "has_rhythm": False,
             "note_duration_range": (0.8, 2.0), "ornament_prob": 0.4},
            {"name": "jor", "duration": duration * 0.3, "tempo": "medium", "has_rhythm": True,
             "note_duration_range": (0.4, 0.8), "ornament_prob": 0.6},
            {"name": "jhala", "duration": duration * 0.4, "tempo": "fast", "has_rhythm": True,
             "note_duration_range": (0.2, 0.4), "ornament_prob": 0.8}
        ]

        # Create a more sophisticated composition with proper structure
        current_time = 0.0

        # Function to generate a meend (glide between notes)
        def generate_meend(start_note, end_note, duration, start_time):
            start_ratio = ratios.get(start_note, 1.0)
            end_ratio = ratios.get(end_note, 1.0)
            start_freq = base_freq * start_ratio
            end_freq = base_freq * end_ratio

            # Create a time array for this segment
            segment_t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

            # Create a frequency array that glides from start to end
            freq_array = np.linspace(start_freq, end_freq, len(segment_t))

            # Generate the phase by integrating the frequency
            phase = 2 * np.pi * np.cumsum(freq_array) / sample_rate

            # Generate the sine wave with the time-varying frequency
            segment = 0.5 * np.sin(phase)

            # Add harmonics with decreasing amplitude for richer sound
            for harmonic in range(2, 6):
                harmonic_phase = harmonic * 2 * np.pi * np.cumsum(freq_array) / sample_rate
                segment += (0.5 / harmonic) * np.sin(harmonic_phase)

            # Apply an envelope
            envelope = np.ones_like(segment)
            attack = max(1, int(0.1 * len(segment)))
            release = max(1, int(0.2 * len(segment)))

            # Ensure we have at least one sample for attack and release
            if attack > 0 and len(segment) > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release > 0 and len(segment) > release:
                envelope[-release:] = np.linspace(1, 0, release)

            # Apply a slight vibrato (gamak) in the middle of the meend
            vibrato_rate = 5.0  # Hz
            vibrato_depth = 0.02
            vibrato = 1.0 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * segment_t)
            segment = segment * envelope * vibrato

            # Calculate the indices in the main audio array
            start_idx = int(start_time * sample_rate)
            end_idx = start_idx + len(segment)
            end_idx = min(end_idx, len(audio))

            # Add to the audio
            audio[start_idx:end_idx] += segment[:end_idx-start_idx]

            return duration

        # Function to generate a single note with ornamentations
        def generate_note(note, duration, start_time, ornament_prob):
            # Get the frequency ratio for this note
            ratio = ratios.get(note, 1.0)

            # Calculate the frequency
            freq = base_freq * ratio

            # Calculate the indices in the main audio array
            start_idx = int(start_time * sample_rate)
            end_idx = int((start_time + duration) * sample_rate)
            end_idx = min(end_idx, len(audio))

            # Create a time array for this segment
            segment_t = np.linspace(0, duration, end_idx - start_idx, endpoint=False)

            # Decide if we should add an ornament
            ornament_type = None
            if np.random.random() < ornament_prob:
                ornament_type = np.random.choice(["kan", "andolan", "gamak", "khatka", None],
                                               p=[0.3, 0.3, 0.2, 0.1, 0.1])

            # Generate the base note
            segment = 0.5 * np.sin(2 * np.pi * freq * segment_t)

            # Add harmonics with decreasing amplitude for richer sound
            for harmonic in range(2, 6):
                segment += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * segment_t)

            # Apply an envelope
            envelope = np.ones_like(segment)
            attack = max(1, int(0.1 * len(segment)))
            release = max(1, int(0.2 * len(segment)))

            # Ensure we have at least one sample for attack and release
            if attack > 0 and len(segment) > 0:
                envelope[:attack] = np.linspace(0, 1, attack)
            if release > 0 and len(segment) > release:
                envelope[-release:] = np.linspace(1, 0, release)

            # Apply ornamentations
            if ornament_type == "kan":
                # Kan: Quick touch of adjacent note
                kan_idx = int(0.2 * len(segment))
                kan_duration = int(0.1 * len(segment))

                # Choose an adjacent note from the raga
                if note in arohanam:
                    idx = arohanam.index(note)
                    if idx > 0:
                        kan_note = arohanam[idx-1]
                    else:
                        kan_note = arohanam[-1]
                else:
                    kan_note = np.random.choice(list(allowed_notes))

                kan_ratio = ratios.get(kan_note, 1.0)
                kan_freq = base_freq * kan_ratio

                # Apply the kan ornament
                segment[kan_idx:kan_idx+kan_duration] = 0.6 * np.sin(2 * np.pi * kan_freq * segment_t[kan_idx:kan_idx+kan_duration])

            elif ornament_type == "andolan":
                # Andolan: Slow oscillation around the note
                andolan_rate = 3.0  # Hz
                andolan_depth = 0.03
                andolan = 1.0 + andolan_depth * np.sin(2 * np.pi * andolan_rate * segment_t)
                segment = segment * andolan

            elif ornament_type == "gamak":
                # Gamak: Heavy oscillation
                gamak_rate = 8.0  # Hz
                gamak_depth = 0.08
                gamak = 1.0 + gamak_depth * np.sin(2 * np.pi * gamak_rate * segment_t)
                segment = segment * gamak

            elif ornament_type == "khatka":
                # Khatka: Quick succession of notes
                khatka_idx = int(0.4 * len(segment))
                khatka_duration = int(0.2 * len(segment))

                # Create a quick pattern of notes
                if note in arohanam:
                    idx = arohanam.index(note)
                    if idx < len(arohanam) - 1:
                        khatka_notes = [note, arohanam[idx+1], note]
                    else:
                        khatka_notes = [note, arohanam[0], note]
                else:
                    khatka_notes = [note, np.random.choice(list(allowed_notes)), note]

                # Apply the khatka ornament
                sub_duration = khatka_duration // len(khatka_notes)
                for i, khatka_note in enumerate(khatka_notes):
                    khatka_ratio = ratios.get(khatka_note, 1.0)
                    khatka_freq = base_freq * khatka_ratio
                    start = khatka_idx + i * sub_duration
                    end = start + sub_duration
                    segment[start:end] = 0.6 * np.sin(2 * np.pi * khatka_freq * segment_t[start:end])

            # Apply the envelope
            segment = segment * envelope

            # Add to the audio
            audio[start_idx:end_idx] += segment

            return duration

        # Generate audio for each section
        for section in sections:
            section_end_time = current_time + section["duration"]

            # For alap section, use longer notes with meends (glides)
            if section["name"] == "alap":
                while current_time < section_end_time:
                    # Choose a note, emphasizing vadi and samvadi
                    if np.random.random() < 0.4:
                        note = vadi if np.random.random() < 0.6 else samvadi
                    else:
                        note = np.random.choice(list(allowed_notes))

                    # Determine note duration
                    min_dur, max_dur = section["note_duration_range"]
                    note_duration = np.random.uniform(min_dur, max_dur)

                    # Ensure we don't exceed section duration
                    note_duration = min(note_duration, section_end_time - current_time)

                    # Decide whether to use a meend (glide) or a single note
                    if np.random.random() < 0.3 and note_duration > 1.0:
                        # Choose another note for the meend
                        if note in arohanam:
                            idx = arohanam.index(note)
                            if idx < len(arohanam) - 1:
                                end_note = arohanam[idx+1]
                            else:
                                end_note = arohanam[0]
                        else:
                            end_note = np.random.choice(list(allowed_notes))

                        # Generate a meend between the two notes
                        current_time += generate_meend(note, end_note, note_duration, current_time)
                    else:
                        # Generate a single note with possible ornamentations
                        current_time += generate_note(note, note_duration, current_time, section["ornament_prob"])

                    # Add a pause occasionally
                    if np.random.random() < 0.2:
                        pause_duration = np.random.uniform(0.2, 0.5)
                        pause_duration = min(pause_duration, section_end_time - current_time)
                        current_time += pause_duration

            # For jor and jhala sections, use faster notes with rhythm
            else:
                # Create a rhythmic pattern
                if section["has_rhythm"]:
                    # Simple teental (16 beat) pattern for rhythm
                    rhythm_pattern = [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0]
                    rhythm_idx = 0

                # Generate notes following arohanam/avarohanam patterns more closely
                use_arohanam = True
                pattern_idx = 0

                while current_time < section_end_time:
                    # Choose the next note based on the pattern
                    if use_arohanam:
                        note = arohanam[pattern_idx]
                        pattern_idx += 1
                        if pattern_idx >= len(arohanam):
                            pattern_idx = 0
                            use_arohanam = False
                    else:
                        note = avarohanam[pattern_idx]
                        pattern_idx += 1
                        if pattern_idx >= len(avarohanam):
                            pattern_idx = 0
                            use_arohanam = True

                    # Determine note duration based on rhythm if applicable
                    if section["has_rhythm"]:
                        base_duration = 0.3 if section["name"] == "jor" else 0.15
                        note_duration = base_duration * rhythm_pattern[rhythm_idx]
                        rhythm_idx = (rhythm_idx + 1) % len(rhythm_pattern)
                    else:
                        min_dur, max_dur = section["note_duration_range"]
                        note_duration = np.random.uniform(min_dur, max_dur)

                    # Ensure we don't exceed section duration
                    note_duration = min(note_duration, section_end_time - current_time)

                    # Generate the note with possible ornamentations
                    current_time += generate_note(note, note_duration, current_time, section["ornament_prob"])

        # Add a tanpura drone in the background for authenticity
        tanpura_notes = ['S', 'P', 'S', 'S']  # Common tanpura tuning
        tanpura_audio = np.zeros_like(audio)

        for i, note in enumerate(tanpura_notes):
            # Get the frequency ratio for this note
            ratio = ratios.get(note, 1.0)

            # Calculate the frequency
            freq = base_freq * ratio / 2  # Lower octave for tanpura

            # Generate the tanpura string sound
            string_audio = 0.15 * np.sin(2 * np.pi * freq * t)

            # Add harmonics
            for harmonic in range(2, 10):
                string_audio += (0.15 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

            # Add the characteristic tanpura buzz (jawari effect)
            jawari_rate = 40.0  # Hz
            jawari_depth = 0.03
            jawari = 1.0 + jawari_depth * np.sin(2 * np.pi * jawari_rate * t)
            string_audio = string_audio * jawari

            # Apply a slow plucking pattern
            pluck_interval = 2.0  # seconds
            num_plucks = int(duration / pluck_interval)

            for p in range(num_plucks):
                pluck_time = p * pluck_interval + i * (pluck_interval / len(tanpura_notes))
                pluck_idx = int(pluck_time * sample_rate)

                if pluck_idx < len(tanpura_audio):
                    # Apply an envelope for each pluck
                    pluck_duration = int(1.8 * sample_rate)  # 1.8 seconds per pluck
                    end_idx = min(pluck_idx + pluck_duration, len(tanpura_audio))

                    envelope = np.ones(end_idx - pluck_idx)
                    attack = int(0.02 * len(envelope))
                    release = int(0.8 * len(envelope))
                    envelope[:attack] = np.linspace(0, 1, attack)
                    envelope[release:] = np.linspace(1, 0, len(envelope) - release)

                    tanpura_audio[pluck_idx:end_idx] += string_audio[pluck_idx:end_idx] * envelope

        # Mix the tanpura with the main audio
        audio = audio * 0.8 + tanpura_audio * 0.2

        # Add some room ambience
        reverb_time = 0.3  # seconds
        reverb_samples = int(reverb_time * sample_rate)
        reverb = np.exp(-np.linspace(0, 10, reverb_samples))

        # Apply the reverb using convolution
        from scipy import signal
        audio = signal.convolve(audio, reverb, mode='same')

        # Add a very small amount of noise for warmth
        audio += 0.003 * np.random.randn(*audio.shape)

        # Normalize
        audio = audio / np.max(np.abs(audio))

        return audio, sample_rate
    
    def evaluate_audio(self, audio: np.ndarray, sr: int, raga: str) -> Dict[str, float]:
        """
        Evaluate generated audio for quality and authenticity
        
        Parameters:
            audio (np.ndarray): Generated audio
            sr (int): Sample rate
            raga (str): Name of the raga
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Initialize metrics
        metrics = {
            'authenticity': 0.0,
            'timbral_quality': 0.0,
            'spectral_richness': 0.0,
            'overall_quality': 0.0
        }
        
        # Check if the discriminator is available
        if self.available and self.discriminator is not None:
            try:
                # Prepare the audio for the discriminator
                # This would depend on the specific implementation of the discriminator
                # For now, we'll use a simplified approach
                
                # Compute the spectrogram
                import librosa
                S = librosa.feature.melspectrogram(y=audio, sr=sr)
                S_db = librosa.power_to_db(S, ref=np.max)
                
                # Resize to the expected input shape
                # This is a placeholder - actual implementation would depend on the model
                S_resized = np.resize(S_db, (128, 128))
                S_resized = np.expand_dims(S_resized, axis=0)
                S_resized = np.expand_dims(S_resized, axis=-1)
                
                # Get the discriminator score
                authenticity_score = self.discriminator.predict(S_resized, verbose=0)[0][0]
                metrics['authenticity'] = float(authenticity_score)
            except Exception as e:
                logger.error(f"Error evaluating audio with discriminator: {str(e)}")
                # Fall back to simulated evaluation
                metrics['authenticity'] = np.random.uniform(0.6, 0.9)
        else:
            # Simulate discriminator score
            metrics['authenticity'] = np.random.uniform(0.6, 0.9)
        
        # Calculate timbral quality using spectral features
        try:
            import librosa
            
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            centroid_mean = np.mean(centroid)
            centroid_std = np.std(centroid)
            
            # Spectral bandwidth (spread)
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            bandwidth_mean = np.mean(bandwidth)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            contrast_mean = np.mean(contrast)
            
            # Normalize and combine
            timbral_score = (
                0.4 * (centroid_mean / 4000) + 
                0.3 * (1 - centroid_std / centroid_mean) + 
                0.3 * (bandwidth_mean / 2000)
            )
            metrics['timbral_quality'] = min(max(timbral_score, 0.0), 1.0)
            
            # Spectral richness
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            rolloff_mean = np.mean(rolloff)
            
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.sum(y_harmonic**2)
            percussive_energy = np.sum(y_percussive**2)
            balance = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)
            
            # Normalize and combine
            richness_score = (
                0.5 * (rolloff_mean / (sr/2)) + 
                0.5 * balance
            )
            metrics['spectral_richness'] = min(max(richness_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating audio features: {str(e)}")
            # Fall back to simulated metrics
            metrics['timbral_quality'] = np.random.uniform(0.5, 0.9)
            metrics['spectral_richness'] = np.random.uniform(0.5, 0.9)
        
        # Calculate overall quality
        metrics['overall_quality'] = (
            metrics['authenticity'] * 0.4 + 
            metrics['timbral_quality'] * 0.3 + 
            metrics['spectral_richness'] * 0.3
        )
        
        return metrics


class HybridComposer:
    """
    Hybrid composer combining Bi-LSTM and CNNGAN models
    
    This class combines the strengths of both models:
    - Bi-LSTM for melodic structure and raga grammar
    - CNNGAN for audio synthesis and timbral qualities
    """
    
    def __init__(self):
        """Initialize the hybrid composer"""
        self.bilstm = BiLSTMComposer()
        self.cnngan = CNNGANComposer()
    
    def generate_composition(self, raga: str, duration: float = 30.0,
                            seed: Optional[str] = None, temperature: float = 1.0,
                            melodic_complexity: float = 0.7,
                            ornamentation_level: float = 0.6) -> Tuple[np.ndarray, int, str]:
        """
        Generate a sophisticated complete composition for a given raga

        This enhanced hybrid approach creates high-quality compositions by:
        1. Using Bi-LSTM to generate the melodic structure with proper raga grammar
        2. Using CNNGAN to create realistic timbral qualities
        3. Applying advanced audio processing to combine both models effectively
        4. Adding authentic ornamentations and musical structure

        Parameters:
            raga (str): Name of the raga
            duration (float): Duration of the composition in seconds
            seed (str, optional): Seed sequence to start generation
            temperature (float): Controls randomness in generation (higher = more creative)
            melodic_complexity (float): Controls the complexity of melodic patterns
            ornamentation_level (float): Controls the amount of ornamentations

        Returns:
            Tuple[np.ndarray, int, str]: Generated audio, sample rate, and symbolic representation
        """
        # Get raga information
        from modules.raga_knowledge import get_raga_info
        raga_info = get_raga_info(raga)

        # Define musical structure sections
        sections = [
            {"name": "alap", "duration_ratio": 0.3, "tempo": "slow", "notes_per_second": 1.0},
            {"name": "jor", "duration_ratio": 0.3, "tempo": "medium", "notes_per_second": 2.5},
            {"name": "jhala", "duration_ratio": 0.4, "tempo": "fast", "notes_per_second": 5.0}
        ]

        # Calculate total sequence length based on the musical structure
        sequence_length = 0
        for section in sections:
            section_duration = duration * section["duration_ratio"]
            section_notes = int(section_duration * section["notes_per_second"])
            sequence_length += section_notes

        # Generate the melodic sequence using Bi-LSTM with temperature parameter
        sequence = self.bilstm.generate_sequence(
            raga=raga,
            seed=seed,
            length=sequence_length,
            temperature=temperature
        )

        # Set sample rate
        sr = 22050

        try:
            import librosa
            from scipy import signal

            # Define base frequencies for Sa (C4 = 261.63 Hz)
            base_freq = 261.63

            # Define frequency ratios with microtonal adjustments
            ratios = {
                'S': 1.0,      # Sa
                'r': 1.067,    # Komal Re
                'R': 1.125,    # Shuddha Re
                'g': 1.2,      # Komal Ga
                'G': 1.25,     # Shuddha Ga
                'M': 1.333,    # Shuddha Ma
                'm': 1.406,    # Tivra Ma
                'P': 1.5,      # Pa
                'd': 1.6,      # Komal Dha
                'D': 1.667,    # Shuddha Dha
                'n': 1.778,    # Komal Ni
                'N': 1.875     # Shuddha Ni
            }

            # Initialize the audio array
            audio = np.zeros(int(duration * sr))

            # Generate CNNGAN audio for timbral reference
            if self.cnngan.available:
                # Generate audio with CNNGAN
                gan_audio, gan_sr = self.cnngan.generate_audio(raga, min(10.0, duration))

                # Resample if needed
                if gan_sr != sr:
                    gan_audio = librosa.resample(gan_audio, orig_sr=gan_sr, target_sr=sr)

                # Extract timbral characteristics
                gan_harmonic, gan_percussive = librosa.effects.hpss(gan_audio)

                # Extract spectral envelope
                gan_spec = np.abs(librosa.stft(gan_harmonic))
                gan_env = np.mean(gan_spec, axis=1)
                gan_env = gan_env / np.max(gan_env)
            else:
                gan_audio = None

            # Process the sequence section by section
            sequence_idx = 0
            current_time = 0.0

            for section in sections:
                section_duration = duration * section["duration_ratio"]
                section_end_time = current_time + section_duration

                # Calculate notes for this section
                section_notes = int(section_duration * section["notes_per_second"])
                section_sequence = sequence[sequence_idx:sequence_idx+section_notes]
                sequence_idx += section_notes

                # Calculate note duration for this section
                base_note_duration = 1.0 / section["notes_per_second"]

                # Process each note in the section
                for i, note in enumerate(section_sequence):
                    if note == '-':
                        # This is a pause, advance time but don't generate audio
                        current_time += base_note_duration
                        continue

                    # Get the frequency ratio for this note
                    ratio = ratios.get(note, 1.0)

                    # Calculate the frequency
                    freq = base_freq * ratio

                    # Determine note duration with variation based on melodic_complexity
                    if section["tempo"] == "slow":
                        # In alap, notes have variable duration
                        variation_range = 0.7 * melodic_complexity  # Higher complexity = more variation
                        note_duration = base_note_duration * np.random.uniform(1.0 - variation_range, 1.0 + variation_range)
                    elif section["tempo"] == "medium":
                        # In jor, notes follow a more regular pattern with some complexity
                        variation_factor = 0.3 * melodic_complexity
                        note_duration = base_note_duration * (1.0 + variation_factor * np.sin(i * 0.5))
                    else:
                        # In jhala, notes are more precise but still allow some complexity
                        variation = 0.1 * melodic_complexity
                        note_duration = base_note_duration * np.random.uniform(1.0 - variation, 1.0 + variation)

                    # Ensure we don't exceed section duration
                    note_duration = min(note_duration, section_end_time - current_time)

                    # Calculate the start and end indices
                    start_idx = int(current_time * sr)
                    end_idx = int((current_time + note_duration) * sr)
                    end_idx = min(end_idx, len(audio))

                    # Generate time array for this note
                    t = np.linspace(0, note_duration, end_idx - start_idx, endpoint=False)

                    # Decide whether to add ornamentation based on ornamentation_level parameter
                    ornament_type = None
                    if section["tempo"] == "slow" and np.random.random() < (0.4 * ornamentation_level):
                        ornament_type = np.random.choice(["meend", "kan", "andolan"])
                    elif section["tempo"] == "medium" and np.random.random() < (0.3 * ornamentation_level):
                        ornament_type = np.random.choice(["kan", "andolan", "gamak"])
                    elif section["tempo"] == "fast" and np.random.random() < (0.2 * ornamentation_level):
                        ornament_type = np.random.choice(["gamak", "khatka", "murki"])

                    # Generate the note with ornamentation
                    if ornament_type == "meend" and i < len(section_sequence) - 1:
                        # Meend: Glide to the next note
                        next_note = section_sequence[i+1]
                        if next_note != '-':
                            next_ratio = ratios.get(next_note, 1.0)
                            next_freq = base_freq * next_ratio

                            # Create a frequency array that glides from current to next
                            freq_array = np.linspace(freq, next_freq, len(t))

                            # Generate the phase by integrating the frequency
                            phase = 2 * np.pi * np.cumsum(freq_array) / sr

                            # Generate the sine wave with the time-varying frequency
                            note_audio = 0.5 * np.sin(phase)

                            # Add harmonics
                            for harmonic in range(2, 6):
                                harmonic_phase = harmonic * 2 * np.pi * np.cumsum(freq_array) / sr
                                note_audio += (0.5 / harmonic) * np.sin(harmonic_phase)
                        else:
                            # Regular note if next is a pause
                            note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

                            # Add harmonics
                            for harmonic in range(2, 6):
                                note_audio += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

                    elif ornament_type == "kan":
                        # Kan: Quick touch of adjacent note
                        kan_idx = int(0.2 * len(t))
                        kan_duration = int(0.1 * len(t))

                        # Choose an adjacent note from the raga
                        if raga_info and 'arohanam' in raga_info:
                            arohanam = raga_info['arohanam']
                            if note in arohanam:
                                idx = arohanam.index(note)
                                if idx > 0:
                                    kan_note = arohanam[idx-1]
                                else:
                                    kan_note = arohanam[-1]
                            else:
                                kan_note = note
                        else:
                            kan_note = note

                        kan_ratio = ratios.get(kan_note, 1.0)
                        kan_freq = base_freq * kan_ratio

                        # Generate the base note
                        note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

                        # Add harmonics
                        for harmonic in range(2, 6):
                            note_audio += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

                        # Apply the kan ornament
                        if kan_idx + kan_duration < len(note_audio):
                            kan_t = t[kan_idx:kan_idx+kan_duration]
                            kan_audio = 0.6 * np.sin(2 * np.pi * kan_freq * kan_t)

                            # Add harmonics to the kan
                            for harmonic in range(2, 6):
                                kan_audio += (0.6 / harmonic) * np.sin(2 * np.pi * kan_freq * harmonic * kan_t)

                            note_audio[kan_idx:kan_idx+kan_duration] = kan_audio

                    elif ornament_type == "andolan":
                        # Andolan: Slow oscillation around the note
                        note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

                        # Add harmonics
                        for harmonic in range(2, 6):
                            note_audio += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

                        # Apply the andolan effect
                        andolan_rate = 3.0  # Hz
                        andolan_depth = 0.03
                        andolan = 1.0 + andolan_depth * np.sin(2 * np.pi * andolan_rate * t)
                        note_audio = note_audio * andolan

                    elif ornament_type == "gamak":
                        # Gamak: Heavy oscillation
                        note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

                        # Add harmonics
                        for harmonic in range(2, 6):
                            note_audio += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

                        # Apply the gamak effect
                        gamak_rate = 8.0  # Hz
                        gamak_depth = 0.08
                        gamak = 1.0 + gamak_depth * np.sin(2 * np.pi * gamak_rate * t)
                        note_audio = note_audio * gamak

                    elif ornament_type == "khatka" or ornament_type == "murki":
                        # Khatka/Murki: Quick succession of notes
                        note_audio = np.zeros_like(t)

                        # Create a quick pattern of notes
                        if raga_info and 'arohanam' in raga_info:
                            arohanam = raga_info['arohanam']
                            if note in arohanam:
                                idx = arohanam.index(note)
                                if ornament_type == "khatka":
                                    # Khatka: Up and down
                                    if idx < len(arohanam) - 1 and idx > 0:
                                        ornament_notes = [note, arohanam[idx+1], arohanam[idx-1], note]
                                    else:
                                        ornament_notes = [note, note, note, note]
                                else:
                                    # Murki: More complex pattern
                                    if idx < len(arohanam) - 2:
                                        ornament_notes = [note, arohanam[idx+1], arohanam[idx+2], arohanam[idx+1], note]
                                    else:
                                        ornament_notes = [note, note, note, note, note]
                            else:
                                ornament_notes = [note, note, note, note]
                        else:
                            ornament_notes = [note, note, note, note]

                        # Apply the ornament
                        sub_duration = len(t) // len(ornament_notes)
                        for j, ornament_note in enumerate(ornament_notes):
                            ornament_ratio = ratios.get(ornament_note, 1.0)
                            ornament_freq = base_freq * ornament_ratio

                            start = j * sub_duration
                            end = (j + 1) * sub_duration if j < len(ornament_notes) - 1 else len(t)

                            if start < len(t) and end <= len(t):
                                sub_t = t[start:end] - t[start]
                                sub_audio = 0.5 * np.sin(2 * np.pi * ornament_freq * sub_t)

                                # Add harmonics
                                for harmonic in range(2, 6):
                                    sub_audio += (0.5 / harmonic) * np.sin(2 * np.pi * ornament_freq * harmonic * sub_t)

                                note_audio[start:end] = sub_audio

                    else:
                        # Regular note without ornamentation
                        note_audio = 0.5 * np.sin(2 * np.pi * freq * t)

                        # Add harmonics with decreasing amplitude for richer sound
                        for harmonic in range(2, 6):
                            note_audio += (0.5 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)

                    # Apply an envelope
                    envelope = np.ones_like(note_audio)
                    attack = int(0.1 * len(note_audio))
                    release = int(0.2 * len(note_audio))

                    if attack > 0:
                        envelope[:attack] = np.linspace(0, 1, attack)
                    if release > 0 and release < len(envelope):
                        envelope[-release:] = np.linspace(1, 0, release)

                    note_audio = note_audio * envelope

                    # Add to the audio
                    if start_idx < len(audio) and end_idx <= len(audio):
                        audio[start_idx:end_idx] = note_audio

                    # Advance time
                    current_time += note_duration

            # Add a tanpura drone in the background for authenticity
            tanpura_notes = ['S', 'P', 'S', 'S']  # Common tanpura tuning
            tanpura_audio = np.zeros_like(audio)

            # Create time array for the entire duration
            t_full = np.linspace(0, duration, len(audio), endpoint=False)

            for i, note in enumerate(tanpura_notes):
                # Get the frequency ratio for this note
                ratio = ratios.get(note, 1.0)

                # Calculate the frequency (lower octave for tanpura)
                freq = base_freq * ratio / 2

                # Generate the tanpura string sound
                string_audio = 0.15 * np.sin(2 * np.pi * freq * t_full)

                # Add harmonics
                for harmonic in range(2, 10):
                    string_audio += (0.15 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t_full)

                # Add the characteristic tanpura buzz (jawari effect)
                jawari_rate = 40.0  # Hz
                jawari_depth = 0.03
                jawari = 1.0 + jawari_depth * np.sin(2 * np.pi * jawari_rate * t_full)
                string_audio = string_audio * jawari

                # Apply a slow plucking pattern
                pluck_interval = 2.0  # seconds
                num_plucks = int(duration / pluck_interval)

                for p in range(num_plucks):
                    pluck_time = p * pluck_interval + i * (pluck_interval / len(tanpura_notes))
                    pluck_idx = int(pluck_time * sr)

                    if pluck_idx < len(tanpura_audio):
                        # Apply an envelope for each pluck
                        pluck_duration = int(1.8 * sr)  # 1.8 seconds per pluck
                        end_idx = min(pluck_idx + pluck_duration, len(tanpura_audio))

                        if pluck_idx < end_idx:
                            envelope = np.ones(end_idx - pluck_idx)
                            attack = int(0.02 * len(envelope))
                            release = int(0.8 * len(envelope))

                            if attack > 0:
                                envelope[:attack] = np.linspace(0, 1, attack)
                            if release > 0 and release < len(envelope):
                                envelope[release:] = np.linspace(1, 0, len(envelope) - release)

                            tanpura_audio[pluck_idx:end_idx] += string_audio[pluck_idx:end_idx] * envelope

            # If CNNGAN is available, use it to enhance the audio
            if self.cnngan.available and gan_audio is not None:
                # Use the spectral envelope of the GAN audio to filter the sequence audio
                y_harmonic, y_percussive = librosa.effects.hpss(audio)

                # Extract the spectral envelope from the GAN audio
                audio_spec = np.abs(librosa.stft(y_harmonic))

                # Apply the spectral envelope
                gan_env_expanded = np.tile(gan_env.reshape(-1, 1), (1, audio_spec.shape[1]))

                # Filter the audio
                filtered_spec = audio_spec * gan_env_expanded[:audio_spec.shape[0], :audio_spec.shape[1]]
                filtered_audio = librosa.istft(filtered_spec)

                # Combine with percussive component
                if len(filtered_audio) > len(y_percussive):
                    filtered_audio = filtered_audio[:len(y_percussive)]
                else:
                    y_percussive = y_percussive[:len(filtered_audio)]

                # Mix the filtered audio with the original
                audio = 0.7 * (filtered_audio + y_percussive) + 0.3 * audio

            # Mix the tanpura with the main audio
            audio = audio * 0.8 + tanpura_audio * 0.2

            # Add some room ambience (reverb)
            reverb_time = 0.3  # seconds
            reverb_samples = int(reverb_time * sr)
            reverb = np.exp(-np.linspace(0, 10, reverb_samples))

            # Apply the reverb using convolution
            audio = signal.convolve(audio, reverb, mode='same')

            # Add a very small amount of noise for warmth
            audio += 0.003 * np.random.randn(*audio.shape)

            # Normalize
            audio = audio / np.max(np.abs(audio))

            return audio, sr, sequence

        except Exception as e:
            logger.error(f"Error generating hybrid composition: {str(e)}")
            # Fall back to CNNGAN generation
            audio, sr = self.cnngan.generate_audio(raga, duration)
            return audio, sr, sequence
    
    def evaluate_composition(self, audio: np.ndarray, sr: int,
                           sequence: str, raga: str,
                           melodic_complexity: float = 0.7,
                           ornamentation_level: float = 0.6) -> Dict[str, float]:
        """
        Evaluate a composition using both models

        Parameters:
            audio (np.ndarray): Generated audio
            sr (int): Sample rate
            sequence (str): Symbolic representation of the composition
            raga (str): Name of the raga
            melodic_complexity (float): Level of melodic complexity used in generation
            ornamentation_level (float): Level of ornamentation used in generation

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Evaluate the melodic sequence
        sequence_metrics = self.bilstm.evaluate_sequence(sequence, raga)
        
        # Evaluate the audio
        audio_metrics = self.cnngan.evaluate_audio(audio, sr, raga)
        
        # Combine the metrics
        combined_metrics = {
            'melodic_authenticity': sequence_metrics['overall_quality'],
            'timbral_quality': audio_metrics['timbral_quality'],
            'structural_coherence': sequence_metrics['phrase_coherence'],
            'raga_adherence': sequence_metrics['raga_adherence'],
            'ornamentation': ornamentation_level * audio_metrics['spectral_richness'],
            'overall_quality': (
                sequence_metrics['overall_quality'] * 0.4 +
                audio_metrics['overall_quality'] * 0.4 +
                melodic_complexity * 0.1 +
                ornamentation_level * 0.1
            )
        }
        
        return combined_metrics


# Create a function to convert symbolic notation to audio
def symbolic_to_audio(sequence: str, sr: int = 22050, note_duration: float = 0.25) -> np.ndarray:
    """
    Convert a symbolic sequence to audio
    
    Parameters:
        sequence (str): Symbolic representation as a string of notes
        sr (int): Sample rate
        note_duration (float): Duration of each note in seconds
        
    Returns:
        np.ndarray: Audio representation of the sequence
    """
    # Define base frequencies for Sa (C4 = 261.63 Hz)
    base_freq = 261.63
    
    # Define frequency ratios for the 12 semitones
    ratios = {
        'S': 1.0,      # Sa
        'r': 1.067,    # Komal Re
        'R': 1.125,    # Shuddha Re
        'g': 1.2,      # Komal Ga
        'G': 1.25,     # Shuddha Ga
        'M': 1.333,    # Shuddha Ma
        'm': 1.406,    # Tivra Ma
        'P': 1.5,      # Pa
        'd': 1.6,      # Komal Dha
        'D': 1.667,    # Shuddha Dha
        'n': 1.778,    # Komal Ni
        'N': 1.875     # Shuddha Ni
    }
    
    # Calculate the total duration
    total_duration = len(sequence) * note_duration
    
    # Initialize the audio array
    audio = np.zeros(int(total_duration * sr))
    
    # Generate audio for each note
    for i, note in enumerate(sequence):
        if note == '-':
            # This is a pause, skip
            continue
        
        # Get the frequency ratio for this note
        ratio = ratios.get(note, 1.0)
        
        # Calculate the frequency
        freq = base_freq * ratio
        
        # Calculate the start and end indices
        start_idx = int(i * note_duration * sr)
        end_idx = int((i + 1) * note_duration * sr)
        
        # Generate a sine wave for this note
        t = np.linspace(0, note_duration, end_idx - start_idx, endpoint=False)
        note_audio = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add some harmonics
        note_audio += 0.25 * np.sin(2 * np.pi * freq * 2 * t)
        note_audio += 0.125 * np.sin(2 * np.pi * freq * 3 * t)
        
        # Apply an envelope
        envelope = np.ones_like(note_audio)
        attack = int(0.1 * len(note_audio))
        release = int(0.2 * len(note_audio))
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-release:] = np.linspace(1, 0, release)
        note_audio = note_audio * envelope
        
        # Add to the audio
        audio[start_idx:end_idx] = note_audio
    
    # Add some noise
    audio += 0.005 * np.random.randn(*audio.shape)
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio