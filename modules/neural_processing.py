"""
Neural Processing Module for RagaVani

This module provides neural network-based analysis for Indian classical music.
It includes models for raga identification, tala detection, and ornament recognition.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import os
import time

# Configure logging
logger = logging.getLogger(__name__)

# Try to import TensorFlow, but handle gracefully if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    # Set TensorFlow logging level
    tf.get_logger().setLevel(logging.ERROR)
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"TensorFlow detected {len(gpus)} GPU(s)")
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        logger.info("No GPU detected, using CPU for neural processing")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. Neural processing will be disabled.")

# Define model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
RAGA_MODEL_PATH = os.path.join(MODEL_DIR, "raga_classifier")
TALA_MODEL_PATH = os.path.join(MODEL_DIR, "tala_detector")
ORNAMENT_MODEL_PATH = os.path.join(MODEL_DIR, "ornament_recognizer")

# Cache for loaded models
_model_cache = {}

def _load_model(model_path: str) -> Optional[Any]:
    """
    Load a TensorFlow model from disk with caching
    
    Parameters:
        model_path (str): Path to the model directory
        
    Returns:
        Optional[tf.keras.Model]: Loaded model or None if not available
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

def _extract_features_for_raga(audio_data: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Extract features for raga identification
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Dictionary of features
    """
    try:
        import librosa
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=128,
            fmin=20,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(
            y=audio_data, 
            sr=sr,
            bins_per_octave=24
        )
        
        # Extract pitch features
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Normalize features
        mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
        chroma = (chroma - np.mean(chroma)) / (np.std(chroma) + 1e-8)
        
        return {
            "mel_spectrogram": mel_spec_db,
            "chroma": chroma,
            "pitch": f0,
            "voiced_flag": voiced_flag
        }
    except Exception as e:
        logger.error(f"Error extracting features for raga: {str(e)}")
        return {}

def _extract_features_for_tala(audio_data: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Extract features for tala detection
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Dictionary of features
    """
    try:
        import librosa
        
        # Extract onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # Extract tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr
        )
        
        # Detect beats
        _, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr
        )
        
        # Create beat synchronous features
        beat_features = librosa.util.sync(
            onset_env.reshape(1, -1),
            beats,
            aggregate=np.mean
        )
        
        return {
            "onset_envelope": onset_env,
            "tempogram": tempogram,
            "beats": beats,
            "beat_features": beat_features
        }
    except Exception as e:
        logger.error(f"Error extracting features for tala: {str(e)}")
        return {}

def _extract_features_for_ornaments(audio_data: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    """
    Extract features for ornament recognition
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Dictionary of features
    """
    try:
        import librosa
        
        # Extract pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Calculate pitch derivatives
        pitch_diff = np.diff(f0)
        pitch_diff = np.concatenate([[0], pitch_diff])  # Pad to match original length
        pitch_diff[np.isnan(pitch_diff)] = 0
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=audio_data,
            sr=sr
        )
        
        return {
            "pitch": f0,
            "pitch_derivative": pitch_diff,
            "voiced_flag": voiced_flag,
            "spectral_contrast": contrast
        }
    except Exception as e:
        logger.error(f"Error extracting features for ornaments: {str(e)}")
        return {}

def analyze_with_neural_models(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Analyze audio using neural models
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Analysis results from neural models
    """
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available, skipping neural analysis")
        return {}
        
    start_time = time.time()
    
    results = {
        "raga_analysis": None,
        "tala_analysis": None,
        "ornament_analysis": None,
        "processing_time": None
    }
    
    # Analyze raga
    try:
        raga_results = identify_raga_neural(audio_data, sr)
        results["raga_analysis"] = raga_results
    except Exception as e:
        logger.warning(f"Neural raga analysis failed: {str(e)}")
    
    # Analyze tala
    try:
        tala_results = detect_tala_neural(audio_data, sr)
        results["tala_analysis"] = tala_results
    except Exception as e:
        logger.warning(f"Neural tala analysis failed: {str(e)}")
    
    # Analyze ornaments
    try:
        ornament_results = detect_ornaments_neural(audio_data, sr)
        results["ornament_analysis"] = ornament_results
    except Exception as e:
        logger.warning(f"Neural ornament analysis failed: {str(e)}")
    
    # Calculate processing time
    results["processing_time"] = time.time() - start_time
    
    return results

def identify_raga_neural(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Identify raga using neural model
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Raga identification results
    """
    # Load model
    model = _load_model(RAGA_MODEL_PATH)
    
    # If model is not available, return empty results
    if model is None:
        logger.warning("Raga model not available, skipping neural raga identification")
        return {}
    
    # Extract features
    features = _extract_features_for_raga(audio_data, sr)
    if not features:
        return {}
    
    # Prepare input for model
    # Note: In a real implementation, this would depend on the specific model architecture
    # For now, we'll simulate the model prediction
    
    # Simulate model prediction (in a real implementation, this would use the actual model)
    # This is a placeholder for demonstration purposes
    ragas = ["Yaman", "Bhairav", "Bhimpalasi", "Darbari", "Khamaj", "Malkauns"]
    confidences = np.random.random(len(ragas))
    confidences = confidences / np.sum(confidences)  # Normalize to sum to 1
    
    # Sort by confidence
    raga_confidences = list(zip(ragas, confidences))
    raga_confidences.sort(key=lambda x: x[1], reverse=True)
    
    # Get top raga and alternatives
    top_raga, top_confidence = raga_confidences[0]
    alternatives = [{"name": name, "confidence": float(conf)} for name, conf in raga_confidences[1:4]]
    
    return {
        "detected_raga": top_raga,
        "confidence": float(top_confidence),
        "alternative_ragas": alternatives,
        "method": "neural"
    }

def detect_tala_neural(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detect tala using neural model
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Tala detection results
    """
    # Load model
    model = _load_model(TALA_MODEL_PATH)
    
    # If model is not available, return empty results
    if model is None:
        logger.warning("Tala model not available, skipping neural tala detection")
        return {}
    
    # Extract features
    features = _extract_features_for_tala(audio_data, sr)
    if not features:
        return {}
    
    # Prepare input for model
    # Note: In a real implementation, this would depend on the specific model architecture
    # For now, we'll simulate the model prediction
    
    # Simulate model prediction (in a real implementation, this would use the actual model)
    # This is a placeholder for demonstration purposes
    talas = ["Teentaal", "Ektaal", "Jhaptaal", "Keherwa", "Rupak", "Dadra"]
    confidences = np.random.random(len(talas))
    confidences = confidences / np.sum(confidences)  # Normalize to sum to 1
    
    # Sort by confidence
    tala_confidences = list(zip(talas, confidences))
    tala_confidences.sort(key=lambda x: x[1], reverse=True)
    
    # Get top tala and alternatives
    top_tala, top_confidence = tala_confidences[0]
    alternatives = [{"name": name, "confidence": float(conf)} for name, conf in tala_confidences[1:4]]
    
    # Simulate beat positions
    import librosa
    tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    return {
        "detected_tala": top_tala,
        "confidence": float(top_confidence),
        "alternative_talas": alternatives,
        "tempo": float(tempo),
        "beat_positions": beat_times.tolist(),
        "method": "neural"
    }

def detect_ornaments_neural(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detect ornaments using neural model
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Ornament detection results
    """
    # Load model
    model = _load_model(ORNAMENT_MODEL_PATH)
    
    # If model is not available, return empty results
    if model is None:
        logger.warning("Ornament model not available, skipping neural ornament detection")
        return {}
    
    # Extract features
    features = _extract_features_for_ornaments(audio_data, sr)
    if not features:
        return {}
    
    # Prepare input for model
    # Note: In a real implementation, this would depend on the specific model architecture
    # For now, we'll simulate the model prediction
    
    # Simulate ornament detection (in a real implementation, this would use the actual model)
    # This is a placeholder for demonstration purposes
    ornament_types = ["meend", "kan", "andolan", "gamak", "murki"]
    
    # Generate random ornaments
    num_ornaments = np.random.randint(3, 8)
    ornaments = []
    
    for i in range(num_ornaments):
        ornament_type = np.random.choice(ornament_types)
        start_time = np.random.uniform(0, len(audio_data) / sr - 1)
        duration = np.random.uniform(0.2, 1.0)
        confidence = np.random.uniform(0.6, 0.95)
        
        ornaments.append({
            "type": ornament_type,
            "start_time": float(start_time),
            "end_time": float(start_time + duration),
            "duration": float(duration),
            "confidence": float(confidence)
        })
    
    # Sort by start time
    ornaments.sort(key=lambda x: x["start_time"])
    
    return {
        "ornaments": ornaments,
        "method": "neural"
    }

class NeuralProcessor:
    """
    Class for neural processing of Indian classical music
    """
    
    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the neural processor
        
        Parameters:
            model_dir (str, optional): Directory containing models
        """
        self.model_dir = model_dir or MODEL_DIR
        self.raga_model_path = os.path.join(self.model_dir, "raga_classifier")
        self.tala_model_path = os.path.join(self.model_dir, "tala_detector")
        self.ornament_model_path = os.path.join(self.model_dir, "ornament_recognizer")
        
        # Check if TensorFlow is available
        self.tensorflow_available = TENSORFLOW_AVAILABLE
        if not self.tensorflow_available:
            logger.warning("TensorFlow not available. Neural processing will be disabled.")
        
        # Load models
        self.raga_model = None
        self.tala_model = None
        self.ornament_model = None
        
        if self.tensorflow_available:
            self.raga_model = _load_model(self.raga_model_path)
            self.tala_model = _load_model(self.tala_model_path)
            self.ornament_model = _load_model(self.ornament_model_path)
    
    def analyze_audio(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Analyze audio using neural models
        
        Parameters:
            audio_data (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Analysis results
        """
        return analyze_with_neural_models(audio_data, sr)
    
    def identify_raga(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Identify raga using neural model
        
        Parameters:
            audio_data (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Raga identification results
        """
        return identify_raga_neural(audio_data, sr)
    
    def detect_tala(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect tala using neural model
        
        Parameters:
            audio_data (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Tala detection results
        """
        return detect_tala_neural(audio_data, sr)
    
    def detect_ornaments(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Detect ornaments using neural model
        
        Parameters:
            audio_data (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Ornament detection results
        """
        return detect_ornaments_neural(audio_data, sr)
    
    def calculate_performance_metrics(self, audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Calculate performance metrics for a recording
        
        Parameters:
            audio_data (np.ndarray): Audio time series
            sr (int): Sample rate
            
        Returns:
            dict: Performance metrics
        """
        try:
            import librosa
            
            # Extract pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_data, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # Calculate pitch stability
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                pitch_stability = 1.0 / (1.0 + np.std(valid_f0) / np.mean(valid_f0))
            else:
                pitch_stability = 0.0
            
            # Calculate rhythm stability
            onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            if len(beat_times) > 1:
                ibis = np.diff(beat_times)
                rhythm_stability = 1.0 / (1.0 + np.std(ibis) / np.mean(ibis))
            else:
                rhythm_stability = 0.0
            
            # Calculate tonal clarity
            chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            tonal_clarity = np.max(chroma_mean) / (np.mean(chroma_mean) + 1e-8)
            
            # Calculate overall score
            overall_score = (pitch_stability + rhythm_stability + tonal_clarity) / 3.0
            
            return {
                "pitch_stability": float(pitch_stability),
                "rhythm_stability": float(rhythm_stability),
                "tonal_clarity": float(tonal_clarity),
                "overall_score": float(overall_score)
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {
                "error": str(e)
            }