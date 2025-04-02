"""
Audio Analysis Module for RagaVani

This module provides functions for analyzing Indian classical music audio.
It combines traditional signal processing with neural and symbolic approaches.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import logging

# Import neural and symbolic processing modules
from modules.neural_processing import analyze_with_neural_models, identify_raga_neural, detect_tala_neural
from modules.symbolic_processing import analyze_composition_symbolic

# Configure logging
logger = logging.getLogger(__name__)

def analyze_audio(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Analyze audio data to extract musical features using a hybrid approach
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Analysis results
    """
    try:
        # Initialize results dictionary
        results = {
            "duration": len(audio_data) / sr,
            "pitch_data": None,
            "detected_raga": None,
            "raga_confidence": None,
            "detected_tala": None,
            "tala_confidence": None,
            "ornaments": None,
            "neural_analysis": None,
            "symbolic_analysis": None,
            "error": None
        }
        
        # Extract pitch contour using traditional signal processing
        pitch_data = extract_pitch_contour(audio_data, sr)
        results["pitch_data"] = pitch_data
        
        # Perform neural analysis
        try:
            neural_results = analyze_with_neural_models(audio_data, sr)
            results["neural_analysis"] = neural_results
            
            # Use neural results for raga and tala if available
            if neural_results and "raga_analysis" in neural_results and neural_results["raga_analysis"]:
                raga_info = neural_results["raga_analysis"]
                results["detected_raga"] = raga_info.get("detected_raga")
                results["raga_confidence"] = raga_info.get("confidence")
                results["raga_details"] = raga_info
            
            if neural_results and "tala_analysis" in neural_results and neural_results["tala_analysis"]:
                tala_info = neural_results["tala_analysis"]
                results["detected_tala"] = tala_info.get("detected_tala")
                results["tala_confidence"] = tala_info.get("confidence")
                results["tala_details"] = tala_info
            
            if neural_results and "ornament_analysis" in neural_results and neural_results["ornament_analysis"]:
                results["ornaments"] = neural_results["ornament_analysis"].get("ornaments", [])
        except Exception as e:
            logger.warning(f"Neural analysis failed, falling back to traditional methods: {str(e)}")
            # Fall back to traditional methods if neural analysis fails
        
        # If neural analysis didn't provide results, use traditional methods
        if not results["detected_raga"]:
            raga_info = identify_raga(audio_data, sr, pitch_data)
            results["detected_raga"] = raga_info["name"]
            results["raga_confidence"] = raga_info["confidence"]
            results["raga_details"] = raga_info
        
        if not results["detected_tala"]:
            tala_info = detect_tala(audio_data, sr)
            results["detected_tala"] = tala_info["name"]
            results["tala_confidence"] = tala_info["confidence"]
            results["tala_details"] = tala_info
        
        if not results["ornaments"]:
            results["ornaments"] = detect_ornaments(audio_data, sr, pitch_data)
        
        # Perform symbolic analysis based on detected raga and tala
        try:
            # Create a composition representation from the analysis
            composition = {
                "raga": results["detected_raga"],
                "tala": results["detected_tala"],
                "swaras": extract_symbolic_representation(audio_data, sr, pitch_data),
                "rhythm": extract_rhythm_pattern(audio_data, sr)
            }
            
            # Analyze the composition using symbolic processing
            symbolic_results = analyze_composition_symbolic(composition)
            results["symbolic_analysis"] = symbolic_results
            
            # Add symbolic feedback to results
            if symbolic_results and "feedback" in symbolic_results:
                if "feedback" not in results:
                    results["feedback"] = []
                results["feedback"].extend(symbolic_results["feedback"])
        except Exception as e:
            logger.warning(f"Symbolic analysis failed: {str(e)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        return {
            "error": str(e),
            "duration": len(audio_data) / sr if audio_data is not None and sr is not None else 0
        }

def extract_pitch_contour(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Extract pitch contour from audio data
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Pitch contour data
    """
    try:
        # Use PYIN algorithm for pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data, 
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Convert to times
        times = librosa.times_like(f0, sr=sr)
        
        # Create pitch contour dictionary
        pitch_data = {
            "times": times.tolist(),
            "frequencies": f0.tolist(),
            "voiced_flag": voiced_flag.tolist(),
            "voiced_probs": voiced_probs.tolist()
        }
        
        # Calculate additional pitch statistics
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            pitch_data["mean_pitch"] = float(np.mean(valid_f0))
            pitch_data["min_pitch"] = float(np.min(valid_f0))
            pitch_data["max_pitch"] = float(np.max(valid_f0))
            pitch_data["pitch_range"] = float(np.max(valid_f0) - np.min(valid_f0))
        
        return pitch_data
        
    except Exception as e:
        logger.error(f"Error extracting pitch contour: {str(e)}")
        return {"error": str(e)}

def identify_raga(audio_data: np.ndarray, sr: int, pitch_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Identify the raga in the audio
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        pitch_data (dict): Pitch contour data
        
    Returns:
        dict: Raga identification results
    """
    try:
        # First try neural approach
        try:
            neural_results = identify_raga_neural(audio_data, sr)
            if neural_results and "detected_raga" in neural_results and neural_results["detected_raga"]:
                return {
                    "name": neural_results["detected_raga"],
                    "confidence": neural_results["confidence"],
                    "alternatives": neural_results.get("alternative_ragas", []),
                    "method": "neural"
                }
        except Exception as e:
            logger.warning(f"Neural raga identification failed, falling back to traditional method: {str(e)}")
        
        # Fall back to traditional approach
        # Extract chroma features
        chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr, bins_per_octave=24)
        
        # Calculate pitch histogram
        f0 = np.array(pitch_data["frequencies"])
        valid_f0 = f0[~np.isnan(f0)]
        
        # Convert to cents relative to tonic (assuming C as tonic for now)
        tonic_freq = librosa.note_to_hz('C4')  # Example tonic
        cents = 1200 * np.log2(valid_f0 / tonic_freq)
        
        # Create histogram
        hist, bin_edges = np.histogram(cents % 1200, bins=24, range=(0, 1200))
        
        # Normalize
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
        
        # Example raga profiles (simplified)
        raga_profiles = {
            "Yaman": np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]),  # S R G M P D N
            "Bhairav": np.array([1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1]),  # S r G M P d N
            "Bhimpalasi": np.array([1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]),  # S g M P n
            "Darbari": np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])   # S R g M P d n
        }
        
        # Resample histogram to 12 bins for comparison
        hist_12 = np.zeros(12)
        for i in range(24):
            hist_12[i // 2] += hist[i]
        
        # Normalize again
        if np.sum(hist_12) > 0:
            hist_12 = hist_12 / np.sum(hist_12)
        
        # Compare with raga profiles
        similarities = {}
        for raga_name, profile in raga_profiles.items():
            # Calculate cosine similarity
            similarity = np.dot(hist_12, profile) / (np.linalg.norm(hist_12) * np.linalg.norm(profile))
            similarities[raga_name] = float(similarity)
        
        # Find the best match
        best_raga = max(similarities, key=similarities.get)
        confidence = similarities[best_raga]
        
        # Get alternative ragas
        alternatives = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[1:4]
        
        return {
            "name": best_raga,
            "confidence": confidence,
            "alternatives": [{"name": name, "confidence": conf} for name, conf in alternatives],
            "pitch_histogram": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "chroma_mean": np.mean(chroma, axis=1).tolist(),
            "method": "traditional"
        }
        
    except Exception as e:
        logger.error(f"Error identifying raga: {str(e)}")
        return {"name": "Unknown", "confidence": 0.0, "error": str(e)}

def detect_tala(audio_data: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Detect the tala in the audio
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        dict: Tala detection results
    """
    try:
        # First try neural approach
        try:
            neural_results = detect_tala_neural(audio_data, sr)
            if neural_results and "detected_tala" in neural_results and neural_results["detected_tala"]:
                return {
                    "name": neural_results["detected_tala"],
                    "confidence": neural_results["confidence"],
                    "alternatives": neural_results.get("alternative_talas", []),
                    "tempo": neural_results.get("tempo"),
                    "beat_positions": neural_results.get("beat_positions", []),
                    "method": "neural"
                }
        except Exception as e:
            logger.warning(f"Neural tala detection failed, falling back to traditional method: {str(e)}")
        
        # Fall back to traditional approach
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # Detect tempo and beats
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Calculate inter-beat intervals
        ibis = np.diff(beat_times)
        
        # Analyze beat patterns
        beat_pattern = analyze_beat_pattern(beat_times, ibis)
        
        # Determine tala based on beat pattern
        tala_info = determine_tala(beat_pattern, tempo)
        
        # Add beat information to results
        tala_info["tempo"] = float(tempo)
        tala_info["beat_times"] = beat_times.tolist()
        tala_info["inter_beat_intervals"] = ibis.tolist()
        tala_info["beat_pattern"] = beat_pattern
        tala_info["method"] = "traditional"
        
        return tala_info
        
    except Exception as e:
        logger.error(f"Error detecting tala: {str(e)}")
        return {"name": "Unknown", "confidence": 0.0, "error": str(e)}

def analyze_beat_pattern(beat_times: np.ndarray, ibis: np.ndarray) -> Dict[str, Any]:
    """
    Analyze beat pattern to find recurring structures
    
    Parameters:
        beat_times (np.ndarray): Beat times in seconds
        ibis (np.ndarray): Inter-beat intervals
        
    Returns:
        dict: Beat pattern analysis
    """
    # Initialize results
    pattern = {
        "total_beats": len(beat_times),
        "regularity": 0.0,
        "cycle_length": 0,
        "cycles": 0
    }
    
    # Check if we have enough beats to analyze
    if len(ibis) < 4:
        return pattern
    
    # Calculate regularity (inverse of coefficient of variation)
    cv = np.std(ibis) / np.mean(ibis)
    pattern["regularity"] = 1.0 / (1.0 + cv)
    
    # Try to find cycle lengths
    for cycle_length in [16, 12, 10, 8, 7, 6, 5, 4, 3]:
        if len(beat_times) >= cycle_length * 2:
            # Check if beat pattern repeats at this cycle length
            cycles = len(beat_times) // cycle_length
            
            # Group IBIs by cycle
            cycle_ibis = []
            for i in range(cycles - 1):
                cycle_ibis.append(ibis[i*cycle_length:(i+1)*cycle_length])
            
            # Calculate similarity between cycles
            similarities = []
            for i in range(len(cycle_ibis) - 1):
                for j in range(i + 1, len(cycle_ibis)):
                    # Calculate correlation
                    corr = np.corrcoef(cycle_ibis[i], cycle_ibis[j])[0, 1]
                    if not np.isnan(corr):
                        similarities.append(corr)
            
            # If we have similarities and they're high enough, we found a cycle
            if similarities and np.mean(similarities) > 0.6:
                pattern["cycle_length"] = cycle_length
                pattern["cycles"] = cycles
                pattern["cycle_similarity"] = float(np.mean(similarities))
                break
    
    return pattern

def determine_tala(beat_pattern: Dict[str, Any], tempo: float) -> Dict[str, Any]:
    """
    Determine the tala based on beat pattern and tempo
    
    Parameters:
        beat_pattern (dict): Beat pattern analysis
        tempo (float): Tempo in BPM
        
    Returns:
        dict: Tala information
    """
    # Initialize results
    tala_info = {
        "name": "Unknown",
        "confidence": 0.0,
        "alternatives": []
    }
    
    # Define tala characteristics
    talas = {
        "Teentaal": {"beats": 16, "tempo_range": [60, 300]},
        "Ektaal": {"beats": 12, "tempo_range": [40, 200]},
        "Jhaptaal": {"beats": 10, "tempo_range": [50, 200]},
        "Keherwa": {"beats": 8, "tempo_range": [70, 280]},
        "Rupak": {"beats": 7, "tempo_range": [60, 180]},
        "Dadra": {"beats": 6, "tempo_range": [80, 200]}
    }
    
    # Calculate match scores for each tala
    scores = {}
    for tala_name, tala_props in talas.items():
        score = 0.0
        
        # Check cycle length
        if beat_pattern["cycle_length"] == tala_props["beats"]:
            score += 0.6
        elif beat_pattern["cycle_length"] > 0 and beat_pattern["cycle_length"] % tala_props["beats"] == 0:
            score += 0.3
        
        # Check tempo range
        if tala_props["tempo_range"][0] <= tempo <= tala_props["tempo_range"][1]:
            score += 0.2
        
        # Check regularity
        if beat_pattern["regularity"] > 0.8:
            score += 0.2
        
        scores[tala_name] = score
    
    # Find the best match
    if scores:
        best_tala = max(scores, key=scores.get)
        confidence = scores[best_tala]
        
        if confidence > 0.3:
            tala_info["name"] = best_tala
            tala_info["confidence"] = confidence
            
            # Get alternative talas
            alternatives = sorted(scores.items(), key=lambda x: x[1], reverse=True)[1:4]
            tala_info["alternatives"] = [{"name": name, "confidence": conf} for name, conf in alternatives]
    
    return tala_info

def detect_ornaments(audio_data: np.ndarray, sr: int, pitch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect ornaments in the audio
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        pitch_data (dict): Pitch contour data
        
    Returns:
        list: Detected ornaments
    """
    try:
        ornaments = []
        
        # Extract pitch contour
        f0 = np.array(pitch_data["frequencies"])
        times = np.array(pitch_data["times"])
        
        # Calculate pitch derivatives
        pitch_diff = np.diff(f0)
        pitch_diff[np.isnan(pitch_diff)] = 0
        
        # Smooth the differences
        from scipy.ndimage import gaussian_filter1d
        smooth_diff = gaussian_filter1d(np.abs(pitch_diff), sigma=5)
        
        # Find peaks in the smoothed differences
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smooth_diff, height=50, distance=50)
        
        # Analyze each peak to identify ornament type
        for i, peak in enumerate(peaks):
            if peak >= len(times):
                continue
                
            # Get a window around the peak
            start_idx = max(0, peak - 25)
            end_idx = min(len(f0) - 1, peak + 25)
            
            window = f0[start_idx:end_idx]
            window_times = times[start_idx:end_idx]
            
            # Skip if too many NaN values
            if np.isnan(window).sum() > len(window) * 0.5:
                continue
            
            # Replace NaN with interpolated values
            window_clean = np.copy(window)
            nan_mask = np.isnan(window_clean)
            window_clean[nan_mask] = np.interp(
                np.flatnonzero(nan_mask), 
                np.flatnonzero(~nan_mask), 
                window_clean[~nan_mask]
            )
            
            # Calculate features for classification
            pitch_range = np.max(window_clean) - np.min(window_clean)
            pitch_std = np.std(window_clean)
            duration = window_times[-1] - window_times[0]
            
            # Classify ornament type
            ornament_type = "unknown"
            confidence = 0.5
            
            if pitch_range > 200 and duration > 0.3:
                ornament_type = "meend"  # Slide
                confidence = min(1.0, pitch_range / 400)
            elif pitch_std > 30 and duration < 0.2:
                ornament_type = "kan"  # Grace note
                confidence = min(1.0, pitch_std / 60)
            elif pitch_std > 20 and 0.2 < duration < 0.5:
                ornament_type = "andolan"  # Oscillation
                confidence = min(1.0, pitch_std / 40)
            elif pitch_std > 40:
                ornament_type = "gamak"  # Heavy oscillation
                confidence = min(1.0, pitch_std / 80)
            elif 10 < pitch_std < 30 and duration < 0.3:
                ornament_type = "murki"  # Quick turn
                confidence = min(1.0, pitch_std / 30)
            
            # Add to results if confidence is high enough
            if confidence > 0.6:
                ornaments.append({
                    "type": ornament_type,
                    "start_time": float(window_times[0]),
                    "end_time": float(window_times[-1]),
                    "duration": float(duration),
                    "pitch_range": float(pitch_range),
                    "confidence": float(confidence)
                })
        
        return ornaments
        
    except Exception as e:
        logger.error(f"Error detecting ornaments: {str(e)}")
        return []

def extract_symbolic_representation(audio_data: np.ndarray, sr: int, pitch_data: Dict[str, Any]) -> str:
    """
    Extract symbolic representation (swaras) from audio data
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        pitch_data (dict): Pitch contour data
        
    Returns:
        str: Symbolic representation as a string of swaras
    """
    try:
        # Extract pitch contour
        f0 = np.array(pitch_data["frequencies"])
        times = np.array(pitch_data["times"])
        
        # Remove NaN values
        valid_indices = ~np.isnan(f0)
        valid_f0 = f0[valid_indices]
        valid_times = times[valid_indices]
        
        if len(valid_f0) == 0:
            return ""
        
        # Estimate tonic frequency (using the most common pitch class)
        cents = 1200 * np.log2(valid_f0 / valid_f0[0])
        cents_mod = cents % 1200
        hist, _ = np.histogram(cents_mod, bins=24, range=(0, 1200))
        tonic_bin = np.argmax(hist)
        tonic_cents = tonic_bin * 50  # 50 cents per bin
        tonic_freq = valid_f0[0] * 2**(tonic_cents/1200)
        
        # Define swara positions in cents
        swara_positions = {
            "S": 0,
            "r": 100,
            "R": 200,
            "g": 300,
            "G": 400,
            "m": 500,
            "M": 600,
            "P": 700,
            "d": 800,
            "D": 900,
            "n": 1000,
            "N": 1100,
            "S'": 1200
        }
        
        # Quantize pitch to swaras
        swaras = []
        current_swara = None
        min_duration = 0.1  # Minimum duration for a swara in seconds
        last_change_time = valid_times[0]
        
        for i in range(len(valid_f0)):
            # Calculate cents relative to tonic
            rel_cents = 1200 * np.log2(valid_f0[i] / tonic_freq)
            
            # Find closest swara
            octave = int(rel_cents / 1200)
            cents_in_octave = rel_cents % 1200
            
            closest_swara = None
            min_dist = float('inf')
            
            for swara, pos in swara_positions.items():
                dist = min(abs(cents_in_octave - pos), abs(cents_in_octave - pos - 1200))
                if dist < min_dist:
                    min_dist = dist
                    closest_swara = swara
            
            # Adjust for octave
            if closest_swara == "S'" and octave > 0:
                closest_swara = "S"  # Use S for higher octave Sa
            elif closest_swara != "S'" and octave > 0:
                closest_swara += "'"  # Add ' for higher octave
            elif octave < 0:
                closest_swara = closest_swara.lower()  # Use lowercase for lower octave
            
            # Check if swara has changed
            if closest_swara != current_swara:
                # Add previous swara if it lasted long enough
                if current_swara and valid_times[i] - last_change_time >= min_duration:
                    swaras.append(current_swara)
                
                current_swara = closest_swara
                last_change_time = valid_times[i]
        
        # Add the last swara if it lasted long enough
        if current_swara and valid_times[-1] - last_change_time >= min_duration:
            swaras.append(current_swara)
        
        return " ".join(swaras)
        
    except Exception as e:
        logger.error(f"Error extracting symbolic representation: {str(e)}")
        return ""

def extract_rhythm_pattern(audio_data: np.ndarray, sr: int) -> str:
    """
    Extract rhythm pattern from audio data
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        sr (int): Sample rate
        
    Returns:
        str: Rhythm pattern as a string
    """
    try:
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        
        # Detect onsets
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units='time'
        )
        
        if len(onsets) == 0:
            return ""
        
        # Detect tempo and beats
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Quantize onsets to beats
        quantized_onsets = []
        for onset in onsets:
            # Find closest beat
            closest_beat = np.argmin(np.abs(beat_times - onset))
            quantized_onsets.append(closest_beat)
        
        # Count occurrences of each beat position
        beat_counts = np.bincount(quantized_onsets)
        
        # Normalize to get beat strength
        if np.sum(beat_counts) > 0:
            beat_strength = beat_counts / np.max(beat_counts)
        else:
            beat_strength = beat_counts
        
        # Convert to rhythm pattern
        rhythm_pattern = []
        
        # Define bol patterns based on beat strength
        for strength in beat_strength:
            if strength > 0.8:
                rhythm_pattern.append("Dha")
            elif strength > 0.6:
                rhythm_pattern.append("Dhin")
            elif strength > 0.4:
                rhythm_pattern.append("Tin")
            elif strength > 0.2:
                rhythm_pattern.append("Na")
            elif strength > 0:
                rhythm_pattern.append("Ta")
        
        # Group into vibhags (sections)
        vibhag_sizes = [4, 4, 4, 4]  # Default to Teentaal pattern
        
        # Try to determine vibhag sizes based on beat pattern
        if len(rhythm_pattern) == 16:
            vibhag_sizes = [4, 4, 4, 4]  # Teentaal
        elif len(rhythm_pattern) == 12:
            vibhag_sizes = [2, 2, 2, 2, 2, 2]  # Ektaal
        elif len(rhythm_pattern) == 10:
            vibhag_sizes = [2, 3, 2, 3]  # Jhaptaal
        elif len(rhythm_pattern) == 8:
            vibhag_sizes = [4, 4]  # Keherwa
        elif len(rhythm_pattern) == 7:
            vibhag_sizes = [3, 2, 2]  # Rupak
        elif len(rhythm_pattern) == 6:
            vibhag_sizes = [3, 3]  # Dadra
        
        # Format the rhythm pattern with vibhag separators
        formatted_pattern = ""
        current_pos = 0
        
        for size in vibhag_sizes:
            if current_pos + size <= len(rhythm_pattern):
                vibhag = rhythm_pattern[current_pos:current_pos+size]
                formatted_pattern += " ".join(vibhag)
                current_pos += size
                
                if current_pos < len(rhythm_pattern):
                    formatted_pattern += " | "
        
        return formatted_pattern
        
    except Exception as e:
        logger.error(f"Error extracting rhythm pattern: {str(e)}")
        return ""