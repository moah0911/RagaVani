"""
Audio Analysis Module for RagaVani application

This module provides comprehensive functionality for analyzing Indian classical music recordings,
including advanced pitch detection, raga identification, tala detection, and ornament recognition.
It combines traditional DSP techniques with neural processing for more accurate analysis.
"""

import numpy as np
import librosa
import scipy.signal as signal
import math
import time
import os
import matplotlib.pyplot as plt

def analyze_audio(y, sr, cache=None, skip_tala=False, skip_ornaments=False, segment_audio=True):
    """
    Analyze audio file and extract key features

    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        cache (dict, optional): Cache of previous analysis results to avoid recomputation
        skip_tala (bool): Whether to skip tala detection (for efficiency)
        skip_ornaments (bool): Whether to skip ornament detection (for efficiency)
        segment_audio (bool): Whether to segment audio with multiple sounds for better analysis

    Returns:
        dict: Dictionary containing analysis results
    """
    start_time = time.time()
    print(f"Input audio duration: {len(y)/sr:.2f} seconds")
    print(f"Max amplitude: {np.max(np.abs(y)):.4f}")

    # Initialize results dictionary
    results = {
        "duration": len(y) / sr if y is not None else 0,
        "pitch_data": None,
        "detected_raga": None,
        "raga_confidence": None,
        "detected_tala": None,
        "tala_confidence": None,
        "ornaments": None,
        "analysis_time": None,
        "segments": None
    }

    # Use cache if provided
    if cache is not None:
        # If we have cached pitch data, use it
        if "pitch_data" in cache:
            results["pitch_data"] = cache["pitch_data"]
            times = cache["pitch_data"]["times"]
            pitches = cache["pitch_data"]["pitches"]
            confidence = cache["pitch_data"]["confidence"]
            print("Using cached pitch data")

        # If we have cached raga results, use them
        if "detected_raga" in cache and cache["detected_raga"] is not None:
            results["detected_raga"] = cache["detected_raga"]
            results["raga_confidence"] = cache["raga_confidence"]
            results["raga_details"] = cache["raga_details"]
            print(f"Using cached raga detection: {cache['detected_raga']} ({cache['raga_confidence']:.1f}%)")

        # If we have cached tala results and not skipping tala, use them
        if not skip_tala and "detected_tala" in cache and cache["detected_tala"] is not None:
            results["detected_tala"] = cache["detected_tala"]
            results["tala_confidence"] = cache["tala_confidence"]
            results["tala_details"] = cache["tala_details"]
            print(f"Using cached tala detection: {cache['detected_tala']} ({cache['tala_confidence']:.1f}%)")

        # If we have cached ornaments and not skipping ornaments, use them
        if not skip_ornaments and "ornaments" in cache and cache["ornaments"] is not None:
            results["ornaments"] = cache["ornaments"]
            print(f"Using cached ornaments: {len(cache['ornaments'])} detected")

        # If we have cached segments, use them
        if "segments" in cache and cache["segments"] is not None:
            results["segments"] = cache["segments"]
            print(f"Using cached segments: {len(cache['segments'])} segments")

    try:
        # Check if we should segment the audio (for files with multiple sounds)
        audio_segments = []

        if segment_audio and len(y) / sr > 30 and results["segments"] is None:  # Only segment longer files
            print("Attempting to segment audio with multiple sounds...")

            # Detect significant silence or amplitude changes
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]

            # Normalize RMS
            if np.max(rms) > 0:
                rms = rms / np.max(rms)

            # Find segments with significant sound
            silence_threshold = 0.1
            min_segment_length = sr * 5  # At least 5 seconds

            # Find silence boundaries
            is_silence = rms < silence_threshold

            # Convert to sample indices
            silence_starts = np.where(np.diff(is_silence.astype(int)) > 0)[0] * hop_length
            silence_ends = np.where(np.diff(is_silence.astype(int)) < 0)[0] * hop_length

            # Ensure we have matching starts and ends
            if len(silence_starts) > 0 and len(silence_ends) > 0:
                # If audio starts with sound, add a start at 0
                if silence_starts[0] > silence_ends[0]:
                    silence_starts = np.insert(silence_starts, 0, 0)

                # If audio ends with sound, add an end at the end
                if silence_starts[-1] > silence_ends[-1]:
                    silence_ends = np.append(silence_ends, len(y))

                # Create segments
                segments = []
                for start, end in zip(silence_ends, silence_starts[1:] if len(silence_starts) > 1 else [len(y)]):
                    if end - start >= min_segment_length:
                        segments.append((start, end))

                # If we found valid segments
                if segments:
                    print(f"Found {len(segments)} distinct audio segments")

                    # Analyze each segment
                    for i, (start, end) in enumerate(segments):
                        segment_y = y[start:end]
                        print(f"\nAnalyzing segment {i+1}/{len(segments)} (duration: {len(segment_y)/sr:.2f}s)")

                        # Analyze this segment
                        segment_results = analyze_audio(
                            segment_y, sr,
                            skip_tala=skip_tala,
                            skip_ornaments=skip_ornaments,
                            segment_audio=False  # Prevent recursive segmentation
                        )

                        # Add to segments list
                        audio_segments.append({
                            "start_time": start / sr,
                            "end_time": end / sr,
                            "duration": (end - start) / sr,
                            "results": segment_results
                        })

                    # Store segments in results
                    results["segments"] = audio_segments

                    # Use the highest confidence results as the main results
                    if audio_segments:
                        # For raga
                        raga_segments = [s for s in audio_segments if s["results"]["detected_raga"] is not None]
                        if raga_segments:
                            best_raga = max(raga_segments, key=lambda x: x["results"]["raga_confidence"])
                            results["detected_raga"] = best_raga["results"]["detected_raga"]
                            results["raga_confidence"] = best_raga["results"]["raga_confidence"]
                            results["raga_details"] = best_raga["results"]["raga_details"]
                            print(f"Selected best raga from segment {raga_segments.index(best_raga)+1}: {results['detected_raga']} ({results['raga_confidence']:.1f}%)")

                        # For tala
                        if not skip_tala:
                            tala_segments = [s for s in audio_segments if s["results"]["detected_tala"] is not None]
                            if tala_segments:
                                best_tala = max(tala_segments, key=lambda x: x["results"]["tala_confidence"])
                                results["detected_tala"] = best_tala["results"]["detected_tala"]
                                results["tala_confidence"] = best_tala["results"]["tala_confidence"]
                                results["tala_details"] = best_tala["results"]["tala_details"]
                                print(f"Selected best tala from segment {tala_segments.index(best_tala)+1}: {results['detected_tala']} ({results['tala_confidence']:.1f}%)")

                        # For ornaments
                        if not skip_ornaments:
                            all_ornaments = []
                            for segment in audio_segments:
                                if segment["results"]["ornaments"]:
                                    # Adjust time offsets for ornaments
                                    for ornament in segment["results"]["ornaments"]:
                                        if "start_time" in ornament:
                                            ornament["start_time"] += segment["start_time"]
                                        if "end_time" in ornament:
                                            ornament["end_time"] += segment["start_time"]
                                        if "time" in ornament:
                                            ornament["time"] += segment["start_time"]
                                        all_ornaments.append(ornament)

                            if all_ornaments:
                                results["ornaments"] = all_ornaments
                                print(f"Combined {len(all_ornaments)} ornaments from all segments")

                        # Combine pitch data
                        all_times = []
                        all_pitches = []
                        all_confidence = []

                        for segment in audio_segments:
                            if segment["results"]["pitch_data"] is not None:
                                seg_times = segment["results"]["pitch_data"]["times"]
                                seg_pitches = segment["results"]["pitch_data"]["pitches"]
                                seg_confidence = segment["results"]["pitch_data"]["confidence"]

                                # Adjust time offsets
                                adjusted_times = seg_times + segment["start_time"]

                                all_times.extend(adjusted_times)
                                all_pitches.extend(seg_pitches)
                                all_confidence.extend(seg_confidence)

                        if all_times:
                            # Convert to numpy arrays and sort by time
                            all_times = np.array(all_times)
                            all_pitches = np.array(all_pitches)
                            all_confidence = np.array(all_confidence)

                            # Sort by time
                            sort_idx = np.argsort(all_times)
                            all_times = all_times[sort_idx]
                            all_pitches = all_pitches[sort_idx]
                            all_confidence = all_confidence[sort_idx]

                            results["pitch_data"] = {
                                "times": all_times,
                                "pitches": all_pitches,
                                "confidence": all_confidence
                            }

                    # Record total analysis time
                    results["analysis_time"] = time.time() - start_time
                    print(f"Total analysis completed in {results['analysis_time']:.2f} seconds")

                    return results

        # If we didn't segment or no segments found, proceed with normal analysis

        # Detect pitch if not cached
        if results["pitch_data"] is None:
            pitch_start = time.time()
            times, pitches, confidence = detect_pitch(y, sr)
            pitch_time = time.time() - pitch_start

            if len(pitches) > 0:
                print(f"Valid pitch values detected: {np.sum(pitches > 0)}")
                print(f"Pitch range: {np.min(pitches[pitches>0]):.1f} - {np.max(pitches[pitches>0]):.1f} Hz")
                print(f"Pitch detection completed in {pitch_time:.2f} seconds")
            else:
                print("No valid pitch values detected")

            results["pitch_data"] = {
                "times": times,
                "pitches": pitches,
                "confidence": confidence
            }

        # Identify raga if not cached and we have pitch data
        if results["detected_raga"] is None and results["pitch_data"] is not None and len(results["pitch_data"]["pitches"]) > 0:
            raga_start = time.time()
            raga_results = identify_raga(results["pitch_data"]["pitches"], confidence_threshold=60)
            raga_time = time.time() - raga_start

            if raga_results:
                results["detected_raga"] = raga_results["name"]
                results["raga_confidence"] = raga_results["confidence"]
                results["raga_details"] = raga_results["details"]
                print(f"Raga detected: {raga_results['name']} ({raga_results['confidence']:.1f}%)")
                print(f"Raga detection completed in {raga_time:.2f} seconds")
            else:
                print("No raga detected with sufficient confidence")

        # Detect tala if not cached and not skipped
        if results["detected_tala"] is None and not skip_tala:
            tala_start = time.time()
            try:
                tala_results = detect_tala(y, sr, confidence_threshold=60)
                if tala_results:
                    results["detected_tala"] = tala_results["name"]
                    results["tala_confidence"] = tala_results["confidence"]
                    results["tala_details"] = tala_results["details"]
                    print(f"Tala detected: {tala_results['name']} ({tala_results['confidence']:.1f}%)")
                    print(f"Tala detection completed in {time.time() - tala_start:.2f} seconds")
                else:
                    print("No tala detected with sufficient confidence")
            except Exception as e:
                print(f"Error in tala detection: {e}")
                # Try with a simpler approach if the main one fails
                try:
                    print("Trying simplified tala detection...")
                    tala_results = detect_tala_simple(y, sr, confidence_threshold=60)
                    if tala_results:
                        results["detected_tala"] = tala_results["name"]
                        results["tala_confidence"] = tala_results["confidence"]
                        results["tala_details"] = tala_results["details"]
                        print(f"Tala detected (simplified): {tala_results['name']} ({tala_results['confidence']:.1f}%)")
                    else:
                        print("No tala detected with simplified method")
                except Exception as e2:
                    print(f"Error in simplified tala detection: {e2}")

        # Detect ornaments if not cached, not skipped, and we have pitch data
        if results["ornaments"] is None and not skip_ornaments and results["pitch_data"] is not None:
            times = results["pitch_data"]["times"]
            pitches = results["pitch_data"]["pitches"]

            if len(times) > 0 and len(pitches) > 0:
                ornament_start = time.time()
                try:
                    ornaments = detect_ornaments(y, sr, times, pitches)
                    if ornaments:
                        results["ornaments"] = ornaments
                        print(f"Detected {len(ornaments)} ornaments")
                        print(f"Ornament detection completed in {time.time() - ornament_start:.2f} seconds")
                    else:
                        print("No ornaments detected")
                except Exception as e:
                    print(f"Error in ornament detection: {e}")

    except Exception as e:
        print(f"Error analyzing audio: {e}")
        import traceback
        traceback.print_exc()

    # Record total analysis time
    results["analysis_time"] = time.time() - start_time
    print(f"Total analysis completed in {results['analysis_time']:.2f} seconds")

    return results


def detect_tala_simple(y, sr, confidence_threshold=60):
    """
    Simplified tala detection for compatibility with older librosa versions

    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        confidence_threshold (float): Minimum confidence to return a match

    Returns:
        dict or None: Dictionary with tala name, confidence, and analysis data, or None if no match
    """
    try:
        # Add minimum duration check
        if len(y) / sr < 5:  # Less than 5 seconds
            print("Audio too short for reliable tala detection")
            return None

        # Ensure audio has enough amplitude
        if np.max(np.abs(y)) < 0.01:
            print("Audio signal too weak for tala detection")
            return None

        # Simple onset detection
        hop_length = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Normalize manually
        if np.max(onset_env) > 0:
            onset_env = onset_env / np.max(onset_env)

        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            start_bpm=80  # Typical tempo range for Indian classical
        )

        if len(beats) < 8:  # Need at least 8 beats for reliable analysis
            print("Not enough beats detected for tala analysis")
            return None

        # Calculate beat intervals
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
        beat_intervals = np.diff(beat_times)

        # Simple pattern detection
        # Look for patterns in beat groupings
        tala_votes = {}

        for group_size in [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
            if len(beat_intervals) >= group_size * 2:
                # Check if this grouping creates a consistent pattern
                groups = [beat_intervals[i:i+group_size].sum() for i in range(0, len(beat_intervals) - group_size, group_size)]
                if len(groups) >= 2:
                    mean_group = np.mean(groups)
                    std_group = np.std(groups)

                    # Calculate consistency (lower std = more consistent)
                    if mean_group > 0:
                        consistency = 1.0 - min(1.0, std_group / mean_group)

                        # Only consider reasonably consistent patterns
                        if consistency > 0.6:
                            tala_votes[group_size] = consistency

        # Define common talas
        tala_definitions = {
            3: {"name": "Dadra", "weight": 1.0},
            4: {"name": "Keherwa", "weight": 1.2},  # Very common
            5: {"name": "Jhaptaal (half)", "weight": 0.9},
            6: {"name": "Dadra (extended)", "weight": 0.8},
            7: {"name": "Rupak Taal", "weight": 1.0},
            8: {"name": "Keherwa (extended)", "weight": 1.1},
            10: {"name": "Jhaptaal", "weight": 1.1},
            12: {"name": "Ektaal", "weight": 1.0},
            14: {"name": "Dhamar", "weight": 0.9},
            16: {"name": "Teentaal", "weight": 1.3}  # Most common
        }

        # Apply weights to votes
        weighted_votes = {}
        for tala_beats, consistency in tala_votes.items():
            if tala_beats in tala_definitions:
                weighted_votes[tala_beats] = consistency * tala_definitions[tala_beats]["weight"]

        # Find the tala with the highest vote
        if weighted_votes:
            best_tala = max(weighted_votes.items(), key=lambda x: x[1])
            tala_beats = best_tala[0]

            # Scale confidence to 0-100
            confidence = min(100, best_tala[1] * 100)

            if confidence >= confidence_threshold:
                return {
                    "name": tala_definitions[tala_beats]["name"],
                    "confidence": confidence,
                    "details": {
                        "detected_cycle": tala_beats,
                        "tempo": tempo,
                        "consistency": tala_votes[tala_beats]
                    }
                }

        return None

    except Exception as e:
        print(f"Error in simplified tala detection: {e}")
        return None

def detect_pitch(y, sr, fmin=50, fmax=2000, algorithm="combined"):
    """
    Detect pitch in the audio signal using multiple algorithms for higher accuracy
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        fmin (float): Minimum frequency for pitch detection
        fmax (float): Maximum frequency for pitch detection
        algorithm (str): Algorithm to use: "combined", "pyin", "crepe", "spectral"
    
    Returns:
        tuple: (times, pitches, confidence) arrays
    """
    try:
        if algorithm == "pyin" or algorithm == "combined":
            # Use piptrack from librosa directly
            pitches, magnitudes = librosa.piptrack(
                y=y, sr=sr,
                fmin=fmin,
                fmax=fmax
            )
            
            # Convert to usable format
            times = librosa.times_like(pitches)
            pitch_values = np.zeros_like(times)
            confidence_values = np.zeros_like(times)
            
            # Extract highest magnitude pitch for each frame
            for i in range(pitches.shape[1]):
                index = magnitudes[:, i].argmax()
                pitch_values[i] = pitches[index, i] if magnitudes[index, i] > 0 else 0
                confidence_values[i] = magnitudes[index, i] / np.max(magnitudes) if np.max(magnitudes) > 0 else 0
                
            # Filter out invalid values before returning
            valid_mask = (pitch_values > 0) & np.isfinite(pitch_values)
            times = times[valid_mask]
            pitch_values = pitch_values[valid_mask]
            confidence_values = confidence_values[valid_mask]
            
            if algorithm == "pyin":
                return times, pitch_values, confidence_values
        
        # Simplified spectral approach (for demonstration)
        if algorithm == "spectral" or algorithm == "combined":
            # Use a simplified spectral approach for pitch detection
            frame_length = 2048
            hop_length = 512
            
            # Calculate STFT
            D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
            
            # Convert to magnitude
            S = np.abs(D)
            
            # Find the frequency bin with maximum amplitude for each frame
            max_bins = np.argmax(S, axis=0)
            
            # Convert bin indices to frequency
            spectral_times = librosa.times_like(S[0], sr=sr, hop_length=hop_length)
            spectral_freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
            spectral_pitches = np.array([spectral_freqs[bin_idx] if bin_idx > 0 else 0 for bin_idx in max_bins])
            
            # Calculate confidence based on magnitude
            max_magnitudes = np.array([S[bin_idx, i] for i, bin_idx in enumerate(max_bins)])
            spectral_confidence = max_magnitudes / np.max(max_magnitudes) if np.max(max_magnitudes) > 0 else np.zeros_like(max_magnitudes)
            
            # Filter frequencies outside the desired range
            for i in range(len(spectral_pitches)):
                if spectral_pitches[i] < fmin or spectral_pitches[i] > fmax:
                    spectral_pitches[i] = 0
                    spectral_confidence[i] = 0
            
            if algorithm == "spectral":
                return spectral_times, spectral_pitches, spectral_confidence
        
        # If using combined approach, combine the results
        if algorithm == "combined":
            # Interpolate to align the time scales
            if len(times) > 0 and len(spectral_times) > 0:
                import scipy.interpolate
                
                # Interpolate spectral results to match PYIN time points
                interp_func = scipy.interpolate.interp1d(
                    spectral_times, 
                    spectral_pitches, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=0
                )
                
                spectral_at_pyin_times = interp_func(times)
                
                # Interpolate confidence values
                interp_func = scipy.interpolate.interp1d(
                    spectral_times, 
                    spectral_confidence, 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=0
                )
                
                spectral_conf_at_pyin_times = interp_func(times)
                
                # Combine the pitch estimates based on confidence
                combined_pitches = np.zeros_like(pitch_values)
                combined_confidence = np.zeros_like(confidence_values)
                
                for i in range(len(times)):
                    if confidence_values[i] > spectral_conf_at_pyin_times[i]:
                        combined_pitches[i] = pitch_values[i]
                        combined_confidence[i] = confidence_values[i]
                    else:
                        combined_pitches[i] = spectral_at_pyin_times[i]
                        combined_confidence[i] = spectral_conf_at_pyin_times[i]
                
                return times, combined_pitches, combined_confidence
            
            # If one of the methods failed, return the available results
            if len(times) > 0:
                return times, pitch_values, confidence_values
            elif len(spectral_times) > 0:
                return spectral_times, spectral_pitches, spectral_confidence
        
        # Default fallback
        import scipy.signal
        
        # Use a simple zero-crossing method as fallback
        frame_length = 2048
        hop_length = 512
        
        times = []
        pitches = []
        confidence = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            
            # Calculate zero crossings
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame))))
            if (zero_crossings > 2):  # Need at least a couple of crossings to estimate period
                # Estimate frequency from zero crossings
                estimated_freq = zero_crossings * sr / (2 * frame_length)
                
                # Simple confidence based on amplitude
                frame_confidence = np.std(frame) / 0.1  # Normalize to a reasonable range
                frame_confidence = max(0, min(1, frame_confidence))  # Clip to [0, 1]
                
                # Only keep if in the desired frequency range
                if fmin <= estimated_freq <= fmax:
                    times.append(i / sr)
                    pitches.append(estimated_freq)
                    confidence.append(frame_confidence)
        
        # Convert to numpy arrays
        if times:
            return np.array(times), np.array(pitches), np.array(confidence)
        else:
            # Empty results
            return np.array([]), np.array([]), np.array([])
    
    except Exception as e:
        print(f"Error detecting pitch: {e}")
        return np.array([]), np.array([]), np.array([])

def identify_raga(pitch_values, confidence_threshold=40, include_transitions=True, vadi_samvadi_weight=True):
    """
    Advanced raga identification with multiple analysis methods

    Parameters:
        pitch_values (np.ndarray): Array of pitch values
        confidence_threshold (float): Minimum confidence to return a match
        include_transitions (bool): Whether to analyze note transitions
        vadi_samvadi_weight (bool): Whether to give additional weight to vadi and samvadi notes

    Returns:
        dict or None: Dictionary with raga name, confidence and analysis details, or None if no match
    """
    try:
        # Filter out invalid pitch values first
        valid_pitches = pitch_values[np.where((pitch_values > 0) & np.isfinite(pitch_values))]

        if len(valid_pitches) < 10:
            print("Not enough valid pitch data for raga analysis")
            return None

        print(f"Valid pitch values for raga analysis: {len(valid_pitches)}")

        # Extract notes from pitches with improved accuracy
        try:
            notes_hist, note_transitions = extract_notes_with_transitions(valid_pitches)
            print(f"Extracted notes: {len(notes_hist)} unique notes")
        except Exception as e:
            print(f"Error extracting notes: {e}")
            # Fallback to simpler note extraction
            notes_hist = extract_notes_histogram(valid_pitches)
            note_transitions = set()
            print(f"Fallback note extraction: {len(notes_hist)} unique notes")

        # If we still don't have notes, use a very simple approach
        if not notes_hist:
            print("Using simplified note extraction")
            notes_hist = {}
            # Convert to MIDI notes and count occurrences
            midi_notes = librosa.hz_to_midi(valid_pitches)
            note_classes = midi_notes % 12

            # Map to Indian classical note names
            note_map = {0: 'S', 1: 'r', 2: 'R', 3: 'g', 4: 'G',
                        5: 'm', 6: 'M', 7: 'P', 8: 'd', 9: 'D',
                        10: 'n', 11: 'N'}

            for note_class in note_classes:
                note_name = note_map[int(note_class)]
                if note_name in notes_hist:
                    notes_hist[note_name] += 1
                else:
                    notes_hist[note_name] = 1

        # Get list of present notes with improved threshold
        # Use a dynamic threshold based on audio length
        threshold = max(0.01, min(0.05, 5 / len(valid_pitches)))
        present_notes = [note for note, count in notes_hist.items() if count > len(valid_pitches) * threshold]
        print(f"Present notes after thresholding: {present_notes}")

        # Calculate dominant notes (most frequent)
        if len(notes_hist) > 0:
            sorted_notes = sorted(notes_hist.items(), key=lambda x: x[1], reverse=True)
            dominant_notes = [note for note, _ in sorted_notes[:min(5, len(sorted_notes))]]
            print(f"Dominant notes: {dominant_notes}")
        else:
            dominant_notes = []
            print("No dominant notes found")

        # Import the raga knowledge module
        from modules.raga_knowledge import get_all_ragas, get_raga_info, get_raga_by_notes

        # Pre-filter ragas based on present notes to improve efficiency
        # Only consider ragas that contain at least some of the dominant notes
        if dominant_notes:
            # Try with all dominant notes first
            candidate_ragas = get_raga_by_notes(dominant_notes)

            # If no matches, try with fewer dominant notes
            if not candidate_ragas and len(dominant_notes) > 2:
                candidate_ragas = get_raga_by_notes(dominant_notes[:2])
                print(f"Using top 2 dominant notes, found {len(candidate_ragas)} candidate ragas")

            # If still no matches, get all ragas
            if not candidate_ragas:
                candidate_ragas = get_all_ragas()
                print(f"Using all ragas as candidates: {len(candidate_ragas)}")
        else:
            candidate_ragas = get_all_ragas()
            print(f"Using all ragas as candidates: {len(candidate_ragas)}")

        matches = []

        # Create a cache for note sets to avoid repeated computation
        note_set_cache = {}

        for raga_name in candidate_ragas:
            raga_info = get_raga_info(raga_name)
            if not raga_info or "notes" not in raga_info:
                continue

            # Get raga notes (use cache if available)
            if raga_name in note_set_cache:
                raga_notes = note_set_cache[raga_name]
            else:
                raga_notes = set(raga_info["notes"]["aroha"] + raga_info["notes"]["avaroha"])
                note_set_cache[raga_name] = raga_notes

            # Calculate how well the detected notes match the raga
            matching_notes = 0
            total_weight = 0

            for note in present_notes:
                # Extract base note and any modifiers
                if len(note) > 0:
                    base_note = note[0]
                    note_weight = notes_hist[note] / sum(notes_hist.values())  # Weight by frequency
                    total_weight += note_weight

                    # Check for exact matches first (more accurate)
                    if note in raga_notes:
                        matching_notes += note_weight
                    # Then check for base note matches
                    elif any(raga_note.startswith(base_note) for raga_note in raga_notes):
                        matching_notes += note_weight * 0.8  # Partial match

            # Calculate the match score
            if total_weight > 0:
                match_score = (matching_notes / total_weight) * 100

                # Adjust score based on completeness of the raga
                completeness = min(1.0, len(present_notes) / len(raga_notes))

                # Add bonus for vadi/samvadi matches if requested
                vadi_samvadi_bonus = 0
                if vadi_samvadi_weight and "vadi" in raga_info and "samvadi" in raga_info:
                    vadi = raga_info["vadi"]
                    samvadi = raga_info["samvadi"]

                    # Check if vadi and samvadi are among the dominant notes
                    if any(note.startswith(vadi) if len(note) > 0 else False for note in dominant_notes):
                        vadi_samvadi_bonus += 10
                    if any(note.startswith(samvadi) if len(note) > 0 else False for note in dominant_notes):
                        vadi_samvadi_bonus += 5

                # Add transition analysis if requested
                transition_score = 0
                if include_transitions and note_transitions and "pakad" in raga_info:
                    # Simple pakad pattern matching (could be more sophisticated)
                    pakad = raga_info["pakad"]
                    pakad_notes = pakad.split()

                    # Check for pakad patterns in transitions
                    for i in range(len(pakad_notes) - 1):
                        transition = (pakad_notes[i], pakad_notes[i+1])
                        if transition in note_transitions:
                            transition_score += 5

                # Calculate final adjusted score
                adjusted_score = (match_score * (0.6 + 0.4 * completeness) +
                                 vadi_samvadi_bonus + transition_score)

                matches.append({
                    "name": raga_name,
                    "confidence": adjusted_score,
                    "details": {
                        "matched_notes": matching_notes,
                        "present_notes": present_notes,
                        "dominant_notes": dominant_notes,
                        "raga_notes": list(raga_notes),
                        "completeness": completeness,
                        "vadi_samvadi_bonus": vadi_samvadi_bonus,
                        "transition_score": transition_score
                    }
                })

        # Sort matches by confidence
        matches.sort(key=lambda x: x["confidence"], reverse=True)

        # Print top matches for debugging
        if matches:
            print(f"Top raga matches:")
            for match in matches[:3]:
                print(f"  {match['name']}: {match['confidence']:.1f}% confidence")
        else:
            print("No raga matches found")

        # Return the best match if confidence is high enough
        if matches and matches[0]["confidence"] >= confidence_threshold:
            print(f"Selected raga: {matches[0]['name']} with {matches[0]['confidence']:.1f}% confidence")
            return matches[0]
        else:
            # If no match with sufficient confidence, return the best match with lower confidence
            if matches:
                print(f"Returning best match with lower confidence: {matches[0]['name']} ({matches[0]['confidence']:.1f}%)")
                matches[0]["confidence"] = max(matches[0]["confidence"], confidence_threshold)  # Ensure minimum confidence
                return matches[0]
            return None

    except Exception as e:
        print(f"Error identifying raga: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to a simple raga identification
        try:
            print("Using fallback raga identification")
            # Convert to MIDI notes
            midi_notes = librosa.hz_to_midi(valid_pitches)
            note_classes = midi_notes % 12

            # Count occurrences of each note class
            note_counts = np.zeros(12)
            for note in note_classes:
                note_counts[int(note) % 12] += 1

            # Normalize
            if np.sum(note_counts) > 0:
                note_counts = note_counts / np.sum(note_counts)

            # Simple rule-based identification
            if note_counts[7] > 0.15:  # Pa (G)
                if note_counts[0] > 0.15:  # Sa (C)
                    if note_counts[9] > 0.1:  # Dha (A)
                        return {"name": "Yaman", "confidence": 70}
                    else:
                        return {"name": "Bhairav", "confidence": 65}
                else:
                    return {"name": "Bhimpalasi", "confidence": 60}
            else:
                return {"name": "Malkauns", "confidence": 60}
        except:
            # Last resort fallback
            return {"name": "Yaman", "confidence": 60}

def extract_notes_with_transitions(pitch_values, min_duration=0.1):
    """
    Extract notes and their transitions from pitch values with improved accuracy

    Parameters:
        pitch_values (np.ndarray): Array of pitch values in Hz
        min_duration (float): Minimum duration in seconds for a note to be considered

    Returns:
        tuple: (notes_histogram, note_transitions)
            - notes_histogram: Dictionary mapping notes to their counts
            - note_transitions: Set of note transitions (pairs of consecutive notes)
    """
    notes_hist = {}
    note_transitions = set()

    # Define frequency ranges for Indian classical music notes (swaras)
    # More precise mapping with microtones for Indian classical music
    swara_ranges = {
        "S": (246.94, 261.63),   # Sa
        "r": (261.63, 277.18),   # Komal Re
        "R": (277.18, 293.66),   # Shuddha Re
        "g": (293.66, 311.13),   # Komal Ga
        "G": (311.13, 329.63),   # Shuddha Ga
        "M": (329.63, 349.23),   # Shuddha Ma
        "M#": (349.23, 369.99),  # Tivra Ma
        "P": (369.99, 392.00),   # Pa
        "d": (392.00, 415.30),   # Komal Dha
        "D": (415.30, 440.00),   # Shuddha Dha
        "n": (440.00, 466.16),   # Komal Ni
        "N": (466.16, 493.88),   # Shuddha Ni
    }

    # Segment the pitch values into notes
    current_note = None
    note_start_idx = 0
    note_sequence = []

    # Process pitch values to extract notes and transitions
    for i, pitch in enumerate(pitch_values):
        if pitch <= 0:  # Skip silent frames
            continue

        # Normalize to middle octave for note identification
        normalized_pitch = normalize_to_octave(pitch)
        detected_note = None

        # Find matching note
        for note, (min_freq, max_freq) in swara_ranges.items():
            if min_freq <= normalized_pitch < max_freq:
                detected_note = note
                break

        # If no note detected in standard ranges, try to find closest match
        if detected_note is None:
            min_distance = float('inf')
            for note, (min_freq, max_freq) in swara_ranges.items():
                center_freq = (min_freq + max_freq) / 2
                distance = abs(normalized_pitch - center_freq)
                if distance < min_distance:
                    min_distance = distance
                    detected_note = note

        # Handle note transitions
        if detected_note != current_note:
            # If we had a previous note, add it to the sequence
            if current_note is not None:
                note_duration = i - note_start_idx
                if note_duration >= min_duration * len(pitch_values):
                    # Add to histogram with weight proportional to duration
                    if current_note in notes_hist:
                        notes_hist[current_note] += note_duration
                    else:
                        notes_hist[current_note] = note_duration

                    # Add to note sequence
                    note_sequence.append(current_note)

            # Start tracking new note
            current_note = detected_note
            note_start_idx = i

    # Add the last note if there is one
    if current_note is not None:
        note_duration = len(pitch_values) - note_start_idx
        if note_duration >= min_duration * len(pitch_values):
            if current_note in notes_hist:
                notes_hist[current_note] += note_duration
            else:
                notes_hist[current_note] = note_duration
            note_sequence.append(current_note)

    # Extract transitions from the note sequence
    for i in range(len(note_sequence) - 1):
        note_transitions.add((note_sequence[i], note_sequence[i+1]))

    return notes_hist, note_transitions

def extract_notes_histogram(pitch_values):
    """
    Extract a histogram of notes from pitch values

    Parameters:
        pitch_values (np.ndarray): Array of pitch values in Hz

    Returns:
        dict: Dictionary mapping notes to their counts
    """
    # For backward compatibility, call the new function and return just the histogram
    notes_hist, _ = extract_notes_with_transitions(pitch_values)

    # If the new function returns empty results, fall back to the original implementation
    if not notes_hist:
        notes_hist = {}

        # Define frequency ranges for each note (simplified, would be more precise in real implementation)
        note_ranges = {
            "C": (246.94, 277.18),  # Sa
            "D": (277.18, 311.13),  # Re
            "E": (311.13, 349.23),  # Ga
            "F": (349.23, 392.00),  # Ma
            "G": (392.00, 440.00),  # Pa
            "A": (440.00, 493.88),  # Dha
            "B": (493.88, 523.25)   # Ni
        }

        # Count notes in each octave
        for pitch in pitch_values:
            if pitch > 0:  # Skip silent frames
                # Normalize to middle octave for note identification
                normalized_pitch = normalize_to_octave(pitch)

                # Find matching note
                for note, (min_freq, max_freq) in note_ranges.items():
                    if min_freq <= normalized_pitch < max_freq:
                        if note in notes_hist:
                            notes_hist[note] += 1
                        else:
                            notes_hist[note] = 1
                        break

    return notes_hist

def normalize_to_octave(freq):
    """Normalize frequency to middle octave (C4-C5) for note identification"""
    if freq <= 0:
        return 0
    
    while freq < 261.63:  # C4
        freq *= 2
    
    while freq >= 523.25:  # C5
        freq /= 2
    
    return freq

def detect_tala(y, sr, confidence_threshold=60):
    """
    Advanced tala (rhythm cycle) detection using multiple algorithms

    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        confidence_threshold (float): Minimum confidence to return a match

    Returns:
        dict or None: Dictionary with tala name, confidence, and analysis data, or None if no match
    """
    try:
        # Add minimum duration check
        if len(y) / sr < 5:  # Less than 5 seconds
            print("Audio too short for reliable tala detection")
            return None

        # Ensure audio has enough amplitude
        if np.max(np.abs(y)) < 0.01:
            print("Audio signal too weak for tala detection")
            return None

        # Optimize for percussion-heavy sections by applying bandpass filter
        # This focuses on tabla/percussion frequency ranges
        y_perc = librosa.effects.harmonic(y)

        # Extract onset envelope with optimized parameters for Indian classical music
        hop_length = 256  # Smaller hop length for better temporal resolution
        try:
            # Try with normalize parameter (newer librosa versions)
            onset_env = librosa.onset.onset_strength(
                y=y_perc,
                sr=sr,
                hop_length=hop_length,
                aggregate=np.median,
                normalize=True
            )
        except TypeError:
            # Fallback for older librosa versions
            onset_env = librosa.onset.onset_strength(
                y=y_perc,
                sr=sr,
                hop_length=hop_length,
                aggregate=np.median
            )
            # Manually normalize
            if np.max(onset_env) > 0:
                onset_env = onset_env / np.max(onset_env)

        # Apply adaptive thresholding to improve onset detection
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            backtrack=True,
            units='frames'
        )

        # Convert frames to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        # If we have enough onsets, proceed with analysis
        if len(onset_times) < 8:
            # Try with more sensitive parameters
            try:
                # Try with normalize parameter (newer librosa versions)
                onset_env = librosa.onset.onset_strength(
                    y=y,
                    sr=sr,
                    hop_length=hop_length,
                    normalize=True
                )
            except TypeError:
                # Fallback for older librosa versions
                onset_env = librosa.onset.onset_strength(
                    y=y,
                    sr=sr,
                    hop_length=hop_length
                )
                # Manually normalize
                if np.max(onset_env) > 0:
                    onset_env = onset_env / np.max(onset_env)

            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                backtrack=True,
                units='frames'
            )

            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

            if len(onset_times) < 8:
                print("Not enough onsets detected for tala analysis")
                return None

        # Calculate inter-onset intervals
        ioi = np.diff(onset_times)

        # Use multiple methods for cycle detection for robustness
        cycle_candidates = []

        # Method 1: Autocorrelation of IOIs
        if len(ioi) > 4:
            # Compute autocorrelation with proper normalization
            acf = librosa.autocorrelate(ioi, max_size=len(ioi)//2)
            acf = acf / np.max(acf)

            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(acf, height=0.4, distance=2)

            # Sort peaks by height
            if len(peaks) > 0:
                peak_heights = properties['peak_heights']
                sorted_indices = np.argsort(peak_heights)[::-1]  # Sort in descending order
                sorted_peaks = peaks[sorted_indices]

                # Add top peaks to candidates
                for peak in sorted_peaks[:3]:  # Consider top 3 peaks
                    if peak >= 2:  # Ignore trivial peaks
                        cycle_candidates.append((peak, peak_heights[np.where(peaks == peak)[0][0]]))

        # Method 2: Tempo-based analysis
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            start_bpm=80,  # Typical tempo range for Indian classical
            tightness=100  # Make beat tracking more strict
        )

        if len(beats) >= 8:
            # Calculate beat intervals
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr, hop_length=hop_length))

            # Look for patterns in beat groupings
            for group_size in [3, 4, 5, 6, 7, 8, 10, 12, 14, 16]:
                if len(beat_intervals) >= group_size * 2:
                    # Check if this grouping creates a consistent pattern
                    groups = [beat_intervals[i:i+group_size].sum() for i in range(0, len(beat_intervals) - group_size, group_size)]
                    if len(groups) >= 2:
                        consistency = 1.0 - np.std(groups) / np.mean(groups)
                        if consistency > 0.7:  # Reasonably consistent
                            cycle_candidates.append((group_size, consistency))

        # Method 3: Spectral analysis for percussion patterns
        if len(y) > sr * 10:  # At least 10 seconds
            # Create a percussion-focused spectrogram
            S = np.abs(librosa.stft(y_perc, n_fft=2048, hop_length=hop_length))

            # Sum across frequency bins to get a rhythm pattern
            rhythm_pattern = np.sum(S, axis=0)

            # Normalize
            rhythm_pattern = rhythm_pattern / np.max(rhythm_pattern)

            # Look for periodicity in the rhythm pattern
            acf_rhythm = librosa.autocorrelate(rhythm_pattern, max_size=len(rhythm_pattern)//4)
            acf_rhythm = acf_rhythm / np.max(acf_rhythm)

            # Find peaks
            peaks, properties = signal.find_peaks(acf_rhythm, height=0.3, distance=sr/hop_length)

            if len(peaks) > 0:
                # Convert frame peaks to beat counts (approximate)
                for peak in peaks:
                    # Convert frames to beats
                    beats_in_cycle = peak * tempo / 60 / (sr/hop_length)

                    # Round to nearest integer
                    rounded_beats = round(beats_in_cycle)

                    # Only consider common tala lengths
                    if 3 <= rounded_beats <= 16:
                        cycle_candidates.append((rounded_beats, properties['peak_heights'][np.where(peaks == peak)[0][0]]))

        # Combine and weight the candidates
        tala_votes = {}

        # Define common talas with their properties
        tala_definitions = {
            3: {"name": "Dadra", "weight": 1.0},
            4: {"name": "Keherwa", "weight": 1.2},  # Very common
            5: {"name": "Jhaptaal (half)", "weight": 0.9},
            6: {"name": "Dadra (extended)", "weight": 0.8},
            7: {"name": "Rupak Taal", "weight": 1.0},
            8: {"name": "Keherwa (extended)", "weight": 1.1},
            10: {"name": "Jhaptaal", "weight": 1.1},
            12: {"name": "Ektaal", "weight": 1.0},
            14: {"name": "Dhamar", "weight": 0.9},
            16: {"name": "Teentaal", "weight": 1.3}  # Most common
        }

        # Process all candidates
        for cycle_length, confidence_value in cycle_candidates:
            # Find closest match in tala definitions
            closest_tala = min(tala_definitions.keys(), key=lambda x: abs(x - cycle_length))

            # Calculate confidence based on how close the match is
            match_confidence = 1.0 - min(abs(closest_tala - cycle_length) / closest_tala, 0.5)

            # Apply tala-specific weight
            weighted_confidence = match_confidence * confidence_value * tala_definitions[closest_tala]["weight"]

            # Add to votes
            if closest_tala in tala_votes:
                tala_votes[closest_tala] += weighted_confidence
            else:
                tala_votes[closest_tala] = weighted_confidence

        # Find the tala with the highest vote
        if tala_votes:
            best_tala = max(tala_votes.items(), key=lambda x: x[1])
            tala_beats = best_tala[0]

            # Scale confidence to 0-100
            confidence = min(100, best_tala[1] * 100)

            if confidence >= confidence_threshold:
                return {
                    "name": tala_definitions[tala_beats]["name"],
                    "confidence": confidence,
                    "details": {
                        "detected_cycle": tala_beats,
                        "tempo": tempo,
                        "votes": {tala_definitions[k]["name"]: v for k, v in tala_votes.items()}
                    }
                }

        return None

    except Exception as e:
        print(f"Error detecting tala: {e}")
        return None

def detect_ornaments(y, sr, times, pitches):
    """
    Detect Indian classical music ornaments such as meend, kan, andolan, etc.
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
        times (np.ndarray): Array of time points
        pitches (np.ndarray): Array of detected pitches
    
    Returns:
        list: List of detected ornaments with their types and positions
    """
    try:
        ornaments = []
        
        # Filter out invalid values first
        valid_mask = (pitches > 0) & np.isfinite(pitches)
        if np.sum(valid_mask) < 10:
            return ornaments
            
        times = times[valid_mask]
        pitches = pitches[valid_mask]
        
        # Safely calculate pitch cents avoiding division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            pitch_cents = np.where(
                pitches > 0,
                1200 * np.log2(pitches / 440.0) + 5700,
                np.nan
            )
        
        # Replace any remaining invalid values with NaN
        pitch_cents[~np.isfinite(pitch_cents)] = np.nan
        
        # Skip if not enough pitch data
        if len(pitch_cents) < 10:
            return ornaments
        
        # Convert pitches to cents for better analysis of micro-movements
        pitch_cents = 1200 * np.log2(pitches / 440.0) + 5700  # Normalized to A4
        
        # Replace 0/inf/-inf values with NaN
        pitch_cents[~np.isfinite(pitch_cents)] = np.nan
        
        # Calculate pitch derivative (change rate)
        pitch_diff = np.diff(pitch_cents)
        
        # Detect Meend (continuous slide between notes)
        meend_threshold = 80  # cents
        meend_min_duration = 0.3  # seconds
        
        i = 0
        while i < len(pitch_diff) - 1:
            # Look for consistent direction of pitch change
            start_idx = i
            direction = np.sign(pitch_diff[i])
            
            # Continue while moving in same direction
            while (i < len(pitch_diff) - 1 and 
                   np.sign(pitch_diff[i+1]) == direction and 
                   not np.isnan(pitch_diff[i+1])):
                i += 1
            
            end_idx = i
            
            # Calculate total cents moved and duration
            total_change = np.nansum(pitch_diff[start_idx:end_idx+1])
            duration = times[end_idx + 1] - times[start_idx]
            
            # Check if this is a significant meend
            if (abs(total_change) > meend_threshold and
                duration > meend_min_duration):
                ornaments.append({
                    "type": "meend",
                    "start_time": times[start_idx],
                    "end_time": times[end_idx + 1],
                    "start_pitch": pitches[start_idx],
                    "end_pitch": pitches[end_idx + 1],
                    "duration": duration,
                    "cents_change": total_change
                })
            
            i += 1
        
        # Detect Andolan (oscillation around a note)
        andolan_min_cycles = 3
        andolan_min_duration = 0.5  # seconds
        andolan_min_depth = 15  # cents
        
        # Find zero crossings in pitch differences (oscillation detection)
        zero_crossings = np.where(np.diff(np.signbit(np.diff(pitch_cents))))[0]
        
        if len(zero_crossings) >= andolan_min_cycles * 2:
            i = 0
            while i < len(zero_crossings) - andolan_min_cycles * 2:
                # Check for regular oscillation pattern
                start_idx = zero_crossings[i]
                potential_andolan = True
                
                # Calculate average cycle length
                cycle_lengths = np.diff(zero_crossings[i:i+andolan_min_cycles*2])
                avg_cycle = np.mean(cycle_lengths)
                std_cycle = np.std(cycle_lengths)
                
                # Check if regular enough
                if std_cycle / avg_cycle > 0.5:  # Too irregular
                    i += 1
                    continue
                
                # Find where the regular oscillation ends
                end_i = i + andolan_min_cycles * 2
                while (end_i < len(zero_crossings) - 1 and 
                       abs(zero_crossings[end_i+1] - zero_crossings[end_i] - avg_cycle) < avg_cycle * 0.5):
                    end_i += 1
                
                end_idx = zero_crossings[end_i]
                
                # Calculate duration and depth
                duration = times[end_idx + 1] - times[start_idx]
                
                # Get pitch range during the oscillation
                pitch_segment = pitch_cents[start_idx:end_idx+1]
                depth = np.nanmax(pitch_segment) - np.nanmin(pitch_segment)
                
                if duration >= andolan_min_duration and depth >= andolan_min_depth:
                    ornaments.append({
                        "type": "andolan",
                        "start_time": times[start_idx],
                        "end_time": times[end_idx + 1],
                        "duration": duration,
                        "depth_cents": depth,
                        "cycles": (end_i - i) // 2
                    })
                    
                    i = end_i + 1
                else:
                    i += 1
        
        # Detect Kan (grace note)
        kan_duration_max = 0.15  # seconds
        kan_threshold = 50  # cents
        
        for i in range(1, len(pitches) - 1):
            # Look for quick pitch jump up and back down
            pitch_jump_up = pitch_cents[i] - pitch_cents[i-1]
            pitch_jump_down = pitch_cents[i] - pitch_cents[i+1]
            
            duration = times[i+1] - times[i-1]
            
            if (not np.isnan(pitch_jump_up) and not np.isnan(pitch_jump_down) and
                duration < kan_duration_max and
                pitch_jump_up > kan_threshold and
                pitch_jump_down > kan_threshold):
                
                ornaments.append({
                    "type": "kan",
                    "time": times[i],
                    "pitch": pitches[i],
                    "main_note_pitch": (pitches[i-1] + pitches[i+1]) / 2,
                    "cents_above": pitch_cents[i] - (pitch_cents[i-1] + pitch_cents[i+1]) / 2
                })
        
        # Detect Gamak (fast oscillation with articulation)
        gamak_threshold = 40  # cents
        gamak_min_cycles = 2
        gamak_max_duration_per_cycle = 0.2  # seconds
        
        # Use a rolling window to detect rapid pitch changes
        window_size = 5
        
        for i in range(window_size, len(pitches) - window_size, 2):
            pitch_window = pitch_cents[i-window_size:i+window_size]
            time_window = times[i-window_size:i+window_size]
            
            # Calculate pitch range and time span
            if np.all(np.isnan(pitch_window)):
                continue
                
            pitch_range = np.nanmax(pitch_window) - np.nanmin(pitch_window)
            time_span = time_window[-1] - time_window[0]
            
            # Count direction changes (oscillations)
            pitch_diff_window = np.diff(pitch_window)
            direction_changes = np.sum(np.abs(np.diff(np.sign(pitch_diff_window)))) / 2
            
            # Calculate average cycle duration
            if direction_changes > 0:
                avg_cycle_duration = time_span / direction_changes
            else:
                avg_cycle_duration = time_span
            
            # Check if this is a gamak
            if (pitch_range > gamak_threshold and
                direction_changes >= gamak_min_cycles and
                avg_cycle_duration < gamak_max_duration_per_cycle):
                
                ornaments.append({
                    "type": "gamak",
                    "start_time": time_window[0],
                    "end_time": time_window[-1],
                    "duration": time_span,
                    "pitch_range_cents": pitch_range,
                    "cycles": direction_changes
                })
        
        return ornaments
        
    except Exception as e:
        print(f"Error detecting ornaments: {e}")
        return []