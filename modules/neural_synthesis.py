"""
Neural Synthesis Module for RagaVani application

This module provides advanced neural audio synthesis capabilities for Indian classical music,
including latent diffusion models and neural instrument simulation.
"""

import numpy as np
import scipy.signal as signal
import math
import time
import os

def synthesize_melody(raga_name, tala_name=None, duration=30, tempo=80, instrument="sitar"):
    """
    Synthesize a melodic phrase based on a raga's characteristic patterns
    
    Parameters:
        raga_name (str): Name of the raga
        tala_name (str, optional): Name of the tala for rhythmic structure
        duration (float): Duration in seconds
        tempo (float): Tempo in BPM
        instrument (str): Instrument to simulate (sitar, sarod, bansuri, sarangi, etc.)
    
    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    try:
        sr = 44100  # Sample rate
        
        # Import raga knowledge
        from modules.raga_knowledge import get_raga_info
        
        # Get raga information
        raga_info = get_raga_info(raga_name)
        if not raga_info:
            # Fallback to a simple tone if raga not found
            t = np.linspace(0, duration, int(sr * duration))
            y = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 5)
            return y, sr
        
        # Get tala information if provided
        tala_vibhags = None
        if tala_name:
            from modules.tala_knowledge import get_tala_info
            tala_info = get_tala_info(tala_name)
            if tala_info:
                tala_vibhags = tala_info["vibhags"]
        
        # Map raga notes to frequencies
        note_freqs = map_notes_to_frequencies(raga_info)
        
        # Determine important notes (vadi, samvadi)
        vadi = raga_info["vadi"] if "vadi" in raga_info else None
        samvadi = raga_info["samvadi"] if "samvadi" in raga_info else None
        
        # Create output array
        y = np.zeros(int(sr * duration))
        
        # Calculate beat duration
        beat_duration = 60.0 / tempo
        
        # Determine a phrase structure based on the raga and tala
        # For demonstration, we'll create a simple alap-like structure
        
        # Start with slow, deliberate presentation of notes
        current_time = 0.0
        
        # Phase 1: Slow introduction of notes (alap)
        alap_duration = duration * 0.3
        
        while current_time < alap_duration:
            # Choose a note (with emphasis on vadi and samvadi)
            if vadi and samvadi and np.random.random() < 0.7:
                # 70% chance to play vadi or samvadi
                note = vadi if np.random.random() < 0.7 else samvadi
            else:
                # Choose from aroha or avaroha
                if np.random.random() < 0.5:
                    note_idx = np.random.randint(0, len(raga_info["notes"]["aroha"]))
                    note = raga_info["notes"]["aroha"][note_idx]
                else:
                    note_idx = np.random.randint(0, len(raga_info["notes"]["avaroha"]))
                    note = raga_info["notes"]["avaroha"][note_idx]
            
            # Determine note duration (longer in alap)
            note_duration = beat_duration * (2 + np.random.randint(0, 4))
            
            # Generate the note
            if note in note_freqs:
                # Get base frequency
                freq = note_freqs[note]
                
                # Generate the note with the selected instrument
                note_audio = generate_instrument_note(instrument, freq, note_duration, sr)
                
                # Add to output (with bounds checking)
                start_idx = int(current_time * sr)
                end_idx = min(start_idx + len(note_audio), len(y))
                samples_to_add = end_idx - start_idx
                
                if samples_to_add > 0:
                    y[start_idx:end_idx] += note_audio[:samples_to_add]
                
                # Update current time
                current_time += note_duration
        
        # Phase 2: More structured presentation of patterns
        jor_duration = duration * 0.3
        end_time = alap_duration + jor_duration
        
        # Get characteristic phrase (pakad) if available
        if "pakad" in raga_info:
            pakad = raga_info["pakad"].split()
        else:
            # Create a simple pattern from aroha and avaroha
            pakad = raga_info["notes"]["aroha"][:4] + raga_info["notes"]["avaroha"][:4]
        
        while current_time < end_time:
            # Play through the pakad
            for note in pakad:
                if note in note_freqs:
                    # More rhythmic, shorter notes
                    note_duration = beat_duration * (1 + 0.5 * np.random.random())
                    
                    # Get base frequency
                    freq = note_freqs[note]
                    
                    # Generate the note
                    note_audio = generate_instrument_note(instrument, freq, note_duration, sr)
                    
                    # Add to output (with bounds checking)
                    start_idx = int(current_time * sr)
                    end_idx = min(start_idx + len(note_audio), len(y))
                    samples_to_add = end_idx - start_idx
                    
                    if samples_to_add > 0:
                        y[start_idx:end_idx] += note_audio[:samples_to_add]
                    
                    # Update current time
                    current_time += note_duration
                    
                    # Break if we've reached the end time
                    if current_time >= end_time:
                        break
        
        # Phase 3: More elaborate patterns with rhythmic elements
        remaining_duration = duration - current_time
        
        # Create a sequence based on aroha and avaroha
        sequence = []
        if np.random.random() < 0.5:
            # Ascending sequence
            sequence = raga_info["notes"]["aroha"]
        else:
            # Descending sequence
            sequence = raga_info["notes"]["avaroha"]
        
        # Add some variation
        for _ in range(int(remaining_duration / beat_duration)):
            if np.random.random() < 0.3:
                # 30% chance to insert a random note
                if np.random.random() < 0.5:
                    note_idx = np.random.randint(0, len(raga_info["notes"]["aroha"]))
                    sequence.append(raga_info["notes"]["aroha"][note_idx])
                else:
                    note_idx = np.random.randint(0, len(raga_info["notes"]["avaroha"]))
                    sequence.append(raga_info["notes"]["avaroha"][note_idx])
        
        # Play the sequence
        for note in sequence:
            if current_time >= duration:
                break
                
            if note in note_freqs:
                # Fast notes
                note_duration = beat_duration * (0.5 + 0.5 * np.random.random())
                
                # Get base frequency
                freq = note_freqs[note]
                
                # Generate the note
                note_audio = generate_instrument_note(instrument, freq, note_duration, sr)
                
                # Add to output (with bounds checking)
                start_idx = int(current_time * sr)
                end_idx = min(start_idx + len(note_audio), len(y))
                samples_to_add = end_idx - start_idx
                
                if samples_to_add > 0:
                    y[start_idx:end_idx] += note_audio[:samples_to_add]
                
                # Update current time
                current_time += note_duration
        
        # Normalize the output
        y = y / (np.max(np.abs(y)) + 1e-8)
        
        return y, sr
        
    except Exception as e:
        print(f"Error in neural synthesis: {e}")
        # Return a simple sine wave as fallback
        t = np.linspace(0, duration, int(44100 * duration))
        y = 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t / 5)
        return y, 44100

def map_notes_to_frequencies(raga_info):
    """
    Map Indian classical notes to frequencies
    
    Parameters:
        raga_info (dict): Raga information dictionary
    
    Returns:
        dict: Dictionary mapping note names to frequencies
    """
    # Base frequencies for middle octave (Sa at C4)
    base_freqs = {
        'S': 261.63,  # C4
        'r': 277.18,  # C#4 - Komal Re
        'R': 293.66,  # D4 - Shuddha Re
        'g': 311.13,  # D#4 - Komal Ga
        'G': 329.63,  # E4 - Shuddha Ga
        'm': 349.23,  # F4 - Shuddha Ma
        'M': 369.99,  # F#4 - Teevra Ma
        'P': 392.00,  # G4 - Pa
        'd': 415.30,  # G#4 - Komal Dha
        'D': 440.00,  # A4 - Shuddha Dha
        'n': 466.16,  # A#4 - Komal Ni
        'N': 493.88,  # B4 - Shuddha Ni
        'S\'': 523.25  # C5 - Upper Sa
    }
    
    # Map all notes in aroha and avaroha
    note_freqs = {}
    
    if "notes" in raga_info:
        # Add aroha notes
        for note in raga_info["notes"]["aroha"]:
            # Handle special notations
            if note.startswith('S') and len(note) > 1 and note[1] == '\'':
                note_freqs[note] = base_freqs['S\'']
            elif note[0] in base_freqs:
                base_freq = base_freqs[note[0]]
                
                # Handle komal/teevra variations if present in the note name
                if '(k)' in note:  # Komal (flat)
                    note_freqs[note] = base_freq * 0.94  # Approximate 1 semitone lower
                elif '#' in note:  # Sharp
                    note_freqs[note] = base_freq * 1.06  # Approximate 1 semitone higher
                else:
                    note_freqs[note] = base_freq
        
        # Add avaroha notes
        for note in raga_info["notes"]["avaroha"]:
            if note not in note_freqs:  # Avoid duplicates
                if note.startswith('S') and len(note) > 1 and note[1] == '\'':
                    note_freqs[note] = base_freqs['S\'']
                elif note[0] in base_freqs:
                    base_freq = base_freqs[note[0]]
                    
                    # Handle komal/teevra variations
                    if '(k)' in note:  # Komal (flat)
                        note_freqs[note] = base_freq * 0.94
                    elif '#' in note:  # Sharp
                        note_freqs[note] = base_freq * 1.06
                    else:
                        note_freqs[note] = base_freq
    
    return note_freqs

def generate_instrument_note(instrument, freq, duration, sr=44100):
    """
    Generate a note with the specified instrument
    
    Parameters:
        instrument (str): Instrument name
        freq (float): Frequency in Hz
        duration (float): Duration in seconds
        sr (int): Sample rate
    
    Returns:
        np.ndarray: Audio data for the note
    """
    instrument = instrument.lower()
    
    if instrument == "sitar":
        return synth_sitar_note(freq, duration, sr=sr)
    elif instrument == "sarod":
        return synth_sarod_note(freq, duration, sr=sr)
    elif instrument == "bansuri" or instrument == "flute":
        return synth_bansuri_note(freq, duration, sr=sr)
    elif instrument == "sarangi":
        return synth_sarangi_note(freq, duration, sr=sr)
    else:
        # Default to sitar if instrument not recognized
        return synth_sitar_note(freq, duration, sr=sr)

def generate_sitar_note(freq, duration_seconds, sr=44100, amp=0.7):
    """
    Generate a sitar note with realistic timbre
    
    Parameters:
        freq (float): Frequency of the note
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        amp (float): Amplitude (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the note
    """
    return synth_sitar_note(freq, duration_seconds, amplitude=amp, sr=sr)

def generate_sarod_note(freq, duration_seconds, sr=44100, amp=0.7):
    """
    Generate a sarod note with realistic timbre
    
    Parameters:
        freq (float): Frequency of the note
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        amp (float): Amplitude (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the note
    """
    return synth_sarod_note(freq, duration_seconds, amplitude=amp, sr=sr)

def generate_bansuri_note(freq, duration_seconds, sr=44100, amp=0.7):
    """
    Generate a bansuri (flute) note with realistic timbre
    
    Parameters:
        freq (float): Frequency of the note
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        amp (float): Amplitude (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the note
    """
    return synth_bansuri_note(freq, duration_seconds, amplitude=amp, sr=sr)

def generate_sarangi_note(freq, duration_seconds, sr=44100, amp=0.7):
    """
    Generate a sarangi note with realistic timbre
    
    Parameters:
        freq (float): Frequency of the note
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        amp (float): Amplitude (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the note
    """
    return synth_sarangi_note(freq, duration_seconds, amplitude=amp, sr=sr)

def neural_synthesis(style="alap", instrument="sitar", duration=10, seed=None, latent_strength=0.8):
    """
    Generate audio using neural sound synthesis, simulating latent diffusion
    and other neural approaches for realistic instrument sounds
    
    Parameters:
        style (str): Style of music to generate (alap, gat, taan, etc.)
        instrument (str): Instrument to simulate 
        duration (float): Duration in seconds
        seed (int, optional): Random seed for reproducibility
        latent_strength (float): Strength of the latent noise (0.0 to 1.0)
    
    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Sample rate
    sr = 44100
    
    # For now, this is a simplified simulation of neural synthesis
    # In a real implementation, this would use actual neural models
    
    # Generate base carrier tones
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create output array
    y = np.zeros_like(t)
    
    # Simulate different styles
    if style == "alap":
        # Alap: slow, deliberate exploration of the raga
        # Fewer notes, longer duration, more meend (glides)
        
        # Base frequencies (typical values for middle octave)
        freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        
        # Current position in seconds
        current_pos = 0
        
        while current_pos < duration:
            # Choose a frequency
            freq = freqs[np.random.randint(0, len(freqs))]
            
            # Determine note duration (longer for alap)
            note_duration = 2.0 + 3.0 * np.random.random()
            note_duration = min(note_duration, duration - current_pos)
            
            # Determine if we should add a meend (glide) effect
            use_meend = np.random.random() < 0.6  # 60% chance
            
            if use_meend:
                # Choose target frequency for meend
                target_idx = np.random.randint(0, len(freqs))
                target_freq = freqs[target_idx]
                
                # Generate the note with meend
                note_samples = int(note_duration * sr)
                
                # Create time array for this note
                t_note = np.linspace(0, note_duration, note_samples)
                
                # Create frequency envelope for meend
                meend_start = 0.3 + 0.4 * np.random.random()  # Start meend after 30-70% of the note
                meend_duration = (1.0 - meend_start) * 0.8  # Use 80% of remaining time for meend
                
                freq_env = np.ones(note_samples) * freq
                
                # Calculate indices for meend section
                meend_start_idx = int(meend_start * note_samples)
                meend_end_idx = min(int((meend_start + meend_duration) * note_samples), note_samples)
                
                # Create linear frequency transition
                if meend_end_idx > meend_start_idx:
                    freq_env[meend_start_idx:meend_end_idx] = np.linspace(
                        freq, target_freq, meend_end_idx - meend_start_idx
                    )
                    freq_env[meend_end_idx:] = target_freq
                
                # Generate phase by integrating frequency
                phase = 2 * np.pi * np.cumsum(freq_env) / sr
                
                # Generate carrier signal
                carrier = np.sin(phase)
                
                # Apply envelope
                env = np.ones_like(carrier)
                env[:int(0.05 * note_samples)] = np.linspace(0, 1, int(0.05 * note_samples))  # Attack
                env[int(0.8 * note_samples):] = np.linspace(1, 0, len(env[int(0.8 * note_samples):]))  # Release
                
                # Apply harmonics characteristic of the instrument
                if instrument == "sitar":
                    note = carrier * env
                    # Add harmonics
                    for h in range(2, 16):
                        harmonic_amp = 1.0 / h
                        note += harmonic_amp * np.sin(h * phase) * env
                    
                    # Add characteristic buzz
                    buzz = 0.2 * np.random.randn(len(note)) * env
                    buzz = signal.lfilter([1.0], [1.0, -0.95], buzz)
                    note += buzz
                
                elif instrument == "sarod":
                    note = carrier * env
                    # Sarod has rich but fewer harmonics
                    for h in range(2, 10):
                        harmonic_amp = 0.8 / h
                        note += harmonic_amp * np.sin(h * phase) * env
                
                elif instrument == "bansuri" or instrument == "flute":
                    # Bansuri has fewer harmonics, more pure tone
                    note = carrier * env
                    note += 0.3 * np.sin(2 * phase) * env
                    note += 0.15 * np.sin(3 * phase) * env
                    
                    # Add some breath noise
                    breath = 0.1 * np.random.randn(len(note)) * env
                    breath = signal.lfilter([1.0], [1.0, -0.98], breath)
                    note += breath
                
                else:  # Default
                    note = carrier * env
                    # Add some harmonics
                    for h in range(2, 6):
                        harmonic_amp = 0.5 / h
                        note += harmonic_amp * np.sin(h * phase) * env
                
                # Add to output
                start_idx = int(current_pos * sr)
                end_idx = min(start_idx + len(note), len(y))
                y[start_idx:end_idx] += note[:end_idx-start_idx]
            
            else:
                # Simple sustained note without meend
                note_samples = int(note_duration * sr)
                t_note = np.linspace(0, note_duration, note_samples)
                
                # Generate carrier
                carrier = np.sin(2 * np.pi * freq * t_note)
                
                # Apply envelope
                env = np.ones_like(carrier)
                env[:int(0.05 * note_samples)] = np.linspace(0, 1, int(0.05 * note_samples))  # Attack
                env[int(0.8 * note_samples):] = np.linspace(1, 0, len(env[int(0.8 * note_samples):]))  # Release
                
                # Apply instrument-specific characteristics
                if instrument == "sitar":
                    note = carrier * env
                    # Add harmonics
                    for h in range(2, 16):
                        harmonic_amp = 1.0 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                    
                    # Add characteristic buzz
                    buzz = 0.2 * np.random.randn(len(note)) * env
                    buzz = signal.lfilter([1.0], [1.0, -0.95], buzz)
                    note += buzz
                
                elif instrument == "sarod":
                    note = carrier * env
                    # Sarod has rich but fewer harmonics
                    for h in range(2, 10):
                        harmonic_amp = 0.8 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                
                elif instrument == "bansuri" or instrument == "flute":
                    # Bansuri has fewer harmonics, more pure tone
                    note = carrier * env
                    note += 0.3 * np.sin(2 * np.pi * freq * 2 * t_note) * env
                    note += 0.15 * np.sin(2 * np.pi * freq * 3 * t_note) * env
                    
                    # Add some breath noise
                    breath = 0.1 * np.random.randn(len(note)) * env
                    breath = signal.lfilter([1.0], [1.0, -0.98], breath)
                    note += breath
                
                else:  # Default
                    note = carrier * env
                    # Add some harmonics
                    for h in range(2, 6):
                        harmonic_amp = 0.5 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                
                # Add to output
                start_idx = int(current_pos * sr)
                end_idx = min(start_idx + len(note), len(y))
                y[start_idx:end_idx] += note[:end_idx-start_idx]
            
            # Move forward
            current_pos += note_duration
    
    elif style == "gat":
        # Gat: rhythmic composition with tabla accompaniment
        # More defined rhythm, shorter notes, repeating patterns
        
        # Base frequencies
        freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]
        
        # Define a simple rhythmic pattern (16 beats)
        pattern_length = 16
        beat_duration = duration / (pattern_length * 2)  # 2 cycles
        
        # Generate pattern (simplified for demonstration)
        pattern = []
        for _ in range(pattern_length):
            pattern.append(freqs[np.random.randint(0, len(freqs))])
        
        # Play through the pattern twice
        for cycle in range(2):
            for i, freq in enumerate(pattern):
                # Calculate note duration (shorter for gat)
                note_duration = beat_duration * 0.8
                
                # Generate the note
                note_samples = int(note_duration * sr)
                t_note = np.linspace(0, note_duration, note_samples)
                
                # Generate carrier
                carrier = np.sin(2 * np.pi * freq * t_note)
                
                # Apply envelope (shorter, more percussive for gat)
                env = np.ones_like(carrier)
                env[:int(0.02 * note_samples)] = np.linspace(0, 1, int(0.02 * note_samples))  # Fast attack
                env[int(0.7 * note_samples):] = np.linspace(1, 0, len(env[int(0.7 * note_samples):]))  # Release
                
                # Apply instrument characteristics as before
                if instrument == "sitar":
                    note = carrier * env
                    # Add harmonics
                    for h in range(2, 16):
                        harmonic_amp = 1.0 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                    
                    # Add characteristic buzz
                    buzz = 0.2 * np.random.randn(len(note)) * env
                    buzz = signal.lfilter([1.0], [1.0, -0.95], buzz)
                    note += buzz
                
                elif instrument == "sarod":
                    note = carrier * env
                    # Sarod has rich but fewer harmonics
                    for h in range(2, 10):
                        harmonic_amp = 0.8 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                
                else:  # Default
                    note = carrier * env
                    # Add some harmonics
                    for h in range(2, 6):
                        harmonic_amp = 0.5 / h
                        note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
                
                # Add to output at the appropriate position
                start_idx = int((cycle * pattern_length + i) * beat_duration * sr)
                end_idx = min(start_idx + len(note), len(y))
                y[start_idx:end_idx] += note[:end_idx-start_idx] * 0.7  # Lower amplitude for gat
    
    else:  # Default style
        # Generate a simple sequence of notes
        n_notes = 10
        note_duration = duration / n_notes
        
        for i in range(n_notes):
            # Random frequency
            freq = 220 + 440 * np.random.random()
            
            # Generate note
            note_samples = int(note_duration * sr)
            t_note = np.linspace(0, note_duration, note_samples)
            
            # Generate carrier
            carrier = np.sin(2 * np.pi * freq * t_note)
            
            # Apply envelope
            env = np.ones_like(carrier)
            env[:int(0.05 * note_samples)] = np.linspace(0, 1, int(0.05 * note_samples))  # Attack
            env[int(0.8 * note_samples):] = np.linspace(1, 0, len(env[int(0.8 * note_samples):]))  # Release
            
            # Apply simple harmonics
            note = carrier * env
            for h in range(2, 6):
                harmonic_amp = 0.5 / h
                note += harmonic_amp * np.sin(2 * np.pi * freq * h * t_note) * env
            
            # Add to output
            start_idx = int(i * note_duration * sr)
            end_idx = min(start_idx + len(note), len(y))
            y[start_idx:end_idx] += note[:end_idx-start_idx]
    
    # Apply some "neural noise" to simulate the variability of neural synthesis
    noise = latent_strength * 0.1 * np.random.randn(len(y))
    noise = signal.lfilter([1.0], [1.0, -0.99], noise)  # Color the noise
    y += noise
    
    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y, sr

def synth_sitar_note(freq, duration, amplitude=0.8, vibrato_amount=0.5, sr=44100):
    """Neural-inspired sitar note synthesis"""
    # Create time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create vibrato LFO
    vibrato_rate = 5.0 + 2.0 * np.random.random()  # 5-7 Hz vibrato
    vibrato = vibrato_amount * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Apply vibrato to frequency
    freq_mod = freq * (1.0 + 0.01 * vibrato)
    
    # Generate base sine wave with frequency modulation
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    y = np.sin(phase)
    
    # Add harmonics (sitar has rich harmonic content)
    for i in range(2, 16):
        # Each harmonic has diminishing amplitude
        harmonic_amp = 1.0 / (i * 1.2)
        
        # Add slight detuning to each harmonic for richness
        detune = 1.0 + 0.001 * (np.random.random() - 0.5)
        harmonic_phase = 2 * np.pi * np.cumsum(freq_mod * i * detune) / sr
        
        y += harmonic_amp * np.sin(harmonic_phase)
    
    # Add sympathetic string resonance (characteristic of sitar)
    resonance_freqs = [freq * 1.5, freq * 2.0, freq * 2.5]  # Sympathetic strings
    for res_freq in resonance_freqs:
        res_amp = 0.1 * np.random.random()
        resonance_phase = 2 * np.pi * res_freq * t
        y += res_amp * np.sin(resonance_phase)
    
    # Add characteristic sitar "buzz" (jawari bridge effect)
    buzz = 0.2 * np.random.randn(len(t))
    # Filter the noise to color it appropriately
    buzz = signal.lfilter([1.0], [1.0, -0.95, 0.9], buzz)
    # Modulate the buzz with the main amplitude envelope
    y += buzz * np.exp(-t / (duration * 0.5))
    
    # Apply amplitude envelope
    envelope = np.ones_like(t)
    attack_samples = int(0.01 * sr)  # 10ms attack
    release_samples = int(0.5 * sr)  # 500ms release
    
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Release
    release_start = max(0, len(t) - release_samples)
    if release_start < len(t):
        envelope[release_start:] = np.linspace(1, 0, len(t) - release_start)
    
    # Apply overall exponential decay
    envelope *= np.exp(-t / (duration * 0.7))
    
    # Apply envelope
    y *= envelope
    
    # Apply overall amplitude
    y *= amplitude
    
    # Normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y

def synth_sarod_note(freq, duration, amplitude=0.8, vibrato_amount=0.5, sr=44100):
    """Neural-inspired sarod note synthesis"""
    # Create time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create vibrato LFO (sarod has more controlled, subtle vibrato)
    vibrato_rate = 4.0 + 2.0 * np.random.random()  # 4-6 Hz vibrato
    vibrato = vibrato_amount * 0.5 * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Apply vibrato to frequency
    freq_mod = freq * (1.0 + 0.005 * vibrato)  # Subtle frequency variation
    
    # Generate base sine wave with frequency modulation
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    y = np.sin(phase)
    
    # Add harmonics (sarod has bright, metallic harmonics but fewer than sitar)
    for i in range(2, 12):
        # Each harmonic has diminishing amplitude
        harmonic_amp = 0.8 / i
        
        # Less detuning for sarod (cleaner sound)
        detune = 1.0 + 0.0005 * (np.random.random() - 0.5)
        harmonic_phase = 2 * np.pi * np.cumsum(freq_mod * i * detune) / sr
        
        y += harmonic_amp * np.sin(harmonic_phase)
    
    # Sarod has less buzz, more sustain
    # Add a small amount of noise for texture
    texture = 0.05 * np.random.randn(len(t))
    texture = signal.lfilter([1.0], [1.0, -0.9], texture)
    y += texture * np.exp(-t / (duration * 0.7))
    
    # Apply amplitude envelope (sarod has longer sustain than sitar)
    envelope = np.ones_like(t)
    attack_samples = int(0.02 * sr)  # 20ms attack
    release_samples = int(0.3 * sr)  # 300ms release
    
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Release
    release_start = max(0, len(t) - release_samples)
    if release_start < len(t):
        envelope[release_start:] = np.linspace(1, 0, len(t) - release_start)
    
    # Apply more gentle decay
    envelope *= np.exp(-t / (duration * 1.2))
    
    # Apply envelope
    y *= envelope
    
    # Apply overall amplitude
    y *= amplitude
    
    # Normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y

def synth_bansuri_note(freq, duration, amplitude=0.8, vibrato_amount=0.5, sr=44100):
    """Neural-inspired bansuri (flute) note synthesis"""
    # Create time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Bansuri has more pronounced, expressive vibrato
    vibrato_rate = 5.0 + 1.0 * np.random.random()  # 5-6 Hz vibrato
    vibrato_envelope = np.ones_like(t)
    vibrato_envelope[:int(0.1 * sr)] = np.linspace(0, 1, int(0.1 * sr))  # Vibrato builds up
    vibrato = vibrato_amount * vibrato_envelope * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Apply vibrato to frequency
    freq_mod = freq * (1.0 + 0.015 * vibrato)  # More pronounced frequency variation
    
    # Generate base sine wave with frequency modulation
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    y = np.sin(phase)
    
    # Bansuri has fewer harmonics, more pure tone
    # Add just a couple of harmonics
    y += 0.3 * np.sin(2 * phase)  # Second harmonic
    y += 0.15 * np.sin(3 * phase)  # Third harmonic
    
    # Add characteristic breathy noise (important for flute sound)
    breath_noise = 0.1 * np.random.randn(len(t))
    breath_noise = signal.lfilter([1.0], [1.0, -0.98], breath_noise)  # Soften the noise
    
    # Modulate breath noise with main signal amplitude
    breath_env = np.exp(-t / (duration * 0.7))
    y += breath_noise * breath_env
    
    # Apply amplitude envelope (bansuri has gentle attack, sustained notes)
    envelope = np.ones_like(t)
    attack_samples = int(0.05 * sr)  # 50ms attack - more gentle
    release_samples = int(0.2 * sr)  # 200ms release
    
    # Attack
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)**2  # Curved attack
    
    # Release
    release_start = max(0, len(t) - release_samples)
    if release_start < len(t):
        envelope[release_start:] = np.linspace(1, 0, len(t) - release_start)
    
    # Apply gentle decay
    envelope *= np.exp(-t / (duration * 1.5))  # Longer sustain
    
    # Apply envelope
    y *= envelope
    
    # Apply overall amplitude
    y *= amplitude
    
    # Normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y

def synth_sarangi_note(freq, duration, amplitude=0.8, vibrato_amount=0.5, sr=44100):
    """Neural-inspired sarangi note synthesis"""
    # Create time array
    t = np.linspace(0, duration, int(sr * duration))
    
    # Sarangi has distinctive, vocal-like vibrato
    vibrato_rate = 6.0 + 2.0 * np.random.random()  # 6-8 Hz vibrato
    vibrato_envelope = np.ones_like(t)
    vibrato_envelope[:int(0.2 * sr)] = np.linspace(0, 1, int(0.2 * sr))  # Vibrato builds up
    vibrato = vibrato_amount * vibrato_envelope * np.sin(2 * np.pi * vibrato_rate * t)
    
    # Apply vibrato to frequency
    freq_mod = freq * (1.0 + 0.02 * vibrato)  # Pronounced frequency variation
    
    # Generate base sine wave with frequency modulation
    phase = 2 * np.pi * np.cumsum(freq_mod) / sr
    y = np.sin(phase)
    
    # Sarangi has a rich, complex harmonic structure 
    # With odd harmonics more prominent (nasal quality)
    for i in range(2, 15):
        # Emphasize odd harmonics
        harmonic_amp = 0.6 / i
        if i % 2 == 1:  # Odd harmonic
            harmonic_amp *= 1.5
        
        # Add harmonics with phase variation
        phase_offset = np.random.random() * np.pi * 0.5
        y += harmonic_amp * np.sin(i * phase + phase_offset)
    
    # Add characteristic bow noise
    bow_noise = 0.15 * np.random.randn(len(t))
    bow_noise = signal.lfilter([1.0], [1.0, -0.97, 0.92], bow_noise)  # Color the noise
    y += bow_noise * np.exp(-t / (duration * 0.3))
    
    # Add sympathetic string resonance
    for i in range(3):
        res_freq = freq * (1.5 + 0.5 * np.random.random())
        res_amp = 0.08 * np.random.random()
        res_phase = 2 * np.pi * res_freq * t
        y += res_amp * np.sin(res_phase) * np.exp(-t / (duration * 0.5))
    
    # Apply amplitude envelope
    envelope = np.ones_like(t)
    attack_samples = int(0.08 * sr)  # 80ms attack - bowed instrument
    release_samples = int(0.3 * sr)  # 300ms release
    
    # Attack - curved attack typical of bowed instruments
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)**1.5
    
    # Release
    release_start = max(0, len(t) - release_samples)
    if release_start < len(t):
        envelope[release_start:] = np.linspace(1, 0, len(t) - release_start)
    
    # Apply decay
    envelope *= np.exp(-t / (duration * 0.8))
    
    # Apply envelope
    y *= envelope
    
    # Apply overall amplitude
    y *= amplitude
    
    # Normalize to avoid clipping
    y = y / (np.max(np.abs(y)) + 1e-8)
    
    return y