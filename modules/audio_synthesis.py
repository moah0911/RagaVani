"""
Audio Synthesis Module for RagaVani application

This module provides comprehensive functionality for generating Indian classical music sounds,
including tanpura drones, tabla patterns, and neural synthesis of Indian classical instruments.
It combines physical modeling, wavetable synthesis, and neural approaches for realistic sound generation.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import math

def generate_tanpura(root_note="C", duration=60, tempo=60, jiva=0.5):
    """
    Generate a tanpura drone sound
    
    Parameters:
        root_note (str): Root note (Sa) in Western notation (default: "C")
        duration (float): Duration in seconds (default: 60)
        tempo (float): Tempo in BPM (default: 60)
        jiva (float): Amount of jawari/jiva effect (0.0 to 1.0)
    
    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    # Sample rate
    sr = 44100
    
    # Note frequencies (in Hz)
    note_frequencies = {
        'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
        'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
        'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
    }
    
    # Get root note frequency
    if root_note not in note_frequencies:
        root_note = 'C'  # Default to C if note not found
    
    root_freq = note_frequencies[root_note]
    
    # Define tanpura string frequencies (common tuning: Pa-Sa-Sa-Sa)
    # For a typical male vocal tuning
    fifth_freq = root_freq * 3/2  # Perfect fifth (Pa)
    root_octave_down_freq = root_freq / 2  # Sa in lower octave 
    
    # String frequencies
    string_freqs = [fifth_freq, root_freq, root_freq, root_octave_down_freq]
    
    # Calculate time between plucks based on tempo
    seconds_per_beat = 60.0 / tempo
    pluck_interval = seconds_per_beat
    
    # Number of plucks
    num_plucks = int(duration / pluck_interval) + 1
    
    # Initialize audio buffer
    y = np.zeros(int(sr * duration))
    
    # Current position in audio buffer
    current_pos = 0
    
    # String decay factor (longer for lower strings)
    decay_factors = [2.5, 3.0, 3.5, 4.0]  # Decay times in seconds
    
    # Generate each pluck
    for i in range(num_plucks):
        # Select string to pluck (cycle through strings)
        string_idx = i % len(string_freqs)
        string_freq = string_freqs[string_idx]
        decay_time = decay_factors[string_idx]
        
        # Generate pluck
        pluck_duration = min(decay_time * 2, pluck_interval * 4)  # Ensure overlap
        pluck_samples = int(pluck_duration * sr)
        
        # Generate the pluck sound
        pluck = generate_string_pluck(string_freq, pluck_duration, sr=sr, jiva=jiva)
        
        # Add to buffer (with overlap)
        end_pos = min(current_pos + len(pluck), len(y))
        samples_to_add = end_pos - current_pos
        y[current_pos:end_pos] += pluck[:samples_to_add]
        
        # Move to next pluck position
        current_pos = int(i * pluck_interval * sr)
        if current_pos >= len(y):
            break
    
    # Normalize
    y = y / np.max(np.abs(y))
    
    return y, sr

def generate_string_pluck(freq, duration_seconds, sr=44100, jiva=0.5, amp=0.5):
    """
    Generate a single tanpura string pluck with jawari/jiva effect
    
    Parameters:
        freq (float): Frequency of the string
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        jiva (float): Amount of jawari/jiva effect (0.0 to 1.0)
        amp (float): Amplitude of the pluck (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the string pluck
    """
    # Number of samples
    num_samples = int(sr * duration_seconds)
    
    # Generate initial pluck using Karplus-Strong algorithm
    # Buffer size for Karplus-Strong
    buffer_size = int(sr / freq)
    
    # Initialize buffer with random noise
    buffer = np.random.random(buffer_size) * 2 - 1
    
    # Output array
    y = np.zeros(num_samples)
    
    # Basic Karplus-Strong algorithm
    for i in range(num_samples):
        # Get the current output sample
        y[i] = buffer[i % buffer_size]
        
        # Update the buffer (with slight lowpass filter)
        if (i + 1) % buffer_size == 0:
            # Apply the basic Karplus-Strong algorithm to update the buffer
            new_buffer = np.zeros(buffer_size)
            for j in range(buffer_size - 1):
                new_buffer[j] = 0.5 * (buffer[j] + buffer[j + 1])
            new_buffer[-1] = 0.5 * (buffer[-1] + buffer[0])
            buffer = new_buffer
    
    # Apply envelope
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Initial attack
    attack_time = 0.01  # 10ms attack
    attack_env = np.minimum(1.0, t / attack_time)
    
    # Decay
    decay_time = duration_seconds * 0.2
    decay_env = np.exp(-t / decay_time)
    
    # Combine for envelope
    envelope = attack_env * decay_env
    
    # Apply envelope
    y = y * envelope
    
    # Apply jawari/jiva effect (sympathetic resonance and buzz)
    if jiva > 0:
        # Generate harmonics with slight detuning
        num_harmonics = 12
        harmonic_gain = jiva * 0.8
        
        for h in range(2, num_harmonics + 2):
            # Slight detuning for each harmonic
            detune_factor = 1.0 + (np.random.random() * 0.01 - 0.005) * jiva
            harmonic_freq = freq * h * detune_factor
            
            # Generate harmonic sine wave
            t_harmonic = np.arange(num_samples) / sr
            harmonic = np.sin(2 * np.pi * harmonic_freq * t_harmonic)
            
            # Apply faster decay for higher harmonics
            harmonic_decay = np.exp(-t * h / decay_time)
            harmonic = harmonic * harmonic_decay
            
            # Add to main signal with decreasing amplitude for higher harmonics
            harmonic_amplitude = harmonic_gain / (h * 0.7)
            y += harmonic * harmonic_amplitude
        
        # Add some buzz/noise modulation
        if jiva > 0.3:
            # Create buzz modulation
            buzz_freq = freq * 0.5  # Lower frequency for buzz
            t_buzz = np.arange(num_samples) / sr
            buzz_mod = 0.5 + 0.5 * np.sin(2 * np.pi * buzz_freq * t_buzz)
            
            # Create noise
            noise = np.random.random(num_samples) * 2 - 1
            noise = signal.lfilter([1.0], [1.0, -0.99], noise)  # Color the noise
            noise = noise * np.exp(-t / (decay_time * 0.5))  # Faster decay for noise
            
            # Apply buzz modulation to noise and add to signal
            buzz_amount = jiva * 0.2
            y += noise * buzz_mod * buzz_amount
    
    # Apply final amplitude
    y = y * amp
    
    # Ensure no clipping
    y = y / max(1.0, np.max(np.abs(y)))
    
    return y

def synthesize_tabla(tala, tempo=60, duration=60):
    """
    Synthesize tabla sounds for a given tala
    
    Parameters:
        tala (str): Name of tala
        tempo (float): Tempo in BPM
        duration (float): Duration in seconds
    
    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    # Sample rate
    sr = 44100
    
    # Get tala information
    from modules.tala_knowledge import get_tala_info
    tala_info = get_tala_info(tala)
    
    if not tala_info:
        # Default to Teentaal if tala not found
        tala_info = {
            "beats": 16,
            "vibhags": [4, 4, 4, 4],
            "clap_pattern": ["X", "2", "0", "3"]
        }
    
    # Total beats in tala
    total_beats = tala_info["beats"]
    
    # Calculate time per beat based on tempo
    seconds_per_beat = 60.0 / tempo
    
    # Calculate number of cycles
    cycle_duration = seconds_per_beat * total_beats
    num_cycles = int(duration / cycle_duration) + 1
    
    # Initialize audio buffer
    y = np.zeros(int(sr * duration))
    
    # Current beat position for mapping
    beat_position = 0
    vibhag_position = 0
    clap_type = tala_info["clap_pattern"][0]
    
    # Process each beat
    for i in range(num_cycles * total_beats):
        # Calculate beat start time
        beat_time = i * seconds_per_beat
        if beat_time >= duration:
            break
        
        # Calculate sample position
        sample_pos = int(beat_time * sr)
        
        # Get appropriate bol for this beat
        bol = get_bol_for_beat(beat_position, clap_type)
        
        # Generate tabla sound for this bol
        tabla_sound = create_tabla_sound(bol, 0.5, sr)  # 0.5 seconds per sound
        
        # Add sound to buffer
        end_pos = min(sample_pos + len(tabla_sound), len(y))
        samples_to_add = end_pos - sample_pos
        y[sample_pos:end_pos] += tabla_sound[:samples_to_add]
        
        # Update beat position
        beat_position = (beat_position + 1) % total_beats
        
        # Update vibhag position and clap type
        if beat_position == 0 or sum(tala_info["vibhags"][:vibhag_position+1]) <= beat_position:
            vibhag_position = (vibhag_position + 1) % len(tala_info["vibhags"])
            clap_type = tala_info["clap_pattern"][vibhag_position]
    
    # Normalize
    y = y / max(1.0, np.max(np.abs(y)))
    
    return y, sr

def create_tabla_sound(bol, duration_seconds, sr=44100, amp=0.7):
    """
    Create a tabla sound for a specific bol (syllable)
    
    Parameters:
        bol (str): Tabla bol/syllable (e.g., "dha", "dhin", "tin", "ta")
        duration_seconds (float): Duration in seconds
        sr (int): Sample rate
        amp (float): Amplitude of the sound (0.0 to 1.0)
    
    Returns:
        np.ndarray: Audio data for the tabla sound
    """
    # Number of samples
    num_samples = int(sr * duration_seconds)
    
    # Initialize output array
    y = np.zeros(num_samples)
    
    # Create time array
    t = np.arange(num_samples) / sr
    
    # Different bols require different synthesis approaches
    if bol.lower() in ["dha", "dhin"]:
        # Dayan (right drum) + Bayan (left drum)
        
        # Dayan component (higher pitched)
        dayan_freq = 420  # Hz
        dayan_decay = 0.2  # seconds
        dayan_env = np.exp(-t / dayan_decay)
        dayan_sound = np.sin(2 * np.pi * dayan_freq * t) * dayan_env
        
        # Bayan component (lower pitched)
        bayan_freq = 80  # Hz base frequency
        bayan_decay = 0.3  # seconds
        bayan_env = np.exp(-t / bayan_decay)
        
        # Add frequency sweep for bayan (characteristic of bass drum)
        bayan_freq_sweep = bayan_freq * (1 + 0.5 * np.exp(-t / 0.05))
        bayan_phase = 2 * np.pi * np.cumsum(bayan_freq_sweep) / sr
        bayan_sound = np.sin(bayan_phase) * bayan_env
        
        # Add resonance to bayan
        resonance_freq = 160  # Hz
        resonance_env = np.exp(-t / 0.15)
        resonance = np.sin(2 * np.pi * resonance_freq * t) * resonance_env * 0.3
        bayan_sound += resonance
        
        # Combine dayan and bayan
        if bol.lower() == "dha":
            y = dayan_sound * 0.6 + bayan_sound * 0.7
        else:  # "dhin" - more emphasis on bayan
            y = dayan_sound * 0.5 + bayan_sound * 0.8
    
    elif bol.lower() in ["ta", "tin", "tun"]:
        # Dayan (right drum) only
        
        # Membrane frequency
        if bol.lower() == "ta":
            freq = 380  # Hz
        elif bol.lower() == "tin":
            freq = 420  # Hz
        else:  # "tun"
            freq = 400  # Hz
        
        # Quick decay
        decay = 0.15  # seconds
        env = np.exp(-t / decay)
        
        # Membrane sound
        membrane = np.sin(2 * np.pi * freq * t) * env
        
        # Add higher harmonics
        for harmonic in [2, 3, 5]:
            harmonic_amp = 1.0 / (harmonic * 2)
            harmonic_decay = decay / harmonic
            harmonic_env = np.exp(-t / harmonic_decay)
            membrane += np.sin(2 * np.pi * freq * harmonic * t) * harmonic_env * harmonic_amp
        
        y = membrane
    
    elif bol.lower() in ["na", "ne"]:
        # Na stroke (muted dayan)
        
        # Higher frequency but very damped
        freq = 350  # Hz
        decay = 0.08  # Very quick decay
        env = np.exp(-t / decay)
        
        # Add noise component for damped sound
        noise = np.random.random(num_samples) * 2 - 1
        noise = signal.lfilter([1.0], [1.0, -0.98], noise)  # Color the noise
        noise_env = np.exp(-t / 0.05)  # Very quick decay for noise
        
        # Combine
        y = (np.sin(2 * np.pi * freq * t) * 0.7 + noise * 0.3) * env
    
    elif bol.lower() in ["ge", "ke"]:
        # Bayan (left drum) only
        
        # Low frequency with sweep
        base_freq = 70  # Hz
        decay = 0.25  # seconds
        env = np.exp(-t / decay)
        
        # Frequency sweep (characteristic of bayan)
        freq_sweep = base_freq * (1 + 0.7 * np.exp(-t / 0.06))
        phase = 2 * np.pi * np.cumsum(freq_sweep) / sr
        
        # Generate sound with sweep
        y = np.sin(phase) * env
        
        # Add some noise for attack
        noise = np.random.random(num_samples) * 2 - 1
        noise = signal.lfilter([1.0], [1.0, -0.95], noise)  # Color the noise
        noise_env = np.exp(-t / 0.03)  # Very quick decay for noise
        y += noise * noise_env * 0.2
    
    else:
        # Default generic tabla sound
        freq = 350  # Hz
        decay = 0.2  # seconds
        env = np.exp(-t / decay)
        y = np.sin(2 * np.pi * freq * t) * env
    
    # Apply overall amplitude
    y = y * amp
    
    # Ensure no clipping
    if np.max(np.abs(y)) > 1.0:
        y = y / np.max(np.abs(y))
    
    return y

def get_bol_for_beat(beat_index, clap_type):
    """
    Get the appropriate tabla bol for a specific beat position

    Parameters:
        beat_index (int): Index of the beat in the tala cycle
        clap_type (int): Type of beat (0 for regular, 1 for sam, 2 for tali, etc.)

    Returns:
        str: Tabla bol (syllable)
    """
    # Sam (first beat of cycle)
    if clap_type == "X":
        return "dha"

    # Khali (empty beat)
    elif clap_type == "0":
        return "ta"

    # Tali (clap beats)
    else:
        # Alternate between different bols for variety
        if beat_index % 3 == 0:
            return "dhin"
        elif beat_index % 2 == 0:
            return "tin"
        else:
            return "na"

# Alias for synthesize_tabla to maintain backward compatibility
def generate_tabla(tala, tempo=60, duration=60):
    """
    Alias for synthesize_tabla function

    Parameters:
        tala (str): Name of tala
        tempo (float): Tempo in BPM
        duration (float): Duration in seconds

    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    return synthesize_tabla(tala, tempo, duration)

def synthesize_melody(raga, instrument="sitar", tempo=60, duration=30, complexity=5):
    """
    Generate a melodic pattern based on a specific raga

    Parameters:
        raga (str): Name of the raga
        instrument (str): Instrument sound to use
        tempo (float): Tempo in BPM
        duration (float): Duration in seconds
        complexity (int): Complexity level (1-10)

    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    # Sample rate
    sr = 44100
    
    # Get raga information
    from modules.raga_knowledge import get_raga_info
    raga_info = get_raga_info(raga)
    
    if not raga_info:
        # Create placeholder audio
        t = np.arange(int(sr * duration)) / sr
        y = np.sin(2 * np.pi * 440 * t) * np.exp(-t / 5)
        return y, sr
    
    # Simplified placeholder implementation
    # In a real implementation, we would use the raga's structure to generate authentic phrases
    
    # Create a simple placeholder melody based on the notes of the raga
    notes = raga_info["notes"]["aroha"] + raga_info["notes"]["avaroha"]
    
    # Map notes to frequencies (very simplified)
    base_freq = 261.63  # C4
    
    # Create time array
    t = np.arange(int(sr * duration)) / sr
    
    # Initialize output
    y = np.zeros_like(t)
    
    # Simple note mapping to semitones (very approximate)
    note_to_semitone = {
        'S': 0, 'R': 2, 'G': 4, 'M': 5, 'P': 7, 'D': 9, 'N': 11, 'S\'': 12,
        'r': 1, 'g': 3, 'n': 10, 'd': 8, 'm': 6
    }
    
    # Current position in seconds
    current_time = 0
    
    # Notes per beat based on complexity
    notes_per_beat = max(1, min(4, complexity // 3 + 1))
    
    # Beat duration in seconds
    beat_duration = 60.0 / tempo
    note_duration = beat_duration / notes_per_beat
    
    # Generate sequence of notes
    while current_time < duration:
        # Select a note (with weighted probability for vadi/samvadi)
        note_idx = np.random.randint(0, len(notes))
        note = notes[note_idx]
        
        # Get base note
        base_note = note[0]
        if base_note == 'S' and len(note) > 1 and note[1] == "'":
            base_note = 'S\''
        
        # Calculate frequency adjustment
        semitone = note_to_semitone.get(base_note, 0)
        
        # Adjust for komal or tivra
        if '(k)' in note:  # Komal (flat)
            semitone -= 1
        elif '#' in note:  # Tivra (sharp)
            semitone += 1
        
        # Calculate frequency
        freq = base_freq * 2 ** (semitone / 12.0)
        
        # Decide note duration (with some variation)
        duration_variation = 0.8 + 0.4 * np.random.random()
        current_note_duration = note_duration * duration_variation
        
        # End time for this note
        end_time = min(current_time + current_note_duration, duration)
        
        # Convert time to samples
        start_sample = int(current_time * sr)
        end_sample = int(end_time * sr)
        
        # Generate note based on instrument
        note_samples = end_sample - start_sample
        t_note = np.arange(note_samples) / sr
        
        # Generate basic tone with envelope
        attack = 0.05  # 50ms attack
        decay = 0.2  # 200ms decay
        sustain = 0.7  # Sustain level
        release = 0.3  # 300ms release
        
        # ADSR envelope
        t_note_norm = t_note / max(t_note[-1], 0.001)  # Normalized time
        env = np.ones_like(t_note_norm)
        
        # Attack phase
        attack_mask = t_note_norm < attack
        env[attack_mask] = t_note_norm[attack_mask] / attack
        
        # Decay phase
        decay_mask = (t_note_norm >= attack) & (t_note_norm < attack + decay)
        env[decay_mask] = 1.0 - (1.0 - sustain) * (t_note_norm[decay_mask] - attack) / decay
        
        # Sustain phase
        sustain_mask = (t_note_norm >= attack + decay) & (t_note_norm < 1.0 - release)
        env[sustain_mask] = sustain
        
        # Release phase
        release_mask = t_note_norm >= 1.0 - release
        env[release_mask] = sustain * (1.0 - (t_note_norm[release_mask] - (1.0 - release)) / release)
        
        # Base note sound
        note_sound = np.sin(2 * np.pi * freq * t_note)
        
        # Add harmonics based on instrument
        if instrument.lower() == "sitar":
            # Sitar has rich harmonics and characteristic buzz
            for harmonic in range(2, 10):
                harmonic_amp = 1.0 / harmonic
                note_sound += np.sin(2 * np.pi * freq * harmonic * t_note) * harmonic_amp
            
            # Add sitar's characteristic buzz/jangle
            buzz_freq = freq * 2.1  # Slightly detuned high harmonic
            buzz = np.sin(2 * np.pi * buzz_freq * t_note) * 0.1
            buzz *= np.random.random(len(buzz)) * 0.5 + 0.5  # Modulate the buzz
            note_sound += buzz
        
        elif instrument.lower() == "sarod":
            # Sarod has a cleaner, more sustained tone
            for harmonic in range(2, 8):
                harmonic_amp = 0.8 / harmonic
                note_sound += np.sin(2 * np.pi * freq * harmonic * t_note) * harmonic_amp
            
            # Less attack noise, more sustain
            env = np.ones_like(t_note_norm)
            env = 0.2 + 0.8 * env  # Higher sustain
        
        elif instrument.lower() == "bansuri" or instrument.lower() == "flute":
            # Bansuri (flute) has fewer harmonics, more pure tone
            note_sound = np.sin(2 * np.pi * freq * t_note)
            # Add just a couple of harmonics
            note_sound += np.sin(2 * np.pi * freq * 2 * t_note) * 0.3
            note_sound += np.sin(2 * np.pi * freq * 3 * t_note) * 0.15
            
            # Add some breath noise
            breath_noise = np.random.random(len(t_note)) * 0.1
            breath_noise = signal.lfilter([1.0], [1.0, -0.98], breath_noise)  # Color the noise
            note_sound += breath_noise
            
            # Smoother envelope
            env = 0.2 + 0.8 * np.ones_like(t_note_norm)  # Higher sustain
            env *= 1.0 - 0.3 * np.random.random(len(env))  # Slight random modulation
        
        else:  # Default synthesized sound
            # Simple tone with some harmonics
            for harmonic in range(2, 5):
                harmonic_amp = 0.5 / harmonic
                note_sound += np.sin(2 * np.pi * freq * harmonic * t_note) * harmonic_amp
        
        # Apply envelope
        note_sound = note_sound * env
        
        # Add to output
        y[start_sample:end_sample] += note_sound
        
        # Move to next note
        current_time = end_time

    # Normalize
    y = y / max(1.0, np.max(np.abs(y)))

    return y, sr

# Alias for synthesize_melody to maintain backward compatibility
def synthesize_raga_phrase(raga, instrument="sitar", tempo=60, duration=30, complexity=5):
    """
    Alias for synthesize_melody function

    Parameters:
        raga (str): Name of the raga
        instrument (str): Instrument sound to use
        tempo (float): Tempo in BPM
        duration (float): Duration in seconds
        complexity (int): Complexity level (1-10)

    Returns:
        tuple: (y, sr) audio time series and sample rate
    """
    return synthesize_melody(raga, instrument, tempo, duration, complexity)