"""
Visualization Module for RagaVani application

This module provides visualization tools for audio data,
including waveforms, spectrograms, pitch contours, and note distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Define Indian-styled colors
COLORS = {
    "primary": "#8b4513",  # Brown
    "accent": "#d4af37",   # Gold
    "background": "#fbf5e6", # Light cream
    "text": "#333333",
    "highlight": "#cc5500"  # Burnt orange
}

def plot_waveform(y, sr):
    """
    Plot the waveform of an audio signal
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate
    
    Returns:
        Figure: Matplotlib figure object
    """
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    
    # Calculate time axis
    times = np.arange(len(y)) / sr
    duration = len(y) / sr
    
    # Plot waveform
    ax.plot(times, y, color=COLORS["primary"], linewidth=0.5)
    
    # Add envelope for better visualization
    n_fft = 2048
    hop_length = 512
    envelope = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    env_times = librosa.times_like(envelope, sr=sr, hop_length=hop_length)
    env = np.max(envelope, axis=0)
    env = env / np.max(env)
    
    # Scale the envelope to match the waveform
    env_scaled = env * np.max(np.abs(y)) * 0.9
    ax.plot(env_times, env_scaled, color=COLORS["accent"], alpha=0.8, linewidth=1.5)
    ax.plot(env_times, -env_scaled, color=COLORS["accent"], alpha=0.8, linewidth=1.5)
    
    # Set labels and title
    ax.set_xlabel("Time (seconds)", color=COLORS["text"])
    ax.set_ylabel("Amplitude", color=COLORS["text"])
    ax.set_title("Audio Waveform", color=COLORS["primary"], fontweight="bold")
    
    # Set axis limits
    ax.set_xlim(0, duration)
    
    # Customize ticks
    ax.tick_params(colors=COLORS["text"])
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_pitch_contour(times, pitches):
    """
    Plot the pitch contour of an audio signal
    
    Parameters:
        times (np.ndarray): Array of time points
        pitches (np.ndarray): Array of pitch values in Hz
    
    Returns:
        Figure: Matplotlib figure object
    """
    # Filter out zero values (where no pitch was detected)
    valid_indices = pitches > 0
    valid_times = times[valid_indices]
    valid_pitches = pitches[valid_indices]
    
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    
    # Plot pitch contour
    ax.plot(valid_times, valid_pitches, color=COLORS["primary"], linewidth=2)
    
    # Add smoothed contour
    if len(valid_pitches) > 10:
        try:
            smoothed_pitches = librosa.util.smooth(valid_pitches, window_length=11, window_type='hann')
            ax.plot(valid_times, smoothed_pitches, color=COLORS["accent"], linewidth=1.5, alpha=0.7)
        except:
            # If smoothing fails, skip this step
            pass
    
    # Set labels and title
    ax.set_xlabel("Time (seconds)", color=COLORS["text"])
    ax.set_ylabel("Frequency (Hz)", color=COLORS["text"])
    ax.set_title("Pitch Contour", color=COLORS["primary"], fontweight="bold")
    
    # Set y-axis to logarithmic scale
    ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set y-axis limits to a reasonable range for music
    if len(valid_pitches) > 0:
        y_min = max(50, valid_pitches.min() * 0.8)
        y_max = min(1000, valid_pitches.max() * 1.2)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(50, 1000)
    
    # Add horizontal lines for reference notes (Sa, Pa)
    if len(valid_pitches) > 0:
        # Estimate tonic (Sa)
        pitch_hist, bins = np.histogram(valid_pitches, bins=60, range=(50, 1000))
        tonic_idx = np.argmax(pitch_hist)
        tonic_freq = bins[tonic_idx]
        
        # Add tonic line (Sa)
        ax.axhline(y=tonic_freq, color=COLORS["highlight"], linestyle='--', alpha=0.8, linewidth=1)
        ax.text(times[-1]*1.01, tonic_freq, "Sa", color=COLORS["highlight"], fontweight="bold")
        
        # Add fifth (Pa)
        fifth_freq = tonic_freq * 1.5
        if fifth_freq < y_max:
            ax.axhline(y=fifth_freq, color=COLORS["highlight"], linestyle='--', alpha=0.8, linewidth=1)
            ax.text(times[-1]*1.01, fifth_freq, "Pa", color=COLORS["highlight"], fontweight="bold")
    
    # Customize ticks
    ax.tick_params(colors=COLORS["text"])
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_note_distribution(pitches):
    """
    Plot the distribution of notes in an audio signal
    
    Parameters:
        pitches (np.ndarray): Array of pitch values in Hz
    
    Returns:
        Figure: Matplotlib figure object
    """
    # Filter out zero values (where no pitch was detected)
    valid_pitches = pitches[pitches > 0]
    
    if len(valid_pitches) == 0:
        # Create an empty figure if no valid pitches
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor(COLORS["background"])
        ax.set_facecolor(COLORS["background"])
        ax.set_title("Note Distribution (No valid pitches detected)", color=COLORS["primary"], fontweight="bold")
        ax.set_xlabel("Note", color=COLORS["text"])
        ax.set_ylabel("Frequency", color=COLORS["text"])
        plt.tight_layout()
        return fig
    
    # Convert from Hz to MIDI note numbers (logarithmic pitch)
    midi_notes = librosa.hz_to_midi(valid_pitches)
    
    # Wrap to single octave (0-11 representing C, C#, D, etc.)
    notes = midi_notes % 12
    
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    
    # Create histogram of notes
    note_bins = np.linspace(0, 12, 13)  # 13 bins for 12 semitones (C to C)
    note_hist, bins = np.histogram(notes, bins=note_bins)
    
    # Normalize histogram
    note_hist = note_hist / np.sum(note_hist)
    
    # Plot histogram as bars
    bar_positions = (bins[:-1] + bins[1:]) / 2  # Center of bins
    bar_width = 0.8
    bars = ax.bar(bar_positions, note_hist, width=bar_width, color=COLORS["primary"], alpha=0.7, edgecolor=COLORS["accent"])
    
    # Highlight the most common notes
    threshold = 0.7 * note_hist.max()
    for i, (bar, height) in enumerate(zip(bars, note_hist)):
        if height >= threshold:
            bar.set_color(COLORS["highlight"])
            bar.set_edgecolor(COLORS["accent"])
    
    # Set labels and title
    ax.set_xlabel("Note", color=COLORS["text"])
    ax.set_ylabel("Relative Frequency", color=COLORS["text"])
    ax.set_title("Note Distribution", color=COLORS["primary"], fontweight="bold")
    
    # Set x-ticks to note names
    note_names = ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']
    ax.set_xticks(range(12))
    ax.set_xticklabels(note_names)
    
    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45)
    
    # Customize ticks
    ax.tick_params(colors=COLORS["text"])
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_spectrogram(y, sr):
    """
    Plot the spectrogram of an audio signal

    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate

    Returns:
        Figure: Matplotlib figure object
    """
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Create spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    # Plot spectrogram
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax, cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.tick_params(colors=COLORS["text"])
    cbar.set_label('Amplitude (dB)', color=COLORS["text"])

    # Set labels and title
    ax.set_xlabel("Time (seconds)", color=COLORS["text"])
    ax.set_ylabel("Frequency (Hz)", color=COLORS["text"])
    ax.set_title("Spectrogram", color=COLORS["primary"], fontweight="bold")

    # Customize ticks
    ax.tick_params(colors=COLORS["text"])

    # Tight layout
    plt.tight_layout()

    return fig

# Alias for plot_spectrogram to maintain backward compatibility
def create_spectrogram(y, sr):
    """
    Alias for plot_spectrogram function

    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sampling rate

    Returns:
        Figure: Matplotlib figure object
    """
    return plot_spectrogram(y, sr)

def plot_raga_distribution(raga_data, reference_raga=None):
    """
    Plot the distribution of notes in a raga compared to a reference raga

    Parameters:
        raga_data (dict): Dictionary containing raga note distribution data
        reference_raga (dict, optional): Dictionary containing reference raga data for comparison

    Returns:
        Figure: Matplotlib figure object
    """
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Note names in Indian classical music
    note_names = ['Sa', 're', 'Re', 'ga', 'Ga', 'ma', 'Ma', 'Pa', 'dha', 'Dha', 'ni', 'Ni']

    # Check if raga_data contains note distribution
    if 'note_distribution' not in raga_data:
        # Create placeholder data
        note_distribution = np.zeros(12)
        raga_name = "Unknown Raga"
    else:
        note_distribution = raga_data['note_distribution']
        raga_name = raga_data.get('name', "Analyzed Raga")

    # Plot the main raga distribution
    x = np.arange(len(note_names))
    bar_width = 0.35
    bars = ax.bar(x, note_distribution, bar_width, color=COLORS["primary"],
                 alpha=0.7, label=raga_name, edgecolor=COLORS["accent"])

    # Highlight the most common notes
    threshold = 0.7 * max(note_distribution) if max(note_distribution) > 0 else 0
    for i, (bar, height) in enumerate(zip(bars, note_distribution)):
        if height >= threshold:
            bar.set_color(COLORS["highlight"])
            bar.set_edgecolor(COLORS["accent"])

    # If reference raga is provided, plot it for comparison
    if reference_raga is not None and 'note_distribution' in reference_raga:
        ref_distribution = reference_raga['note_distribution']
        ref_name = reference_raga.get('name', "Reference Raga")

        ax.bar(x + bar_width, ref_distribution, bar_width, color=COLORS["accent"],
              alpha=0.7, label=ref_name, edgecolor=COLORS["primary"])

    # Set labels and title
    ax.set_xlabel("Notes", color=COLORS["text"], fontsize=12)
    ax.set_ylabel("Relative Frequency", color=COLORS["text"], fontsize=12)
    ax.set_title("Raga Note Distribution", color=COLORS["primary"], fontweight="bold", fontsize=14)

    # Set x-ticks to note names
    ax.set_xticks(x + bar_width/2 if reference_raga else x)
    ax.set_xticklabels(note_names)

    # Rotate x-tick labels
    plt.setp(ax.get_xticklabels(), rotation=45)

    # Add legend if reference raga is provided
    if reference_raga is not None:
        ax.legend(loc='upper right', facecolor=COLORS["background"])

    # Customize ticks
    ax.tick_params(colors=COLORS["text"])

    # Add grid
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')

    # Tight layout
    plt.tight_layout()

    return fig

def plot_tala_pattern(tala_data):
    """
    Plot the rhythmic pattern of a tala

    Parameters:
        tala_data (dict): Dictionary containing tala information

    Returns:
        Figure: Matplotlib figure object
    """
    # Create figure with Indian-styled aesthetics
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Check if tala_data contains required information
    if not tala_data or 'beats' not in tala_data or 'vibhags' not in tala_data:
        # Create placeholder
        ax.text(0.5, 0.5, "Tala data not available",
                ha='center', va='center', fontsize=14, color=COLORS["primary"])
        ax.set_title("Tala Pattern", color=COLORS["primary"], fontweight="bold", fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        return fig

    # Extract tala information
    total_beats = tala_data['beats']
    vibhags = tala_data['vibhags']
    clap_pattern = tala_data.get('clap_pattern', [])
    tala_name = tala_data.get('name', "Unknown Tala")

    # Create beat positions
    beat_positions = np.arange(1, total_beats + 1)

    # Create beat markers (1 for all beats)
    beat_markers = np.ones(total_beats)

    # Plot beats as markers
    ax.stem(beat_positions, beat_markers, linefmt='-', markerfmt='o',
            basefmt=' ', label='Beats', use_line_collection=True)

    # Customize markers based on clap pattern
    if clap_pattern:
        # Calculate vibhag boundaries
        vibhag_boundaries = [0]
        for v in vibhags:
            vibhag_boundaries.append(vibhag_boundaries[-1] + v)

        # Assign clap types to each beat
        beat_types = []
        for i in range(len(vibhags)):
            start = vibhag_boundaries[i]
            end = vibhag_boundaries[i+1]
            clap_type = clap_pattern[i] if i < len(clap_pattern) else "0"

            for _ in range(end - start):
                beat_types.append(clap_type)

        # Plot different markers for different beat types
        for i, beat_type in enumerate(beat_types):
            pos = i + 1  # Beat position (1-indexed)

            if beat_type == "X":  # Sam (first beat)
                ax.plot(pos, 1, 'o', markersize=15, color=COLORS["highlight"],
                        markeredgecolor=COLORS["primary"], markeredgewidth=2)
            elif beat_type == "0":  # Khali (wave)
                ax.plot(pos, 1, 's', markersize=10, color=COLORS["background"],
                        markeredgecolor=COLORS["primary"], markeredgewidth=2)
            else:  # Tali (clap)
                ax.plot(pos, 1, 'o', markersize=10, color=COLORS["accent"],
                        markeredgecolor=COLORS["primary"], markeredgewidth=1.5)

    # Add vibhag separators
    if vibhags:
        current_pos = 0
        for v in vibhags[:-1]:
            current_pos += v
            ax.axvline(x=current_pos + 0.5, color=COLORS["primary"],
                      linestyle='--', alpha=0.7, linewidth=1.5)

    # Set labels and title
    ax.set_title(f"Tala Pattern: {tala_name}", color=COLORS["primary"],
                fontweight="bold", fontsize=14)
    ax.set_xlabel("Beat Number", color=COLORS["text"], fontsize=12)

    # Set y-axis limits and remove y-ticks
    ax.set_ylim(0, 1.5)
    ax.set_yticks([])

    # Set x-ticks to beat numbers
    ax.set_xticks(beat_positions)
    ax.set_xticklabels([str(i) for i in beat_positions])

    # Add legend for beat types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["highlight"],
               markersize=10, label='Sam (X)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["accent"],
               markersize=10, label='Tali (Clap)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=COLORS["background"],
               markeredgecolor=COLORS["primary"], markersize=10, label='Khali (Wave)')
    ]
    ax.legend(handles=legend_elements, loc='upper center',
             bbox_to_anchor=(0.5, -0.15), ncol=3, facecolor=COLORS["background"])

    # Customize ticks
    ax.tick_params(colors=COLORS["text"])

    # Tight layout
    plt.tight_layout()

    return fig