"""
Audio Utility Functions for RagaVani application

This module provides utility functions for audio processing, including loading,
saving, playing, recording, and preprocessing audio data.
"""

import numpy as np
import soundfile as sf
import librosa
import time
import io
import matplotlib.pyplot as plt
import base64
import os

def load_audio(file_path, sr=None):
    """
    Load audio from file
    
    Parameters:
        file_path (str): Path to audio file
        sr (int, optional): Target sample rate. If None, uses the file's sample rate
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    try:
        # Try loading with librosa first (handles more formats)
        y, sr_orig = librosa.load(file_path, sr=sr, mono=True)
        return y, sr_orig
    except Exception as e:
        try:
            # Fall back to soundfile
            y, sr_orig = sf.read(file_path)
            
            # Convert to mono if stereo
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.mean(y, axis=1)
            
            # Resample if needed
            if sr is not None and sr != sr_orig:
                y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
                sr_orig = sr
                
            return y, sr_orig
        except Exception as e2:
            print(f"Error loading audio file: {e} / {e2}")
            return None, None

def save_audio(audio_data, file_path, sr=44100, format='wav'):
    """
    Save audio to file
    
    Parameters:
        audio_data (np.ndarray): Audio time series
        file_path (str): Path to save the audio file
        sr (int): Sample rate
        format (str): Audio format ('wav', 'mp3', etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Save the file
        sf.write(file_path, audio_data, sr, format=format)
        return True
    except Exception as e:
        print(f"Error saving audio file: {e}")
        return False


def trim_silence(y, sr, threshold_db=-50, frame_length=2048, hop_length=512):
    """
    Trim leading and trailing silence from an audio signal
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        threshold_db (float): Threshold in decibels
        frame_length (int): Frame length for silence detection
        hop_length (int): Hop length for silence detection
    
    Returns:
        np.ndarray: Trimmed audio time series
    """
    try:
        # Trim audio
        y_trimmed, _ = librosa.effects.trim(
            y, frame_length=frame_length, hop_length=hop_length, top_db=-threshold_db
        )
        return y_trimmed
    except Exception as e:
        print(f"Error trimming silence: {e}")
        return y

def normalize_audio(y, target_db=-3.0):
    """
    Normalize audio to a target dB level
    
    Parameters:
        y (np.ndarray): Audio time series
        target_db (float): Target peak level in dB
    
    Returns:
        np.ndarray: Normalized audio time series
    """
    try:
        # Calculate the current peak
        peak = np.max(np.abs(y))
        
        # Calculate the desired peak
        target_peak = 10 ** (target_db / 20.0)
        
        # Apply gain
        if peak > 0:
            normalized = y * (target_peak / peak)
            return normalized
        else:
            return y
    except Exception as e:
        print(f"Error normalizing audio: {e}")
        return y

def convert_to_mono(y):
    """
    Convert stereo audio to mono
    
    Parameters:
        y (np.ndarray): Audio time series (can be stereo or mono)
    
    Returns:
        np.ndarray: Mono audio time series
    """
    try:
        # Check if input is stereo
        if len(y.shape) > 1 and y.shape[1] > 1:
            return np.mean(y, axis=1)
        else:
            return y
    except Exception as e:
        print(f"Error converting to mono: {e}")
        return y

def get_audio_base64(y, sr, format='wav'):
    """
    Convert audio data to base64 string for web playback
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        format (str): Audio format ('wav', 'mp3', etc.)
    
    Returns:
        str: Base64 encoded audio data
    """
    try:
        # Buffer for writing
        buffer = io.BytesIO()
        
        # Write to the buffer
        sf.write(buffer, y, sr, format=format)
        
        # Get the buffer content
        audio_bytes = buffer.getvalue()
        
        # Encode as base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return audio_b64
    except Exception as e:
        print(f"Error converting audio to base64: {e}")
        return None

def audio_to_html(y, sr, format='wav'):
    """
    Generate HTML audio element for audio data
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        format (str): Audio format ('wav', 'mp3', etc.)
    
    Returns:
        str: HTML audio element
    """
    try:
        audio_b64 = get_audio_base64(y, sr, format)
        if audio_b64:
            mime_type = f'audio/{format}'
            html = f'<audio controls><source src="data:{mime_type};base64,{audio_b64}" type="{mime_type}"></audio>'
            return html
        else:
            return "<p>Error creating audio player</p>"
    except Exception as e:
        print(f"Error creating audio HTML: {e}")
        return "<p>Error creating audio player</p>"

def plot_waveform(y, sr, figsize=(10, 4)):
    """
    Create a waveform plot for audio visualization
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        figsize (tuple): Figure size (width, height) in inches
    
    Returns:
        matplotlib.figure.Figure: Figure containing the waveform plot
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, facecolor="#FFF8DC")
        
        # Time axis
        times = np.arange(len(y)) / float(sr)
        
        # Plot waveform
        ax.plot(times, y, color="#800000", linewidth=1)
        
        # Add labels
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Waveform", color="#800000")
        
        # Set background color
        ax.set_facecolor("#FFF8DC")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"Error plotting waveform: {e}")
        # Return empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error plotting waveform: {e}", 
                horizontalalignment='center', verticalalignment='center')
        return fig

def plot_spectrogram(y, sr, figsize=(10, 6), n_fft=2048, hop_length=512):
    """
    Create a spectrogram plot for audio visualization
    
    Parameters:
        y (np.ndarray): Audio time series
        sr (int): Sample rate
        figsize (tuple): Figure size (width, height) in inches
        n_fft (int): FFT window size
        hop_length (int): Hop length
    
    Returns:
        matplotlib.figure.Figure: Figure containing the spectrogram plot
    """
    try:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize, facecolor="#FFF8DC")
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        
        # Plot spectrogram
        img = librosa.display.specshow(
            D, x_axis='time', y_axis='log', sr=sr, hop_length=hop_length, ax=ax
        )
        
        # Add colorbar
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        
        # Add labels
        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("Spectrogram", color="#800000")
        
        # Set background color
        ax.set_facecolor("#FFF8DC")
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        print(f"Error plotting spectrogram: {e}")
        # Return empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Error plotting spectrogram: {e}",
                horizontalalignment='center', verticalalignment='center')
        return fig

