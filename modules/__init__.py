"""
RagaVani Modules Package

This package contains the core modules for the RagaVani application.
"""

from modules.raga_knowledge import get_raga_info, get_all_ragas, get_raga_by_mood, get_raga_by_time, compare_ragas
from modules.tala_knowledge import get_tala_info, get_all_talas, get_tala_by_beats, get_tala_clap_pattern, compare_talas
from modules.audio_analysis import analyze_audio
from modules.audio_analysis_simplified import analyze_audio as analyze_audio_simple
from modules.audio_analysis_hybrid import (
    analyze_audio as analyze_audio_hybrid,
    extract_pitch_contour,
    identify_raga,
    detect_tala,
    detect_ornaments
)
from modules.audio_synthesis import generate_tanpura, synthesize_tabla, generate_tabla, synthesize_melody, synthesize_raga_phrase
from modules.visualization import plot_pitch_contour, plot_raga_distribution, plot_tala_pattern, create_spectrogram
from modules.neural_processing import analyze_with_neural_models, identify_raga_neural, detect_tala_neural
from modules.symbolic_processing import analyze_composition_symbolic, generate_composition_symbolic, convert_to_notation