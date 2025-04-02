"""
Music Composer Page for RagaVani

This page provides a sophisticated user interface for generating and evaluating
high-quality Indian classical music compositions using advanced deep learning models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import soundfile as sf
import time
import os

# Import the music composition module
from modules.music_composition import (
    BiLSTMComposer,
    CNNGANComposer,
    HybridComposer,
    symbolic_to_audio
)

# Import raga knowledge
from modules.raga_knowledge import get_all_ragas, get_raga_info

def render_decorative_divider():
    """Render a decorative divider"""
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <span style="display: inline-block; height: 1px; width: 10%; background-color: #800000; margin: 0 10px;"></span>
        <span style="color: #800000; font-size: 20px;">♪</span>
        <span style="display: inline-block; height: 1px; width: 30%; background-color: #800000; margin: 0 10px;"></span>
        <span style="color: #800000; font-size: 20px;">♪</span>
        <span style="display: inline-block; height: 1px; width: 30%; background-color: #800000; margin: 0 10px;"></span>
        <span style="color: #800000; font-size: 20px;">♪</span>
        <span style="display: inline-block; height: 1px; width: 10%; background-color: #800000; margin: 0 10px;"></span>
    </div>
    """, unsafe_allow_html=True)

def render_music_composer_page():
    """Render the music composer page"""
    st.markdown("## Music Composer")
    
    st.markdown("""
    <div style="background-color: rgba(72, 61, 139, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #483D8B;">
        <p>Create authentic Indian classical music compositions using our advanced deep learning models.
        Choose between different composition approaches, customize parameters, and generate high-quality
        music that respects the traditional grammar and aesthetics of ragas.
        Our models have been trained on extensive collections of classical performances to capture
        the nuances, ornamentations, and structural elements that define each raga.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all ragas
    all_ragas = get_all_ragas()
    raga_names = all_ragas  # all_ragas is already a list of raga names
    
    # Create tabs for different composition models
    tab1, tab2, tab3 = st.tabs(["Melodic Composer (Bi-LSTM)", "Audio Composer (CNNGAN)", "Hybrid Composer"])
    
    with tab1:
        st.markdown("### Melodic Composition with Bi-LSTM")
        
        st.markdown("""
        The Bi-LSTM model generates sophisticated melodic sequences that authentically follow the grammar and structure of specific ragas.
        It excels at creating coherent musical phrases while respecting the rules of the chosen raga, including:

        - Proper note progressions (arohanam/avarohanam)
        - Characteristic phrases (pakad)
        - Emphasis on important notes (vadi/samvadi)
        - Natural musical phrasing and development

        This model has been trained on thousands of traditional compositions to capture the essence of each raga.
        """)
        
        # Raga selection
        selected_raga = st.selectbox("Select Raga:", raga_names, key="bilstm_raga")

        # Get raga info
        raga_info = get_raga_info(selected_raga)

        # Display raga information
        if raga_info:
            aroha_str = ' '.join(raga_info.get('aroha', []))
            avaroha_str = ' '.join(raga_info.get('avaroha', []))
            st.markdown(f"""
            **Raga**: {selected_raga}
            **Arohanam**: {aroha_str}
            **Avarohanam**: {avaroha_str}
            **Time**: {raga_info.get('time', 'N/A')}
            """)
        
        # Composition parameters
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_length = st.slider("Sequence Length:", 32, 256, 64, key="bilstm_length")
            
        with col2:
            temperature = st.slider("Temperature:", 0.1, 2.0, 1.0, 0.1, key="bilstm_temp")
        
        # Seed input
        seed = st.text_input("Seed Sequence (optional):", key="bilstm_seed", 
                            help="Enter a sequence of notes (S, R, G, M, P, D, N) to start the composition")
        
        # Generate button
        if st.button("Generate Melodic Sequence", key="bilstm_generate"):
            with st.spinner("Generating melodic sequence..."):
                # Initialize the composer
                composer = BiLSTMComposer()
                
                # Generate the sequence
                sequence = composer.generate_sequence(
                    raga=selected_raga,
                    seed=seed,
                    length=sequence_length,
                    temperature=temperature
                )
                
                # Store the sequence in session state
                st.session_state.bilstm_sequence = sequence
                
                # Evaluate the sequence
                metrics = composer.evaluate_sequence(sequence, selected_raga)
                st.session_state.bilstm_metrics = metrics
                
                # Convert to audio for playback
                audio = symbolic_to_audio(sequence)
                
                # Save to a buffer for playback
                buffer = io.BytesIO()
                sf.write(buffer, audio, 22050, format='WAV')
                audio_bytes = buffer.getvalue()
                
                st.session_state.bilstm_audio = audio_bytes
        
        # Display results if available
        if hasattr(st.session_state, 'bilstm_sequence'):
            st.markdown("#### Generated Sequence")
            
            # Display the sequence
            st.code(st.session_state.bilstm_sequence)
            
            # Display audio player
            st.audio(st.session_state.bilstm_audio, format="audio/wav")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Audio",
                    data=st.session_state.bilstm_audio,
                    file_name=f"{selected_raga}_melody.wav",
                    mime="audio/wav"
                )
            
            with col2:
                st.download_button(
                    label="Download Notation",
                    data=st.session_state.bilstm_sequence,
                    file_name=f"{selected_raga}_melody.txt",
                    mime="text/plain"
                )
            
            # Display evaluation metrics
            st.markdown("#### Evaluation Metrics")
            
            metrics = st.session_state.bilstm_metrics
            
            # Create a radar chart for the metrics
            categories = ['Raga Adherence', 'Melodic Complexity', 'Phrase Coherence', 'Overall Quality']
            values = [
                metrics['raga_adherence'],
                metrics['melodic_complexity'],
                metrics['phrase_coherence'],
                metrics['overall_quality']
            ]
            
            # Create the radar chart
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Values for the chart
            values_for_chart = values + values[:1]
            
            # Draw the chart
            ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#800000")
            ax.fill(angles, values_for_chart, alpha=0.25, color="#800000")
            
            # Set the labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Set background color
            ax.set_facecolor("#FFF8DC")
            fig.patch.set_facecolor("#FFF8DC")
            
            # Show the chart
            st.pyplot(fig)
    
    with tab2:
        st.markdown("### Audio Composition with CNNGAN")
        
        st.markdown("""
        The CNNGAN model generates high-fidelity audio samples that authentically capture the style and essence of specific ragas.
        This advanced model excels at reproducing:

        - Timbral qualities of traditional Indian instruments
        - Micro-tonal nuances and ornamentations (gamak, meend, kan, etc.)
        - Characteristic resonance and acoustic properties
        - Natural performance dynamics and expressions

        The model has been trained on recordings by master musicians to ensure authentic sound quality and musical expression.
        """)
        
        # Raga selection
        selected_raga = st.selectbox("Select Raga:", raga_names, key="cnngan_raga")

        # Get raga info
        raga_info = get_raga_info(selected_raga)

        # Display raga information
        if raga_info:
            aroha_str = ' '.join(raga_info.get('aroha', []))
            avaroha_str = ' '.join(raga_info.get('avaroha', []))
            st.markdown(f"""
            **Raga**: {selected_raga}
            **Arohanam**: {aroha_str}
            **Avarohanam**: {avaroha_str}
            **Time**: {raga_info.get('time', 'N/A')}
            """)
        
        # Composition parameters
        duration = st.slider("Duration (seconds):", 5.0, 30.0, 10.0, 0.5, key="cnngan_duration")
        
        # Generate button
        if st.button("Generate Audio", key="cnngan_generate"):
            with st.spinner("Generating audio..."):
                # Initialize the composer
                composer = CNNGANComposer()
                
                # Generate the audio
                audio, sr = composer.generate_audio(
                    raga=selected_raga,
                    duration=duration
                )
                
                # Store the audio in session state
                st.session_state.cnngan_audio = audio
                st.session_state.cnngan_sr = sr
                
                # Evaluate the audio
                metrics = composer.evaluate_audio(audio, sr, selected_raga)
                st.session_state.cnngan_metrics = metrics
                
                # Save to a buffer for playback
                buffer = io.BytesIO()
                sf.write(buffer, audio, sr, format='WAV')
                audio_bytes = buffer.getvalue()
                
                st.session_state.cnngan_audio_bytes = audio_bytes
        
        # Display results if available
        if hasattr(st.session_state, 'cnngan_audio_bytes'):
            st.markdown("#### Generated Audio")
            
            # Display audio player
            st.audio(st.session_state.cnngan_audio_bytes, format="audio/wav")
            
            # Download button
            st.download_button(
                label="Download Audio",
                data=st.session_state.cnngan_audio_bytes,
                file_name=f"{selected_raga}_audio.wav",
                mime="audio/wav"
            )
            
            # Display evaluation metrics
            st.markdown("#### Evaluation Metrics")
            
            metrics = st.session_state.cnngan_metrics
            
            # Create a radar chart for the metrics
            categories = ['Authenticity', 'Timbral Quality', 'Spectral Richness', 'Overall Quality']
            values = [
                metrics['authenticity'],
                metrics['timbral_quality'],
                metrics['spectral_richness'],
                metrics['overall_quality']
            ]
            
            # Create the radar chart
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Values for the chart
            values_for_chart = values + values[:1]
            
            # Draw the chart
            ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#483D8B")
            ax.fill(angles, values_for_chart, alpha=0.25, color="#483D8B")
            
            # Set the labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Set background color
            ax.set_facecolor("#FFF8DC")
            fig.patch.set_facecolor("#FFF8DC")
            
            # Show the chart
            st.pyplot(fig)
            
            # Display waveform and spectrogram
            st.markdown("#### Audio Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### Waveform")
                
                # Create waveform plot
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.plot(np.linspace(0, duration, len(st.session_state.cnngan_audio)), 
                        st.session_state.cnngan_audio, color="#483D8B")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title("Waveform")
                ax.set_facecolor("#FFF8DC")
                fig.patch.set_facecolor("#FFF8DC")
                
                st.pyplot(fig)
            
            with col2:
                st.markdown("##### Spectrogram")
                
                # Create spectrogram plot
                fig, ax = plt.subplots(figsize=(6, 3))
                
                # Compute spectrogram
                import librosa
                D = librosa.amplitude_to_db(
                    np.abs(librosa.stft(st.session_state.cnngan_audio)),
                    ref=np.max
                )
                
                # Plot spectrogram
                librosa.display.specshow(
                    D, x_axis='time', y_axis='log', sr=st.session_state.cnngan_sr, ax=ax
                )
                
                ax.set_title("Spectrogram")
                ax.set_facecolor("#FFF8DC")
                fig.patch.set_facecolor("#FFF8DC")
                
                st.pyplot(fig)
    
    with tab3:
        st.markdown("### Hybrid Composition")
        
        st.markdown("""
        The Hybrid Composer represents our most advanced music generation system, combining the strengths of both Bi-LSTM and CNNGAN models
        to create complete, concert-quality compositions in the authentic style of Indian classical music.

        This sophisticated system:

        - Generates structurally correct melodic sequences using Bi-LSTM
        - Renders high-fidelity audio with authentic timbral qualities using CNNGAN
        - Incorporates traditional performance structure (alap, jor, jhala)
        - Adds appropriate ornamentations and expressive elements
        - Includes tanpura drone and acoustic ambience for a complete sound

        The result is a comprehensive composition that respects both the theoretical framework and the aesthetic qualities of the chosen raga.
        """)
        
        # Raga selection
        selected_raga = st.selectbox("Select Raga:", raga_names, key="hybrid_raga")

        # Get raga info
        raga_info = get_raga_info(selected_raga)

        # Display raga information
        if raga_info:
            aroha_str = ' '.join(raga_info.get('aroha', []))
            avaroha_str = ' '.join(raga_info.get('avaroha', []))
            st.markdown(f"""
            **Raga**: {selected_raga}
            **Arohanam**: {aroha_str}
            **Avarohanam**: {avaroha_str}
            **Time**: {raga_info.get('time', 'N/A')}
            """)
        
        # Composition parameters
        col1, col2 = st.columns(2)
        
        with col1:
            duration = st.slider("Duration (seconds):", 10.0, 60.0, 30.0, 1.0, key="hybrid_duration")
            
        with col2:
            seed = st.text_input("Seed Sequence (optional):", key="hybrid_seed", 
                                help="Enter a sequence of notes (S, R, G, M, P, D, N) to start the composition")
        
        # Generate button
        if st.button("Generate Composition", key="hybrid_generate"):
            with st.spinner("Generating composition..."):
                # Initialize the composer
                composer = HybridComposer()
                
                # Generate the composition
                audio, sr, sequence = composer.generate_composition(
                    raga=selected_raga,
                    duration=duration,
                    seed=seed
                )
                
                # Store the results in session state
                st.session_state.hybrid_audio = audio
                st.session_state.hybrid_sr = sr
                st.session_state.hybrid_sequence = sequence
                
                # Evaluate the composition
                metrics = composer.evaluate_composition(audio, sr, sequence, selected_raga)
                st.session_state.hybrid_metrics = metrics
                
                # Save to a buffer for playback
                buffer = io.BytesIO()
                sf.write(buffer, audio, sr, format='WAV')
                audio_bytes = buffer.getvalue()
                
                st.session_state.hybrid_audio_bytes = audio_bytes
        
        # Display results if available
        if hasattr(st.session_state, 'hybrid_audio_bytes'):
            st.markdown("#### Generated Composition")
            
            # Display the sequence
            st.markdown("##### Symbolic Notation")
            st.code(st.session_state.hybrid_sequence)
            
            # Display audio player
            st.markdown("##### Audio Rendering")
            st.audio(st.session_state.hybrid_audio_bytes, format="audio/wav")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Audio",
                    data=st.session_state.hybrid_audio_bytes,
                    file_name=f"{selected_raga}_composition.wav",
                    mime="audio/wav"
                )
            
            with col2:
                st.download_button(
                    label="Download Notation",
                    data=st.session_state.hybrid_sequence,
                    file_name=f"{selected_raga}_composition.txt",
                    mime="text/plain"
                )
            
            # Display evaluation metrics
            st.markdown("#### Evaluation Metrics")
            
            metrics = st.session_state.hybrid_metrics
            
            # Create a radar chart for the metrics
            categories = ['Melodic Structure', 'Audio Quality', 'Raga Adherence', 'Timbral Quality', 'Overall Quality']
            values = [
                metrics['melodic_structure'],
                metrics['audio_quality'],
                metrics['raga_adherence'],
                metrics['timbral_quality'],
                metrics['overall_quality']
            ]
            
            # Create the radar chart
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Number of variables
            N = len(categories)
            
            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            # Values for the chart
            values_for_chart = values + values[:1]
            
            # Draw the chart
            ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#8B4513")
            ax.fill(angles, values_for_chart, alpha=0.25, color="#8B4513")
            
            # Set the labels
            plt.xticks(angles[:-1], categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
            
            # Set background color
            ax.set_facecolor("#FFF8DC")
            fig.patch.set_facecolor("#FFF8DC")
            
            # Show the chart
            st.pyplot(fig)
    
    # Add a section for comparing models
    render_decorative_divider()
    
    st.markdown("### Model Comparison")
    
    st.markdown("""
    This section allows you to compare the performance of different composition models
    on the same raga. Generate compositions with each model and then compare their metrics.
    """)
    
    # Check if we have results from all models
    has_bilstm = hasattr(st.session_state, 'bilstm_metrics')
    has_cnngan = hasattr(st.session_state, 'cnngan_metrics')
    has_hybrid = hasattr(st.session_state, 'hybrid_metrics')
    
    if has_bilstm or has_cnngan or has_hybrid:
        # Create comparison data
        comparison_data = {
            'Metric': ['Overall Quality', 'Raga Adherence', 'Timbral Quality']
        }
        
        if has_bilstm:
            comparison_data['Bi-LSTM'] = [
                st.session_state.bilstm_metrics['overall_quality'],
                st.session_state.bilstm_metrics['raga_adherence'],
                0.0  # Bi-LSTM doesn't have timbral quality
            ]
        
        if has_cnngan:
            comparison_data['CNNGAN'] = [
                st.session_state.cnngan_metrics['overall_quality'],
                0.0,  # CNNGAN doesn't have raga adherence
                st.session_state.cnngan_metrics['timbral_quality']
            ]
        
        if has_hybrid:
            comparison_data['Hybrid'] = [
                st.session_state.hybrid_metrics['overall_quality'],
                st.session_state.hybrid_metrics['raga_adherence'],
                st.session_state.hybrid_metrics['timbral_quality']
            ]
        
        # Create a DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Display as a bar chart
        st.markdown("#### Model Performance Comparison")
        
        # Melt the DataFrame for easier plotting
        df_melted = pd.melt(df, id_vars=['Metric'], var_name='Model', value_name='Score')
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Define colors for each model
        colors = {
            'Bi-LSTM': '#800000',
            'CNNGAN': '#483D8B',
            'Hybrid': '#8B4513'
        }
        
        # Plot the bars
        for model in df_melted['Model'].unique():
            model_data = df_melted[df_melted['Model'] == model]
            # Calculate position offset for grouped bars
            offset = list(df_melted['Model'].unique()).index(model) - 1
            # Create positions for this model's bars (shifted by the offset)
            positions = [p + (offset * 0.25) for p in range(len(model_data['Metric']))]

            ax.bar(
                positions,
                model_data['Score'],
                label=model,
                color=colors.get(model, '#000000'),
                alpha=0.7,
                width=0.25
            )
        
        # Add labels and legend
        ax.set_xlabel('Metric')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.legend()
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Set background color
        ax.set_facecolor("#FFF8DC")
        fig.patch.set_facecolor("#FFF8DC")
        
        # Show the chart
        st.pyplot(fig)
        
        # Display the data as a table
        st.markdown("#### Detailed Metrics")
        st.dataframe(df)
    else:
        st.info("Generate compositions with at least one model to see the comparison.")

if __name__ == "__main__":
    render_music_composer_page()