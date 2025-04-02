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

# Set minimal page configuration
st.set_page_config(
    layout="wide",
    page_title="RagaVani Music Composer",
    page_icon="🎵"
)

# No decorative divider function needed

def render_music_composer_page():
    """Render the music composer page"""
    # Get all ragas
    all_ragas = get_all_ragas()
    raga_names = all_ragas  # all_ragas is already a list of raga names

    # Create a clean header with title and subtitle
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("RagaVani Music Composer")
        st.subheader("Create authentic Indian classical music with AI")

    # Add a simple separator
    st.markdown("---")

    # Simple model comparison section
    st.header("Our Composition Models")
    st.write("Each model is designed to capture different aspects of Indian classical music")

    # Create simple model cards
    model_cols = st.columns(3)

    with model_cols[0]:
        st.subheader("🎵 Bi-LSTM Model")
        st.write("Specializes in creating authentic melodic patterns that follow the grammar and structure of traditional ragas.")
        st.markdown("""
        - Authentic melodic patterns
        - Precise raga structure
        - Sophisticated phrase development
        """)

    with model_cols[1]:
        st.subheader("🔊 CNNGAN Model")
        st.write("Generates high-fidelity audio with authentic timbral qualities and micro-tonal nuances of Indian instruments.")
        st.markdown("""
        - High-fidelity audio synthesis
        - Authentic instrument timbres
        - Natural performance dynamics
        """)

    with model_cols[2]:
        st.subheader("🎹 Hybrid Model")
        st.write("Combines melodic structure with ornamentations to create complete, performance-ready compositions.")
        st.markdown("""
        - Complete melodic structure
        - Sophisticated ornamentation
        - Balanced composition development
        """)

    # Simple tabs for the different models
    tab_titles = ["🎵 Melodic Composer (Bi-LSTM)", "🔊 Audio Composer (CNNGAN)", "🎹 Hybrid Composer"]
    model_tabs = st.tabs(tab_titles)

    # Store the selected model in session state if not already set
    if 'selected_model_tab' not in st.session_state:
        st.session_state.selected_model_tab = 0

    # Get the active tab
    active_tab = st.session_state.selected_model_tab

    # Simple separator
    st.markdown("---")

    # Simple description
    st.info("Create authentic Indian classical music with our advanced AI composition models")

    # Simple model selection section
    st.header("Choose Your Composition Model")
    st.write("Select the model that best suits your composition needs")

    # Create simple model selection buttons
    model_cols = st.columns(3)

    with model_cols[0]:
        melodic_selected = st.button("🎵 Select Bi-LSTM", key="select_bilstm", use_container_width=True)

    with model_cols[1]:
        audio_selected = st.button("🔊 Select CNNGAN", key="select_cnngan", use_container_width=True)

    with model_cols[2]:
        hybrid_selected = st.button("🎹 Select Hybrid", key="select_hybrid", use_container_width=True)

    # Store the selected model in session state
    if 'selected_model_tab' not in st.session_state:
        st.session_state.selected_model_tab = 0

    if melodic_selected:
        st.session_state.selected_model_tab = 0
    elif audio_selected:
        st.session_state.selected_model_tab = 1
    elif hybrid_selected:
        st.session_state.selected_model_tab = 2

    # Elegant decorative divider
    st.markdown("""
    <div style="text-align: center; margin: 3.5rem 0;">
        <div style="height: 1px; background: linear-gradient(to right, transparent, #4A2545, transparent); max-width: 800px; margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    # Modern raga selection section
    st.markdown("""
    <div style="margin: 3rem 0 2rem 0;">
        <h2 style="text-align: center; color: #4A2545; font-size: 2rem; margin-bottom: 1rem; font-weight: 600;">
            Select a Raga for Your Composition
        </h2>
        <div style="width: 80px; height: 3px; background: linear-gradient(to right, transparent, #4A2545, transparent);
                    margin: 0 auto 1.5rem auto;"></div>
        <p style="text-align: center; max-width: 700px; margin: 0 auto 2rem auto; color: #555; font-weight: 300; font-size: 1.1rem;">
            Choose from our collection of traditional Indian ragas
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Modern search box
    search_col1, search_col2, search_col3 = st.columns([1, 2, 1])
    with search_col2:
        st.markdown("""
        <style>
        div[data-baseweb="input"] {
            border-radius: 30px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
        }
        div[data-baseweb="input"]:focus-within {
            border: 1px solid #4A2545;
            box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        div[data-baseweb="input"] input {
            padding-left: 15px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        raga_search = st.text_input("", key="raga_search",
                                  placeholder="🔍 Search for a raga...",
                                  label_visibility="collapsed")

    # Filter ragas based on search
    filtered_ragas = [raga for raga in raga_names if raga_search.lower() in raga.lower()] if raga_search else raga_names

    # Create a modern container for raga selection
    st.markdown("""
    <div style="border-radius: 12px; padding: 2rem; margin: 2rem 0;
                background: linear-gradient(145deg, #ffffff, #f5f5f5); box-shadow: 0 6px 15px rgba(0,0,0,0.05);">
        <h3 style="color: #4A2545; text-align: center; margin-bottom: 2rem; font-size: 1.5rem; font-weight: 600;">
            Popular Ragas
        </h3>
    """, unsafe_allow_html=True)

    # Display popular ragas with clean buttons
    popular_ragas = ["Yaman", "Bhairav", "Darbari", "Malkauns", "Bhimpalasi", "Todi", "Bhairavi", "Kafi"]

    # Create a grid for popular ragas
    popular_cols = st.columns(4)
    for i, raga in enumerate(popular_ragas):
        col_index = i % 4
        with popular_cols[col_index]:
            # Modern styling for popular raga buttons
            st.markdown(f"""
            <style>
            div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child({i+1}) button {{
                background-color: white;
                color: #4A2545;
                border: 2px solid #4A2545;
                border-radius: 8px;
                padding: 10px 5px;
                margin-bottom: 15px;
                font-weight: 500;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }}
            div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child({i+1}) button:hover {{
                background-color: #4A2545;
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(0,0,0,0.1);
            }}
            </style>
            """, unsafe_allow_html=True)

            if st.button(f"{raga}", key=f"popular_{raga}", use_container_width=True):
                # Set the raga in all selectboxes
                st.session_state.bilstm_raga = raga
                st.session_state.cnngan_raga = raga
                st.session_state.hybrid_raga = raga

    # Modern "More Ragas" section
    if filtered_ragas and len(filtered_ragas) > len(popular_ragas):
        st.markdown("""
        <h3 style="color: #4A2545; text-align: center; margin: 3rem 0 2rem 0; font-size: 1.5rem; font-weight: 600;">
            More Ragas
        </h3>
        """, unsafe_allow_html=True)

        # Create a grid for more ragas
        more_cols = st.columns(4)
        raga_count = 0

        for raga in filtered_ragas[:20]:  # Limit to 20 for performance
            if raga not in popular_ragas:
                col_index = raga_count % 4
                with more_cols[col_index]:
                    # Modern styling for more raga buttons
                    st.markdown(f"""
                    <style>
                    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child({raga_count+len(popular_ragas)+1}) button {{
                        background-color: white;
                        color: #555;
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 8px 5px;
                        margin-bottom: 10px;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
                    }}
                    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] > div:nth-child({raga_count+len(popular_ragas)+1}) button:hover {{
                        background-color: #f8f8f8;
                        border: 1px solid #4A2545;
                        color: #4A2545;
                        transform: translateY(-2px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
                    }}
                    </style>
                    """, unsafe_allow_html=True)

                    if st.button(f"{raga}", key=f"more_{raga}", use_container_width=True):
                        # Set the raga in all selectboxes
                        st.session_state.bilstm_raga = raga
                        st.session_state.cnngan_raga = raga
                        st.session_state.hybrid_raga = raga
                raga_count += 1

        st.markdown("</div>", unsafe_allow_html=True)

    # Close the container
    st.markdown("</div>", unsafe_allow_html=True)

    # Decorative divider
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <div style="height: 1px; background: linear-gradient(to right, transparent, #DAA520, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Simple decorative divider
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <div style="height: 1px; background: linear-gradient(to right, transparent, #4A2545, transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Add a beautiful header to the main content area
   

    # We're using the model_tabs defined earlier
    tabs = model_tabs

    # Get the active tab
    active_tab = st.session_state.selected_model_tab

    # Tab 1: Melodic Composer (Bi-LSTM)
    with tabs[0]:
        if active_tab == 0:  # Only show content if this tab is active
            st.markdown("### Melodic Composition with Bi-LSTM")

            # Raga selection section
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("""
                <div style="background-color: rgba(128, 0, 0, 0.05); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                    <h4 style="color: #800000; margin-top: 0; margin-bottom: 10px;">Select Raga</h4>
                </div>
                """, unsafe_allow_html=True)

                # Search box for ragas
                raga_search = st.text_input("🔍 Search", key="bilstm_raga_search",
                                          placeholder="Type to search...")

                # Filter ragas based on search
                filtered_ragas = [raga for raga in raga_names if raga_search.lower() in raga.lower()] if raga_search else raga_names

                # Display filtered ragas in a selectbox
                selected_raga = st.selectbox("Select a raga", filtered_ragas, key="bilstm_raga")

            with col2:
                # Create a horizontal list of popular ragas as buttons
                popular_ragas = ["Yaman", "Bhairav", "Bhairavi", "Darbari", "Malkauns", "Todi"]
                st.markdown("<p style='margin-bottom: 5px; font-size: 0.9rem;'>Popular Ragas:</p>", unsafe_allow_html=True)
                popular_cols = st.columns(len(popular_ragas))

                for i, raga in enumerate(popular_ragas):
                    with popular_cols[i]:
                        if st.button(raga, key=f"popular_bilstm_{raga}", use_container_width=True):
                            st.session_state.bilstm_raga = raga

            st.markdown("""
            <div style="background-color: rgba(128, 0, 0, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                The Bi-LSTM model generates sophisticated melodic sequences that authentically follow the grammar and structure of specific ragas.
                It excels at creating coherent musical phrases while respecting the rules of the chosen raga, including:
                <ul>
                    <li>Proper note progressions (arohanam/avarohanam)</li>
                    <li>Characteristic phrases (pakad)</li>
                    <li>Emphasis on important notes (vadi/samvadi)</li>
                    <li>Natural musical phrasing and development</li>
                </ul>
                This model has been trained on thousands of traditional compositions to capture the essence of each raga.
            </div>
            """, unsafe_allow_html=True)

            # Raga selection with enhanced styling
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h4 style="color: #800000; font-family: 'Palatino Linotype', serif;">
                    Select a Raga for Composition
                </h4>
            </div>
            """, unsafe_allow_html=True)

            selected_raga = st.selectbox("", raga_names, key="bilstm_raga_main")

            # Get raga info
            raga_info = get_raga_info(selected_raga)

            # Display raga information with enhanced styling
            if raga_info:
                aroha_str = ' '.join(raga_info.get('aroha', []))
                avaroha_str = ' '.join(raga_info.get('avaroha', []))

                st.markdown("""
                <div style="background-color: rgba(218, 165, 32, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <div style="display: flex; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <p><strong style="color: #8B4513;">Raga:</strong> {}</p>
                            <p><strong style="color: #8B4513;">Time:</strong> {}</p>
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <p><strong style="color: #8B4513;">Arohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                            <p><strong style="color: #8B4513;">Avarohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                        </div>
                    </div>
                </div>
                """.format(
                    selected_raga,
                    raga_info.get('time', 'N/A'),
                    aroha_str,
                    avaroha_str
                ), unsafe_allow_html=True)

            # Composition parameters with enhanced styling
            st.markdown("""
            <div style="margin: 20px 0 15px 0;">
                <h4 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">
                    Composition Parameters
                </h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                sequence_length = st.slider("Sequence Length:", 32, 256, 64, key="bilstm_length",
                                          help="Controls the length of the generated sequence. Longer sequences create more developed compositions.")

            with col2:
                temperature = st.slider("Temperature:", 0.1, 2.0, 1.0, 0.1, key="bilstm_temp",
                                      help="Controls randomness in generation. Higher values (>1) create more creative but potentially less authentic compositions.")
        
            # Seed input with enhanced styling
            st.markdown("""
            <div style="margin: 20px 0 10px 0;">
                <h4 style="color: #800000; font-family: 'Palatino Linotype', serif;">
                    Starting Sequence (Optional)
                </h4>
                <p style="font-size: 0.9rem; color: #4B3621;">
                    Enter a sequence of notes (S, R, G, M, P, D, N) to guide the composition
                </p>
            </div>
            """, unsafe_allow_html=True)

            seed = st.text_input("", key="bilstm_seed", placeholder="Example: S R G M P D N S")

            # Generate button with enhanced styling
            st.markdown("<div style='margin: 25px 0 15px 0;'></div>", unsafe_allow_html=True)

            generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
            with generate_col2:
                if st.button("✨ Generate Melodic Sequence", key="bilstm_generate", use_container_width=True):
                    with st.spinner("Creating your melodic sequence..."):
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
                # Add a decorative divider
                st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="height: 1px; background: linear-gradient(to right, transparent, #800000, transparent);"></div>
                    <div style="font-size: 24px; color: #800000; margin: -14px 0;">♪</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">
                        Your Generated Composition
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # Create tabs for different views of the result
                result_tabs = st.tabs(["Audio Player", "Notation", "Analysis"])

                with result_tabs[0]:
                    st.markdown("""
                    <div style="background-color: rgba(72, 61, 139, 0.05); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                        <h4 style="color: #483D8B; margin-top: 0;">Listen to Your Composition</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            This audio rendering uses synthesized instruments to play the generated melodic sequence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display audio player with some styling
                    st.audio(st.session_state.bilstm_audio, format="audio/wav")

                    # Download button with better styling
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="💾 Download Audio",
                            data=st.session_state.bilstm_audio,
                            file_name=f"{selected_raga}_melody.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )

                with result_tabs[1]:
                    st.markdown("""
                    <div style="background-color: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #8B4513; margin-top: 0;">Symbolic Notation</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            The sequence uses Indian classical notation: S (Sa), R (Re), G (Ga), M (Ma), P (Pa), D (Dha), N (Ni)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the sequence in a code block with custom styling
                    st.code(st.session_state.bilstm_sequence)

                    # Download button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📄 Download Notation",
                            data=st.session_state.bilstm_sequence,
                            file_name=f"{selected_raga}_melody.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                with result_tabs[2]:
                    st.markdown("""
                    <div style="background-color: rgba(128, 0, 0, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #800000; margin-top: 0;">Composition Analysis</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            Our AI evaluates the quality and authenticity of the generated composition
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display evaluation metrics
                    metrics = st.session_state.bilstm_metrics

                    # Create a radar chart for the metrics
                    # Print the available keys for debugging
                    st.write("Available metrics keys:", list(metrics.keys()))

                    # Use the correct keys based on what's available in the metrics dictionary
                    categories = ['Raga Adherence', 'Melodic Authenticity', 'Structural Coherence', 'Overall Quality']
                    values = [
                        metrics.get('raga_adherence', 0.0),
                        metrics.get('melodic_authenticity', 0.0),
                        metrics.get('structural_coherence', 0.0),
                        metrics.get('overall_quality', 0.0)
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
    
    # Tab 2: Audio Composer (CNNGAN)
    with tabs[1]:
        if active_tab == 1:  # Only show content if this tab is active
            st.markdown("### Audio Composition with CNNGAN")

            # Raga selection section
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("""
                <div style="background-color: rgba(72, 61, 139, 0.05); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                    <h4 style="color: #483D8B; margin-top: 0; margin-bottom: 10px;">Select Raga</h4>
                </div>
                """, unsafe_allow_html=True)

                # Search box for ragas
                raga_search = st.text_input("🔍 Search", key="cnngan_raga_search",
                                          placeholder="Type to search...")

                # Filter ragas based on search
                filtered_ragas = [raga for raga in raga_names if raga_search.lower() in raga.lower()] if raga_search else raga_names

                # Display filtered ragas in a selectbox
                selected_raga = st.selectbox("Select a raga", filtered_ragas, key="cnngan_raga")

            with col2:
                # Create a horizontal list of popular ragas as buttons
                popular_ragas = ["Yaman", "Bhairav", "Bhairavi", "Darbari", "Malkauns", "Todi"]
                st.markdown("<p style='margin-bottom: 5px; font-size: 0.9rem;'>Popular Ragas:</p>", unsafe_allow_html=True)
                popular_cols = st.columns(len(popular_ragas))

                for i, raga in enumerate(popular_ragas):
                    with popular_cols[i]:
                        if st.button(raga, key=f"popular_cnngan_{raga}", use_container_width=True):
                            st.session_state.cnngan_raga = raga

            st.markdown("""
            <div style="background-color: rgba(72, 61, 139, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                The CNNGAN model generates high-fidelity audio samples that authentically capture the style and essence of specific ragas.
                This advanced model excels at reproducing:
                <ul>
                    <li>Timbral qualities of traditional Indian instruments</li>
                    <li>Micro-tonal nuances and ornamentations (gamak, meend, kan, etc.)</li>
                    <li>Characteristic resonance and acoustic properties</li>
                    <li>Natural performance dynamics and expressions</li>
                </ul>
                The model has been trained on recordings by master musicians to ensure authentic sound quality and musical expression.
            </div>
            """, unsafe_allow_html=True)

            # Raga selection with enhanced styling
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h4 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">
                    Select a Raga for Audio Generation
                </h4>
            </div>
            """, unsafe_allow_html=True)

            selected_raga = st.selectbox("", raga_names, key="cnngan_raga_main")

            # Get raga info
            raga_info = get_raga_info(selected_raga)

            # Display raga information with enhanced styling
            if raga_info:
                aroha_str = ' '.join(raga_info.get('aroha', []))
                avaroha_str = ' '.join(raga_info.get('avaroha', []))

                st.markdown("""
                <div style="background-color: rgba(72, 61, 139, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <div style="display: flex; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <p><strong style="color: #483D8B;">Raga:</strong> {}</p>
                            <p><strong style="color: #483D8B;">Time:</strong> {}</p>
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <p><strong style="color: #483D8B;">Arohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                            <p><strong style="color: #483D8B;">Avarohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                        </div>
                    </div>
                </div>
                """.format(
                    selected_raga,
                    raga_info.get('time', 'N/A'),
                    aroha_str,
                    avaroha_str
                ), unsafe_allow_html=True)

            # Composition parameters with enhanced styling
            st.markdown("""
            <div style="margin: 20px 0 15px 0;">
                <h4 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">
                    Audio Parameters
                </h4>
            </div>
            """, unsafe_allow_html=True)

            # Duration slider with better styling
            duration = st.slider("Duration (seconds):", 5.0, 30.0, 10.0, 0.5, key="cnngan_duration",
                               help="Controls the length of the generated audio. Longer durations allow for more musical development.")
        
            # Generate button with enhanced styling
            st.markdown("<div style='margin: 25px 0 15px 0;'></div>", unsafe_allow_html=True)

            generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
            with generate_col2:
                if st.button("🔊 Generate Audio", key="cnngan_generate", use_container_width=True):
                    with st.spinner("Creating your audio composition..."):
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
                # Add a decorative divider
                st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="height: 1px; background: linear-gradient(to right, transparent, #483D8B, transparent);"></div>
                    <div style="font-size: 24px; color: #483D8B; margin: -14px 0;">♪</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">
                        Your Generated Audio
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # Create tabs for different views of the result
                result_tabs = st.tabs(["Audio Player", "Visualizations", "Analysis"])

                with result_tabs[0]:
                    st.markdown("""
                    <div style="background-color: rgba(72, 61, 139, 0.05); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                        <h4 style="color: #483D8B; margin-top: 0;">Listen to Your Composition</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            This audio has been generated to capture the authentic timbral qualities of the selected raga
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display audio player with some styling
                    st.audio(st.session_state.cnngan_audio_bytes, format="audio/wav")

                    # Download button with better styling
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="💾 Download Audio",
                            data=st.session_state.cnngan_audio_bytes,
                            file_name=f"{selected_raga}_audio.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
            
                with result_tabs[1]:
                    st.markdown("""
                    <div style="background-color: rgba(72, 61, 139, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #483D8B; margin-top: 0;">Audio Visualizations</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            Visual representations of your audio composition's waveform and frequency content
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <h5 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">Waveform</h5>
                        </div>
                        """, unsafe_allow_html=True)

                        # Create waveform plot with enhanced styling
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(np.linspace(0, duration, len(st.session_state.cnngan_audio)),
                                st.session_state.cnngan_audio, color="#483D8B")
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        ax.grid(alpha=0.2)
                        ax.set_facecolor("#FFF8DC")
                        fig.patch.set_facecolor("#FFF8DC")

                        st.pyplot(fig)

                    with col2:
                        st.markdown("""
                        <div style="text-align: center; margin-bottom: 10px;">
                            <h5 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">Spectrogram</h5>
                        </div>
                        """, unsafe_allow_html=True)

                        # Create spectrogram plot with enhanced styling
                        fig, ax = plt.subplots(figsize=(6, 3))

                        # Compute spectrogram
                        import librosa
                        D = librosa.amplitude_to_db(
                            np.abs(librosa.stft(st.session_state.cnngan_audio)),
                            ref=np.max
                        )

                        # Plot spectrogram with better color map
                        librosa.display.specshow(
                            D, x_axis='time', y_axis='log', sr=st.session_state.cnngan_sr,
                            ax=ax, cmap='viridis'
                        )

                        ax.set_facecolor("#FFF8DC")
                        fig.patch.set_facecolor("#FFF8DC")

                        st.pyplot(fig)

                with result_tabs[2]:
                    st.markdown("""
                    <div style="background-color: rgba(72, 61, 139, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #483D8B; margin-top: 0;">Audio Analysis</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            Our AI evaluates the quality and authenticity of the generated audio
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display evaluation metrics with enhanced styling
                    metrics = st.session_state.cnngan_metrics

                    # Create metric cards
                    metric_cols = st.columns(4)

                    # Print the available keys for debugging
                    st.write("Available CNNGAN metrics keys:", list(metrics.keys()))

                    metric_data = [
                        {"name": "Authenticity", "value": metrics.get('authenticity', 0.0), "color": "#483D8B"},
                        {"name": "Timbral Quality", "value": metrics.get('timbral_quality', 0.0), "color": "#800000"},
                        {"name": "Spectral Richness", "value": metrics.get('spectral_richness', 0.0), "color": "#8B4513"},
                        {"name": "Overall Quality", "value": metrics.get('overall_quality', 0.0), "color": "#006400"}
                    ]

                    for i, col in enumerate(metric_cols):
                        with col:
                            metric = metric_data[i]
                            st.markdown(f"""
                            <div style="background-color: rgba({','.join(str(int(c)) for c in tuple(int(metric['color'][1:3], 16), int(metric['color'][3:5], 16), int(metric['color'][5:7], 16)))}, 0.1);
                                        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                                <h2 style="color: {metric['color']}; margin: 0; font-size: 2rem;">{metric['value']:.2f}</h2>
                                <p style="margin: 5px 0 0 0; font-size: 0.8rem; color: #4B3621;">{metric['name']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Create a radar chart for the metrics
                    st.markdown("""
                    <div style="text-align: center; margin: 20px 0 10px 0;">
                        <h5 style="color: #483D8B; font-family: 'Palatino Linotype', serif;">Quality Assessment</h5>
                    </div>
                    """, unsafe_allow_html=True)

                    categories = ['Authenticity', 'Timbral Quality', 'Spectral Richness', 'Overall Quality']
                    values = [
                        metrics.get('authenticity', 0.0),
                        metrics.get('timbral_quality', 0.0),
                        metrics.get('spectral_richness', 0.0),
                        metrics.get('overall_quality', 0.0)
                    ]

                    # Create the radar chart with enhanced styling
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111, polar=True)

                    # Number of variables
                    N = len(categories)

                    # What will be the angle of each axis in the plot
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]

                    # Values for the chart
                    values_for_chart = values + values[:1]

                    # Draw the chart with gradient fill
                    ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#483D8B")
                    ax.fill(angles, values_for_chart, alpha=0.25, color="#483D8B")

                    # Add grid lines with better styling
                    ax.grid(color='gray', alpha=0.2)

                    # Set the labels with better styling
                    plt.xticks(angles[:-1], categories, size=10, color='#4B3621')

                    # Set y-axis limits
                    ax.set_ylim(0, 1)

                    # Set background color
                    ax.set_facecolor("#FFF8DC")
                    fig.patch.set_facecolor("#FFF8DC")

                    # Show the chart
                    st.pyplot(fig)
    
    # Tab 3: Hybrid Composer
    with tabs[2]:
        if active_tab == 2:  # Only show content if this tab is active
            st.markdown("### Hybrid Composition")

            # Raga selection section
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("""
                <div style="background-color: rgba(139, 69, 19, 0.05); padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                    <h4 style="color: #8B4513; margin-top: 0; margin-bottom: 10px;">Select Raga</h4>
                </div>
                """, unsafe_allow_html=True)

                # Search box for ragas
                raga_search = st.text_input("🔍 Search", key="hybrid_raga_search",
                                          placeholder="Type to search...")

                # Filter ragas based on search
                filtered_ragas = [raga for raga in raga_names if raga_search.lower() in raga.lower()] if raga_search else raga_names

                # Display filtered ragas in a selectbox
                selected_raga = st.selectbox("Select a raga", filtered_ragas, key="hybrid_raga")

            with col2:
                # Create a horizontal list of popular ragas as buttons
                popular_ragas = ["Yaman", "Bhairav", "Bhairavi", "Darbari", "Malkauns", "Todi"]
                st.markdown("<p style='margin-bottom: 5px; font-size: 0.9rem;'>Popular Ragas:</p>", unsafe_allow_html=True)
                popular_cols = st.columns(len(popular_ragas))

                for i, raga in enumerate(popular_ragas):
                    with popular_cols[i]:
                        if st.button(raga, key=f"popular_hybrid_{raga}", use_container_width=True):
                            st.session_state.hybrid_raga = raga

            st.markdown("""
            <div style="background-color: rgba(139, 69, 19, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                The Hybrid Composer represents our most advanced music generation system, combining the strengths of both Bi-LSTM and CNNGAN models
                to create complete, concert-quality compositions in the authentic style of Indian classical music.
                <br><br>
                This sophisticated system:
                <ul>
                    <li>Generates structurally correct melodic sequences using Bi-LSTM</li>
                    <li>Renders high-fidelity audio with authentic timbral qualities using CNNGAN</li>
                    <li>Incorporates traditional performance structure (alap, jor, jhala)</li>
                    <li>Adds appropriate ornamentations and expressive elements</li>
                    <li>Includes tanpura drone and acoustic ambience for a complete sound</li>
                </ul>
                The result is a comprehensive composition that respects both the theoretical framework and the aesthetic qualities of the chosen raga.
            </div>
            """, unsafe_allow_html=True)

            # Raga selection with enhanced styling
            st.markdown("""
            <div style="margin-bottom: 15px;">
                <h4 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">
                    Select a Raga for Complete Composition
                </h4>
            </div>
            """, unsafe_allow_html=True)

            selected_raga = st.selectbox("", raga_names, key="hybrid_raga_main")

            # Get raga info
            raga_info = get_raga_info(selected_raga)

            # Display raga information with enhanced styling
            if raga_info:
                aroha_str = ' '.join(raga_info.get('aroha', []))
                avaroha_str = ' '.join(raga_info.get('avaroha', []))

                st.markdown("""
                <div style="background-color: rgba(139, 69, 19, 0.1); padding: 15px; border-radius: 10px; margin: 15px 0;">
                    <div style="display: flex; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <p><strong style="color: #8B4513;">Raga:</strong> {}</p>
                            <p><strong style="color: #8B4513;">Time:</strong> {}</p>
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <p><strong style="color: #8B4513;">Arohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                            <p><strong style="color: #8B4513;">Avarohanam:</strong> <span style="font-family: monospace;">{}</span></p>
                        </div>
                    </div>
                </div>
                """.format(
                    selected_raga,
                    raga_info.get('time', 'N/A'),
                    aroha_str,
                    avaroha_str
                ), unsafe_allow_html=True)

            # Composition parameters with enhanced styling
            st.markdown("""
            <div style="margin: 20px 0 15px 0;">
                <h4 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">
                    Composition Parameters
                </h4>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                duration = st.slider("Duration (seconds):", 10.0, 60.0, 30.0, 1.0, key="hybrid_duration",
                                   help="Controls the length of the composition. Longer durations allow for more complete musical development.")

            with col2:
                seed = st.text_input("Seed Sequence (optional):", key="hybrid_seed",
                                    placeholder="Example: S R G M P D N S",
                                    help="Enter a sequence of notes (S, R, G, M, P, D, N) to guide the composition")

            # Advanced parameters with enhanced styling
            st.markdown("""
            <div style="margin: 20px 0 10px 0;">
                <h4 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">
                    Advanced Parameters
                </h4>
                <p style="font-size: 0.9rem; color: #4B3621;">
                    Fine-tune your composition with these advanced controls
                </p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                melodic_complexity = st.slider("Melodic Complexity:", 0.1, 1.0, 0.7, 0.1, key="hybrid_complexity",
                                             help="Controls the complexity of melodic patterns. Higher values create more intricate melodic structures.")

            with col2:
                ornamentation_level = st.slider("Ornamentation Level:", 0.1, 1.0, 0.6, 0.1, key="hybrid_ornamentation",
                                              help="Controls the amount of ornamentations like meend, gamak, kan, etc. Higher values add more expressive elements.")

            # Generate button with enhanced styling
            st.markdown("<div style='margin: 25px 0 15px 0;'></div>", unsafe_allow_html=True)

            generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
            with generate_col2:
                if st.button("🎹 Generate Complete Composition", key="hybrid_generate", use_container_width=True):
                    with st.spinner("Creating your complete composition..."):
                        # Initialize the composer
                        composer = HybridComposer()

                        # Generate the composition
                        audio, sr, sequence = composer.generate_composition(
                            raga=selected_raga,
                            duration=duration,
                            seed=seed,
                            melodic_complexity=melodic_complexity,
                            ornamentation_level=ornamentation_level
                        )

                        # Store the results in session state
                        st.session_state.hybrid_audio = audio
                        st.session_state.hybrid_sr = sr
                        st.session_state.hybrid_sequence = sequence

                        # Evaluate the composition
                        metrics = composer.evaluate_composition(audio, sr, sequence, selected_raga,
                                                              melodic_complexity=melodic_complexity,
                                                              ornamentation_level=ornamentation_level)
                        st.session_state.hybrid_metrics = metrics

                        # Save to a buffer for playback
                        buffer = io.BytesIO()
                        sf.write(buffer, audio, sr, format='WAV')
                        audio_bytes = buffer.getvalue()

                        st.session_state.hybrid_audio_bytes = audio_bytes
        
            # Display results if available
            if hasattr(st.session_state, 'hybrid_audio_bytes'):
                # Add a decorative divider
                st.markdown("""
                <div style="text-align: center; margin: 30px 0;">
                    <div style="height: 1px; background: linear-gradient(to right, transparent, #8B4513, transparent);"></div>
                    <div style="font-size: 24px; color: #8B4513; margin: -14px 0;">♪</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">
                        Your Complete Composition
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                # Create tabs for different views of the result
                result_tabs = st.tabs(["Audio Player", "Notation", "Structure", "Analysis"])

                with result_tabs[0]:
                    st.markdown("""
                    <div style="background-color: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 15px;">
                        <h4 style="color: #8B4513; margin-top: 0;">Listen to Your Composition</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            This complete composition includes traditional structure, ornamentations, and tanpura accompaniment
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display audio player with some styling
                    st.audio(st.session_state.hybrid_audio_bytes, format="audio/wav")

                    # Download button with better styling
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="💾 Download Audio",
                            data=st.session_state.hybrid_audio_bytes,
                            file_name=f"{selected_raga}_composition.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )

                with result_tabs[1]:
                    st.markdown("""
                    <div style="background-color: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #8B4513; margin-top: 0;">Symbolic Notation</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            The sequence uses Indian classical notation: S (Sa), R (Re), G (Ga), M (Ma), P (Pa), D (Dha), N (Ni)
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the sequence in a code block with custom styling
                    st.code(st.session_state.hybrid_sequence)

                    # Download button
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📄 Download Notation",
                            data=st.session_state.hybrid_sequence,
                            file_name=f"{selected_raga}_composition.txt",
                            mime="text/plain",
                            use_container_width=True
                        )

                with result_tabs[2]:
                    st.markdown("""
                    <div style="background-color: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #8B4513; margin-top: 0;">Composition Structure</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            The composition follows the traditional structure of Indian classical music
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display the structure information
                    st.markdown("""
                    <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-top: 20px;">
                        <div style="flex: 1; min-width: 200px; background-color: rgba(139, 69, 19, 0.05); padding: 15px; border-radius: 10px; border-left: 3px solid #8B4513;">
                            <h5 style="color: #8B4513; margin-top: 0;">Alap (30%)</h5>
                            <p style="font-size: 0.9rem; color: #4B3621; margin-bottom: 0;">
                                Slow, rhythmless introduction that explores the raga's characteristics
                            </p>
                        </div>
                        <div style="flex: 1; min-width: 200px; background-color: rgba(139, 69, 19, 0.05); padding: 15px; border-radius: 10px; border-left: 3px solid #8B4513;">
                            <h5 style="color: #8B4513; margin-top: 0;">Jor (30%)</h5>
                            <p style="font-size: 0.9rem; color: #4B3621; margin-bottom: 0;">
                                Medium-tempo section with rhythmic development
                            </p>
                        </div>
                        <div style="flex: 1; min-width: 200px; background-color: rgba(139, 69, 19, 0.05); padding: 15px; border-radius: 10px; border-left: 3px solid #8B4513;">
                            <h5 style="color: #8B4513; margin-top: 0;">Jhala (40%)</h5>
                            <p style="font-size: 0.9rem; color: #4B3621; margin-bottom: 0;">
                                Fast, rhythmic climactic section
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display ornamentations information
                    st.markdown("""
                    <div style="margin: 25px 0 15px 0;">
                        <h5 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">
                            Ornamentations Used
                        </h5>
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;">
                        <div style="flex: 1; min-width: 150px; background-color: rgba(139, 69, 19, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                            <p style="font-weight: bold; color: #8B4513; margin: 0;">Meend</p>
                            <p style="font-size: 0.8rem; color: #4B3621; margin: 5px 0 0 0;">Gliding between notes</p>
                        </div>
                        <div style="flex: 1; min-width: 150px; background-color: rgba(139, 69, 19, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                            <p style="font-weight: bold; color: #8B4513; margin: 0;">Gamak</p>
                            <p style="font-size: 0.8rem; color: #4B3621; margin: 5px 0 0 0;">Heavy oscillations</p>
                        </div>
                        <div style="flex: 1; min-width: 150px; background-color: rgba(139, 69, 19, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                            <p style="font-weight: bold; color: #8B4513; margin: 0;">Kan</p>
                            <p style="font-size: 0.8rem; color: #4B3621; margin: 5px 0 0 0;">Touch of adjacent note</p>
                        </div>
                        <div style="flex: 1; min-width: 150px; background-color: rgba(139, 69, 19, 0.05); padding: 10px; border-radius: 8px; text-align: center;">
                            <p style="font-weight: bold; color: #8B4513; margin: 0;">Andolan</p>
                            <p style="font-size: 0.8rem; color: #4B3621; margin: 5px 0 0 0;">Slow oscillation</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
                with result_tabs[3]:
                    st.markdown("""
                    <div style="background-color: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                        <h4 style="color: #8B4513; margin-top: 0;">Composition Analysis</h4>
                        <p style="font-size: 0.9rem; color: #4B3621;">
                            Our AI evaluates the quality and authenticity of the generated composition
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display evaluation metrics with enhanced styling
                    metrics = st.session_state.hybrid_metrics

                    # Create metric cards
                    metric_cols = st.columns(5)

                    # Print the available keys for debugging
                    st.write("Available Hybrid metrics keys:", list(metrics.keys()))

                    metric_data = [
                        {"name": "Melodic Authenticity", "value": metrics.get('melodic_authenticity', 0.0), "color": "#800000"},
                        {"name": "Structural Coherence", "value": metrics.get('structural_coherence', 0.0), "color": "#483D8B"},
                        {"name": "Raga Adherence", "value": metrics.get('raga_adherence', 0.0), "color": "#8B4513"},
                        {"name": "Timbral Quality", "value": metrics.get('timbral_quality', 0.0), "color": "#006400"},
                        {"name": "Overall Quality", "value": metrics.get('overall_quality', 0.0), "color": "#DAA520"}
                    ]

                    for i, col in enumerate(metric_cols):
                        with col:
                            metric = metric_data[i]
                            st.markdown(f"""
                            <div style="background-color: rgba({','.join(str(int(c)) for c in tuple(int(metric['color'][1:3], 16), int(metric['color'][3:5], 16), int(metric['color'][5:7], 16)))}, 0.1);
                                        padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
                                <h2 style="color: {metric['color']}; margin: 0; font-size: 1.8rem;">{metric['value']:.2f}</h2>
                                <p style="margin: 5px 0 0 0; font-size: 0.7rem; color: #4B3621;">{metric['name']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                    # Create a radar chart for the metrics
                    st.markdown("""
                    <div style="text-align: center; margin: 20px 0 10px 0;">
                        <h5 style="color: #8B4513; font-family: 'Palatino Linotype', serif;">Quality Assessment</h5>
                    </div>
                    """, unsafe_allow_html=True)

                    categories = ['Melodic Authenticity', 'Structural Coherence', 'Raga Adherence', 'Timbral Quality', 'Overall Quality']
                    values = [
                        metrics.get('melodic_authenticity', 0.0),
                        metrics.get('structural_coherence', 0.0),
                        metrics.get('raga_adherence', 0.0),
                        metrics.get('timbral_quality', 0.0),
                        metrics.get('overall_quality', 0.0)
                    ]

                    # Create the radar chart with enhanced styling
                    fig = plt.figure(figsize=(6, 6))
                    ax = fig.add_subplot(111, polar=True)

                    # Number of variables
                    N = len(categories)

                    # What will be the angle of each axis in the plot
                    angles = [n / float(N) * 2 * np.pi for n in range(N)]
                    angles += angles[:1]

                    # Values for the chart
                    values_for_chart = values + values[:1]

                    # Draw the chart with gradient fill
                    ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#8B4513")
                    ax.fill(angles, values_for_chart, alpha=0.25, color="#8B4513")

                    # Add grid lines with better styling
                    ax.grid(color='gray', alpha=0.2)

                    # Set the labels with better styling
                    plt.xticks(angles[:-1], categories, size=9, color='#4B3621')

                    # Set y-axis limits
                    ax.set_ylim(0, 1)

                    # Set background color
                    ax.set_facecolor("#FFF8DC")
                    fig.patch.set_facecolor("#FFF8DC")

                    # Show the chart
                    st.pyplot(fig)
    
    # Add a section for comparing models if any compositions have been generated
    has_bilstm = hasattr(st.session_state, 'bilstm_metrics')
    has_cnngan = hasattr(st.session_state, 'cnngan_metrics')
    has_hybrid = hasattr(st.session_state, 'hybrid_metrics')

    if has_bilstm or has_cnngan or has_hybrid:
        # Add a decorative divider
        st.markdown("""
        <div style="text-align: center; margin: 40px 0 20px 0;">
            <div style="height: 1px; background: linear-gradient(to right, transparent, #DAA520, transparent);"></div>
            <div style="font-size: 24px; color: #DAA520; margin: -14px 0;">♪</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #800000; font-family: 'Palatino Linotype', serif;">
                Model Comparison
            </h2>
            <p style="color: #4B3621; font-size: 1rem; max-width: 800px; margin: 10px auto;">
                Compare the performance of different composition models on the same raga
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Create comparison data
        comparison_data = {
            'Metric': ['Overall Quality', 'Raga Adherence', 'Timbral Quality']
        }

        if has_bilstm:
            bilstm_metrics = st.session_state.bilstm_metrics
            comparison_data['Bi-LSTM'] = [
                bilstm_metrics.get('overall_quality', 0.0),
                bilstm_metrics.get('raga_adherence', 0.0),
                0.0  # Bi-LSTM doesn't have timbral quality
            ]

        if has_cnngan:
            cnngan_metrics = st.session_state.cnngan_metrics
            comparison_data['CNNGAN'] = [
                cnngan_metrics.get('overall_quality', 0.0),
                0.0,  # CNNGAN doesn't have raga adherence
                cnngan_metrics.get('timbral_quality', 0.0)
            ]

        if has_hybrid:
            hybrid_metrics = st.session_state.hybrid_metrics
            comparison_data['Hybrid'] = [
                hybrid_metrics.get('overall_quality', 0.0),
                hybrid_metrics.get('raga_adherence', 0.0),
                hybrid_metrics.get('timbral_quality', 0.0)
            ]

        # Create a DataFrame
        df = pd.DataFrame(comparison_data)

        # Create tabs for different comparison views
        comparison_tabs = st.tabs(["Bar Chart", "Radar Chart", "Data Table"])

        with comparison_tabs[0]:
            st.markdown("""
            <div style="background-color: rgba(218, 165, 32, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #DAA520; margin-top: 0;">Model Performance Comparison</h4>
                <p style="font-size: 0.9rem; color: #4B3621;">
                    This chart compares the performance of each model across key metrics
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Melt the DataFrame for easier plotting
            df_melted = pd.melt(df, id_vars=['Metric'], var_name='Model', value_name='Score')

            # Create the bar chart with enhanced styling
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
                    alpha=0.8,
                    width=0.25,
                    edgecolor='white',
                    linewidth=0.5
                )

            # Add labels and legend with enhanced styling
            ax.set_xlabel('Metric', fontsize=12, color='#4B3621')
            ax.set_ylabel('Score', fontsize=12, color='#4B3621')
            ax.set_title('Model Performance Comparison', fontsize=14, color='#800000', pad=20)

            # Add a legend with better styling
            legend = ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('#FFF8DC')
            frame.set_edgecolor('#DAA520')

            # Set y-axis limits
            ax.set_ylim(0, 1)

            # Add grid lines
            ax.grid(axis='y', linestyle='--', alpha=0.3)

            # Set background color
            ax.set_facecolor("#FFF8DC")
            fig.patch.set_facecolor("#FFF8DC")

            # Set x-tick labels
            ax.set_xticks(range(len(comparison_data['Metric'])))
            ax.set_xticklabels(comparison_data['Metric'], color='#4B3621')

            # Show the chart
            st.pyplot(fig)

        with comparison_tabs[1]:
            st.markdown("""
            <div style="background-color: rgba(218, 165, 32, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #DAA520; margin-top: 0;">Radar Chart Comparison</h4>
                <p style="font-size: 0.9rem; color: #4B3621;">
                    This visualization shows the strengths of each model across different dimensions
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Create a radar chart for comparing models
            categories = ['Overall Quality', 'Raga Adherence', 'Timbral Quality']

            # Create the radar chart with enhanced styling
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)

            # Number of variables
            N = len(categories)

            # What will be the angle of each axis in the plot
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            # Set the labels with better styling
            plt.xticks(angles[:-1], categories, size=12, color='#4B3621')

            # Draw the chart for each model
            if has_bilstm:
                values = [
                    comparison_data['Bi-LSTM'][0],  # Overall Quality
                    comparison_data['Bi-LSTM'][1],  # Raga Adherence
                    comparison_data['Bi-LSTM'][2]   # Timbral Quality
                ]
                values_for_chart = values + values[:1]
                ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#800000", label="Bi-LSTM")
                ax.fill(angles, values_for_chart, alpha=0.25, color="#800000")

            if has_cnngan:
                values = [
                    comparison_data['CNNGAN'][0],  # Overall Quality
                    comparison_data['CNNGAN'][1],  # Raga Adherence
                    comparison_data['CNNGAN'][2]   # Timbral Quality
                ]
                values_for_chart = values + values[:1]
                ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#483D8B", label="CNNGAN")
                ax.fill(angles, values_for_chart, alpha=0.25, color="#483D8B")

            if has_hybrid:
                values = [
                    comparison_data['Hybrid'][0],  # Overall Quality
                    comparison_data['Hybrid'][1],  # Raga Adherence
                    comparison_data['Hybrid'][2]   # Timbral Quality
                ]
                values_for_chart = values + values[:1]
                ax.plot(angles, values_for_chart, linewidth=2, linestyle='solid', color="#8B4513", label="Hybrid")
                ax.fill(angles, values_for_chart, alpha=0.25, color="#8B4513")

            # Add grid lines with better styling
            ax.grid(color='gray', alpha=0.2)

            # Set y-axis limits
            ax.set_ylim(0, 1)

            # Add a legend with better styling
            legend = ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, fancybox=True, framealpha=0.9)
            frame = legend.get_frame()
            frame.set_facecolor('#FFF8DC')
            frame.set_edgecolor('#DAA520')

            # Set background color
            ax.set_facecolor("#FFF8DC")
            fig.patch.set_facecolor("#FFF8DC")

            # Add a title
            plt.title('Model Comparison Across Metrics', size=14, color='#800000', pad=20)

            # Show the chart
            st.pyplot(fig)

        with comparison_tabs[2]:
            st.markdown("""
            <div style="background-color: rgba(218, 165, 32, 0.05); padding: 20px; border-radius: 10px; margin-bottom: 15px;">
                <h4 style="color: #DAA520; margin-top: 0;">Detailed Metrics</h4>
                <p style="font-size: 0.9rem; color: #4B3621;">
                    Numerical comparison of model performance across key metrics
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Style the dataframe
            st.dataframe(df, use_container_width=True)

            # Add a download button for the data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Comparison Data",
                data=csv,
                file_name="model_comparison.csv",
                mime="text/csv",
            )

# Simple footer function
def render_footer():
    """Render a simple footer for the app"""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("RagaVani - Exploring Indian classical music through AI")
    with col2:
        st.write("© 2023 RagaVani. All rights reserved.")

# Simple main function
if __name__ == "__main__":
    # Initialize session state variables if they don't exist
    if 'selected_model_tab' not in st.session_state:
        st.session_state.selected_model_tab = 0

    # Render the main page
    render_music_composer_page()

    # Add the footer
    render_footer()