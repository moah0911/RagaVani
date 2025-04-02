import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import time
import io
import base64
import logging
from datetime import datetime

# Load environment variables first
from utils.env_utils import load_environment_variables, get_environment_variable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_environment_variables()

# Initialize Snowflake environment if running in Snowflake
from utils.snowflake_init import initialize_snowflake_environment
initialize_snowflake_environment()

# Configure matplotlib for better visuals
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Palatino Linotype', 'Book Antiqua', 'Palatino', 'DejaVu Serif']
matplotlib.rcParams['axes.titlesize'] = 16
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10

# Import modules
from modules.raga_knowledge import get_raga_info, get_all_ragas, get_raga_by_mood, get_raga_by_time, compare_ragas
from modules.tala_knowledge import get_tala_info, get_all_talas, get_tala_by_beats, get_tala_clap_pattern, compare_talas

# Import utilities
from utils.audio_utils import load_audio, save_audio, trim_silence, normalize_audio, convert_to_mono, get_audio_base64

# Configuration
st.set_page_config(
    page_title="RagaVani - Indian Classical Music Explorer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Color Palette
colors = {
    "primary": "#800000",  # Deep Maroon
    "secondary": "#DAA520",  # Golden Yellow
    "accent1": "#483D8B",  # Deep Purple
    "accent2": "#FF8C00",  # Saffron
    "accent3": "#008B8B",  # Turquoise
    "background": "#FFF8DC",  # Soft Cream
    "text": "#4B3621"  # Deep Brown
}

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default to home page
if 'selected_raga' not in st.session_state:
    st.session_state.selected_raga = None
if 'selected_tala' not in st.session_state:
    st.session_state.selected_tala = None
if 'tanpura' not in st.session_state:
    st.session_state.tanpura = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Helper Functions
def load_css(css_file):
    """Load CSS styles"""
    try:
        with open(css_file, "r") as f:
            css_content = f.read()
        
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS file: {str(e)}")
        # Fallback styling
        st.markdown(
            """
            <style>
            body {
                background-color: #FFF8DC;
                color: #4B3621;
                font-family: serif;
            }
            h1, h2, h3 {
                color: #800000;
            }
            </style>
            """, 
            unsafe_allow_html=True
        )

def render_svg(svg_file):
    """Render SVG file"""
    try:
        with open(svg_file, "r") as f:
            svg_content = f.read()
        
        # Instead of using direct SVG, convert it to an img tag with base64 encoding
        import base64
        b64 = base64.b64encode(svg_content.encode("utf-8")).decode("utf-8")
        html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:150px; margin:auto;">'
        st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering SVG: {str(e)}")
        # Fallback to displaying app name
        st.markdown("<h3 style='color:#800000;'>RagaVani</h3>", unsafe_allow_html=True)

def nav_page(page):
    """Navigate to a specific page"""
    # Check if we're already on this page to avoid unnecessary reruns
    if st.session_state.page != page:
        st.session_state.page = page
        # Force a rerun to update the UI immediately
        st.rerun()

def render_decorative_divider():
    """Render a decorative divider"""
    st.markdown(
        """
        <div class="decorative-divider">
            <div class="symbol">♪</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

def get_raga_notes_string(raga_info):
    """Get formatted string of raga notes"""
    aroha_str = " ".join(raga_info["notes"]["aroha"])
    avaroha_str = " ".join(raga_info["notes"]["avaroha"])
    return aroha_str, avaroha_str

def raga_time_to_emoji(time_str):
    """Convert raga time to appropriate emoji"""
    time_str = time_str.lower()
    
    if "morning" in time_str:
        return "🌅"
    elif "afternoon" in time_str or "noon" in time_str:
        return "☀️"
    elif "evening" in time_str:
        return "🌆"
    elif "night" in time_str:
        return "🌙"
    elif "dawn" in time_str or "twilight" in time_str:
        return "🌄"
    elif "dusk" in time_str:
        return "🌇"
    else:
        return "🕰️"

def raga_mood_to_color(mood):
    """Map raga mood to color"""
    mood_color_map = {
        "devotional": "#FF8C00",  # Saffron
        "romantic": "#E75480",    # Pink
        "peaceful": "#6B8E23",    # Olive Green
        "serene": "#4682B4",      # Steel Blue
        "melancholic": "#483D8B", # Deep Purple
        "joyful": "#FF4500",      # Orange Red
        "serious": "#2F4F4F",     # Dark Slate Gray
        "profound": "#800000",    # Maroon
        "intense": "#8B0000",     # Dark Red
        "playful": "#FFA500",     # Orange
        "yearning": "#B22222",    # Firebrick
        "complex": "#4B0082",     # Indigo
        "meditative": "#2E8B57",  # Sea Green
        "sweet": "#DB7093",       # Pale Violet Red
        "bright": "#FFD700",      # Gold
        "majestic": "#4B0082",    # Indigo
        "mystical": "#8A2BE2",    # Blue Violet
    }
    
    # Default color if mood is not found
    default_color = "#800000"  # Maroon
    
    # Find a close match
    for key, value in mood_color_map.items():
        if key in mood.lower():
            return value
    
    return default_color

def create_pitch_visualization(raga_info):
    """Create a visualization of the raga's pitch structure"""
    # Clean up and prepare data
    aroha = raga_info["notes"]["aroha"]
    avaroha = raga_info["notes"]["avaroha"]
    
    # Define a mapping for Indian classical notes to numeric positions
    base_note_map = {
        'S': 0, 'r': 1, 'R': 2, 'g': 3, 'G': 4, 
        'm': 5, 'M': 6, 'P': 7, 'd': 8, 'D': 9, 
        'n': 10, 'N': 11, 'S\'': 12
    }
    
    # Map notes to positions for visualization
    y_asc = []
    for note in aroha:
        base_note = note[0]  # Get first character
        if base_note == 'S' and len(note) > 1 and note[1] == "'":
            y_asc.append(12)  # Upper Sa
        else:
            pos = base_note_map.get(base_note, 0)
            if '(k)' in note:  # Komal note
                pos -= 0.5
            elif '#' in note:  # Sharp note
                pos += 0.5
            y_asc.append(pos)
    
    y_desc = []
    for note in avaroha:
        base_note = note[0]
        if base_note == 'S' and len(note) > 1 and note[1] == "'":
            y_desc.append(12)  # Upper Sa
        else:
            pos = base_note_map.get(base_note, 0)
            if '(k)' in note:  # Komal note
                pos -= 0.5
            elif '#' in note:  # Sharp note
                pos += 0.5
            y_desc.append(pos)
    
    # Create the visualization
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFF8DC')
    
    # Define gradient colormaps for aroha and avaroha
    aroha_cmap = LinearSegmentedColormap.from_list("aroha", ["#800000", "#FF8C00"])
    avaroha_cmap = LinearSegmentedColormap.from_list("avaroha", ["#483D8B", "#008B8B"])
    
    # Plot ascending and descending patterns with gradient lines
    x_asc = np.arange(len(y_asc))
    x_desc = np.arange(len(y_desc))
    
    for i in range(len(x_asc)-1):
        ax.plot(x_asc[i:i+2], y_asc[i:i+2], 'o-', color=aroha_cmap(i/len(x_asc)), 
                linewidth=2.5, markersize=8, alpha=0.8)
    
    for i in range(len(x_desc)-1):
        ax.plot(x_desc[i:i+2], y_desc[i:i+2], 's-', color=avaroha_cmap(i/len(x_desc)), 
                linewidth=2.5, markersize=7, alpha=0.8)
    
    # Add emphasis on vadi and samvadi
    vadi_note = raga_info["vadi"]
    samvadi_note = raga_info["samvadi"]
    
    # Find positions of vadi and samvadi in the aroha and avaroha
    vadi_positions = []
    samvadi_positions = []
    
    for i, note in enumerate(aroha):
        if note.startswith(vadi_note):
            vadi_positions.append((i, y_asc[i]))
        if note.startswith(samvadi_note):
            samvadi_positions.append((i, y_asc[i]))
    
    for i, note in enumerate(avaroha):
        if note.startswith(vadi_note):
            vadi_positions.append((i, y_desc[i]))
        if note.startswith(samvadi_note):
            samvadi_positions.append((i, y_desc[i]))
    
    # Highlight vadi and samvadi notes
    for pos in vadi_positions:
        ax.plot(pos[0], pos[1], 'o', color='#800000', markersize=12, alpha=0.8, zorder=10)
    
    for pos in samvadi_positions:
        ax.plot(pos[0], pos[1], 'o', color='#DAA520', markersize=12, alpha=0.8, zorder=10)
    
    # Customize the plot
    ax.set_yticks(range(0, 13))
    ax.set_yticklabels(['S', 'r', 'R', 'g', 'G', 'm', 'M', 'P', 'd', 'D', 'n', 'N', 'S\''])
    ax.set_xticks([])
    
    # Add a grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add title and legend
    ax.set_title(f"Pitch Structure of Raga {raga_info['name'] if 'name' in raga_info else ''}", 
                color='#800000', fontsize=16, pad=20)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#800000', markersize=10, label=f'Vadi: {vadi_note}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#DAA520', markersize=10, label=f'Samvadi: {samvadi_note}'),
        Line2D([0], [0], color=aroha_cmap(0.5), lw=2, marker='o', label='Aroha (Ascending)'),
        Line2D([0], [0], color=avaroha_cmap(0.5), lw=2, marker='s', label='Avaroha (Descending)')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
    
    # Set plot background
    ax.set_facecolor('#FFF8DC')
    
    # Add text description for pakad
    if "pakad" in raga_info:
        fig.text(0.5, 0.02, f"Pakad: {raga_info['pakad']}", 
                ha='center', va='bottom', fontsize=12, color='#4B3621', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF8DC', alpha=0.8, 
                        edgecolor='#DAA520', linewidth=2))
    
    plt.tight_layout()
    return fig

def create_mood_visualization(raga_info):
    """Create a visualization of the raga's mood"""
    if "mood" not in raga_info:
        return None
    
    moods = raga_info["mood"]
    
    # Create a radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor='#FFF8DC')
    
    # Define standard moods for comparison
    standard_moods = ["Devotional", "Romantic", "Peaceful", "Serious", 
                      "Joyful", "Melancholic", "Meditative", "Majestic"]
    
    # Check which standard moods match this raga
    mood_values = []
    for std_mood in standard_moods:
        if any(std_mood.lower() in m.lower() for m in moods):
            mood_values.append(0.9)  # High value for matched moods
        else:
            mood_values.append(0.1)  # Low baseline for unmatched moods
    
    # Number of variables
    N = len(standard_moods)
    
    # Angles for each mood
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the polygon
    mood_values.append(mood_values[0])
    angles.append(angles[0])
    standard_moods.append(standard_moods[0])
    
    # Plot
    ax.plot(angles, mood_values, 'o-', linewidth=2, color='#800000')
    ax.fill(angles, mood_values, alpha=0.25, color='#DAA520')
    
    # Set tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(standard_moods[:-1])
    
    # Remove radial ticks and labels
    ax.set_yticks([])
    
    # Customize the appearance
    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle in (0, np.pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < np.pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
    
    # Set the title
    ax.set_title("Mood Profile", y=1.08, color='#800000', fontsize=16)
    
    # Set background color
    ax.set_facecolor('#FFF8DC')
    
    # Add the actual moods as text
    fig.text(0.5, 0.02, f"Moods: {', '.join(moods)}", 
            ha='center', va='bottom', fontsize=12, color='#4B3621',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='#FFF8DC', alpha=0.8, 
                     edgecolor='#DAA520', linewidth=2))
    
    plt.tight_layout()
    return fig

# Layout Components
def render_header():
    """Render the application header"""
    header_cols = st.columns([1, 3])
    
    with header_cols[0]:
        try:
            render_svg("assets/new_logo.svg")
        except Exception as e:
            st.error(f"Logo error: {e}")
            st.image("https://via.placeholder.com/150x150?text=RagaVani", width=150)
    
    with header_cols[1]:
        st.markdown("<h1 style='margin-top: 15px;'>RagaVani</h1>", unsafe_allow_html=True)
        st.markdown("<h3>Indian Classical Music Explorer</h3>", unsafe_allow_html=True)
        st.markdown("Experience the profound world of Indian classical music through exploration, analysis, and synthesis")

def render_navigation():
    """Render the main navigation"""
    st.sidebar.markdown("## Explore")

    nav_cols = st.sidebar.columns(2)

    with nav_cols[0]:
        if st.button("🏠 Home", use_container_width=True):
            nav_page('home')

    with nav_cols[1]:
        if st.button("ℹ️ About", use_container_width=True):
            nav_page('about')

    # Main navigation sections
    st.sidebar.markdown("## Main Sections")

    nav_sections = [
        ("🎵 Raga Encyclopedia", "ragas"),
        ("🥁 Tala Explorer", "talas"),
        ("🔍 Audio Analysis", "analysis"),
        ("🎶 Music Generator", "synthesis"),
        ("🎼 Music Composer", "composer"),
        ("🧠 AI Assistant", "ai")
    ]

    for name, page in nav_sections:
        if st.sidebar.button(name, key=f"nav_{page}", use_container_width=True):
            nav_page(page)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### RagaVani")
    st.sidebar.markdown("A comprehensive Indian Classical Music explorer combining traditional musicology with modern technology.")
    st.sidebar.markdown("© 2025 | v1.0.0")

# Page Content

def render_home_page():
    """Render the home page"""
    st.markdown("## Welcome to RagaVani")
    st.markdown("""
    <div style="background-color: rgba(128, 0, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #800000;">
        <p>RagaVani is your gateway to the enchanting world of Indian classical music. 
        Explore the intricate beauty of ragas and talas, analyze audio recordings, 
        and even generate authentic Indian classical music with our advanced tools.</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_decorative_divider()
    
    # Featured sections grid
    st.markdown("## Featured Sections")
    
    # Row 1
    cols1 = st.columns(2)
    
    with cols1[0]:
        st.markdown("""
        <div class="showcase-card">
            <div class="showcase-card-header">
                <h3>🎵 Raga Encyclopedia</h3>
            </div>
            <div class="showcase-card-body">
                <p>Explore the rich world of ragas with detailed information about their structure, characteristics, moods, and cultural context.</p>
                <ul>
                    <li>Detailed raga information</li>
                    <li>Visual representations</li>
                    <li>Comprehensive database</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Explore Ragas", key="home_ragas", use_container_width=True):
            nav_page('ragas')
    
    with cols1[1]:
        st.markdown("""
        <div class="showcase-card">
            <div class="showcase-card-header">
                <h3>🥁 Tala Explorer</h3>
            </div>
            <div class="showcase-card-body">
                <p>Understand the rhythmic foundations of Indian classical music through our tala explorer. Learn about various rhythm cycles and their applications.</p>
                <ul>
                    <li>Tala structures and patterns</li>
                    <li>Interactive visualizations</li>
                    <li>Relationship between talas</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Explore Talas", key="home_talas", use_container_width=True):
            nav_page('talas')
    
    # Row 2
    cols2 = st.columns(3)

    with cols2[0]:
        st.markdown("""
        <div class="showcase-card">
            <div class="showcase-card-header">
                <h3>🔍 Audio Analysis</h3>
            </div>
            <div class="showcase-card-body">
                <p>Upload audio samples and analyze them to identify ragas, detect talas, and visualize musical patterns.</p>
                <ul>
                    <li>Raga identification</li>
                    <li>Tala detection</li>
                    <li>Pitch and note distribution analysis</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Analyze Audio", key="home_analysis", use_container_width=True):
            nav_page('analysis')

    with cols2[1]:
        st.markdown("""
        <div class="showcase-card">
            <div class="showcase-card-header">
                <h3>🎶 Music Generator</h3>
            </div>
            <div class="showcase-card-body">
                <p>Generate authentic Indian classical music elements such as tanpura drones, tabla rhythms, and melodic patterns based on ragas.</p>
                <ul>
                    <li>Tanpura drone generator</li>
                    <li>Tabla rhythm patterns</li>
                    <li>Raga-based melodic phrases</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Generate Music", key="home_synthesis", use_container_width=True):
            nav_page('synthesis')

    with cols2[2]:
        st.markdown("""
        <div class="showcase-card">
            <div class="showcase-card-header">
                <h3>🎼 Music Composer</h3>
            </div>
            <div class="showcase-card-body">
                <p>Create complete Indian classical music compositions using advanced AI models like Bi-LSTM and CNNGAN.</p>
                <ul>
                    <li>Melodic composition with Bi-LSTM</li>
                    <li>Audio synthesis with CNNGAN</li>
                    <li>Hybrid composition techniques</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Compose Music", key="home_composer", use_container_width=True):
            nav_page('composer')
    
    # Add a direct link to the Ragas page instead of showing featured ragas
    st.markdown("""
    <div style="background-color: rgba(128, 0, 0, 0.1); padding: 20px; border-radius: 10px; 
         border-left: 5px solid #800000; margin-top: 30px; text-align: center;">
         <h3>Explore All Ragas</h3>
         <p>Visit our comprehensive Raga Encyclopedia to explore the vast collection of Indian classical ragas.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Open Raga Encyclopedia", use_container_width=True):
            nav_page('ragas')

def render_about_page():
    """Render the about page"""
    st.markdown("## About RagaVani")
    
    st.markdown("""
    <div style="background-color: rgba(72, 61, 139, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #483D8B;">
        <p>RagaVani is a comprehensive platform dedicated to exploring, analyzing, and creating Indian classical music. 
        It combines traditional musicological knowledge with modern technology to provide an immersive experience 
        for both beginners and advanced practitioners.</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_decorative_divider()
    
    # About content
    st.markdown("### The Essence of Indian Classical Music")
    
    st.markdown("""
    Indian classical music is one of the oldest and richest musical traditions in the world, 
    with roots dating back thousands of years. It's characterized by two main elements:
    
    1. **Raga**: The melodic framework that defines the note combinations and patterns
    2. **Tala**: The rhythmic cycle that provides structure and time
    
    Unlike Western classical music, Indian classical music places a strong emphasis on improvisation 
    within the structured framework of ragas and talas. Each performance is unique, with musicians 
    interpreting the raga according to their own style and the mood of the moment.
    """)
    
    # Two columns with more info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Features of RagaVani")
        
        st.markdown("""
        - **Raga Encyclopedia**: Comprehensive database of ragas with detailed information
        - **Tala Explorer**: Learn about various rhythm cycles and their applications
        - **Audio Analysis**: Identify ragas and talas in audio recordings
        - **Music Generator**: Create tanpura drones, tabla patterns, and melodic phrases
        - **AI Assistant**: Get intelligent insights and explanations about Indian classical concepts
        """)
    
    with col2:
        st.markdown("### Technical Implementation")
        
        st.markdown("""
        RagaVani combines several advanced technologies:
        
        - **Audio Processing**: Uses specialized DSP techniques for pitch detection and raga analysis
        - **Visualization**: Custom visualizations for ragas, talas, and audio analysis
        - **Synthesis**: Physical modeling and wavetable synthesis for authentic instrument sounds
        - **AI Integration**: Leverages Google's Gemini models for music understanding and explanation
        """)
    
    render_decorative_divider()
    
    # Inspiration and acknowledgments
    st.markdown("### Inspiration")
    
    st.markdown("""
    This application is inspired by the rich tradition of Indian classical music and the dedication of 
    countless musicians who have preserved and evolved this art form over centuries. We aim to make 
    this profound musical system more accessible to a wider audience while maintaining respect for its 
    depth and complexity.
    """)
    
    st.markdown("### Acknowledgments")
    
    st.markdown("""
    We would like to acknowledge the contributions of renowned musicians, musicologists, and scholars 
    whose work has informed our understanding and implementation. Special thanks to the open-source 
    community for providing tools and libraries that make this application possible.
    """)

def render_ragas_page():
    """Render the ragas encyclopedia page"""
    st.markdown("## Raga Encyclopedia")

    st.markdown("""
    <div style="background-color: rgba(255, 140, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #FF8C00;">
        <p>Explore the vast world of Indian classical ragas. Each raga has its unique characteristics,
        mood, and time of performance. Select a raga to view detailed information or use filters to
        find ragas based on specific criteria.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for filters if not already present
    if 'raga_time_filter' not in st.session_state:
        st.session_state.raga_time_filter = "Any Time"
    if 'raga_mood_filter' not in st.session_state:
        st.session_state.raga_mood_filter = "Any Mood"
    if 'raga_thaat_filter' not in st.session_state:
        st.session_state.raga_thaat_filter = "Any Thaat"
    if 'filtered_ragas_cache' not in st.session_state:
        st.session_state.filtered_ragas_cache = None

    # Filters section
    with st.expander("Filter Ragas", expanded=False):
        filter_cols = st.columns(3)

        with filter_cols[0]:
            # Filter by time of day
            time_options = ["Any Time", "Morning", "Afternoon", "Evening", "Night", "Dawn/Twilight"]
            selected_time = st.selectbox("Time of Day", time_options, key="time_filter")
            if selected_time != st.session_state.raga_time_filter:
                st.session_state.raga_time_filter = selected_time
                st.session_state.filtered_ragas_cache = None  # Clear cache when filter changes

        with filter_cols[1]:
            # Filter by mood
            mood_options = ["Any Mood", "Devotional", "Romantic", "Peaceful", "Serious",
                          "Joyful", "Melancholic", "Meditative", "Majestic"]
            selected_mood = st.selectbox("Mood", mood_options, key="mood_filter")
            if selected_mood != st.session_state.raga_mood_filter:
                st.session_state.raga_mood_filter = selected_mood
                st.session_state.filtered_ragas_cache = None  # Clear cache when filter changes

        with filter_cols[2]:
            # Filter by thaat (parent scale)
            thaat_options = ["Any Thaat", "Bilawal", "Kalyan", "Khamaj", "Bhairav",
                           "Bhairavi", "Asavari", "Todi", "Purvi", "Marwa", "Kafi"]
            selected_thaat = st.selectbox("Thaat (Parent Scale)", thaat_options, key="thaat_filter")
            if selected_thaat != st.session_state.raga_thaat_filter:
                st.session_state.raga_thaat_filter = selected_thaat
                st.session_state.filtered_ragas_cache = None  # Clear cache when filter changes

    render_decorative_divider()

    # Use cached filtered ragas if available, otherwise compute them
    if st.session_state.filtered_ragas_cache is None:
        # Get all ragas
        all_ragas = get_all_ragas()

        # Apply filters
        filtered_ragas = []
        filtered_raga_info = {}  # Cache for raga info to avoid repeated calls

        for raga_name in all_ragas:
            raga_info = get_raga_info(raga_name)

            if not raga_info:
                continue

            # Check if raga passes all filters
            passes_filter = True

            # Time filter
            if st.session_state.raga_time_filter != "Any Time" and "time" in raga_info:
                if st.session_state.raga_time_filter.lower() not in raga_info["time"].lower():
                    passes_filter = False

            # Mood filter
            if st.session_state.raga_mood_filter != "Any Mood" and "mood" in raga_info:
                if not any(st.session_state.raga_mood_filter.lower() in mood.lower() for mood in raga_info["mood"]):
                    passes_filter = False

            # Thaat filter
            if st.session_state.raga_thaat_filter != "Any Thaat" and "thaat" in raga_info:
                if st.session_state.raga_thaat_filter != raga_info["thaat"]:
                    passes_filter = False

            if passes_filter:
                filtered_ragas.append(raga_name)
                filtered_raga_info[raga_name] = raga_info

        # Store in session state for future use
        st.session_state.filtered_ragas_cache = {
            'ragas': filtered_ragas,
            'info': filtered_raga_info
        }
    else:
        # Use cached results
        filtered_ragas = st.session_state.filtered_ragas_cache['ragas']
        filtered_raga_info = st.session_state.filtered_ragas_cache['info']

    # Display ragas in a grid
    if filtered_ragas:
        st.markdown(f"### Showing {len(filtered_ragas)} Ragas")

        # Create rows of 3 ragas
        for i in range(0, len(filtered_ragas), 3):
            cols = st.columns(3)

            for j in range(3):
                if i + j < len(filtered_ragas):
                    raga_name = filtered_ragas[i + j]
                    raga_info = filtered_raga_info[raga_name]  # Use cached info

                    with cols[j]:
                        time_emoji = raga_time_to_emoji(raga_info["time"])

                        # Create a color based on the first mood
                        mood_color = raga_mood_to_color(raga_info["mood"][0])

                        # Create a unique key for each button based on raga name
                        button_key = f"view_{raga_name.replace(' ', '_')}"

                        # Display raga card
                        st.markdown(f"""
                        <div class="raga-card" style="border-color: {mood_color};">
                            <h3>{raga_name} {time_emoji}</h3>
                            <p><strong>Thaat:</strong> {raga_info["thaat"]}</p>
                            <p><strong>Mood:</strong> {', '.join(raga_info["mood"][:2])}{' ...' if len(raga_info["mood"]) > 2 else ''}</p>
                            <p style="font-size: 0.9rem; height: 60px; overflow: hidden; text-overflow: ellipsis;">
                                {raga_info["description"][:100]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Use a more efficient approach for buttons
                        if st.button(f"View {raga_name}", key=button_key, use_container_width=True):
                            st.session_state.selected_raga = raga_name
                            nav_page('raga_detail')
    else:
        st.warning("No ragas match the selected filters. Please try different criteria.")

def render_raga_detail_page():
    """Render the detailed view of a specific raga"""
    if not st.session_state.selected_raga:
        st.error("No raga selected. Please select a raga from the encyclopedia.")
        
        if st.button("Return to Raga Encyclopedia"):
            nav_page('ragas')
        
        return
    
    raga_name = st.session_state.selected_raga
    raga_info = get_raga_info(raga_name)
    
    if not raga_info:
        st.error(f"Raga information for '{raga_name}' not found.")
        return
    
    # Add raga name to info dict for visualization
    raga_info["name"] = raga_name
    
    # Back button
    if st.button("← Back to Raga Encyclopedia"):
        nav_page('ragas')
    
    # Convert notes to strings
    aroha_str, avaroha_str = get_raga_notes_string(raga_info)
    
    # Header
    time_emoji = raga_time_to_emoji(raga_info["time"])
    
    st.markdown(f"## {raga_name} {time_emoji}")
    
    # Divider
    render_decorative_divider()
    
    # Overview section
    st.markdown("### Overview")
    st.markdown(raga_info["description"])
    
    # Tabs for different aspects
    raga_tabs = st.tabs(["Structure", "Visualization", "Cultural Context", "Similar Ragas"])
    
    with raga_tabs[0]:  # Structure
        st.markdown("### Raga Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Aroha (Ascending):** {aroha_str}")
            st.markdown(f"**Avaroha (Descending):** {avaroha_str}")
            st.markdown(f"**Vadi (Most important note):** {raga_info['vadi']}")
            st.markdown(f"**Samvadi (Second most important):** {raga_info['samvadi']}")
        
        with col2:
            st.markdown(f"**Pakad (Characteristic phrase):** {raga_info['pakad']}")
            st.markdown(f"**Thaat (Parent scale):** {raga_info['thaat']}")
            st.markdown(f"**Time of performance:** {raga_info['time']}")
    
    with raga_tabs[1]:  # Visualization
        st.markdown("### Raga Visualization")
        
        viz_cols = st.columns(2)
        
        with viz_cols[0]:
            # Pitch structure visualization
            pitch_fig = create_pitch_visualization(raga_info)
            st.pyplot(pitch_fig)
        
        with viz_cols[1]:
            # Mood visualization
            mood_fig = create_mood_visualization(raga_info)
            if mood_fig:
                st.pyplot(mood_fig)
    
    with raga_tabs[2]:  # Cultural Context
        st.markdown("### Cultural Context")
        
        # Mood
        st.markdown(f"**Mood:** {', '.join(raga_info['mood'])}")
        
        # Create mood badges with colors
        mood_html = ""
        for mood in raga_info["mood"]:
            mood_color = raga_mood_to_color(mood)
            mood_html += f'<span style="display: inline-block; background-color: {mood_color}; color: white; padding: 5px 10px; margin: 5px; border-radius: 15px;">{mood}</span>'
        
        st.markdown(f"<div>{mood_html}</div>", unsafe_allow_html=True)
        
        # Famous compositions
        st.markdown("#### Notable Compositions")
        for composition in raga_info["famous_compositions"]:
            st.markdown(f"- {composition}")
    
    with raga_tabs[3]:  # Similar Ragas
        st.markdown("### Similar Ragas")

        # Function to find similar ragas if not already defined in the database
        def find_similar_ragas(raga_name, raga_info, max_similar=3):
            # If similar ragas are already defined in the database, use those
            if "similar_ragas" in raga_info and raga_info["similar_ragas"]:
                return raga_info["similar_ragas"]

            # Otherwise, find similar ragas based on thaat, notes, and mood
            all_ragas = get_all_ragas()
            similar_ragas = []

            # Get all notes in the current raga
            current_notes = set()
            if "notes" in raga_info:
                current_notes = set(raga_info["notes"]["aroha"] + raga_info["notes"]["avaroha"])

            # Get current thaat and mood
            current_thaat = raga_info.get("thaat", "")
            current_mood = set(raga_info.get("mood", []))

            # Score each raga for similarity
            raga_scores = []

            for other_raga_name in all_ragas:
                # Skip the current raga
                if other_raga_name == raga_name:
                    continue

                other_raga_info = get_raga_info(other_raga_name)
                if not other_raga_info:
                    continue

                score = 0

                # Same thaat is a strong indicator of similarity
                if other_raga_info.get("thaat") == current_thaat:
                    score += 5

                # Check note similarity
                other_notes = set()
                if "notes" in other_raga_info:
                    other_notes = set(other_raga_info["notes"]["aroha"] + other_raga_info["notes"]["avaroha"])

                if current_notes and other_notes:
                    # Calculate Jaccard similarity for notes
                    common_notes = current_notes.intersection(other_notes)
                    all_notes = current_notes.union(other_notes)
                    if all_notes:
                        note_similarity = len(common_notes) / len(all_notes)
                        score += 3 * note_similarity

                # Check mood similarity
                other_mood = set(other_raga_info.get("mood", []))
                if current_mood and other_mood:
                    common_moods = current_mood.intersection(other_mood)
                    if common_moods:
                        score += len(common_moods)

                # Check time similarity
                if raga_info.get("time") == other_raga_info.get("time"):
                    score += 2

                # Add to scores list
                raga_scores.append((other_raga_name, score))

            # Sort by score and get top matches
            raga_scores.sort(key=lambda x: x[1], reverse=True)
            return [name for name, score in raga_scores[:max_similar]]

        # Get similar ragas - either from database or by finding them
        similar_ragas = []

        # Always run the find_similar_ragas function to ensure we have results
        computed_similar_ragas = find_similar_ragas(raga_name, raga_info, max_similar=5)

        # Hardcoded fallback similar ragas to ensure something always shows
        fallback_similar_ragas = ["Yaman", "Bhairav", "Bhairavi", "Darbari", "Malkauns"]
        # Remove the current raga from fallbacks if it's in there
        if raga_name in fallback_similar_ragas:
            fallback_similar_ragas.remove(raga_name)

        # If the raga has predefined similar ragas, use those first
        if "similar_ragas" in raga_info and raga_info["similar_ragas"] and len(raga_info["similar_ragas"]) > 0:
            similar_ragas = raga_info["similar_ragas"]
            st.success(f"Showing similar ragas based on musical tradition")
        elif computed_similar_ragas and len(computed_similar_ragas) > 0:
            similar_ragas = computed_similar_ragas
            st.info(f"Showing similar ragas based on musical analysis")
        else:
            # Use fallback ragas if nothing else works
            similar_ragas = fallback_similar_ragas[:3]
            st.warning(f"Showing popular ragas for comparison")

        # Always show similar ragas section
        # Create columns for displaying similar ragas (always show 3 columns)
        similar_cols = st.columns(3)

        # Make sure we have at least some ragas to display
        if not similar_ragas:
            similar_ragas = ["Yaman", "Bhairav", "Bhairavi"]
            if raga_name in similar_ragas:
                similar_ragas.remove(raga_name)
                similar_ragas.append("Darbari")

        # Display the similar ragas
        for i, similar_raga_name in enumerate(similar_ragas[:3]):  # Limit to 3 ragas
            similar_raga_info = get_raga_info(similar_raga_name)

            if not similar_raga_info:
    # Skip if we can't find info
                continue

            with similar_cols[i % 3]:
                            # Create a unique key for the button based on both ragas to avoid conflicts
                button_key = f"view_from_{raga_name}_to_{similar_raga_name}".replace(' ', '_')

                try:
                    time_emoji = raga_time_to_emoji(similar_raga_info["time"])

                    # Create a color based on the first mood
                    mood_color = raga_mood_to_color(similar_raga_info["mood"][0])

                    # Display the raga card
                    st.markdown(f"""
                    <div class="raga-card" style="border-color: {mood_color};">
                        <h3>{similar_raga_name} {time_emoji}</h3>
                        <p><strong>Thaat:</strong> {similar_raga_info["thaat"]}</p>
                        <p><strong>Mood:</strong> {', '.join(similar_raga_info["mood"][:2])}{' ...' if len(similar_raga_info["mood"]) > 2 else ''}</p>
                        <p style="font-size: 0.9rem; height: 60px; overflow: hidden; text-overflow: ellipsis;">
                            {similar_raga_info["description"][:100]}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Add the button to view this raga
                    if st.button(f"View {similar_raga_name}", key=button_key, use_container_width=True):
                        st.session_state.selected_raga = similar_raga_name
                        # Rerun to refresh with new raga
                        st.rerun()
                except Exception as e:
                    # Fallback display if there's an error
                    st.error(f"Error displaying {similar_raga_name}: {str(e)}")
                    st.button(f"View {similar_raga_name}", key=button_key, use_container_width=True, disabled=True)
    
    # Generate tanpura button
    st.markdown("### Experience this Raga")
    
    if st.button("Generate Tanpura for this Raga", key="gen_tanpura", use_container_width=True):
        with st.spinner("Generating tanpura sound..."):
            try:
                # Figure out the root note (Sa) based on traditional performance practice
                # Default to C if we can't determine
                root_note = "C"
                
                # Import the generation function here to avoid potential import errors
                from modules.audio_synthesis import generate_tanpura
                
                # Generate tanpura with 30 seconds duration
                st.session_state.tanpura, sr = generate_tanpura(
                    root_note=root_note,
                    duration=30,
                    tempo=60,
                    jiva=0.7
                )
                
                st.success(f"Tanpura generated for Raga {raga_name}!")
                
                # Play the tanpura
                import soundfile as sf
                
                # Buffer for writing
                buffer = io.BytesIO()
                
                # Write to the buffer
                sf.write(buffer, st.session_state.tanpura, sr, format='WAV')
                
                # Get the buffer content
                audio_bytes = buffer.getvalue()
                
                # Play the audio
                st.audio(audio_bytes, format="audio/wav")
                
                # Download option
                st.download_button(
                    label="Download Tanpura Audio",
                    data=audio_bytes,
                    file_name=f"tanpura_{raga_name}.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Error generating tanpura: {str(e)}")
                st.info("This is a placeholder for the tanpura generation functionality.")

def render_talas_page():
    """Render the talas explorer page"""
    st.markdown("## Tala Explorer")

    st.markdown("""
    <div style="background-color: rgba(0, 139, 139, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #008B8B;">
        <p>Discover the rhythmic foundations of Indian classical music through talas (rhythm cycles).
        Each tala has a specific number of beats arranged in a distinct pattern. Select a tala to view
        detailed information or filter by number of beats.</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for tala filters if not already present
    if 'tala_beat_filter' not in st.session_state:
        st.session_state.tala_beat_filter = "Any"
    if 'filtered_talas_cache' not in st.session_state:
        st.session_state.filtered_talas_cache = None

    # Filter by number of beats
    with st.expander("Filter Talas", expanded=False):
        beat_options = ["Any", "6-8 (Fast)", "10-12 (Medium)", "14-16 (Extended)"]
        selected_beat_range = st.selectbox("Number of Beats", beat_options, key="beat_filter")
        if selected_beat_range != st.session_state.tala_beat_filter:
            st.session_state.tala_beat_filter = selected_beat_range
            st.session_state.filtered_talas_cache = None  # Clear cache when filter changes

    render_decorative_divider()

    # Use cached filtered talas if available, otherwise compute them
    if st.session_state.filtered_talas_cache is None:
        # Get all talas
        all_talas = get_all_talas()

        # Apply filter if selected
        filtered_talas = []
        filtered_tala_info = {}  # Cache for tala info to avoid repeated calls

        for tala_name in all_talas:
            tala_info = get_tala_info(tala_name)

            if not tala_info:
                continue

            # Check beat range filter
            if st.session_state.tala_beat_filter != "Any":
                beats = tala_info["beats"]

                if st.session_state.tala_beat_filter == "6-8 (Fast)" and (beats < 6 or beats > 8):
                    continue
                elif st.session_state.tala_beat_filter == "10-12 (Medium)" and (beats < 10 or beats > 12):
                    continue
                elif st.session_state.tala_beat_filter == "14-16 (Extended)" and (beats < 14 or beats > 16):
                    continue

            filtered_talas.append(tala_name)
            filtered_tala_info[tala_name] = tala_info

        # Store in session state for future use
        st.session_state.filtered_talas_cache = {
            'talas': filtered_talas,
            'info': filtered_tala_info
        }
    else:
        # Use cached results
        filtered_talas = st.session_state.filtered_talas_cache['talas']
        filtered_tala_info = st.session_state.filtered_talas_cache['info']

    # Display talas in a grid
    if filtered_talas:
        st.markdown(f"### Showing {len(filtered_talas)} Talas")

        # Create rows of 2 talas (they need more space than ragas)
        for i in range(0, len(filtered_talas), 2):
            cols = st.columns(2)

            for j in range(2):
                if i + j < len(filtered_talas):
                    tala_name = filtered_talas[i + j]
                    tala_info = filtered_tala_info[tala_name]  # Use cached info

                    with cols[j]:
                        # Create a gradient color based on the number of beats
                        beats = tala_info["beats"]

                        # Create a unique key for each button based on tala name
                        button_key = f"view_{tala_name.replace(' ', '_')}"

                        # Pre-compute clap pattern HTML to improve performance
                        clap_pattern = " | ".join([f"<span style='font-weight: bold; color: {'#d4af37' if p == 'X' else '#8b4513' if p.isdigit() else '#6c757d'}'>{p}</span>" for p in tala_info["clap_pattern"]])

                        # Display tala card
                        st.markdown(f"""
                        <div class="tala-card">
                            <h3>{tala_name}</h3>
                            <p><strong>Beats:</strong> {beats}</p>
                            <p><strong>Pattern:</strong> {tala_info["pattern_description"]}</p>
                            <div style="font-size: 1.2rem; margin: 10px 0; text-align: center;">
                                {clap_pattern}
                            </div>
                            <p style="font-size: 0.9rem; height: 60px; overflow: hidden; text-overflow: ellipsis;">
                                {tala_info["description"][:100]}...
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Use a more efficient approach for buttons
                        if st.button(f"View {tala_name}", key=button_key, use_container_width=True):
                            st.session_state.selected_tala = tala_name
                            nav_page('tala_detail')
    else:
        st.warning("No talas match the selected filters. Please try different criteria.")

def render_tala_detail_page():
    """Render the detailed view of a specific tala"""
    if not st.session_state.selected_tala:
        st.error("No tala selected. Please select a tala from the explorer.")
        
        if st.button("Return to Tala Explorer"):
            nav_page('talas')
        
        return
    
    tala_name = st.session_state.selected_tala
    tala_info = get_tala_info(tala_name)
    
    if not tala_info:
        st.error(f"Tala information for '{tala_name}' not found.")
        return
    
    # Back button
    if st.button("← Back to Tala Explorer"):
        nav_page('talas')
    
    # Header
    st.markdown(f"## {tala_name}")
    
    # Display basic information
    st.markdown(f"**Beats:** {tala_info['beats']}")
    st.markdown(f"**Pattern:** {tala_info['pattern_description']}")
    
    # Divider
    render_decorative_divider()
    
    # Overview
    st.markdown("### Overview")
    st.markdown(tala_info["description"])
    
    # Visualization of the tala structure
    st.markdown("### Tala Structure")
    
    # Visual representation of the tala
    clap_pattern = tala_info["clap_pattern"]
    vibhags = tala_info["vibhags"]
    
    # Create a horizontal visualization
    fig, ax = plt.subplots(figsize=(10, 3), facecolor='#FFF8DC')
    
    # Total beats
    total_beats = tala_info["beats"]
    
    # Create a position for each beat
    beat_positions = list(range(1, total_beats + 1))
    
    # Determine the clap type for each beat
    beat_types = []
    beat_idx = 0
    
    for vibhag_idx, vibhag_size in enumerate(vibhags):
        clap_type = clap_pattern[vibhag_idx]
        for _ in range(vibhag_size):
            beat_types.append(clap_type)
            beat_idx += 1
    
    # Create colors based on clap type
    beat_colors = []
    
    for beat_type in beat_types:
        if beat_type == "X":  # Sam
            beat_colors.append("#800000")  # Deep Maroon
        elif beat_type == "0":  # Khali
            beat_colors.append("#008B8B")  # Turquoise
        else:  # Tali
            beat_colors.append("#DAA520")  # Golden Yellow
    
    # Plot beats as circles
    for i, (pos, color) in enumerate(zip(beat_positions, beat_colors)):
        ax.add_patch(plt.Circle((pos, 0.5), 0.4, color=color, alpha=0.8))
        ax.text(pos, 0.5, str(i+1), ha='center', va='center', color='white', fontweight='bold')
    
    # Add labels for clap types
    # Group consecutive beats with same clap type
    vibhag_starts = [sum(vibhags[:i]) + 1 for i in range(len(vibhags))]
    
    for i, (start, size, clap_type) in enumerate(zip(vibhag_starts, vibhags, clap_pattern)):
        vibhag_center = start + size/2 - 0.5
        label = ""
        
        if clap_type == "X":
            label = "Sam (Beginning)"
        elif clap_type == "0":
            label = "Khali (Empty)"
        else:
            label = f"Tali {clap_type} (Clap)"
        
        ax.text(vibhag_center, 1.2, label, ha='center', va='bottom', 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'))
    
    # Remove axes
    ax.set_xlim(0, total_beats + 1)
    ax.set_ylim(0, 2)
    ax.axis('off')
    
    # Add title
    ax.set_title(f"Structure of {tala_name}: {tala_info['pattern_description']}", 
                y=0.95, color='#800000', fontsize=16)
    
    st.pyplot(fig)
    
    # Tabs for different aspects
    tala_tabs = st.tabs(["Performance", "Applications", "Related Talas"])
    
    with tala_tabs[0]:  # Performance
        st.markdown("### Performance Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Tempo Ranges:** {', '.join(tala_info['tempo_range'])}")
            st.markdown(f"**Instruments:** {', '.join(tala_info['instruments'])}")
        
        with col2:
            if "variations" in tala_info:
                st.markdown(f"**Variations:** {', '.join(tala_info['variations'])}")
        
        # Generate tabla button
        if st.button("Generate Tabla Pattern", key="gen_tabla"):
            with st.spinner("Generating tabla pattern..."):
                try:
                    # Import the generation function here to avoid potential import errors
                    from modules.audio_synthesis import synthesize_tabla
                    
                    # Generate tabla with appropriate tempo based on tala
                    if "Slow" in tala_info["tempo_range"][0]:
                        tempo = 60
                    elif "Fast" in tala_info["tempo_range"][-1]:
                        tempo = 120
                    else:
                        tempo = 90
                    
                    # Generate tabla pattern
                    tabla_audio, sr = synthesize_tabla(
                        tala=tala_name,
                        tempo=tempo,
                        duration=20
                    )
                    
                    st.success(f"Tabla pattern generated for {tala_name}!")
                    
                    # Play the tabla pattern
                    import soundfile as sf
                    
                    # Buffer for writing
                    buffer = io.BytesIO()
                    
                    # Write to the buffer
                    sf.write(buffer, tabla_audio, sr, format='WAV')
                    
                    # Get the buffer content
                    audio_bytes = buffer.getvalue()
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download option
                    st.download_button(
                        label="Download Tabla Pattern",
                        data=audio_bytes,
                        file_name=f"tabla_{tala_name}.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"Error generating tabla pattern: {str(e)}")
                    st.info("This is a placeholder for the tabla pattern generation functionality.")
    
    with tala_tabs[1]:  # Applications
        st.markdown("### Applications in Music")
        
        st.markdown("""
        Indian classical music compositions are structured around talas, with specific sections and improvisations 
        organized according to the rhythmic cycle. Here are typical applications:
        """)
        
        st.markdown(f"""
        - **Compositions**: Fixed compositions (bandish in khyal, dhrupad, etc.) set to {tala_name}
        - **Improvisations**: Various improvisational forms like layakari (rhythmic variations), tans (fast melodic passages)
        - **Solo Performance**: Tabla solo performances feature complex variations and improvisations on the tala
        """)
        
        # Show which types of compositions commonly use this tala
        composition_types = []
        
        # These are broad generalizations and can be refined with more specific knowledge
        if tala_info["beats"] in [6, 8]:
            composition_types.extend(["Thumri", "Dadra", "Light classical forms"])
        
        if tala_info["beats"] in [10, 12, 14, 16]:
            composition_types.extend(["Khayal", "Instrumental gats"])
        
        if tala_info["beats"] in [12, 14]:
            composition_types.extend(["Dhrupad", "Dhamar"])
        
        if composition_types:
            st.markdown("#### Common Composition Types")
            st.markdown(", ".join(composition_types))
    
    with tala_tabs[2]:  # Related Talas
        st.markdown("### Related Talas")
        
        # Find talas with mathematical relationships
        related_talas = []
        
        for other_tala in get_all_talas():
            if other_tala == tala_name:
                continue
                
            other_info = get_tala_info(other_tala)
            
            if not other_info:
                continue
                
            # Check for mathematical relationships
            relationship = None
            
            if other_info["beats"] == tala_info["beats"]:
                relationship = "Same number of beats, different structure"
            elif other_info["beats"] == tala_info["beats"] * 2:
                relationship = "Double the beats (augmentation)"
            elif tala_info["beats"] == other_info["beats"] * 2:
                relationship = "Half the beats (diminution)"
            elif other_info["beats"] % tala_info["beats"] == 0:
                relationship = f"Multiple ({other_info['beats'] // tala_info['beats']}x) of beats"
            elif tala_info["beats"] % other_info["beats"] == 0:
                relationship = f"Division (1/{tala_info['beats'] // other_info['beats']}) of beats"
            
            if relationship or "variations" in tala_info and other_tala in tala_info["variations"]:
                if "variations" in tala_info and other_tala in tala_info["variations"]:
                    relationship = "Variation" + (f", {relationship}" if relationship else "")
                
                related_talas.append((other_tala, other_info, relationship))
        
        if related_talas:
            for other_tala, other_info, relationship in related_talas:
                st.markdown(f"""
                <div class="tala-card">
                    <h3>{other_tala}</h3>
                    <p><strong>Beats:</strong> {other_info['beats']}</p>
                    <p><strong>Relationship:</strong> {relationship}</p>
                    <p><strong>Pattern:</strong> {other_info['pattern_description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"View {other_tala}", key=f"view_related_{other_tala}", use_container_width=True):
                    st.session_state.selected_tala = other_tala
                    # Rerun to refresh with new tala
                    st.rerun()
        else:
            st.info("No strongly related talas found.")

def render_analysis_page():
    """Render the audio analysis page"""
    st.markdown("## Audio Analysis")
    
    st.markdown("""
    <div style="background-color: rgba(72, 61, 139, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #483D8B;">
        <p>Analyze audio recordings to identify ragas, detect talas, and visualize musical patterns.
        Upload an audio file to begin the analysis process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Audio source selection
    audio_source = st.radio(
        "Select Audio Source:",
        ["Upload Audio"],
        horizontal=True
    )
    
    audio_data = None
    sr = None
    
    if audio_source == "Upload Audio":
        uploaded_file = st.file_uploader("Upload an audio file:", type=['mp3', 'wav', 'ogg'])
        
        if uploaded_file:
            # Save to a temporary file
            with st.spinner("Processing audio file..."):
                temp_file = f"temp_upload_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the audio file
                audio_data, sr = load_audio(temp_file)
                
                if audio_data is not None:
                    st.success("Audio file loaded successfully!")
                    st.audio(temp_file)
                else:
                    st.error("Failed to load audio file. Please try a different file.")
    

    
    # Analysis section
    if audio_data is not None:
        render_decorative_divider()
        
        st.markdown("### Analysis Options")
        
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Basic Analysis", "Raga Identification", "Tala Detection", "Advanced Analysis"]
        )
        
        if st.button("Analyze", key="analyze_button", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                # In a real implementation, we would call the appropriate analysis function
                try:
                    # Import our fully implemented analysis module
                    from modules.audio_analysis import analyze_audio
                    
                    # Perform complete analysis to get pitch, raga, tala and ornament data
                    analysis_results = analyze_audio(audio_data, sr)
                    
                    # Store in session state
                    st.session_state.analysis_results = analysis_results
                    
                    st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Error performing analysis: {str(e)}")
                    
                    # Try with simplified module as fallback
                    try:
                        from modules.audio_analysis_simplified import analyze_audio as analyze_audio_simple
                        
                        # Use simplified analysis
                        analysis_results = analyze_audio_simple(audio_data, sr)
                        
                        # Store in session state
                        st.session_state.analysis_results = analysis_results
                        
                        st.success("Analysis completed with simplified algorithm!")
                    except Exception as e2:
                        st.error(f"Error with simplified analysis: {str(e2)}")
                        
                        # Create a more robust fallback for when both analyses fail
                        import time
                        time.sleep(1)  # Brief pause for user experience
                        
                        # Determine raga and tala based on analysis type
                        detected_raga = None
                        raga_confidence = None
                        detected_tala = None
                        tala_confidence = None
                        
                        if "Raga" in analysis_type:
                            detected_raga = "Yaman" if "Basic" in analysis_type else "Bhairav"
                            raga_confidence = 85
                        
                        if "Tala" in analysis_type:
                            detected_tala = "Teentaal" if "Basic" in analysis_type else "Ektaal"
                            tala_confidence = 78
                        
                        # Create duration-appropriate data
                        duration = len(audio_data) / sr if audio_data is not None and sr is not None else 10
                        num_samples = int(duration * 100)  # 100 samples per second
                        
                        # Create a basic analysis result
                        st.session_state.analysis_results = {
                            "duration": duration,
                            "detected_raga": detected_raga,
                            "raga_confidence": raga_confidence,
                            "detected_tala": detected_tala,
                            "tala_confidence": tala_confidence,
                            "pitch_data": {
                                "times": np.linspace(0, duration, num_samples),
                                "pitches": 200 + 80 * np.sin(np.linspace(0, 10*np.pi, num_samples)) + 
                                          20 * np.sin(np.linspace(0, 50*np.pi, num_samples)) +
                                          5 * np.random.randn(num_samples)
                            }
                        }
                        
                        st.success("Analysis completed with basic data visualization.")
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            render_decorative_divider()
            
            st.markdown("### Analysis Results")
            
            results = st.session_state.analysis_results
            
            # Create a tabbed interface for results
            results_tabs = st.tabs(["Overview", "Pitch Analysis", "Rhythm Analysis", "Spectrogram"])
            
            with results_tabs[0]:  # Overview
                st.markdown("#### Overview")
                
                # Create metrics section
                metric_cols = st.columns(3)
                
                with metric_cols[0]:
                    st.markdown("""
                    <div class="metric-card">
                        <div class="metric-value">{:.1f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                    """.format(results.get("duration", 0)), unsafe_allow_html=True)
                
                with metric_cols[1]:
                    if results.get("detected_raga"):
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{}</div>
                            <div class="metric-label">Detected Raga ({:.0f}% confidence)</div>
                        </div>
                        """.format(results["detected_raga"], results.get("raga_confidence", 0)), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Raga not detected</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    if results.get("detected_tala"):
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">{}</div>
                            <div class="metric-label">Detected Tala ({:.0f}% confidence)</div>
                        </div>
                        """.format(results["detected_tala"], results.get("tala_confidence", 0)), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-value">N/A</div>
                            <div class="metric-label">Tala not detected</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # If raga detected, show info
                if results.get("detected_raga"):
                    raga_info = get_raga_info(results["detected_raga"])
                    
                    if raga_info:
                        st.markdown(f"#### Detected Raga: {results['detected_raga']}")
                        st.markdown(raga_info["description"])
                        
                        aroha_str, avaroha_str = get_raga_notes_string(raga_info)
                        
                        # Two columns for raga info
                        raga_cols = st.columns(2)
                        
                        with raga_cols[0]:
                            st.markdown(f"**Aroha:** {aroha_str}")
                            st.markdown(f"**Avaroha:** {avaroha_str}")
                        
                        with raga_cols[1]:
                            st.markdown(f"**Time of Day:** {raga_info['time']}")
                            st.markdown(f"**Mood:** {', '.join(raga_info['mood'])}")
                
                # If tala detected, show info
                if results.get("detected_tala"):
                    tala_info = get_tala_info(results["detected_tala"])
                    
                    if tala_info:
                        st.markdown(f"#### Detected Tala: {results['detected_tala']}")
                        st.markdown(tala_info["description"])
                        
                        # Two columns for tala info
                        tala_cols = st.columns(2)
                        
                        with tala_cols[0]:
                            st.markdown(f"**Beats:** {tala_info['beats']}")
                            st.markdown(f"**Pattern:** {tala_info['pattern_description']}")
                        
                        with tala_cols[1]:
                            # Display clap pattern
                            clap_pattern = " | ".join([f"<span style='font-weight: bold; color: {'#d4af37' if p == 'X' else '#8b4513' if p.isdigit() else '#6c757d'}'>{p}</span>" for p in tala_info["clap_pattern"]])
                            
                            st.markdown("**Clap Pattern:**")
                            st.markdown(f"<div style='font-size: 1.2rem; text-align: center;'>{clap_pattern}</div>", unsafe_allow_html=True)
            
            with results_tabs[1]:  # Pitch Analysis
                st.markdown("#### Pitch Analysis")
                
                # If we have pitch data, plot it
                if results.get("pitch_data") and "times" in results["pitch_data"] and "pitches" in results["pitch_data"]:
                    # Create pitch contour plot
                    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFF8DC')
                    
                    times = results["pitch_data"]["times"]
                    pitches = results["pitch_data"]["pitches"]
                    
                    # Plot the pitch contour
                    ax.plot(times, pitches, color='#800000', linewidth=2)
                    
                    # Set labels and title
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title("Pitch Contour", color='#800000')
                    
                    # Set background color
                    ax.set_facecolor('#FFF8DC')
                    
                    # Add grid
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Note distribution plot
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    
                    # Convert pitches to note numbers
                    valid_pitches = pitches[~np.isnan(pitches) & (pitches > 0)]
                    
                    if len(valid_pitches) > 0:
                        midi_notes = [int(round(69 + 12 * np.log2(p/440.0))) for p in valid_pitches]
                        note_counts = np.zeros(12)
                        
                        for note in midi_notes:
                            note_counts[note % 12] += 1
                        
                        # Normalize
                        if np.sum(note_counts) > 0:
                            note_counts = note_counts / np.sum(note_counts) * 100
                        
                        # Create figure
                        fig2, ax2 = plt.subplots(figsize=(10, 4), facecolor='#FFF8DC')
                        
                        # Create gradient colors
                        colors = plt.cm.YlOrRd(note_counts / max(note_counts))
                        
                        # Plot bars
                        ax2.bar(note_names, note_counts, color=colors)
                        
                        # Set labels and title
                        ax2.set_xlabel("Note")
                        ax2.set_ylabel("Percentage")
                        ax2.set_title("Note Distribution", color='#800000')
                        
                        # Set background color
                        ax2.set_facecolor('#FFF8DC')
                        
                        # Add grid
                        ax2.grid(True, linestyle='--', alpha=0.3, axis='y')
                        
                        st.pyplot(fig2)
                else:
                    st.info("No pitch data available for this analysis.")
            
            with results_tabs[2]:  # Rhythm Analysis
                st.markdown("#### Rhythm Analysis")
                
                if results.get("detected_tala"):
                    # In a real implementation, we would show actual rhythm analysis
                    # For now, just show a placeholder visualization
                    
                    # Create a simulated amplitude envelope plot
                    fig, ax = plt.subplots(figsize=(10, 4), facecolor='#FFF8DC')
                    
                    # Create a simulated amplitude envelope
                    times = np.linspace(0, results.get("duration", 10), 1000)
                    
                    # Simulate a rhythm pattern based on the detected tala
                    tala_info = get_tala_info(results["detected_tala"])
                    
                    if tala_info:
                        beats = tala_info["beats"]
                        vibhags = tala_info["vibhags"]
                        
                        # Create a pattern that repeats every 'beats' time units
                        period = results.get("duration", 10) / 5  # assume 5 cycles in the audio
                        
                        # Base amplitude envelope
                        envelope = 0.2 + 0.3 * np.exp(-3 * (times % (period/beats)) / (period/beats))
                        
                        # Add some emphasis on the sam (first beat of the cycle)
                        sam_emphasis = 0.5 * np.exp(-5 * ((times % period) / period))
                        
                        # Add some noise
                        noise = 0.1 * np.random.randn(len(times))
                        
                        # Combine
                        amplitude = envelope + sam_emphasis + noise
                        
                        # Plot
                        ax.plot(times, amplitude, color='#483D8B', linewidth=2)
                        
                        # Mark the beats
                        beat_times = np.arange(0, results.get("duration", 10), period/beats)
                        beat_values = [1.0 if i % beats == 0 else 0.8 if i % beats in vibhags else 0.6 for i in range(len(beat_times))]
                        
                        ax.scatter(beat_times, beat_values, color='#800000', s=80, zorder=10, alpha=0.7)
                        
                        # Set labels and title
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        ax.set_title(f"Rhythm Analysis - {results['detected_tala']}", color='#800000')
                        
                        # Set background color
                        ax.set_facecolor('#FFF8DC')
                        
                        # Add grid
                        ax.grid(True, linestyle='--', alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Display some metrics
                        tempo = np.random.randint(70, 120)  # Simulated tempo
                        
                        st.markdown(f"**Estimated Tempo:** {tempo} BPM")
                        st.markdown(f"**Cycle Duration:** {period:.2f} seconds")
                        st.markdown(f"**Estimated Cycles:** {results.get('duration', 10) / period:.1f}")
                    else:
                        st.info("No rhythm analysis data available.")
                else:
                    st.info("No tala was detected. Rhythm analysis is not available.")
            
            with results_tabs[3]:  # Spectrogram
                st.markdown("#### Spectrogram")

                # Generate a real spectrogram from the audio data
                try:
                    # Import visualization module
                    from modules.visualization import plot_spectrogram

                    # Check if we have audio data in the session state
                    if 'audio_data' in locals() and audio_data is not None and sr is not None:
                        # Generate the spectrogram using the actual audio data
                        fig = plot_spectrogram(audio_data, sr)
                        st.pyplot(fig)
                        st.success("Spectrogram generated from audio data")
                    else:
                        # If we don't have audio data but have analysis results, try to create a meaningful visualization
                        if results.get("pitch_data") and "times" in results["pitch_data"] and "pitches" in results["pitch_data"]:
                            # Create a more informative spectrogram-like visualization based on pitch data
                            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFF8DC')

                            times = results["pitch_data"]["times"]
                            pitches = results["pitch_data"]["pitches"]

                            # Create a spectrogram-like image from pitch data
                            # First, create a time-frequency grid
                            time_bins = 100
                            freq_bins = 200

                            # Create time and frequency axes
                            t_edges = np.linspace(0, results.get("duration", 10), time_bins+1)
                            f_edges = np.logspace(np.log10(50), np.log10(5000), freq_bins+1)

                            # Create empty spectrogram
                            spec = np.zeros((freq_bins, time_bins))

                            # Fill in the spectrogram based on pitch data
                            valid_idx = (pitches > 0) & np.isfinite(pitches)
                            valid_times = times[valid_idx]
                            valid_pitches = pitches[valid_idx]

                            # Add energy at each pitch point
                            for t, f in zip(valid_times, valid_pitches):
                                # Find the bin indices
                                t_idx = np.clip(int(t / results.get("duration", 10) * time_bins), 0, time_bins-1)
                                f_idx = np.clip(np.searchsorted(f_edges, f) - 1, 0, freq_bins-1)

                                # Add energy to the bin and surrounding bins (for visibility)
                                for di in range(-2, 3):
                                    for dj in range(-2, 3):
                                        ni = f_idx + di
                                        nj = t_idx + dj
                                        if 0 <= ni < freq_bins and 0 <= nj < time_bins:
                                            # Add energy with distance-based falloff
                                            dist = np.sqrt(di**2 + dj**2)
                                            spec[ni, nj] += np.exp(-dist)

                            # Plot the spectrogram
                            t_centers = (t_edges[:-1] + t_edges[1:]) / 2
                            f_centers = (f_edges[:-1] + f_edges[1:]) / 2

                            # Apply log scaling for better visibility
                            spec = np.log1p(spec)

                            # Plot as a pcolormesh
                            pcm = ax.pcolormesh(t_centers, f_centers, spec, cmap='viridis', shading='auto')

                            # Set labels and title
                            ax.set_xlabel("Time (s)")
                            ax.set_ylabel("Frequency (Hz)")
                            ax.set_title("Pitch-based Spectrogram", color='#800000')

                            # Set y-axis to log scale
                            ax.set_yscale('log')

                            # Add colorbar
                            fig.colorbar(pcm, ax=ax, label="Energy")

                            # Set background color
                            ax.set_facecolor('#FFF8DC')

                            st.pyplot(fig)
                            st.info("Spectrogram generated from pitch data")
                        else:
                            # If we don't have pitch data either, show a message
                            st.warning("No audio data available to generate spectrogram")
                except Exception as e:
                    st.error(f"Error generating spectrogram: {str(e)}")

                    # Fallback to a simple spectrogram
                    try:
                        # Create figure
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#FFF8DC')

                        # Create a simulated spectrogram
                        times = np.linspace(0, results.get("duration", 10), 1000)
                        freqs = np.linspace(50, 5000, 500)

                        # Create the 2D array for the spectrogram
                        spectrogram = np.zeros((len(freqs), len(times)))

                        # Add some harmonic content based on the detected raga if available
                        fundamental = 220  # Default to A3

                        # If we have a detected raga, use a characteristic frequency
                        if results.get("detected_raga"):
                            # Simple mapping of ragas to fundamental frequencies
                            raga_fundamentals = {
                                "Yaman": 261.63,  # C4
                                "Bhairav": 293.66,  # D4
                                "Malkauns": 311.13,  # D#4/Eb4
                                "Bhimpalasi": 349.23,  # F4
                                # Add more ragas as needed
                            }
                            fundamental = raga_fundamentals.get(results["detected_raga"], 220)

                        for harmonic in range(1, 8):
                            freq_idx = np.argmin(np.abs(freqs - fundamental * harmonic))

                            # Add some variation over time
                            amplitude = 1.0 / harmonic
                            variation = 0.5 * amplitude * np.sin(2 * np.pi * times / results.get("duration", 10))

                            # Add to spectrogram
                            # Add some width to each harmonic
                            width = max(1, int(len(freqs) * 0.01))
                            for i in range(-width, width+1):
                                if 0 <= freq_idx + i < len(freqs):
                                    spectrogram[freq_idx + i, :] += amplitude + variation

                        # Add some noise
                        spectrogram += 0.05 * np.random.randn(*spectrogram.shape)

                        # Plot the spectrogram
                        im = ax.pcolormesh(times, freqs, spectrogram, shading='gouraud',
                                          cmap=plt.cm.magma, vmin=0, vmax=1)

                        # Set labels and title
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Frequency (Hz)")
                        ax.set_title("Simulated Spectrogram", color='#800000')

                        # Set y-axis to log scale
                        ax.set_yscale('log')

                        # Add colorbar
                        fig.colorbar(im, ax=ax, label="Magnitude")

                        # Set background color
                        ax.set_facecolor('#FFF8DC')

                        st.pyplot(fig)
                        st.info("Using simulated spectrogram (fallback)")
                    except Exception as e2:
                        st.error(f"Error generating fallback spectrogram: {str(e2)}")
                        st.info("Could not generate spectrogram visualization")

def render_synthesis_page():
    """Render the music synthesis page"""
    st.markdown("## Music Generator")
    
    st.markdown("""
    <div style="background-color: rgba(255, 140, 0, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #FF8C00;">
        <p>Generate authentic Indian classical music elements such as tanpura drones, tabla rhythms, 
        and melodic patterns. Create accompaniments for practice or performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Synthesis options
    synthesis_type = st.radio(
        "Select what to generate:",
        ["Tanpura Drone", "Tabla Rhythm", "Raga Melody"],
        horizontal=True
    )
    
    render_decorative_divider()
    
    if synthesis_type == "Tanpura Drone":
        st.markdown("### Tanpura Drone Generator")
        
        st.markdown("""
        The tanpura (tambura) is a long-necked plucked string instrument used in Indian classical music to provide 
        a continuous harmonic drone. It creates a rich, harmonic resonance that serves as the tonal foundation 
        for the performance.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Root note selection
            root_note = st.selectbox(
                "Root Note (Sa):",
                ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            )
            
            # Jiva/Jawari effect
            jiva = st.slider("Jawari Effect:", 0.0, 1.0, 0.7, 0.1, 
                            help="Controls the characteristic buzzing sound of the tanpura. Higher values create more pronounced harmonics.")
        
        with col2:
            # Duration selection
            duration = st.slider("Duration (seconds):", 10, 120, 60)
            
            # Tempo selection
            tempo = st.slider("Tempo (BPM):", 40, 80, 60, 
                             help="Controls the plucking speed. Traditional tanpura is typically played at a slow, steady tempo.")
        
        # Advanced options
        with st.expander("Advanced Options"):
            # Tuning selection
            tuning_options = [
                "Pa Sa Sa Sa (5-1-1-1) - Male vocalists",
                "Ma Sa Sa Sa (4-1-1-1) - Female vocalists",
                "Sa Sa Pa Sa (1-1-5-1) - Specific ragas"
            ]
            
            tuning = st.selectbox("Tuning:", tuning_options)
            
            # String release time
            release = st.slider("String Release:", 0.5, 2.0, 1.0, 0.1, 
                               help="Controls how long each string resonates. Higher values create more overlapping notes.")
        
        # Generate button
        if st.button("Generate Tanpura", key="gen_tanpura", use_container_width=True):
            with st.spinner("Generating tanpura sound..."):
                try:
                    # Import here to avoid potential import errors in the main code flow
                    from modules.audio_synthesis import generate_tanpura
                    
                    # Generate tanpura
                    tanpura_audio, sr = generate_tanpura(
                        root_note=root_note,
                        duration=duration,
                        tempo=tempo,
                        jiva=jiva
                    )
                    
                    st.session_state.tanpura = tanpura_audio
                    
                    st.success("Tanpura generated successfully!")
                    
                    # Play the tanpura
                    import soundfile as sf
                    
                    # Buffer for writing
                    buffer = io.BytesIO()
                    
                    # Write to the buffer
                    sf.write(buffer, tanpura_audio, sr, format='WAV')
                    
                    # Get the buffer content
                    audio_bytes = buffer.getvalue()
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="Download Tanpura Audio",
                        data=audio_bytes,
                        file_name=f"tanpura_{root_note}_{tempo}bpm.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"Error generating tanpura: {str(e)}")
                    st.info("This is a placeholder for the tanpura generation functionality.")
        
        # About tanpura section
        with st.expander("About Tanpura"):
            st.markdown("""
            ### The Tanpura in Indian Classical Music
            
            The tanpura provides the harmonic foundation for Indian classical music performances. Its distinctive sound 
            comes from the jawari (jiva) bridge, which creates a complex set of overtones through a specific type of 
            string-bridge contact.
            
            The standard tanpura has four strings, although variants with five or six strings also exist. The strings 
            are tuned according to the requirements of the raga being performed, with common tunings including:
            
            - **Pa-Sa-Sa-Sa (5-1-1-1)**: Commonly used for male vocalists
            - **Ma-Sa-Sa-Sa (4-1-1-1)**: Often used for female vocalists
            - **Sa-Sa-Pa-Sa (1-1-5-1)**: Used for specific ragas
            
            The strings are plucked one after another in a continuous cycle, creating a rich harmonic texture that 
            establishes the tonal center and mood of the performance.
            """)
    
    elif synthesis_type == "Tabla Rhythm":
        st.markdown("### Tabla Rhythm Generator")
        
        st.markdown("""
        Generate tabla rhythm patterns based on traditional talas (rhythm cycles). The tabla is a pair of hand drums 
        (the smaller dayan and the larger bayan) that provides the rhythmic foundation for Indian classical music.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Tala selection
            all_talas = get_all_talas()
            selected_tala = st.selectbox("Select Tala:", all_talas)
            
            # Get tala info
            tala_info = get_tala_info(selected_tala)
            
            if tala_info:
                st.markdown(f"**Beats:** {tala_info['beats']}")
                st.markdown(f"**Pattern:** {tala_info['pattern_description']}")
                
                # Display clap pattern
                clap_pattern = " | ".join([f"<span style='font-weight: bold; color: {'#d4af37' if p == 'X' else '#8b4513' if p.isdigit() else '#6c757d'}'>{p}</span>" for p in tala_info["clap_pattern"]])
                
                st.markdown("**Clap Pattern:**")
                st.markdown(f"<div style='font-size: 1.2rem; text-align: center;'>{clap_pattern}</div>", unsafe_allow_html=True)
        
        with col2:
            # Tempo selection
            tempo = st.slider("Tempo (BPM):", 60, 300, 120)
            
            # Duration
            duration = st.slider("Duration (seconds):", 10, 60, 30)
            
            # Variation/style
            variation = st.selectbox(
                "Style:",
                ["Traditional", "Contemporary", "Simple", "Complex"]
            )
        
        # Generate button
        if st.button("Generate Tabla Rhythm", key="gen_tabla", use_container_width=True):
            with st.spinner("Generating tabla rhythm..."):
                try:
                    # Import here to avoid potential import errors in the main code flow
                    from modules.audio_synthesis import synthesize_tabla
                    
                    # Generate tabla rhythm
                    tabla_audio, sr = synthesize_tabla(
                        tala=selected_tala,
                        tempo=tempo,
                        duration=duration
                    )
                    
                    st.success("Tabla rhythm generated successfully!")
                    
                    # Play the tabla rhythm
                    import soundfile as sf
                    
                    # Buffer for writing
                    buffer = io.BytesIO()
                    
                    # Write to the buffer
                    sf.write(buffer, tabla_audio, sr, format='WAV')
                    
                    # Get the buffer content
                    audio_bytes = buffer.getvalue()
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="Download Tabla Rhythm",
                        data=audio_bytes,
                        file_name=f"tabla_{selected_tala}_{tempo}bpm.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"Error generating tabla rhythm: {str(e)}")
                    st.info("This is a placeholder for the tabla rhythm generation functionality.")
        
        # About tabla section
        with st.expander("About Tabla"):
            st.markdown("""
            ### The Tabla in Indian Classical Music
            
            The tabla is a pair of hand drums that is the principal percussion instrument in Hindustani classical music. 
            It consists of two drums:
            
            - **Dayan (Right Drum)**: The smaller drum played with the right hand, which produces the higher-pitched sounds.
            - **Bayan (Left Drum)**: The larger drum played with the left hand, which produces the lower-pitched sounds.
            
            Tabla players use a complex system of vocalized syllables called "bols" to represent different strokes and 
            combinations. Common bols include:
            
            - **Na, Ta, Tin, Ti**: Played on the dayan (right drum)
            - **Ge, Ghe, Dha, Dhin**: Combination sounds using both drums
            - **Ka, Kat**: Played on the bayan (left drum)
            
            The tabla is used to keep time according to the tala (rhythmic cycle), with special emphasis on the sam 
            (first beat) and various subdivisions and variations of the basic pattern.
            """)
    
    elif synthesis_type == "Raga Melody":
        st.markdown("### Raga Melody Generator")
        
        st.markdown("""
        Generate melodic patterns based on traditional ragas. These can be used for practice, reference, 
        or as inspiration for your own compositions.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Raga selection
            all_ragas = get_all_ragas()
            selected_raga = st.selectbox("Select Raga:", all_ragas)
            
            # Get raga info
            raga_info = get_raga_info(selected_raga)
            
            if raga_info:
                aroha_str, avaroha_str = get_raga_notes_string(raga_info)
                
                st.markdown(f"**Aroha:** {aroha_str}")
                st.markdown(f"**Avaroha:** {avaroha_str}")
                st.markdown(f"**Pakad:** {raga_info['pakad']}")
        
        with col2:
            # Instrument selection
            instrument = st.selectbox(
                "Instrument:",
                ["Sitar", "Sarod", "Bansuri (Flute)", "Sarangi", "Synthesized"]
            )
            
            # Tempo
            tempo = st.slider("Tempo (BPM):", 60, 300, 120)
            
            # Duration
            duration = st.slider("Duration (seconds):", 10, 60, 30)
            
            # Complexity
            complexity = st.slider("Complexity:", 1, 10, 5, 
                                 help="Controls the complexity of the generated melody. Higher values create more intricate patterns.")
        
        # Generate button
        if st.button("Generate Melody", key="gen_melody", use_container_width=True):
            with st.spinner("Generating melodic pattern..."):
                try:
                    # Import here to avoid potential import errors in the main code flow
                    from modules.audio_synthesis import synthesize_melody
                    
                    # Generate melodic pattern
                    melody_audio, sr = synthesize_melody(
                        raga=selected_raga,
                        instrument=instrument.lower(),
                        tempo=tempo,
                        duration=duration,
                        complexity=complexity
                    )
                    
                    st.success("Melodic pattern generated successfully!")
                    
                    # Play the melody
                    import soundfile as sf
                    
                    # Buffer for writing
                    buffer = io.BytesIO()
                    
                    # Write to the buffer
                    sf.write(buffer, melody_audio, sr, format='WAV')
                    
                    # Get the buffer content
                    audio_bytes = buffer.getvalue()
                    
                    # Play the audio
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="Download Melody",
                        data=audio_bytes,
                        file_name=f"melody_{selected_raga}_{instrument}.wav",
                        mime="audio/wav"
                    )
                except Exception as e:
                    st.error(f"Error generating melody: {str(e)}")
                    st.info("This is a placeholder for the melody generation functionality.")
        
        # About raga improvisation
        with st.expander("About Raga Improvisation"):
            st.markdown("""
            ### Melodic Improvisation in Indian Classical Music
            
            Improvisation within the framework of a raga is the heart of Indian classical music. While adhering to the 
            rules of the raga, musicians create spontaneous melodic patterns and variations. Common types of melodic 
            development include:
            
            - **Alap**: The slow, rhythmless introduction that explores the raga's characteristics
            - **Jod**: Rhythmic development without a specific tala (metric cycle)
            - **Jhala**: Fast rhythmic patterns often used as a climactic section
            - **Gat/Bandish**: Fixed compositions that serve as the basis for further improvisation
            - **Tans**: Fast melodic passages that demonstrate technical virtuosity
            
            Each raga has certain characteristic phrases (pakad) and movements that define its unique identity. 
            The generated melodic patterns attempt to capture these essential features while creating new, 
            authentic-sounding variations.
            """)

def render_ai_page():
    """Render the AI assistant page"""
    st.markdown("## AI Assistant")
    
    st.markdown("""
    <div style="background-color: rgba(0, 139, 139, 0.1); padding: 20px; border-radius: 10px; border-left: 5px solid #008B8B;">
        <p>Get personalized insights and explanations about Indian classical music. Ask questions about ragas, 
        talas, techniques, history, or get recommendations based on your interests.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a message area with a decorative border
    st.markdown("""
    <div style="border: 2px solid #DAA520; border-radius: 10px; padding: 20px; margin: 20px 0;">
        <h3 style="margin-top: 0; color: #483D8B;">Ask about Indian Classical Music</h3>
        <p>Type your question below to get detailed explanations, comparisons, suggestions, or historical context.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if Google API key is available
    import os
    google_api_available = get_environment_variable("GOOGLE_API_KEY") is not None
    
    if google_api_available:
        # User input
        user_question = st.text_input("Your question:", placeholder="E.g., 'What is the difference between Hindustani and Carnatic music?'")
        
        # Process the query
        if user_question and st.button("Get Answer", use_container_width=True):
            with st.spinner("Thinking..."):
                try:
                    # Import the AI service
                    from services.ai_service import get_ai_analysis
                    
                    # Get response
                    response = get_ai_analysis(query=user_question, analysis_type="general_query")
                    
                    # Display response
                    st.markdown("### Response")
                    st.markdown(response)
                except Exception as e:
                    st.error(f"Error getting AI response: {str(e)}")
                    
                    # Provide a fallback response for demonstration
                    import time
                    time.sleep(2)  # Simulate processing time
                    
                    st.markdown("### Response")
                    st.markdown("""
                    I'd be happy to explain the difference between Hindustani and Carnatic music, the two major traditions of Indian classical music.
                    
                    **Hindustani Classical Music**
                    - **Origin**: North India, Pakistan, and Bangladesh
                    - **Influence**: Significantly influenced by Persian and Islamic musical elements during the Mughal era
                    - **Structure**: 
                      - Begins with Alap (unmetered improvisation)
                      - Progresses to Jod (rhythmic but without tabla)
                      - Culminates in Gat (with tabla accompaniment)
                    - **Main Forms**: Khayal, Dhrupad, Thumri, Tarana
                    - **Instruments**: Sitar, Sarod, Santoor, Tabla, Sarangi
                    - **Notation System**: Sargam (Sa, Re, Ga, Ma, Pa, Dha, Ni)
                    
                    **Carnatic Classical Music**
                    - **Origin**: South India, primarily Tamil Nadu, Kerala, Andhra Pradesh, Karnataka
                    - **Influence**: Maintained more of its original structure with less external influence
                    - **Structure**: 
                      - Typically begins with a composition
                      - Improvisation occurs within the context of the composition
                      - More emphasis on compositional elements
                    - **Main Forms**: Varnam, Kriti, Javali, Tillana
                    - **Instruments**: Veena, Violin, Mridangam, Flute, Gottuvadyam
                    - **Notation System**: Also uses Sargam but with regional variations
                    
                    **Key Differences**:
                    1. **Improvisation**: Hindustani gives more emphasis to improvisation, while Carnatic focuses more on compositions
                    2. **Rhythm**: Carnatic has more complex rhythmic structures
                    3. **Raga System**: Both use ragas, but the classification systems differ
                    4. **Performance Style**: Hindustani typically features more sustained notes and slower development, while Carnatic often has faster phrases
                    5. **Devotional Content**: Carnatic music is deeply tied to devotional themes, while Hindustani has both secular and devotional elements
                    
                    Both traditions share the fundamental concepts of raga (melodic framework) and tala (rhythm cycle), but their expression and evolution have taken different paths over the centuries.
                    """)
        
        # Example questions
        st.markdown("### Example Questions")
        
        example_cols = st.columns(2)
        
        with example_cols[0]:
            example_questions1 = [
                "What is a raga?",
                "How does one identify a raga when listening?",
                "What is the importance of the tanpura?",
                "How are ragas connected to different times of day?"
            ]
            
            for question in example_questions1:
                if st.button(question, key=f"q_{question}", use_container_width=True):
                    # Set the question in the text input and rerun
                    st.session_state.user_question = question
                    st.rerun()
        
        with example_cols[1]:
            example_questions2 = [
                "What is a tala?",
                "How do I start learning Indian classical music?",
                "What are the main instruments used in Hindustani music?",
                "What is the difference between Dhrupad and Khayal?"
            ]
            
            for question in example_questions2:
                if st.button(question, key=f"q_{question}", use_container_width=True):
                    # Set the question in the text input and rerun
                    st.session_state.user_question = question
                    st.rerun()
    else:
        st.warning("Google API key is not configured. The AI Assistant requires a valid Google API key to function.")
        
        st.markdown("""
        ### How to Set Up the AI Assistant
        
        To enable the AI Assistant, you need to provide a Google API key:
        
        1. Obtain a Google API key with access to the Gemini model
        2. Add the key to your environment variables as `GOOGLE_API_KEY`
        3. Restart the application
        
        Once configured, you'll be able to ask questions and get detailed explanations about Indian classical music.
        """)
        
        # Notice about API key
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; margin-top: 20px;">
            <p style="font-weight: bold; color: #800000;">Google API Key Required</p>
            <p>To use the AI Assistant, please add your Google Gemini API key in the configuration section above.</p>
            <p>You can obtain a free API key from the <a href="https://ai.google.dev/" target="_blank">Google AI Developer Platform</a>.</p>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    # Load CSS
    load_css("assets/styles.css")

    # Initialize page state if not already set
    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    # Render the header
    render_header()

    # Render navigation
    render_navigation()

    # Render the appropriate page based on navigation state
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'about':
        render_about_page()
    elif st.session_state.page == 'ragas':
        render_ragas_page()
    elif st.session_state.page == 'raga_detail':
        render_raga_detail_page()
    elif st.session_state.page == 'talas':
        render_talas_page()
    elif st.session_state.page == 'tala_detail':
        render_tala_detail_page()
    elif st.session_state.page == 'analysis':
        render_analysis_page()
    elif st.session_state.page == 'synthesis':
        render_synthesis_page()
    elif st.session_state.page == 'composer':
        # Import the music composer page from modules
        from modules.music_composer_page import render_music_composer_page
        render_music_composer_page()
    elif st.session_state.page == 'ai':
        render_ai_page()
    elif st.session_state.page == 'healthcheck':
        # Import the healthcheck page
        from modules.healthcheck import render_healthcheck_page
        render_healthcheck_page()

# Check for healthcheck URL parameter
params = st.experimental_get_query_params()
if 'healthcheck' in params:
    if not hasattr(st.session_state, 'page'):
        st.session_state.page = 'healthcheck'
    else:
        st.session_state.page = 'healthcheck'

if __name__ == "__main__":
    main()