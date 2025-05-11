import streamlit as st
import base64
from pathlib import Path
import os

# Make sure the audio directory exists
def ensure_audio_dir():
    os.makedirs('assets/audio', exist_ok=True)

# Function to play audio in Streamlit
def autoplay_audio(file_path):
    """
    Autoplay audio file in Streamlit
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    """
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

# Function to play audio with control
def play_audio(file_path, autoplay=False):
    """
    Play audio file in Streamlit with controls
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    autoplay : bool
        Whether to autoplay the audio
    """
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        if autoplay:
            md = f"""
                <audio controls autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
        else:
            md = f"""
                <audio controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
        st.markdown(md, unsafe_allow_html=True)

# Create sound effects for different themes
def create_sound_bytes():
    """
    Create base64 encoded sound bytes to use in the app
    """
    # Make sure the directory exists
    ensure_audio_dir()
    
    # Create theme-specific audio files if they don't exist
    # (In a real app, you would include actual audio files instead of these placeholders)
    
    # Dictionary to store sound file paths for each theme
    sound_files = {
        "welcome": "assets/audio/welcome.mp3",
        "zombie": "assets/audio/zombie.mp3",
        "futuristic": "assets/audio/futuristic.mp3",
        "got": "assets/audio/got.mp3",
        "gaming": "assets/audio/gaming.mp3",
        "model_success": "assets/audio/model_success.mp3"
    }
    
    return sound_files

# Initialize session state for audio
def init_audio_state():
    """
    Initialize session state for audio control
    """
    if 'sound_enabled' not in st.session_state:
        st.session_state.sound_enabled = True
    if 'current_theme_sound' not in st.session_state:
        st.session_state.current_theme_sound = None
    if 'sound_files' not in st.session_state:
        st.session_state.sound_files = create_sound_bytes()

# Play theme transition sound
def play_theme_sound(theme_name):
    """
    Play sound when switching to a specific theme
    
    Parameters:
    -----------
    theme_name : str
        Name of the theme
    """
    if not st.session_state.sound_enabled:
        return
    
    # Only play if we're changing to a new theme
    if st.session_state.current_theme_sound != theme_name:
        st.session_state.current_theme_sound = theme_name
        
        # Play the appropriate sound for the theme
        sound_file = st.session_state.sound_files.get(theme_name)
        if sound_file and Path(sound_file).exists():
            autoplay_audio(sound_file)