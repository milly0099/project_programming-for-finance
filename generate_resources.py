import os
import shutil
from utils.generate_audio import generate_all_sounds

def main():
    """Script to generate all resource files for the application"""
    print("Generating audio resources...")
    
    # Create audio directory if it doesn't exist
    os.makedirs('assets/audio', exist_ok=True)
    
    # Generate all sound effects
    try:
        generate_all_sounds()
        print("Audio files generated successfully!")
    except Exception as e:
        print(f"Error generating audio files: {e}")
        
        # Create blank mp3 files as fallback
        print("Creating placeholder audio files...")
        for sound_type in ['welcome', 'zombie', 'futuristic', 'got', 'gaming', 'model_success']:
            mp3_path = f'assets/audio/{sound_type}.mp3'
            with open(mp3_path, 'wb') as f:
                # MP3 header - very minimal but makes it a valid file
                f.write(b'\xFF\xFB\x90\x04\x00')
            print(f"Created empty placeholder for {mp3_path}")

if __name__ == "__main__":
    main()