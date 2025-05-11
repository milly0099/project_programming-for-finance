import os

def create_empty_mp3s():
    """Create minimal MP3 files for each theme sound"""
    # Create audio directory
    os.makedirs('assets/audio', exist_ok=True)
    
    # List of sound types
    sound_types = ['welcome', 'zombie', 'futuristic', 'got', 'gaming', 'model_success']
    
    # Create minimal MP3 files
    for sound_type in sound_types:
        mp3_path = f'assets/audio/{sound_type}.mp3'
        with open(mp3_path, 'wb') as f:
            # MP3 header - very minimal
            f.write(b'\xFF\xFB\x90\x04\x00')
        print(f"Created placeholder for {mp3_path}")

if __name__ == "__main__":
    create_empty_mp3s()