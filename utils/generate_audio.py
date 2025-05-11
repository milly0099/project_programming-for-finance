import wave
import struct
import math
import os
import numpy as np

def ensure_audio_dir():
    """Make sure the audio directory exists"""
    os.makedirs('assets/audio', exist_ok=True)

def generate_sine_wave(frequency, duration, volume=1.0, sample_rate=44100):
    """
    Generate sine wave audio data
    
    Parameters:
    -----------
    frequency : float
        Frequency of the sine wave in Hz
    duration : float
        Duration of the audio in seconds
    volume : float
        Volume of the audio (0.0 to 1.0)
    sample_rate : int
        Sample rate of the audio in Hz
        
    Returns:
    --------
    list
        List of audio samples
    """
    num_samples = int(duration * sample_rate)
    samples = []
    
    for i in range(num_samples):
        sample = volume * math.sin(2 * math.pi * frequency * i / sample_rate)
        samples.append(sample)
        
    return samples

def create_wave_file(filename, samples, sample_rate=44100):
    """
    Create a wave file from audio samples
    
    Parameters:
    -----------
    filename : str
        Output filename
    samples : list
        List of audio samples
    sample_rate : int
        Sample rate of the audio in Hz
    """
    # Scale samples to 16-bit range
    scaled_samples = [int(sample * 32767) for sample in samples]
    
    # Create wave file
    with wave.open(filename, 'w') as wave_file:
        # Set parameters
        wave_file.setparams((1, 2, sample_rate, len(samples), 'NONE', 'not compressed'))
        
        # Write samples
        for sample in scaled_samples:
            wave_file.writeframes(struct.pack('h', sample))

def generate_welcome_sound():
    """Generate welcome sound (positive major chord)"""
    ensure_audio_dir()
    filename = 'assets/audio/welcome.wav'
    
    # Major chord frequencies (C, E, G)
    frequencies = [261.63, 329.63, 392.00]
    
    # Generate samples
    all_samples = []
    for freq in frequencies:
        samples = generate_sine_wave(freq, 1.5, 0.3)
        all_samples.append(samples)
        
    # Mix samples
    mixed_samples = [sum(sample) / len(frequencies) for sample in zip(*all_samples)]
    
    # Apply fade-in and fade-out
    for i in range(int(0.1 * 44100)):
        mixed_samples[i] *= (i / (0.1 * 44100))
        mixed_samples[-(i+1)] *= (i / (0.1 * 44100))
    
    # Create wave file
    create_wave_file(filename, mixed_samples)
    return filename

def generate_zombie_sound():
    """Generate zombie theme sound (dissonant, eerie)"""
    ensure_audio_dir()
    filename = 'assets/audio/zombie.wav'
    
    # Dissonant frequencies
    frequencies = [100, 103, 220, 223]
    
    # Generate samples with tremolo effect
    all_samples = []
    for freq in frequencies:
        base_samples = generate_sine_wave(freq, 2.0, 0.2)
        
        # Add tremolo effect
        tremolo_samples = []
        for i, sample in enumerate(base_samples):
            tremolo = 0.5 + 0.5 * math.sin(2 * math.pi * 5 * i / 44100)
            tremolo_samples.append(sample * tremolo)
        
        all_samples.append(tremolo_samples)
        
    # Mix samples
    mixed_samples = [sum(sample) / len(frequencies) for sample in zip(*all_samples)]
    
    # Apply fade-in and fade-out
    for i in range(int(0.2 * 44100)):
        mixed_samples[i] *= (i / (0.2 * 44100))
        mixed_samples[-(i+1)] *= (i / (0.2 * 44100))
    
    # Create wave file
    create_wave_file(filename, mixed_samples)
    return filename

def generate_futuristic_sound():
    """Generate futuristic theme sound (electronic, sci-fi)"""
    ensure_audio_dir()
    filename = 'assets/audio/futuristic.wav'
    
    sample_rate = 44100
    duration = 2.0
    num_samples = int(duration * sample_rate)
    
    # Start with a sine sweep from low to high
    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        freq = 200 + 2000 * t / duration
        sample = 0.5 * math.sin(2 * math.pi * freq * t)
        samples.append(sample)
    
    # Add some filtered noise
    for i in range(num_samples):
        t = i / sample_rate
        noise = 0.1 * (np.random.random() - 0.5)
        # Filter the noise based on time
        if t < duration / 2:
            noise *= t / (duration / 2)
        else:
            noise *= (duration - t) / (duration / 2)
        samples[i] += noise
    
    # Apply amplitude envelope
    for i in range(num_samples):
        t = i / sample_rate
        # ADSR envelope (Attack, Decay, Sustain, Release)
        if t < 0.1:  # Attack
            envelope = t / 0.1
        elif t < 0.3:  # Decay
            envelope = 1.0 - 0.3 * (t - 0.1) / 0.2
        elif t < duration - 0.5:  # Sustain
            envelope = 0.7
        else:  # Release
            envelope = 0.7 * (duration - t) / 0.5
        samples[i] *= envelope
    
    # Create wave file
    create_wave_file(filename, samples)
    return filename

def generate_got_sound():
    """Generate Game of Thrones theme sound (medieval, fantasy)"""
    ensure_audio_dir()
    filename = 'assets/audio/got.wav'
    
    # Medieval sounding scale (D minor)
    frequencies = [293.66, 349.23, 392.00, 440.00, 493.88]
    
    all_samples = []
    for i, freq in enumerate(frequencies):
        # Each note plays for 0.3 seconds with a slight delay
        start_time = i * 0.2
        end_time = start_time + 0.4
        
        note_samples = [0] * int(start_time * 44100)
        note_samples.extend(generate_sine_wave(freq, 0.4, 0.3))
        note_samples.extend([0] * int((2.0 - end_time) * 44100))
        all_samples.append(note_samples[:int(2.0 * 44100)])  # Ensure all are same length
    
    # Mix samples
    mixed_samples = [sum(sample) / len(frequencies) for sample in zip(*all_samples)]
    
    # Add some reverb effect (simple approximation)
    reverb_samples = mixed_samples.copy()
    reverb_delay = int(0.1 * 44100)  # 100ms delay
    reverb_decay = 0.5
    
    for i in range(reverb_delay, len(mixed_samples)):
        reverb_samples[i] += mixed_samples[i - reverb_delay] * reverb_decay
    
    # Normalize to prevent clipping
    max_val = max(abs(min(reverb_samples)), abs(max(reverb_samples)))
    normalized_samples = [sample / max_val * 0.9 for sample in reverb_samples]
    
    # Create wave file
    create_wave_file(filename, normalized_samples)
    return filename

def generate_gaming_sound():
    """Generate gaming theme sound (8-bit, retro)"""
    ensure_audio_dir()
    filename = 'assets/audio/gaming.wav'
    
    sample_rate = 44100
    duration = 1.5
    num_samples = int(duration * sample_rate)
    
    # Generate 8-bit style melody (simple upward arpeggio)
    notes = [261.63, 329.63, 392.00, 523.25]  # C, E, G, C (octave up)
    note_duration = duration / len(notes)
    
    samples = []
    for note in notes:
        # Square wave (very 8-bit sounding)
        for i in range(int(note_duration * sample_rate)):
            t = i / sample_rate
            # Square wave
            if math.sin(2 * math.pi * note * t) >= 0:
                sample = 0.3
            else:
                sample = -0.3
            samples.append(sample)
    
    # Add bit-crushing effect (reduce effective bit depth)
    bit_depth = 4  # Effectively 4-bit sound
    levels = 2 ** bit_depth
    for i in range(len(samples)):
        samples[i] = round(samples[i] * levels) / levels
    
    # Apply simple envelope
    attack = int(0.01 * sample_rate)
    release = int(0.05 * sample_rate)
    
    # For each note
    for i in range(len(notes)):
        start = int(i * note_duration * sample_rate)
        end = int((i + 1) * note_duration * sample_rate)
        
        # Attack
        for j in range(min(attack, end - start)):
            samples[start + j] *= j / attack
            
        # Release
        for j in range(min(release, end - start)):
            if start + end - start - j < len(samples):
                samples[start + end - start - j - 1] *= j / release
    
    # Create wave file
    create_wave_file(filename, samples)
    return filename

def generate_model_success_sound():
    """Generate sound for successful model training"""
    ensure_audio_dir()
    filename = 'assets/audio/model_success.wav'
    
    # Success chord (major chord with high note)
    frequencies = [523.25, 659.26, 783.99, 1046.50]  # C, E, G, C (high)
    
    # Generate samples
    all_samples = []
    for freq in frequencies:
        samples = generate_sine_wave(freq, 1.0, 0.2)
        all_samples.append(samples)
        
    # Mix samples
    mixed_samples = [sum(sample) / len(frequencies) for sample in zip(*all_samples)]
    
    # Apply fade-in and fade-out
    for i in range(int(0.05 * 44100)):
        mixed_samples[i] *= (i / (0.05 * 44100))
    
    for i in range(int(0.2 * 44100)):
        idx = len(mixed_samples) - i - 1
        if idx < len(mixed_samples):
            mixed_samples[idx] *= (i / (0.2 * 44100))
    
    # Create wave file
    create_wave_file(filename, mixed_samples)
    return filename

def convert_wav_to_mp3():
    """
    Convert generated WAV files to MP3 format
    This is a placeholder function - in a real application, you would use
    a library like pydub to convert WAV to MP3, or directly create MP3 files
    
    For this demo, we'll just print information and keep the WAV files as they are
    """
    print("In a production environment, the WAV files would be converted to MP3 for better compatibility.")
    print("For the purpose of this demo, we'll use the WAV files as they are created.")
    
    # Rename .wav to .mp3 just to match the expected file paths
    for sound_type in ['welcome', 'zombie', 'futuristic', 'got', 'gaming', 'model_success']:
        wav_path = f'assets/audio/{sound_type}.wav'
        mp3_path = f'assets/audio/{sound_type}.mp3'
        if os.path.exists(wav_path):
            # In a real app, convert wav to mp3. Here we just copy to mp3
            with open(wav_path, 'rb') as wav_file:
                with open(mp3_path, 'wb') as mp3_file:
                    mp3_file.write(wav_file.read())
            print(f"Created {mp3_path}")

def generate_all_sounds():
    """Generate all sound effects for the application"""
    generate_welcome_sound()
    generate_zombie_sound()
    generate_futuristic_sound()
    generate_got_sound()
    generate_gaming_sound()
    generate_model_success_sound()
    convert_wav_to_mp3()
    
if __name__ == "__main__":
    generate_all_sounds()