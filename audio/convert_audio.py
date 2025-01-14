import argparse
from pydub import AudioSegment
import os

def convert_to_whisper_wav(input_file, output_file):
    """
    Converts an audio file (MP3/WAV) to WAV format compatible with whisper.cpp.
    The output WAV will have 16 kHz sample rate, mono channel, and PCM S16LE encoding.
    
    Args:
        input_file (str): Path to the input audio file (MP3/WAV).
        output_file (str): Path to save the converted WAV file.
    """
    # Check file format
    file_ext = os.path.splitext(input_file)[-1].lower()
    if file_ext not in ['.mp3', '.wav']:
        raise ValueError("Only MP3 and WAV files are supported.")
    
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Convert to the required format (16 kHz, mono, PCM S16LE)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # 16-bit PCM (2 bytes per sample)
    
    # Export as WAV
    audio.export(output_file, format='wav')
    print(f"Converted file saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert an MP3/WAV audio file to a whisper-compatible WAV format.")
    
    parser.add_argument("input_file", type=str, help="Path to the input audio file (MP3 or WAV).")
    parser.add_argument("output_file", type=str, help="Path to save the converted WAV file.")
    
    args = parser.parse_args()
    
    try:
        convert_to_whisper_wav(args.input_file, args.output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
