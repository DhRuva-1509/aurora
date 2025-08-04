from src.whisper import record_audio, transcribe_audio

if __name__ == "__main__":
    audio = record_audio()
    result = transcribe_audio(audio)
    print("Transcription Result:", result)