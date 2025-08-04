import sounddevice as sd
import numpy as np
import whisper

#chose base for better accuracy
model = whisper.load_model("base")

'''
Function to record audio from the microphone and return it as a numpy array.
Args:
    duration (int): Duration of the recording in seconds. Default is 5 seconds.
    sr (int): Sample rate for the recording. Default is 16000 Hz.
Returns:
    np.ndarray: Recorded audio as a numpy array.
'''
def record_audio(duration=5, sr = 16000):
    print("Speak now...")
    try:
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        return audio.squeeze()
    except Exception as e:
        print(f"An error occurred while recording audio: {e}")
        return None
    
'''
Function to transcribe audio using the Whisper model.
Args:
    audio_array (np.ndarray): Audio data as a numpy array.
'''
def transcribe_audio(audio_array: np.ndarray):
    if audio_array is None:
        return "No audio captured."
    print("Transcribing audio...")
    return model.transcribe(audio_array, fp16=False)["text"]