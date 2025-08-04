from src.whisper import record_audio, transcribe_audio
from src.sentiment import predict_sentiment

if __name__ == "__main__":
    audio = record_audio()
    result = transcribe_audio(audio)
    print("Transcription Result:", result)
    #text = "I am depressed and sad."
    sentiment = predict_sentiment(result)
    print("Sentiment Result:", sentiment)