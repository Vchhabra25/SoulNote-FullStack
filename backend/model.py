import os
from groq import Groq
from transformers import pipeline

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

# Emotion model (loaded only once)
emotion_classifier = None


def get_emotion_classifier():
    global emotion_classifier

    if emotion_classifier is None:
        emotion_classifier = pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )

    return emotion_classifier


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:

        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="text"
        )

    return transcription


def detect_emotion(text):

    result = get_emotion_classifier()(text)

    emotion = result[0][0]["label"]
    confidence = result[0][0]["score"]

    return {
        "emotion": emotion,
        "confidence": round(confidence * 100, 2)
    }


def get_recommendation(emotion):

    recommendations = {
        "joy": "Keep doing activities that make you happy.",
        "sadness": "Consider talking to a trusted friend or taking a short walk.",
        "fear": "Try deep breathing and grounding exercises.",
        "anger": "Take a pause and allow yourself time to cool down.",
        "surprise": "Take a moment to process what happened.",
        "disgust": "Focus on what is within your control.",
        "neutral": "You seem emotionally balanced right now."
    }

    return recommendations.get(
        emotion,
        "Take care of your mental well-being."
    )