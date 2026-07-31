import whisper
from transformers import pipeline

# Load Whisper model
whisper_model = whisper.load_model("tiny")

# Load Emotion Detection model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def detect_emotion(text):
    result = emotion_classifier(text)
    emotion = result[0][0]["label"]
    score = result[0][0]["score"]

    return {
        "emotion": emotion,
        "confidence": round(score * 100, 2)
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