import whisper
from transformers import pipeline

# Models are loaded only once when first used
whisper_model = None
emotion_classifier = None


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
    return whisper_model


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
    result = get_whisper_model().transcribe(audio_path)
    return result["text"]


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