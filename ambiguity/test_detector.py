import joblib
from model import detect_ambiguity

# Load the saved ambiguity detector.
detector = joblib.load("ambiguity/ambiguity_detector.joblib")

# Extract the vectorizer, classifier, and threshold.
tfidf = detector["vectorizer"]
model = detector["model"]
threshold = detector["threshold"]

test_questions = [
    "When did the Simpsons first air on television?",
    "Who played Batman?",
    "What is the capital of France?",
    "When was the first computer invented?"
]

for question in test_questions:
    print(question)
    print(detect_ambiguity(question))
    print()