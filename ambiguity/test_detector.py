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
    "What has been on longer wheel of fortune or jeopardy?"
]

for question in test_questions:
    print(question)
    print(detect_ambiguity(question))
    print()