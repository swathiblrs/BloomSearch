import joblib

# Load the saved ambiguity detector.
detector = joblib.load("ambiguity/ambiguity_detector.joblib")

# Extract the vectorizer, classifier, and threshold.
tfidf = detector["vectorizer"]
model = detector["model"]
threshold = detector["threshold"]

def detect_ambiguity(query):
    # Convert the question into TF-IDF features.
    query_tfidf = tfidf.transform([query])

    # Predict the probability that the question is ambiguous.
    ambiguity_score = model.predict_proba(query_tfidf)[0, 1]

    # Determine whether clarification is needed.
    needs_clarification = ambiguity_score >= threshold

    # Return the detection results.
    return {
        "ambiguity_score": float(ambiguity_score),
        "needs_clarification": bool(needs_clarification),
        "clarification_question": None
    }