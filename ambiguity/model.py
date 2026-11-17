
import joblib

# Load the saved model components.
detector = joblib.load("ambiguity/ambiguity_detector.joblib")

# Extract the vectorizer, classifier, and threshold.
tfidf = detector["vectorizer"]
model = detector["model"]
threshold = detector["threshold"]


def detect_ambiguity(query):
    # Convert the input question into TF-IDF features.
    query_tfidf = tfidf.transform([query])

    # Predict the probability that the question is ambiguous.
    ambiguity_score = model.predict_proba(query_tfidf)[0, 1]

    # Determine whether clarification is needed.
    needs_clarification = ambiguity_score >= threshold

    # Add clarification question if the query is ambiguous.
    clarification_question = (
        "Could you clarify exactly what you're asking?"
        if needs_clarification
        else None
    )

    # Return the detection and clarification results.
    return {
        "ambiguity_score": float(ambiguity_score),
        "needs_clarification": bool(needs_clarification),
        "clarification_question": clarification_question
    }
