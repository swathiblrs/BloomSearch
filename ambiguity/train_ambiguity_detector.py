
"""
Ambiguity Detection: train and evaluate using preprocessed CSV files.

This script starts AFTER data preprocessing. It expects these files
to already exist:

    data/processed/train.csv
    data/processed/validation.csv

The CSV files must contain:
    query
    ambiguity_label

Install dependencies:
    pip install pandas scikit-learn joblib

Run from the project directory:
    python ambiguity_detector.py

Change the settings in the CONFIGURATION section before rerunning.
"""

# ============================================================
# CONFIGURATION: change these values between runs
# ============================================================

# Probability threshold for classifying a question as ambiguous.
# Lower values generally increase recall; higher values generally
# increase precision.
THRESHOLD = 0.3

# TF-IDF settings.
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95

# Logistic Regression settings.
MAX_ITER = 2000
RANDOM_STATE = 42

# Set to True to save the trained model and TF-IDF vectorizer.
SAVE_MODEL = True
MODEL_OUTPUT_PATH = "ambiguity_detector.joblib"

# Location of the preprocessed CSV files, relative to this script.
TRAIN_DATA_PATH = "data/processed/train.csv"
VALIDATION_DATA_PATH = "data/processed/validation.csv"


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

def load_train_validation_data():
    """Load the CSV files created by data_preprocessing.ipynb."""

    # Resolve file paths relative to this Python script's location.
    project_directory = Path(__file__).resolve().parent
    train_path = project_directory / TRAIN_DATA_PATH
    validation_path = project_directory / VALIDATION_DATA_PATH

    # Check that both preprocessed files exist before loading them.
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Run data_preprocessing.ipynb first."
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found: {validation_path}\n"
            "Run data_preprocessing.ipynb first."
        )

    # Load the preprocessed training and validation data.
    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)

    # Check that the columns required for classification are present.
    required_columns = {"query", "ambiguity_label"}

    if not required_columns.issubset(train_df.columns):
        raise ValueError(
            f"Training CSV must contain these columns: {required_columns}"
        )

    if not required_columns.issubset(validation_df.columns):
        raise ValueError(
            f"Validation CSV must contain these columns: {required_columns}"
        )

    # Remove rows with missing labels and normalize the question text.
    train_df = train_df.dropna(subset=["ambiguity_label"]).copy()
    validation_df = validation_df.dropna(subset=["ambiguity_label"]).copy()

    train_df["query"] = train_df["query"].fillna("").astype(str)
    validation_df["query"] = validation_df["query"].fillna("").astype(str)

    # Ensure labels are integers representing clear (0) or ambiguous (1).
    train_df["ambiguity_label"] = train_df["ambiguity_label"].astype(int)
    validation_df["ambiguity_label"] = validation_df["ambiguity_label"].astype(int)

    print(f"Training examples:   {len(train_df):,}")
    print(f"Validation examples: {len(validation_df):,}")

    print("\nTraining label distribution:")
    print(
        train_df["ambiguity_label"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Clear", 1: "Ambiguous"})
    )

    return train_df, validation_df


# ============================================================
# BUILD AND TRAIN THE AMBIGUITY DETECTOR
# ============================================================

def train_detector(train_df):
    """Fit the TF-IDF vectorizer and Logistic Regression classifier."""

    # Separate the question text and target labels.
    X_train = train_df["query"]
    y_train = train_df["ambiguity_label"]

    # Convert questions into TF-IDF features.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    # Learn the vocabulary and IDF weights from training data only.
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Initialize the Logistic Regression classifier.
    model = LogisticRegression(
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
    )

    # Train the classifier using the TF-IDF features.
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


# ============================================================
# EVALUATE THE AMBIGUITY DETECTOR
# ============================================================

def evaluate_detector(validation_df, vectorizer, model, threshold):
    """Evaluate the classifier using the selected probability threshold."""

    # Extract validation questions and their ground-truth labels.
    X_val = validation_df["query"]
    y_val = validation_df["ambiguity_label"]

    # Convert validation questions using the training vocabulary.
    X_val_tfidf = vectorizer.transform(X_val)

    # Get the model's estimated probability of the ambiguous class.
    ambiguity_scores = model.predict_proba(X_val_tfidf)[:, 1]

    # Apply the configured threshold to obtain binary predictions.
    y_pred = (ambiguity_scores >= threshold).astype(int)

    # Calculate evaluation metrics.
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Print the evaluation results.
    print("\n" + "=" * 55)
    print("AMBIGUITY DETECTOR EVALUATION")
    print("=" * 55)
    print(f"Classification threshold: {threshold:.2f}")
    print(f"Accuracy:                 {accuracy:.4f}")
    print(f"Precision:                {precision:.4f}")
    print(f"Recall:                   {recall:.4f}")
    print(f"F1-score:                 {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            y_pred,
            labels=[0, 1],
            target_names=["Clear", "Ambiguous"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_val, y_pred, labels=[0, 1]))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# OPTIONAL: SAVE THE TRAINED DETECTOR
# ============================================================

def save_detector(vectorizer, model, threshold):
    """Save the vectorizer, classifier, and threshold for later use."""

    import joblib

    # Resolve the output path relative to this script.
    project_directory = Path(__file__).resolve().parent
    output_path = project_directory / MODEL_OUTPUT_PATH

    # Save all components needed to make predictions later.
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "threshold": threshold,
        },
        output_path,
    )

    print(f"\nSaved trained detector to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # Load the already-preprocessed training and validation data.
    train_df, validation_df = load_train_validation_data()

    # Train the ambiguity detector using the training data.
    vectorizer, model = train_detector(train_df)

    # Evaluate the detector on the validation data.
    evaluate_detector(
        validation_df,
        vectorizer,
        model,
        THRESHOLD,
    )

    # Save the trained detector if requested in the configuration.
    if SAVE_MODEL:
        save_detector(vectorizer, model, THRESHOLD)


if __name__ == "__main__":
    main()
