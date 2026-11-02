
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from scripts.train_rewrite_benefit_predictor import features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "qrecc"

TEST_FILE = DATA / "test_3000.json"
RESULTS_FILE = DATA / "test_3000_results.json"
MODEL_FILE = DATA / "rewrite_hybrid_validated.joblib"
OUTPUT_FILE = DATA / "selective_rewriting_validated_metrics.json"

def main():
    samples = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    results = json.loads(
        RESULTS_FILE.read_text(encoding="utf-8")
    )["results"]

    # Match test queries to their BM25 retrieval results.
    result_lookup = {
        (str(r["conversation_id"]), str(r["turn_id"])): r
        for r in results
    }

    bundle = joblib.load(MODEL_FILE)
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]
    model = bundle["model"]

    texts = []
    numeric_features = []
    matched_results = []

    for row in samples:
        key = (
            str(row["conversation_id"]),
            str(row["turn_id"]),
        )

        if key not in result_lookup:
            continue

        query = row["original_query"]
        context = row.get("context", [])

        if isinstance(context, list):
            context_text = " ".join(map(str, context[-3:]))
        else:
            context_text = str(context)

        texts.append(
            context_text[-500:] + " [QUERY] " + query
        )

        numeric = features(row)
        if isinstance(numeric, dict):
            numeric = list(numeric.values())

        numeric_features.append(numeric)
        matched_results.append(result_lookup[key])

    # Apply the same transformations used during training.
    X_text = vectorizer.transform(texts)
    X_numeric = scaler.transform(
        np.asarray(numeric_features, dtype=float)
    )

    X = hstack(
        [X_text, csr_matrix(X_numeric)],
        format="csr",
    )

    threshold = bundle["threshold"]
    
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, class_index]
    predictions = (
    probabilities >= threshold
    ).astype(int)

    keep_hits = []
    rewrite_hits = []
    selective_hits = []

    for prediction, result in zip(predictions, matched_results):
        # Update these keys if your result JSON uses different names.
        original_hit = int(result["original_hit_at_10"])
        rewritten_hit = int(result["rewrite_hit_at_10"])

        keep_hits.append(original_hit)
        rewrite_hits.append(rewritten_hit)

        selective_hits.append(
            rewritten_hit if prediction == 1 else original_hit
        )

    metrics = {
        "test_examples": len(matched_results),
        "keep_all_hit_at_10": float(np.mean(keep_hits)),
        "rewrite_all_hit_at_10": float(np.mean(rewrite_hits)),
        "selective_hit_at_10": float(np.mean(selective_hits)),
        "rewrites_recommended": int(predictions.sum()),
        "rewrite_rate": float(np.mean(predictions)),
    }

    print("\nSelective Rewriting Evaluation")
    print("------------------------------")
    print("Test examples:", metrics["test_examples"])
    print(
        f"KEEP all Hit@10: "
        f"{metrics['keep_all_hit_at_10'] * 100:.2f}%"
    )
    print(
        f"REWRITE all Hit@10: "
        f"{metrics['rewrite_all_hit_at_10'] * 100:.2f}%"
    )
    print(
        f"SELECTIVE Hit@10: "
        f"{metrics['selective_hit_at_10'] * 100:.2f}%"
    )
    print(
        "Rewrites recommended:",
        metrics["rewrites_recommended"]
    )
    print(
        f"Rewrite rate: {metrics['rewrite_rate'] * 100:.2f}%"
    )

    OUTPUT_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    print("Saved metrics:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
