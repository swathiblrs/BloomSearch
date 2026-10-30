
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
MODEL_FILE = DATA / "rewrite_hybrid_model.joblib"
OUTPUT_FILE = DATA / "rewrite_threshold_metrics.json"

THRESHOLDS = np.arange(0.1, 1.0, 0.1)


def main():
    samples = json.loads(TEST_FILE.read_text(encoding="utf-8"))
    results = json.loads(
        RESULTS_FILE.read_text(encoding="utf-8")
    )["results"]

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

    X_text = vectorizer.transform(texts)
    X_numeric = scaler.transform(
        np.asarray(numeric_features, dtype=float)
    )

    X = hstack(
        [X_text, csr_matrix(X_numeric)],
        format="csr",
    )

    # Probability that rewriting will help.
    class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X)[:, class_index]

    original_hits = np.array([
        int(r["original_hit_at_10"])
        for r in matched_results
    ])

    rewrite_hits = np.array([
        int(r["rewrite_hit_at_10"])
        for r in matched_results
    ])

    print("\nThreshold Evaluation")
    print("-" * 65)
    print(
        f"{'Threshold':<12}"
        f"{'Hit@10':<12}"
        f"{'Rewrite Rate':<16}"
        f"{'Rewrites':<12}"
    )

    metrics = []

    for threshold in THRESHOLDS:
        decisions = probabilities >= threshold

        selective_hits = np.where(
            decisions,
            rewrite_hits,
            original_hits,
        )

        hit_rate = float(np.mean(selective_hits))
        rewrite_rate = float(np.mean(decisions))
        rewrite_count = int(np.sum(decisions))

        print(
            f"{threshold:<12.2f}"
            f"{hit_rate * 100:<12.2f}"
            f"{rewrite_rate * 100:<16.2f}"
            f"{rewrite_count:<12}"
        )

        metrics.append({
            "threshold": round(float(threshold), 2),
            "hit_at_10": hit_rate,
            "rewrite_rate": rewrite_rate,
            "rewrites": rewrite_count,
        })

    OUTPUT_FILE.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nKEEP all Hit@10:",
          f"{np.mean(original_hits) * 100:.2f}%")
    print("REWRITE all Hit@10:",
          f"{np.mean(rewrite_hits) * 100:.2f}%")
    print("Saved metrics:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
