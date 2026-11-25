"""Complete the automatic (non-human) BloomSearch evaluation.

The evaluation is deliberately split by layer because the available public
artifacts use QReCC for query rewriting and SciFact for reranking.  It reports
paired comparisons, feature ablations, confidence intervals, significance
tests, call/token usage, and optional local timing.  It makes no hosted-model
calls and does not claim a single cross-dataset retrieval score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
from time import perf_counter
from typing import Iterable, Sequence

from bloom_search.index import SearchIndex
from bloom_search.nlp_policy import JointActionPolicy, PolicyFeatures


DEFAULT_HISTORICAL_COMMIT = "19459c6c"
VARIANTS = ("linguistic_only", "bm25_only", "bloom_only", "joint_policy")


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def _git_json(commit: str, path: str) -> object:
    return json.loads(_git_bytes(commit, path))


def _historical_index(commit: str, path: str) -> SearchIndex:
    payload = _git_bytes(commit, path)
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        handle.write(payload)
        handle.flush()
        return SearchIndex.load(handle.name)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    position = (len(rows) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    weight = position - lower
    return rows[lower] * (1.0 - weight) + rows[upper] * weight


def _binary_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    true_positive = sum(a and p for a, p in zip(actual, predicted))
    false_positive = sum(not a and p for a, p in zip(actual, predicted))
    false_negative = sum(a and not p for a, p in zip(actual, predicted))
    true_negative = len(actual) - true_positive - false_positive - false_negative
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    negative_f1 = 2 * true_negative / max(
        1, 2 * true_negative + false_positive + false_negative
    )
    return {
        "accuracy": (true_positive + true_negative) / max(1, len(actual)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _bootstrap_difference(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | list[float]]:
    """Paired bootstrap for mean(first - second)."""
    if len(first) != len(second) or not first:
        raise ValueError("paired samples must be non-empty and have equal length")
    differences = [a - b for a, b in zip(first, second)]
    rng = random.Random(seed)
    count = len(differences)
    samples = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    ]
    return {
        "mean_difference": _mean(differences),
        "confidence_interval_95": [
            _percentile(samples, 0.025),
            _percentile(samples, 0.975),
        ],
        "bootstrap_iterations": iterations,
    }


def _sign_flip_p_value(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test for a mean difference."""
    differences = [a - b for a, b in zip(first, second)]
    observed = abs(_mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = abs(_mean(value if rng.random() < 0.5 else -value for value in differences))
        extreme += int(permuted >= observed - 1e-15)
    return (extreme + 1) / (iterations + 1)


def _mcnemar_exact(first: Sequence[bool], second: Sequence[bool]) -> dict[str, float | int]:
    """Exact two-sided McNemar test for paired binary retrieval success."""
    first_only = sum(a and not b for a, b in zip(first, second))
    second_only = sum(not a and b for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, value) for value in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only_successes": first_only,
        "second_only_successes": second_only,
        "discordant_pairs": discordant,
        "p_value_two_sided": p_value,
    }


def _ablation_action(
    variant: str,
    features: PolicyFeatures,
    *,
    joint_action: str,
    ambiguity_threshold: float,
) -> str:
    if variant == "joint_policy":
        return joint_action
    if variant == "linguistic_only":
        if features.ambiguity >= ambiguity_threshold:
            return "clarify"
        return "rewrite" if features.context_dependence >= 0.35 else "keep"
    if variant == "bm25_only":
        return "rewrite" if features.bm25_confidence < 0.2 else "keep"
    if variant == "bloom_only":
        return "rewrite" if features.bloom_coverage < 0.5 else "keep"
    raise ValueError(f"unknown ablation variant: {variant}")


def _retrieval_outcome(action: str, original_hit: bool, rewrite_hit: bool) -> bool:
    if action == "keep":
        return original_hit
    if action == "rewrite":
        return rewrite_hit
    # QReCC contains no simulated response to a clarification question.
    return False


def _cross_validated_feature_ablation(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[bool],
    *,
    seed: int,
) -> dict[str, object]:
    """Compare signal families with the same repeated cross-validation protocol."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    from sklearn.model_selection import RepeatedStratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    matrix = np.asarray(feature_rows, dtype=float)
    target = np.asarray(labels, dtype=int)
    groups = {
        "linguistic_only": [0, 1, 2, 3],
        "bm25_only": [4, 5],
        "bloom_only": [6],
        "linguistic_plus_bm25": [0, 1, 2, 3, 4, 5],
        "all_signals": [0, 1, 2, 3, 4, 5, 6],
    }
    results: dict[str, object] = {}
    for name, columns in groups.items():
        selected = matrix[:, columns]
        splitter = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            random_state=seed,
        )
        aucs: list[float] = []
        average_precisions: list[float] = []
        positive_f1s: list[float] = []
        macro_f1s: list[float] = []
        for train, test in splitter.split(selected, target):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            )
            model.fit(selected[train], target[train])
            probabilities = model.predict_proba(selected[test])[:, 1]
            predicted = probabilities >= 0.5
            aucs.append(float(roc_auc_score(target[test], probabilities)))
            average_precisions.append(float(average_precision_score(target[test], probabilities)))
            positive_f1s.append(float(f1_score(target[test], predicted, zero_division=0)))
            macro_f1s.append(
                float(f1_score(target[test], predicted, average="macro", zero_division=0))
            )
        results[name] = {
            "features": columns,
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "average_precision_mean": float(np.mean(average_precisions)),
            "positive_f1_mean": float(np.mean(positive_f1s)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "fold_evaluations": len(aucs),
        }
    return {
        "protocol": "5-fold stratified cross-validation repeated 10 times",
        "target": "whether the QReCC gold rewrite improves Hit@10",
        "feature_order": [
            "token_count",
            "meaningful_ratio",
            "context_dependence",
            "ambiguity",
            "bm25_confidence",
            "bm25_margin",
            "bloom_coverage",
        ],
        "results": results,
    }


def evaluate_layer_one(
    commit: str,
    *,
    bootstrap_iterations: int,
    seed: int,
    measure_latency: bool,
    timing_sample_size: int,
) -> dict[str, object]:
    examples = _git_json(commit, "data/qrecc/test_3000.json")
    retrieval_payload = _git_json(commit, "data/qrecc/test_3000_results.json")
    if not isinstance(examples, list) or not isinstance(retrieval_payload, dict):
        raise ValueError("unexpected QReCC artifact format")
    retrieval = retrieval_payload.get("results")
    if not isinstance(retrieval, list):
        raise ValueError("QReCC retrieval results are missing")
    index = _historical_index(commit, "data/qrecc/qrecc_retrieval_index.json")
    policy = JointActionPolicy.runtime()
    if policy.ambiguity_detector is None:
        raise RuntimeError(
            "the trained ambiguity detector could not be loaded; install the research extras"
        )
    ambiguity_threshold = float(getattr(policy.ambiguity_detector, "threshold", 0.5))

    by_turn = {
        (int(row["conversation_id"]), int(row["turn_id"])): row for row in examples
    }
    labels: list[bool] = []
    original_outcomes: list[bool] = []
    always_rewrite_outcomes: list[bool] = []
    outcomes: dict[str, list[bool]] = {variant: [] for variant in VARIANTS}
    predictions: dict[str, list[bool]] = {variant: [] for variant in VARIANTS}
    actions: dict[str, Counter[str]] = {variant: Counter() for variant in VARIANTS}
    sources: Counter[str] = Counter()
    feature_rows: list[tuple[float, ...]] = []

    for row_number, stored in enumerate(retrieval, start=1):
        key = (int(stored["conversation_id"]), int(stored["turn_id"]))
        example = by_turn[key]
        query = str(example["original_query"])
        context_rows = example.get("context", [])
        context = "\n".join(str(turn) for turn in context_rows) if context_rows else ""
        decision = policy.decide_layer_one(index, query, context=context)
        sources[decision.source] += 1
        feature_rows.append(decision.features.values())

        original_hit = bool(stored["original_hit_at_10"])
        rewrite_hit = bool(stored["rewrite_hit_at_10"])
        labels.append(bool(stored["rewrite_helped"]))
        original_outcomes.append(original_hit)
        always_rewrite_outcomes.append(rewrite_hit)

        for variant in VARIANTS:
            action = _ablation_action(
                variant,
                decision.features,
                joint_action=decision.action,
                ambiguity_threshold=ambiguity_threshold,
            )
            actions[variant][action] += 1
            predictions[variant].append(action == "rewrite")
            outcomes[variant].append(_retrieval_outcome(action, original_hit, rewrite_hit))

        if row_number % 250 == 0:
            print(f"Layer 1 ablation: {row_number}/{len(retrieval)} queries", flush=True)

    methods: dict[str, object] = {
        "bm25_original": {
            "hit_at_10": _mean(float(value) for value in original_outcomes),
            "rewrite_call_rate": 0.0,
        },
        "always_gold_rewrite": {
            "hit_at_10": _mean(float(value) for value in always_rewrite_outcomes),
            "rewrite_call_rate": 1.0,
        },
    }
    for variant in VARIANTS:
        count = len(outcomes[variant])
        methods[variant] = {
            "hit_at_10": _mean(float(value) for value in outcomes[variant]),
            "actions": dict(actions[variant]),
            "action_rates": {name: value / count for name, value in actions[variant].items()},
            "rewrite_benefit": _binary_metrics(labels, predictions[variant]),
        }

    comparisons: dict[str, object] = {}
    for variant in VARIANTS:
        comparison = _bootstrap_difference(
            [float(value) for value in outcomes[variant]],
            [float(value) for value in original_outcomes],
            iterations=bootstrap_iterations,
            seed=seed,
        )
        comparison["mcnemar"] = _mcnemar_exact(outcomes[variant], original_outcomes)
        comparisons[f"{variant}_minus_bm25_original"] = comparison

    non_joint = [variant for variant in VARIANTS if variant != "joint_policy"]
    best_single = max(non_joint, key=lambda name: _mean(float(v) for v in outcomes[name]))
    joint_vs_single = _bootstrap_difference(
        [float(value) for value in outcomes["joint_policy"]],
        [float(value) for value in outcomes[best_single]],
        iterations=bootstrap_iterations,
        seed=seed + 1,
    )
    joint_vs_single["mcnemar"] = _mcnemar_exact(
        outcomes["joint_policy"], outcomes[best_single]
    )

    timing: dict[str, object] | None = None
    if measure_latency:
        baseline_latencies: list[float] = []
        joint_latencies: list[float] = []
        timing_count = min(timing_sample_size, len(retrieval))
        timing_rng = random.Random(seed)
        timing_rows = timing_rng.sample(retrieval, timing_count)
        for row_number, stored in enumerate(timing_rows, start=1):
            key = (int(stored["conversation_id"]), int(stored["turn_id"]))
            example = by_turn[key]
            query = str(example["original_query"])
            context_rows = example.get("context", [])
            context = "\n".join(str(turn) for turn in context_rows) if context_rows else ""

            started = perf_counter()
            index.search_bm25(query, limit=10)
            baseline_latencies.append(perf_counter() - started)

            started = perf_counter()
            decision = policy.decide_layer_one(index, query, context=context)
            if decision.action == "keep":
                index.search_bm25(query, limit=10)
            elif decision.action == "rewrite":
                index.search_bm25(str(example["gold_rewrite"]), limit=10)
            joint_latencies.append(perf_counter() - started)
            if row_number % 50 == 0:
                print(f"Layer 1 timing: {row_number}/{timing_count} queries", flush=True)

        timing = {
            "scope": "local Python decision and BM25 time; hosted generation is excluded",
            "sample_size": timing_count,
            "sampling": f"deterministic random sample with seed {seed}",
            "bm25_original": {
                "mean_ms": 1000.0 * _mean(baseline_latencies),
                "median_ms": 1000.0 * statistics.median(baseline_latencies),
                "p95_ms": 1000.0 * _percentile(baseline_latencies, 0.95),
            },
            "joint_policy_gold_rewrite_replay": {
                "mean_ms": 1000.0 * _mean(joint_latencies),
                "median_ms": 1000.0 * statistics.median(joint_latencies),
                "p95_ms": 1000.0 * _percentile(joint_latencies, 0.95),
            },
        }

    return {
        "dataset": "QReCC held-out retrieval set",
        "queries": len(retrieval),
        "ambiguity_threshold": ambiguity_threshold,
        "decision_sources": dict(sources),
        "methods": methods,
        "cross_validated_signal_ablation": _cross_validated_feature_ablation(
            feature_rows,
            labels,
            seed=seed,
        ),
        "paired_comparisons": comparisons,
        "best_single_signal_by_hit_at_10": best_single,
        "joint_minus_best_single": joint_vs_single,
        "timing": timing,
        "limitations": [
            "Gold rewrites test action selection, not newly generated rewrite quality.",
            "Clarification is unresolved because QReCC has no simulated user response.",
            "The binary rewrite-benefit label is not a human three-class action label.",
        ],
    }


def evaluate_layer_two(
    bm25_path: Path,
    rerank_path: Path,
    analysis_path: Path,
    *,
    margin_threshold: float,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    reranked = json.loads(rerank_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    policy = JointActionPolicy.runtime(rerank_margin_threshold=margin_threshold)

    bm25_pool: list[float] = []
    rerank_all_pool: list[float] = []
    selective_pool: list[float] = []
    selected_tokens = 0
    all_tokens = 0
    selected_latency = 0.0
    calls = 0

    for query_id, stored in reranked.items():
        if stored.get("error") or query_id not in bm25:
            continue
        scores = [float(value) for value in bm25[query_id]["bm25_scores"]]
        decision = policy.decide_layer_two([(object(), score) for score in scores])
        original = float(stored["bm25_ndcg"])
        reranked_value = float(stored["reranked_ndcg"])
        selected = reranked_value if decision.should_rerank else original
        bm25_pool.append(original)
        rerank_all_pool.append(reranked_value)
        selective_pool.append(selected)
        all_tokens += int(stored.get("total_tokens", 0))
        if decision.should_rerank:
            calls += 1
            selected_tokens += int(stored.get("total_tokens", 0))
            selected_latency += float(stored.get("latency_seconds", 0.0))

    selective_vs_bm25 = _bootstrap_difference(
        selective_pool,
        bm25_pool,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    selective_vs_bm25["sign_flip_p_value_two_sided"] = _sign_flip_p_value(
        selective_pool,
        bm25_pool,
        iterations=bootstrap_iterations,
        seed=seed + 2,
    )
    selective_vs_all = _bootstrap_difference(
        selective_pool,
        rerank_all_pool,
        iterations=bootstrap_iterations,
        seed=seed + 3,
    )
    selective_vs_all["sign_flip_p_value_two_sided"] = _sign_flip_p_value(
        selective_pool,
        rerank_all_pool,
        iterations=bootstrap_iterations,
        seed=seed + 4,
    )

    predictor = analysis.get("predictor", {})
    feature_ablation = {
        name: value
        for name, value in predictor.items()
        if name in {"BM25 scores only", "Query features only", "Scores + query features"}
    }
    count = len(bm25_pool)
    return {
        "dataset": "BEIR SciFact test split",
        "queries": count,
        "retrieved_pool_ndcg_at_10": {
            "bm25_only": _mean(bm25_pool),
            "rerank_all": _mean(rerank_all_pool),
            "selective_margin_rule": _mean(selective_pool),
        },
        "paired_comparisons": {
            "selective_minus_bm25": selective_vs_bm25,
            "selective_minus_rerank_all": selective_vs_all,
        },
        "hosted_usage_replay": {
            "reranking_calls": calls,
            "reranking_call_rate": calls / count,
            "reranking_call_reduction_vs_always": 1.0 - calls / count,
            "selected_tokens": selected_tokens,
            "always_rerank_tokens": all_tokens,
            "token_reduction_vs_always": 1.0 - selected_tokens / max(1, all_tokens),
            "average_hosted_latency_across_all_queries_seconds": selected_latency / count,
            "average_latency_per_selected_call_seconds": selected_latency / max(1, calls),
        },
        "standard_ndcg_context_from_published_analysis": {
            "bm25_only": analysis["comparison"]["BM25 only"]["ndcg"],
            "rerank_all": analysis["comparison"]["Rerank every query"]["ndcg"],
            "rerank_all_gain_ci95": analysis["reranking"]["ci95"],
            "held_out_selective_margin_rule": 0.7023,
            "held_out_selective_call_rate": 0.43,
        },
        "rerank_benefit_feature_ablation": feature_ablation,
        "limitations": [
            "Stored reranker outputs are replayed; no new hosted call is made.",
            "Retrieved-pool NDCG is used for paired selective-policy tests because per-query full qrels were removed during repository cleanup.",
            "The published standard NDCG values remain the correct cross-system results.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default=DEFAULT_HISTORICAL_COMMIT)
    parser.add_argument(
        "--bm25", type=Path, default=Path("experiments/reranking/bm25_results.json")
    )
    parser.add_argument(
        "--reranked",
        type=Path,
        default=Path("experiments/reranking/reranking_progress_v2.json"),
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("experiments/reranking/analysis_results.json"),
    )
    parser.add_argument("--margin-threshold", type=float, default=3.84)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--timing-sample-size", type=int, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/nonhuman_completion.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_iterations <= 0:
        parser.error("--bootstrap-iterations must be positive")
    if args.timing_sample_size <= 0:
        parser.error("--timing-sample-size must be positive")

    result = {
        "evaluation": "BloomSearch non-human research-question completion",
        "new_hosted_api_calls": 0,
        "cross_dataset_policy": (
            "Layers are evaluated separately; no invalid combined QReCC/SciFact score is reported."
        ),
        "layer_one": evaluate_layer_one(
            args.historical_commit,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            measure_latency=args.measure_latency,
            timing_sample_size=args.timing_sample_size,
        ),
        "layer_two": evaluate_layer_two(
            args.bm25,
            args.reranked,
            args.analysis,
            margin_threshold=args.margin_threshold,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
