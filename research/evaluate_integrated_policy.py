"""Offline replay of the integrated Layer 1 and Layer 2 policies.

This evaluation makes no hosted API calls. Layer 1 replays current policy
decisions on QReCC and uses the dataset's gold rewrite as a controlled proxy
when the policy selects ``rewrite``. Layer 2 replays the current BM25-margin
decision on the stored SciFact hosted-reranking outputs.

The two layers use different public datasets and are therefore reported
separately. The script intentionally does not claim one combined end-to-end
quality score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence

from bloom_search.index import SearchIndex
from bloom_search.nlp_policy import JointActionPolicy


DEFAULT_HISTORICAL_COMMIT = "19459c6c"


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


def _binary_metrics(actual: Sequence[bool], predicted: Sequence[bool]) -> dict[str, float | int]:
    if len(actual) != len(predicted):
        raise ValueError("label and prediction counts differ")
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


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _simulate_layer_one(
    policy: JointActionPolicy,
    index: SearchIndex,
    examples: Sequence[dict[str, object]],
    retrieval: Sequence[dict[str, object]],
) -> dict[str, object]:
    by_turn = {
        (int(example["conversation_id"]), int(example["turn_id"])): example
        for example in examples
    }

    actions: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    successes = 0
    rewrite_labels: list[bool] = []
    rewrite_predictions: list[bool] = []

    for row_number, result in enumerate(retrieval, start=1):
        key = (int(result["conversation_id"]), int(result["turn_id"]))
        try:
            example = by_turn[key]
        except KeyError as error:
            raise ValueError(f"QReCC context is missing for turn {key}") from error
        query = str(example["original_query"])
        context_value = example.get("context", [])
        context = "\n".join(str(turn) for turn in context_value) if context_value else ""
        decision = policy.decide_layer_one(index, query, context=context)
        actions[decision.action] += 1
        sources[decision.source] += 1

        original_hit = bool(result["original_hit_at_10"])
        rewrite_hit = bool(result["rewrite_hit_at_10"])
        rewrite_helped = bool(result["rewrite_helped"])
        rewrite_labels.append(rewrite_helped)
        rewrite_predictions.append(decision.action == "rewrite")

        # Clarification is counted as unresolved in this one-turn retrieval
        # benchmark. Rewrites use the human gold rewrite, not generated text.
        if decision.action == "keep":
            successes += int(original_hit)
        elif decision.action == "rewrite":
            successes += int(rewrite_hit)
        if row_number % 250 == 0:
            print(
                f"Layer 1 replay: {row_number}/{len(retrieval)} queries",
                file=sys.stderr,
                flush=True,
            )

    count = len(retrieval)
    return {
        "queries": count,
        "hit_at_10_gold_rewrite_replay": successes / count,
        "actions": dict(actions),
        "action_rates": {key: value / count for key, value in actions.items()},
        "decision_sources": dict(sources),
        "rewrite_benefit": _binary_metrics(rewrite_labels, rewrite_predictions),
        "hosted_rewrite_call_rate_if_api_supplied": actions["rewrite"] / count,
        "clarification_unresolved_rate": actions["clarify"] / count,
    }


def evaluate_layer_one(commit: str) -> dict[str, object]:
    examples = _git_json(commit, "data/qrecc/test_3000.json")
    retrieval_payload = _git_json(commit, "data/qrecc/test_3000_results.json")
    previous_metrics = _git_json(commit, "data/qrecc/selective_rewriting_validated_metrics.json")
    if not isinstance(examples, list) or not isinstance(retrieval_payload, dict):
        raise ValueError("unexpected QReCC artifact format")
    retrieval = retrieval_payload.get("results")
    if not isinstance(retrieval, list) or not isinstance(previous_metrics, dict):
        raise ValueError("unexpected QReCC result format")
    index = _historical_index(commit, "data/qrecc/qrecc_retrieval_index.json")

    integrated = _simulate_layer_one(JointActionPolicy.runtime(), index, examples, retrieval)
    return {
        "dataset": "QReCC held-out retrieval set",
        "evaluation_type": "offline action replay with human gold rewrites",
        "limitations": [
            "Gold rewrites measure action-selection potential, not hosted rewrite quality.",
            "QReCC has no keep/rewrite/clarify three-class labels; Macro-F1 is not reported.",
            "Clarification is treated as unresolved because the dataset has no user response.",
            "Layer 1 and Layer 2 use different datasets, so no combined quality score is claimed.",
        ],
        "baselines": {
            "bm25_original_hit_at_10": float(retrieval_payload["original_hit_at_10"]),
            "always_gold_rewrite_hit_at_10": float(retrieval_payload["gold_rewrite_hit_at_10"]),
            "previous_selective_hit_at_10": float(previous_metrics["selective_hit_at_10"]),
            "previous_selective_rewrite_rate": float(previous_metrics["rewrite_rate"]),
        },
        "integrated_policy": integrated,
    }


def evaluate_layer_two(
    bm25_path: Path,
    rerank_path: Path,
    margin_threshold: float,
) -> dict[str, object]:
    bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
    reranked = json.loads(rerank_path.read_text(encoding="utf-8"))
    policy = JointActionPolicy.runtime(rerank_margin_threshold=margin_threshold)

    bm25_scores: list[float] = []
    rerank_all_scores: list[float] = []
    selective_scores: list[float] = []
    benefit_labels: list[bool] = []
    call_predictions: list[bool] = []
    selected_tokens = 0
    all_tokens = 0
    selected_latency = 0.0
    selected_gains: list[float] = []

    for query_id, stored in reranked.items():
        if stored.get("error") or query_id not in bm25:
            continue
        scores = [float(value) for value in bm25[query_id]["bm25_scores"]]
        candidates = [(object(), score) for score in scores]
        decision = policy.decide_layer_two(candidates)
        bm25_ndcg = float(stored["bm25_ndcg"])
        reranked_ndcg = float(stored["reranked_ndcg"])
        helped = reranked_ndcg > bm25_ndcg

        bm25_scores.append(bm25_ndcg)
        rerank_all_scores.append(reranked_ndcg)
        selective_scores.append(reranked_ndcg if decision.should_rerank else bm25_ndcg)
        benefit_labels.append(helped)
        call_predictions.append(decision.should_rerank)
        all_tokens += int(stored.get("total_tokens", 0))
        if decision.should_rerank:
            selected_tokens += int(stored.get("total_tokens", 0))
            selected_latency += float(stored.get("latency_seconds", 0.0))
            selected_gains.append(reranked_ndcg - bm25_ndcg)

    count = len(bm25_scores)
    calls = sum(call_predictions)
    return {
        "dataset": "BEIR SciFact test split",
        "evaluation_type": "offline replay of stored hosted-reranker outputs",
        "queries": count,
        "margin_threshold": margin_threshold,
        "metric_note": (
            "Per-query stored NDCG is normalized by the ideal ordering of retrieved "
            "documents; use the published standard-NDCG results for cross-system reporting."
        ),
        "retrieved_pool_ndcg_at_10": {
            "bm25_only": _mean(bm25_scores),
            "rerank_all": _mean(rerank_all_scores),
            "integrated_selective": _mean(selective_scores),
        },
        "rerank_benefit": _binary_metrics(benefit_labels, call_predictions),
        "reranking_calls": calls,
        "reranking_call_rate": calls / count,
        "reranking_call_reduction": 1.0 - calls / count,
        "token_reduction": 1.0 - selected_tokens / max(1, all_tokens),
        "selected_total_tokens": selected_tokens,
        "all_queries_total_tokens": all_tokens,
        "average_latency_per_selected_query_seconds": selected_latency / max(1, calls),
        "average_latency_across_all_queries_seconds": selected_latency / count,
        "average_ndcg_gain_per_selected_call": _mean(selected_gains),
        "published_standard_ndcg_context": {
            "bm25_only": 0.663454305453947,
            "rerank_all": 0.7015109698689124,
            "held_out_selective_rule": 0.7023,
            "held_out_selective_call_rate": 0.43,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default=DEFAULT_HISTORICAL_COMMIT)
    parser.add_argument(
        "--bm25",
        type=Path,
        default=Path("experiments/reranking/bm25_results.json"),
    )
    parser.add_argument(
        "--reranked",
        type=Path,
        default=Path("experiments/reranking/reranking_progress_v2.json"),
    )
    parser.add_argument("--margin-threshold", type=float, default=3.84)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/integrated_policy_replay.json"),
    )
    args = parser.parse_args()

    output = {
        "evaluation": "Integrated JointActionPolicy offline replay",
        "new_hosted_api_calls": 0,
        "layer_one": evaluate_layer_one(args.historical_commit),
        "layer_two": evaluate_layer_two(args.bm25, args.reranked, args.margin_threshold),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
