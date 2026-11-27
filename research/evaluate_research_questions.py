"""Produce the final evidence table for the four BloomSearch research questions.

The evaluator makes no hosted-model calls. It recomputes QReCC Recall@10 and
NDCG@10 from the historical index, combines the stored SciFact reranking
experiment with the P01/P02 human study, and reports limitations explicitly.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

from bloom_search.nlp_policy import JointActionPolicy
from bloom_search.intent_guard import preserves_intent
from research.evaluate_nonhuman_completion import (
    _bootstrap_difference,
    _git_json,
    _historical_index,
    _mcnemar_exact,
    _sign_flip_p_value,
)


DEFAULT_HISTORICAL_COMMIT = "19459c6c"


def _rank(results: list[tuple[object, float]], relevant_id: str) -> int | None:
    for position, (document, _) in enumerate(results, start=1):
        if str(getattr(document, "id", "")) == relevant_id:
            return position
    return None


def _recall(rank: int | None) -> float:
    # Each QReCC index record has one designated relevant passage.
    return float(rank is not None)


def _ndcg(rank: int | None) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None else 0.0


def _estimated_input_tokens(query: str, context: str) -> int:
    """Provider-neutral input estimate used only for relative call policies."""
    return max(1, math.ceil(len(query + "\n" + context) / 4))


def evaluate_qrecc(commit: str, *, max_queries: int | None = None) -> dict[str, object]:
    examples = _git_json(commit, "data/qrecc/test_3000.json")
    mapping = _git_json(commit, "data/qrecc/qrecc_test_mapping.json")
    if not isinstance(examples, list) or not isinstance(mapping, dict):
        raise ValueError("unexpected QReCC example format")
    policy_index = _historical_index(commit, "data/qrecc/qrecc_retrieval_index.json")
    evaluation_index = _historical_index(commit, "data/qrecc/qrecc_test_index.json")
    present_ids = {
        document.id
        for segment in evaluation_index.segments
        for document in segment.documents
    }
    policy = JointActionPolicy.runtime()

    original_recall: list[float] = []
    original_ndcg: list[float] = []
    rewrite_recall: list[float] = []
    rewrite_ndcg: list[float] = []
    selective_recall: list[float] = []
    selective_ndcg: list[float] = []
    actions: Counter[str] = Counter()
    always_input_tokens = 0
    selective_input_tokens = 0

    evaluated = [
        (query_index, example)
        for query_index, example in enumerate(examples)
        if str(query_index) in mapping and str(mapping[str(query_index)]) in present_ids
    ]
    eligible_count = len(evaluated)
    if max_queries is not None:
        evaluated = evaluated[:max_queries]
    for row_number, (query_index, example) in enumerate(evaluated, start=1):
        query = str(example["original_query"])
        rewrite = str(example["gold_rewrite"])
        context_rows = example.get("context", [])
        context = "\n".join(str(turn) for turn in context_rows)
        relevant_id = str(mapping[str(query_index)])

        policy_candidates, _ = policy_index.search_bm25(query, limit=10)
        original_results, _ = evaluation_index.search_bm25(query, limit=10)
        rewrite_results, _ = evaluation_index.search_bm25(rewrite, limit=10)
        original_rank = _rank(original_results, relevant_id)
        rewrite_rank = _rank(rewrite_results, relevant_id)
        decision = policy.decide_layer_one(
            policy_index,
            query,
            context=context,
            initial_candidates=policy_candidates,
        )
        actions[decision.action] += 1

        original_recall.append(_recall(original_rank))
        original_ndcg.append(_ndcg(original_rank))
        rewrite_recall.append(_recall(rewrite_rank))
        rewrite_ndcg.append(_ndcg(rewrite_rank))
        if decision.action == "rewrite":
            selected_rank = rewrite_rank
        elif decision.action == "keep":
            selected_rank = original_rank
        else:
            selected_rank = None
        selective_recall.append(_recall(selected_rank))
        selective_ndcg.append(_ndcg(selected_rank))

        tokens = _estimated_input_tokens(query, context)
        always_input_tokens += tokens
        if decision.action == "rewrite":
            selective_input_tokens += tokens

        if row_number % 250 == 0:
            print(f"Final QReCC metrics: {row_number}/{len(evaluated)}", flush=True)

    count = len(evaluated)
    rewrite_calls = actions["rewrite"]
    paired_comparisons = {}
    for comparison_name, first_recall, second_recall, first_ndcg, second_ndcg in (
        (
            "selective_minus_bm25",
            selective_recall,
            original_recall,
            selective_ndcg,
            original_ndcg,
        ),
        (
            "selective_minus_always_rewrite",
            selective_recall,
            rewrite_recall,
            selective_ndcg,
            rewrite_ndcg,
        ),
    ):
        recall_test = _bootstrap_difference(
            first_recall,
            second_recall,
            iterations=10_000,
            seed=42,
        )
        recall_test["mcnemar"] = _mcnemar_exact(
            [bool(value) for value in first_recall],
            [bool(value) for value in second_recall],
        )
        ndcg_test = _bootstrap_difference(
            first_ndcg,
            second_ndcg,
            iterations=10_000,
            seed=43,
        )
        ndcg_test["sign_flip_p_value_two_sided"] = _sign_flip_p_value(
            first_ndcg,
            second_ndcg,
            iterations=10_000,
            seed=44,
        )
        paired_comparisons[comparison_name] = {
            "recall_at_10": recall_test,
            "ndcg_at_10": ndcg_test,
        }
    return {
        "dataset": "QReCC held-out retrieval set",
        "queries": count,
        "excluded_missing_targets_after_index_deduplication": len(mapping)
        - eligible_count,
        "recomputation_note": (
            "Metrics were recomputed from the checked-in test index and mapping. "
            "Targets removed by index deduplication were excluded."
        ),
        "relevance_note": (
            "Each query has one designated relevant passage, so Recall@10 equals Hit@10."
        ),
        "methods": {
            "bm25_original": {
                "recall_at_10": mean(original_recall),
                "ndcg_at_10": mean(original_ndcg),
                "hosted_rewrite_calls": 0,
            },
            "always_gold_rewrite": {
                "recall_at_10": mean(rewrite_recall),
                "ndcg_at_10": mean(rewrite_ndcg),
                "hosted_rewrite_calls_if_generated": count,
            },
            "selective_gold_rewrite_replay": {
                "recall_at_10": mean(selective_recall),
                "ndcg_at_10": mean(selective_ndcg),
                "actions": dict(actions),
                "hosted_rewrite_calls_if_api_supplied": rewrite_calls,
            },
        },
        "estimated_rewrite_usage": {
            "method": "ceil((query characters + context characters) / 4); fixed prompts excluded",
            "always_input_tokens": always_input_tokens,
            "selective_input_tokens": selective_input_tokens,
            "call_reduction_vs_always": 1.0 - rewrite_calls / count,
            "input_token_reduction_vs_always": 1.0
            - selective_input_tokens / always_input_tokens,
        },
        "paired_comparisons": paired_comparisons,
    }


def retrieval_harm_comparison(commit: str) -> dict[str, object]:
    payload = _git_json(commit, "data/qrecc/system_rewrite_evaluation_final.json")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("unexpected stored rewrite format")
    records = payload["records"]
    accepted = [row for row in records if row.get("accepted")]

    def lost_hit(row: dict[str, object]) -> bool:
        return bool(row.get("original_hit")) and not bool(row.get("rewritten_hit"))

    def reduced_rank(row: dict[str, object]) -> bool:
        return float(row.get("rewritten_rr", 0.0)) < float(row.get("original_rr", 0.0))

    always_lost = sum(lost_hit(row) for row in records)
    selective_lost = sum(lost_hit(row) for row in accepted)
    always_rank_harm = sum(reduced_rank(row) for row in records)
    selective_rank_harm = sum(reduced_rank(row) for row in accepted)
    return {
        "dataset": "QReCC stored generated-rewrite evaluation",
        "queries": len(records),
        "selective_rewrite_calls": len(accepted),
        "always_rewrite_retrieval_harm": {
            "lost_hit_at_5_count": always_lost,
            "lost_hit_at_5_rate_per_query": always_lost / len(records),
            "reduced_reciprocal_rank_count": always_rank_harm,
            "reduced_reciprocal_rank_rate_per_query": always_rank_harm / len(records),
        },
        "selective_rewrite_retrieval_harm": {
            "lost_hit_at_5_count": selective_lost,
            "lost_hit_at_5_rate_per_query": selective_lost / len(records),
            "reduced_reciprocal_rank_count": selective_rank_harm,
            "reduced_reciprocal_rank_rate_per_query": selective_rank_harm / len(records),
        },
        "finding": (
            "Selection did not reduce retrieval-harm counts in the stored run; every "
            "observed lost hit and reciprocal-rank decrease occurred among accepted rewrites."
        ),
        "semantic_limitation": (
            "P01/P02 measured 45% semantic harm for selected rewrites, but no human "
            "always-rewrite baseline exists. Retrieval harm is not equivalent to intent harm."
        ),
    }


def automatic_intent_harm_comparison(
    study_items_path: Path,
    always_rewrite_path: Path,
) -> dict[str, object]:
    """Compare both rewrite policies with the same deterministic intent guard.

    The Phase C packet contains an always-rewrite output for each of the 20
    QReCC pilot items. Ten of those items were selected for rewriting by the
    policy. Applying one guard to both conditions produces a reproducible
    intent-safety proxy without inventing participant judgments.
    """
    with study_items_path.open(newline="", encoding="utf-8") as handle:
        study_items = {row["item_id"]: row for row in csv.DictReader(handle)}
    with always_rewrite_path.open(newline="", encoding="utf-8") as handle:
        always_rows = list(csv.DictReader(handle))

    def evaluate(
        item_id: str,
        original_query: str,
        context_text: str,
        rewrite: str,
    ) -> dict[str, object]:
        context = context_text.split(" | ") if context_text else None
        preserved, problems = preserves_intent(original_query, rewrite, context)
        return {
            "item_id": item_id,
            "preserves_intent": preserved,
            "problems": problems,
        }

    always = [
        evaluate(
            row["item_id"],
            row["original_query"],
            row["context"],
            row["always_rewrite_output"],
        )
        for row in always_rows
    ]
    always_ids = {row["item_id"] for row in always_rows}
    selective = [
        evaluate(
            row["item_id"],
            row["original_query"],
            row["context"],
            row["proposed_rewrite"],
        )
        for row in study_items.values()
        if row["item_id"] in always_ids and row["system_action"] == "rewrite"
    ]
    if not always or not selective:
        raise ValueError("automatic intent comparison requires both rewrite conditions")

    def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
        rejected = [row for row in rows if not row["preserves_intent"]]
        return {
            "evaluated_rewrites": len(rows),
            "harmful_rewrites": len(rejected),
            "harmful_rewrite_rate": len(rejected) / len(rows),
            "rejected_items": rejected,
        }

    always_summary = summarize(always)
    selective_summary = summarize(selective)
    difference = (
        selective_summary["harmful_rewrite_rate"]
        - always_summary["harmful_rewrite_rate"]
    )
    return {
        "dataset": "20 QReCC items from the blinded P01/P02 pilot",
        "method": "The same deterministic intent guard is applied to both conditions.",
        "always_rewrite": always_summary,
        "selective_rewrite": selective_summary,
        "selective_minus_always_rate": difference,
        "finding": (
            "The automatic intent guard does not support the hypothesis that selection "
            "reduces harmful rewrites."
        ),
        "limitation": (
            "This is a rule-based semantic proxy, not a replacement for paired human "
            "ratings of the always-rewrite outputs."
        ),
    }


def build_final_result(
    qrecc: dict[str, object],
    nonhuman: dict[str, object],
    integrated: dict[str, object],
    human: dict[str, object],
    reranking_analysis: dict[str, object],
    harm: dict[str, object],
    automatic_intent_harm: dict[str, object],
) -> dict[str, object]:
    layer_one = nonhuman["layer_one"]
    layer_two = nonhuman["layer_two"]
    standard = layer_two["standard_ndcg_context_from_published_analysis"]
    rerank_usage = layer_two["hosted_usage_replay"]
    scifact_recall = float(reranking_analysis["bm25_all"]["recall"])
    return {
        "evaluation": "Final BloomSearch research-question evidence",
        "new_hosted_api_calls": 0,
        "rq1_action_selection": {
            "status": "answered as a two-stage policy",
            "architecture_note": (
                "Keep/rewrite/clarify is Layer 1; rerank/return-BM25 is Layer 2. "
                "Reranking is not mutually exclusive with rewriting."
            ),
            "layer_one_human_three_class": human["action_policy"],
            "layer_one_rewrite_benefit_signal_ablation": layer_one[
                "cross_validated_signal_ablation"
            ],
            "layer_two_rerank_benefit": {
                "classification": integrated["layer_two"]["rerank_benefit"],
                "feature_ablation": layer_two["rerank_benefit_feature_ablation"],
            },
            "conclusion": (
                "Linguistic features drive Layer 1; Bloom coverage is near chance. "
                "BM25 score features are the strongest tested Layer 2 predictor."
            ),
        },
        "rq2_retrieval_effectiveness": {
            "status": "answered per layer on task-appropriate public datasets",
            "qrecc_layer_one": qrecc,
            "scifact_layer_two": {
                "dataset": "BEIR SciFact test split",
                "recall_at_10": {
                    "bm25_only": scifact_recall,
                    "rerank_all": scifact_recall,
                    "selective_reranking": scifact_recall,
                    "note": "All methods reorder the same top-10 pool, so recall is identical.",
                },
                "standard_ndcg_at_10": {
                    "bm25_only": standard["bm25_only"],
                    "rerank_all": standard["rerank_all"],
                    "selective_reranking_held_out": standard[
                        "held_out_selective_margin_rule"
                    ],
                },
            },
            "cross_dataset_limitation": (
                "No single public artifact supplies conversational rewrites, "
                "clarification responses, and stored hosted reranking for the same queries."
            ),
        },
        "rq3_intent_preservation": {
            "status": "answered negatively under retrieval and automatic intent-harm definitions",
            "retrieval_harm_comparison": harm,
            "automatic_intent_guard_comparison": automatic_intent_harm,
            "human_selected_rewrite_harmful_rate": human["intent_preservation"],
            "conclusion": (
                "RQ3's reduction hypothesis is not supported. Selection did not reduce "
                "retrieval-harm events; the common intent-guard proxy was also worse for "
                "selective rewriting; and 45% of selected rewrites were judged semantically "
                "harmful by P01/P02. A paired human always-rewrite rate was not collected."
            ),
        },
        "rq4_efficiency": {
            "status": "answered per layer with a mixed latency result",
            "layer_one_rewriting": {
                **qrecc["estimated_rewrite_usage"],
                "local_latency": layer_one["timing"],
            },
            "layer_two_reranking": {
                **rerank_usage,
                "always_rerank_average_latency_seconds": reranking_analysis["reranking"][
                    "avg_latency_s"
                ],
                "average_hosted_latency_reduction_vs_always": 1.0
                - rerank_usage["average_hosted_latency_across_all_queries_seconds"]
                / reranking_analysis["reranking"]["avg_latency_s"],
            },
            "conclusion": (
                "Selective policies reduce hosted calls and tokens. Layer 2 also reduces "
                "average hosted latency while preserving retrieval quality, but Layer 1 adds "
                "local decision latency and therefore is not faster than BM25 alone."
            ),
        },
        "overall_conclusion": (
            "All four research questions have an empirical answer. RQ3 is answered "
            "negatively by retrieval harm and a common automatic intent-safety proxy, with "
            "Bansari's P01/P02 human study providing selected-rewrite evidence. A paired "
            "human always-rewrite comparison remains an optional stronger extension."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-commit", default=DEFAULT_HISTORICAL_COMMIT)
    parser.add_argument(
        "--nonhuman",
        type=Path,
        default=Path("research/results/nonhuman_completion.json"),
    )
    parser.add_argument(
        "--human", type=Path, default=Path("research/results/human_study.json")
    )
    parser.add_argument(
        "--integrated",
        type=Path,
        default=Path("research/results/integrated_policy_replay.json"),
    )
    parser.add_argument(
        "--reranking-analysis",
        type=Path,
        default=Path("experiments/reranking/analysis_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results/final_research_questions.json"),
    )
    parser.add_argument(
        "--study-items",
        type=Path,
        default=Path("research/human_study/study_items.csv"),
    )
    parser.add_argument(
        "--always-rewrite-packet",
        type=Path,
        default=Path(
            "research/human_study/participant_packets/P01_phase_c_always_rewrite.csv"
        ),
    )
    args = parser.parse_args()

    qrecc = evaluate_qrecc(args.historical_commit)
    nonhuman = json.loads(args.nonhuman.read_text(encoding="utf-8"))
    integrated = json.loads(args.integrated.read_text(encoding="utf-8"))
    human = json.loads(args.human.read_text(encoding="utf-8"))
    reranking_analysis = json.loads(args.reranking_analysis.read_text(encoding="utf-8"))
    harm = retrieval_harm_comparison(args.historical_commit)
    automatic_intent_harm = automatic_intent_harm_comparison(
        args.study_items,
        args.always_rewrite_packet,
    )
    result = build_final_result(
        qrecc,
        nonhuman,
        integrated,
        human,
        reranking_analysis,
        harm,
        automatic_intent_harm,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
