"""Hosted conversational query rewriting with intent-preserving fallback.

Production rewriting is provider-neutral and API-only. Training and evaluating
rewrite-benefit predictors lives under ``scripts/`` and may use the optional
``research`` dependencies; importing this module never imports or downloads a
local language model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol, Sequence

from .intent_guard import preserves_intent


class HostedRewriteClient(Protocol):
    def chat_json(self, *, system: str, user: str, temperature: float = 0) -> object: ...


@dataclass(frozen=True, slots=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    accepted: bool
    problems: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "accepted": self.accepted,
            "problems": list(self.problems),
        }


def normalize_context(context: str | Sequence[str] | None) -> list[str]:
    if context is None:
        return []
    if isinstance(context, str):
        cleaned = context.strip()
        return [cleaned] if cleaned else []
    return [str(turn).strip() for turn in context if str(turn).strip()]


def prepare_context(context: str | Sequence[str] | None, max_turns: int = 6) -> str:
    """Format recent conversation turns without assuming a dataset-specific layout."""
    turns = normalize_context(context)[-max_turns:]
    return "\n".join(f"Previous turn: {turn}" for turn in turns)


def needs_context_resolution(query: str) -> bool:
    """Detect common references that make a query dependent on earlier turns."""
    patterns = (
        r"\b(he|him|his|she|her|hers|they|them|their)\b",
        r"\b(it|its|one|ones|those|these|that|this)\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bhow much longer\b",
    )
    return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)


def build_rewrite_prompt(
    original_query: str,
    context: str | Sequence[str] | None,
    prompt_version: str = "improved",
) -> str:
    turns = normalize_context(context)
    if prompt_version == "baseline":
        context_text = " ".join(turns[-3:])
        return (
            "Rewrite the final question as a standalone search query. "
            "Use the context only to resolve references. Preserve names, dates, locations, "
            "numbers, comparisons, and constraints. Do not answer the question.\n"
            f"Context: {context_text}\n"
            f"Question: {original_query}\n"
            "Standalone query:"
        )
    if prompt_version != "improved":
        raise ValueError(f"unknown prompt version: {prompt_version}")

    context_text = prepare_context(turns)
    return (
        "Rewrite the current question into one concise standalone search question.\n"
        "Rules:\n"
        "1. Return only the rewritten question.\n"
        "2. Preserve the user's meaning and question type.\n"
        "3. Use context only to resolve references.\n"
        "4. Preserve names, dates, locations, numbers, comparisons, and constraints.\n"
        "5. Do not introduce facts, assumptions, explanations, or answers.\n"
        "6. If context is insufficient, return the original question unchanged.\n"
        f"{context_text}\n"
        f"Current question: {original_query}\n"
        "Standalone question:"
    )


def rewrite_query(
    original_query: str,
    context: str | Sequence[str] | None = None,
    model: str | None = None,
    force_rewrite: bool = False,
    prompt_version: str = "improved",
    hosted_client: HostedRewriteClient | None = None,
) -> RewriteResult:
    """Rewrite through a caller-supplied hosted API and validate the result.

    ``model`` and ``force_rewrite`` remain for compatibility with earlier
    experiments. Model selection belongs to the hosted client. Without a
    hosted client, the original query is returned safely and no model is
    downloaded or executed.
    """
    del model, force_rewrite
    turns = normalize_context(context)
    if hosted_client is None:
        return RewriteResult(
            original_query=original_query,
            rewritten_query=original_query,
            accepted=False,
            problems=("Hosted API is required for query rewriting",),
        )

    response = hosted_client.chat_json(
        system=(
            "You rewrite conversational search queries. Preserve intent, entities, dates, "
            "locations, numbers, comparisons, and constraints. Return JSON only with one "
            "string field named rewritten_query."
        ),
        user=build_rewrite_prompt(original_query, turns, prompt_version),
        temperature=0,
    )
    if not isinstance(response, dict) or not isinstance(response.get("rewritten_query"), str):
        raise ValueError("hosted API returned an invalid query rewrite")

    rewritten = response["rewritten_query"].strip()
    accepted, problems = preserves_intent(original_query, rewritten, turns)
    problems = list(problems)
    if not rewritten:
        accepted = False
        problems.append("Empty rewrite generated")
    elif rewritten.casefold() == original_query.strip().casefold():
        accepted = False
        problems.append("Query unchanged")

    return RewriteResult(
        original_query=original_query,
        rewritten_query=rewritten if accepted else original_query,
        accepted=accepted,
        problems=tuple(dict.fromkeys(problems)),
    )
