
from .api import HostedModelClient
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from .intent_guard import preserves_intent

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = (
    ROOT / "data" / "qrecc" / "rewrite_hybrid_model.joblib"
)

THRESHOLD = 0.50
def prepare_context(context, max_turns=6):
    if not context:
        return ""

    # QReCC context alternates questions and answers.
    # Keep only previous questions.
    questions = context[::2]

    recent_questions = questions[-max_turns:]

    return "\n".join(
        f"Previous question: {question}"
        for question in recent_questions
    )

@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    accepted: bool
    problems: list[str]


@lru_cache(maxsize=1)
def load_predictor():
    return joblib.load(MODEL_FILE)


@lru_cache(maxsize=1)
def load_rewriter():
    from transformers import pipeline

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1,
    )

import re

def needs_context_resolution(query: str) -> bool:
    """Detect queries likely to depend on earlier turns."""
    patterns = [
        r"\b(he|him|his|she|her|hers|they|them|their)\b",
        r"\b(it|its|one|ones|those|these)\b",
        r"\bwhat about\b",
        r"\bhow about\b",
        r"\bwhat about in\b",
        r"\bhow much longer\b",
    ]

    return any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in patterns
    )


def build_rewrite_prompt(
    original_query: str,
    context: list[str],
    prompt_version: str = "baseline",
) -> str:

    if prompt_version == "baseline":
        context_text = " ".join(map(str, context[-3:]))

        return (
            "Rewrite the final question as a standalone search query. "
            "Use the context to resolve pronouns. "
            "Preserve names, dates, and constraints. "
            "Do not answer the question.\n"
            f"Context: {context_text}\n"
            f"Question: {original_query}\n"
            "Standalone query:"
        )

    if prompt_version == "improved":
        context_text = prepare_context(context)

        return (
            "Task: Rewrite the current question into one clear, "
            "standalone question for information retrieval.\n"
            "Rules:\n"
            "1. Output exactly one question, not an answer.\n"
            "2. Keep the original meaning and question type.\n"
            "3. Use conversation context only to resolve references.\n"
            "4. Preserve names, dates, locations, and numbers.\n"
            "5. Do not introduce new facts or assumptions.\n"
            "6. Do not add explanations or extra sentences.\n"
            "7. Never include phrases like 'standalone query'.\n"
            "8. Keep the rewrite short and focused.\n"
            "9. If the question is already standalone, "
            "return it unchanged.\n"
            "10. If context is insufficient, do not guess.\n"
            "\n"
            "Examples:\n"
            "Context: Alan Turing was a mathematician.\n"
            "Current question: When was he born?\n"
            "Standalone question: When was Alan Turing born?\n"
            "\n"
            "Context: We are discussing Marie Curie.\n"
            "Current question: Where did she study?\n"
            "Standalone question: Where did Marie Curie study?\n"
            "\n"
            "Context: We are discussing depression.\n"
            "Current question: What causes it?\n"
            "Standalone question: What causes depression?\n"
            "\n"
            "Now rewrite only the following question.\n"
            f"{context_text}\n"
            f"Current question: {original_query}\n"
            "Standalone question:"
        )

    raise ValueError(
        f"Unknown prompt version: {prompt_version}"
    )


def rewrite_query(
    original_query: str,
    context: list[str] | None = None,
    model: str | None = None,
    force_rewrite: bool = False,
    prompt_version: str = "baseline",
    hosted_client: HostedModelClient | None = None,
) -> RewriteResult:
    context = context or []

    # Use the same context formatting as training.
    context_text = " ".join(map(str, context[-3:]))
    text = context_text[-500:] + " [QUERY] " + original_query
    rewrite_context = prepare_context(context)

    # Only load the predictor when needed.
    if not force_rewrite:
        bundle = load_predictor()

        from scripts.train_rewrite_benefit_predictor import features

        row = {
            "original_query": original_query,
            "context": context,
        }

        numeric = features(row)

        if isinstance(numeric, dict):
            numeric = list(numeric.values())

        X_text = bundle["vectorizer"].transform([text])

        X_num = bundle["scaler"].transform(
            np.asarray([numeric], dtype=float)
        )

        X = hstack(
            [X_text, csr_matrix(X_num)],
            format="csr",
        )

        classifier = bundle["model"]
        class_index = list(classifier.classes_).index(1)

        probability = classifier.predict_proba(
            X
        )[0, class_index]

        # Skip rewriting if predicted benefit is low.
        needs_resolution = (
             bool(context)
             and needs_context_resolution(original_query)
        )
        if probability < THRESHOLD and not needs_resolution:
            return RewriteResult(
                original_query=original_query,
                rewritten_query=original_query,
                accepted=False,
                problems=["Rewrite not recommended by predictor"],
            )

    # Generate a rewrite using the free local model.
    # Generate the prompt using the selected version.
    # Generate a rewrite using the selected backend.
    # 
    prompt = build_rewrite_prompt(
        original_query,
        context,
        prompt_version=prompt_version,
    )
    if hosted_client is not None:
        response = hosted_client.chat_json(
            system=(
                "You rewrite conversational search queries. "
                "Preserve the user's intent, entities, dates, "
                "locations, and constraints. "
                "Return JSON only with one string field "
                "named rewritten_query."
            ),
            user=prompt,
            temperature=0,
        )
        
        if (
            not isinstance(response, dict)
            or not isinstance(response.get("rewritten_query"), str)
        ):
            raise ValueError(
                "Hosted API returned an invalid query rewrite"
            )
        rewritten = response["rewritten_query"].strip()
    else:
        generator = load_rewriter()
        output = generator(
            prompt,
            max_new_tokens=64,
            do_sample=False,
        )
        rewritten = output[0]["generated_text"].strip()
    # Check that rewriting preserves user intent.
    accepted, problems = preserves_intent(
        original_query,
        rewritten,
        context,
    )

    # Reject empty or unchanged rewrites.
    if not rewritten:
        accepted = False
        problems = list(problems) + ["Empty rewrite generated"]

    elif rewritten.casefold() == original_query.strip().casefold():
        accepted = False
        problems = list(problems) + ["Query unchanged"]

    return RewriteResult(
        original_query=original_query,
        rewritten_query=rewritten if accepted else original_query,
        accepted=accepted,
        problems=list(problems),
    )
