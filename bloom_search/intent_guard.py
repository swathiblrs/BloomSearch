
import re


QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how",
    "which",
}

RESTRICTIONS = {
    "city", "country", "state", "year", "month",
    "before", "after", "only", "exactly", "first",
    "last", "most", "least",
}

CONTENT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "of", "for", "to",
    "in", "on", "at", "by", "with", "from", "about", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "can", "could", "should", "would", "will",
    "may", "might", "must", "what", "who", "when", "where", "why",
    "how", "which", "this", "that", "these", "those", "it", "he",
    "she", "they", "him", "her", "them", "his", "hers", "their",
    "its", "i", "you", "we", "me", "us", "my", "your", "our",
}


def _content_stem(token: str) -> str:
    """Apply small, dependency-free normalization for anchor comparison."""
    token = token.lower().rstrip("'s")
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _content_anchors(text: str) -> set[str]:
    return {
        _content_stem(token)
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z']+\b", text)
        if token.lower() not in CONTENT_STOPWORDS
        and token.lower() not in RESTRICTIONS
    }


def _numbers_and_dates(text: str) -> set[str]:
    pattern = (
        r"\b\d+(?:\.\d+)?%?\b|"
        r"\b(?:january|february|march|april|may|june|"
        r"july|august|september|october|november|december)\b"
    )
    return set(re.findall(pattern, text.lower()))


def _question_type(text: str) -> str | None:
    words = re.findall(r"\b[a-z]+\b", text.lower())

    for word in words[:4]:
        if word in QUESTION_WORDS:
            return word

    return None

def _important_entities(text: str) -> set[str]:
    """Find capitalized names, excluding sentence-start words."""
    entities = set()

    for match in re.finditer(r"\b[A-Z][a-zA-Z]+\b", text):
        if match.start() == 0:
            continue

        word = match.group().lower()

        if word not in QUESTION_WORDS:
            entities.add(word)

    return entities
def _capitalized_name_phrases(text: str) -> set[str]:
    """Extract consecutive capitalized words as possible names."""
    return {
        match.group().lower()
        for match in re.finditer(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
            text,
        )
    }


def _question_form(text: str) -> str | None:
    """Identify yes/no questions versus WH questions."""
    words = re.findall(r"\b[a-z]+\b", text.lower())

    if not words:
        return None

    if words[0] in {
        "is", "are", "was", "were", "do", "does",
        "did", "can", "could", "should", "would",
        "will", "has", "have", "had",
    }:
        return "yes_no"

    if words[0] in QUESTION_WORDS:
        return "wh"

    return None

def is_answer_like_rewrite(
    original_query: str,
    rewritten_query: str,
) -> bool:
    """Detect when a question is rewritten as a statement."""

    original = original_query.strip()
    rewritten = rewritten_query.strip()

    question_starters = {
        "who", "what", "when", "where", "why",
        "how", "which", "is", "are", "do",
        "does", "did", "can", "could", "would",
        "should", "will", "was", "were",
    }

    original_first = re.findall(
        r"\b\w+\b", original.lower()
    )
    rewritten_first = re.findall(
        r"\b\w+\b", rewritten.lower()
    )

    if not original_first or not rewritten_first:
        return False

    original_is_question = (
        original.endswith("?")
        or original_first[0] in question_starters
    )

    rewritten_is_question = (
        rewritten.endswith("?")
        or rewritten_first[0] in question_starters
    )

    return original_is_question and not rewritten_is_question



def _rewrite_quality_problems(
    original_query: str,
    rewritten_query: str,
) -> list[str]:
    problems = []

    original_words = original_query.split()
    rewritten_words = rewritten_query.split()

    # Reject excessively long generated text.
    if len(rewritten_words) > 40:
        problems.append("Rewrite is too long")

    # Reject unusually large expansions.
    if len(rewritten_words) > max(
        20, len(original_words) * 4
    ):
        problems.append("Rewrite expanded excessively")

    # Reject multiple sentences that may contain
    # copied answer text or explanatory passages.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        rewritten_query.strip()
    )

    if len(sentences) > 1:
        problems.append(
            "Rewrite contains multiple sentences"
        )

    return problems


def preserves_intent(
    original_query: str,
    rewritten_query: str,
    context: list[str] | None = None,
) -> tuple[bool, list[str]]:

    problems = []
    problems.extend(
        _rewrite_quality_problems(
            original_query,
            rewritten_query,
        )
    )
    if is_answer_like_rewrite(original_query, rewritten_query):
        problems.append(
            "Rewrite changed a question into a statement"
        )

    original_type = _question_type(original_query)
    rewritten_type = _question_type(rewritten_query)

    if (
        original_type
        and rewritten_type
        and original_type != rewritten_type
    ):
        problems.append(
            f"Question type changed: "
            f"{original_type} -> {rewritten_type}"
        )

    original_values = _numbers_and_dates(original_query)
    rewritten_values = _numbers_and_dates(rewritten_query)

    missing_values = original_values - rewritten_values

    if missing_values:
        problems.append(
            "Missing dates or numbers: "
            + ", ".join(sorted(missing_values))
        )

    original_words = set(
        re.findall(r"\b[a-z]+\b", original_query.lower())
    )

    rewritten_words = set(
        re.findall(r"\b[a-z]+\b", rewritten_query.lower())
    )

    added_restrictions = (
        rewritten_words & RESTRICTIONS
    ) - (
        original_words & RESTRICTIONS
    )

    if added_restrictions:
        problems.append(
            "Unsupported restrictions added: "
            + ", ".join(sorted(added_restrictions))
        )

    missing_restrictions = (
        original_words & RESTRICTIONS
    ) - (
        rewritten_words & RESTRICTIONS
    )

    if missing_restrictions:
        problems.append(
            "Original restrictions removed: "
            + ", ".join(sorted(missing_restrictions))
        )

    # Preserve the query's main lexical anchors. Contextual rewrites may add an
    # entity, but they should not silently discard half or more of a query's
    # topic-bearing words (for example, dropping "moves" from "What moves was
    # he known for?").
    original_anchors = _content_anchors(original_query)
    rewritten_anchors = _content_anchors(rewritten_query)
    missing_anchors = original_anchors - rewritten_anchors
    if (
        len(original_anchors) >= 2
        and bool(original_anchors & rewritten_anchors)
        and len(missing_anchors) / len(original_anchors) >= 0.5
    ):
        problems.append(
            "Rewrite dropped important query terms: "
            + ", ".join(sorted(missing_anchors))
        )

    # "Else" signals that the user wants an additional fact, not a repetition
    # of a previous question. A standalone rewrite must retain that constraint.
    if (
        "else" in original_words
        and not rewritten_words.intersection({"else", "other", "another", "additional"})
    ):
        problems.append("Rewrite removed the continuation meaning of 'else'")

    # Protect the semantic roles in a common but damaging reversal:
    # "Who was he accused of assaulting?" asks for the object, whereas
    # "Who was accused of assaulting Jack White?" asks for the subject.
    original_object_question = re.match(
        r"^\s*who\s+was\s+(?!accused\b).+?\s+accused\s+of\b",
        original_query,
        flags=re.IGNORECASE,
    )
    rewritten_subject_question = re.match(
        r"^\s*who\s+was\s+accused\s+of\b",
        rewritten_query,
        flags=re.IGNORECASE,
    )
    if original_object_question and rewritten_subject_question:
        problems.append("Rewrite reversed who performed and received the action")

    # Check whether the question form changed.
    original_form = _question_form(original_query)
    rewritten_form = _question_form(rewritten_query)

    if (
        original_form
        and rewritten_form
        and original_form != rewritten_form
    ):
        problems.append(
            f"Question form changed: "
            f"{original_form} -> {rewritten_form}"
        )

    # Check whether important named entities disappeared.
    original_entities = _important_entities(original_query)
    rewritten_entities = _important_entities(rewritten_query)

    missing_entities = original_entities - rewritten_entities

    if missing_entities:
        problems.append(
            "Missing named entities: "
            + ", ".join(sorted(missing_entities))
        )
        # Reject unsupported person/name substitutions from context.
    if context:
        context_text = " ".join(map(str, context))

        context_names = _capitalized_name_phrases(
            context_text
        )
        original_names = _capitalized_name_phrases(
            original_query
        )
        rewritten_names = _capitalized_name_phrases(
            rewritten_query
        )

        # Names appearing in the rewrite should come from
        # either the original question or the conversation.
        unsupported_names = (
            rewritten_names - context_names - original_names
        )

        if unsupported_names:
            problems.append(
                "Unsupported names introduced: "
                + ", ".join(sorted(unsupported_names))
            )
    # 1. Preserve comparisons.
    # For example, "easier than Spanish" should not
    # become simply "easier to learn Spanish".
    original_comparisons = re.findall(
        r"\b(?:than|versus|vs\.?)\b",
        original_query.lower(),
    )
    rewritten_comparisons = re.findall(
        r"\b(?:than|versus|vs\.?)\b",
        rewritten_query.lower(),
    )

    if original_comparisons and not rewritten_comparisons:
        problems.append(
            "Rewrite removed a comparison"
        )
    
    # 2. Detect prompt artifacts in generated rewrites.
    forbidden_patterns = [
        r"\bstandalone\b",
        r"\bcurrent question\b",
        r"\bprevious question\b",
        r"\brewritten query\b",
        r"\bcontext\s*:",
        r"\bquestion\s*:",
    ]

    for pattern in forbidden_patterns:
        if re.search(
            pattern,
            rewritten_query,
            flags=re.IGNORECASE,
        ):
            problems.append(
                "Rewrite contains prompt artifact"
            )
            break
    # Reject rewrites that narrow plural questions.
    if (
        re.search(r"\bdisadvantages\b", original_query.lower())
        and re.search(
            r"\b(main|primary|biggest|major)\s+disadvantage\b",
            rewritten_query.lower()
        )
    ):
        problems.append("Rewrite narrowed a plural question")



    return len(problems) == 0, problems
