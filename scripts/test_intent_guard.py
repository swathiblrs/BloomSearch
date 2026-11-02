
from bloom_search.intent_guard import preserves_intent

tests = [
    (
        "When was he born?",
        "When was Alan Turing born?",
        ["Who is Alan Turing?"],
    ),
    (
        "Where did she study?",
        "What city did Marie Curie study?",
        ["Tell me about Marie Curie."],
    ),
    (
        "Where did she study?",
        "Where did Marie Curie study?",
        ["Tell me about Marie Curie."],
    ),
]

for original, rewritten, context in tests:
    accepted, problems = preserves_intent(
        original,
        rewritten,
        context,
    )

    print("\nOriginal:", original)
    print("Rewritten:", rewritten)
    print("Accepted:", accepted)
    print("Problems:", problems)
