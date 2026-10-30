
from bloom_search.rewrite import rewrite_query

examples = [
    {
        "query": "When was he born?",
        "context": ["Who is Alan Turing?"],
    },
    {
        "query": "What is machine learning?",
        "context": [],
    },
    {
        "query": "Where did she study?",
        "context": ["Tell me about Marie Curie."],
    },
]

for example in examples:
    result = rewrite_query(
        original_query=example["query"],
        context=example["context"],
    )

    print("\nOriginal:", result.original_query)
    print("Final:", result.rewritten_query)
    print("Accepted:", result.accepted)
    print("Problems:", result.problems)
