Goal — determine whether conversational query rewriting improves retrieval.
Dataset — QReCC; 200 training examples and 188 test examples.
Method — BM25 retrieval, gold human rewrites, intent-preservation guard, and rewrite-benefit predictor.
Results — 64.9% original Hit@10 vs. 80.3% rewritten Hit@10; 29 rewrite-only wins; predictor accuracy 60.1%.
Limitation — the free version uses QReCC human rewrites; the paid hosted OpenAI call was implemented but not run.