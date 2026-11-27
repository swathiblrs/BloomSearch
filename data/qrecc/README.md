# QReCC research data

The QReCC dataset and all generated artifacts are intentionally excluded from
Git because they are large, reproducible, and not needed by the BloomSearch
application.

To reproduce the query-rewriting experiments:

1. Install the optional research dependencies:

   ```bash
   python -m pip install -e ".[research]"
   ```

2. Download QReCC from its official distribution and place the training and
   test JSON files in this directory as `qrecc-training.json` and
   `qrecc-test.json`.
3. Prepare fixed training and test subsets:

   ```bash
   python scripts/prepare_qrecc.py --source data/qrecc/qrecc-training.json --limit 12000 --output data/qrecc/training_12000.json
   python scripts/prepare_qrecc.py --source data/qrecc/qrecc-test.json --limit 3000 --output data/qrecc/test_3000.json
   ```

4. Measure original-versus-gold-rewrite retrieval for both subsets:

   ```bash
   python scripts/evaluate_gold_rewrites.py --input data/qrecc/training_12000.json --output data/qrecc/training_12000_results.json
   python scripts/evaluate_gold_rewrites.py --input data/qrecc/test_3000.json --output data/qrecc/test_3000_results.json
   ```

5. Train and validate the benefit predictor, then evaluate it on the fixed test
   subset:

   ```bash
   python scripts/train_hybrid_with_validation.py
   python scripts/evaluate_selective_rewriting.py
   ```

Files such as raw datasets, prepared samples, indexes, mappings, joblib models,
and per-query result dumps remain ignored. Only consolidated aggregate metrics
are versioned under `research/results/query_rewriting/`.
