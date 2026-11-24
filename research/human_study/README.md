# BloomSearch NLP Human Study

## Purpose

This small within-subject pilot supplies the three metrics that cannot be
obtained honestly from retrieval logs alone:

1. three-class `keep` / `rewrite` / `clarify` Macro-F1;
2. semantic harmful-rewrite rate;
3. clarification usefulness and task success.

Recruit at least two participants, as required by the course pilot-study
instructions. Do not use invented responses.

The repository includes a balanced 30-item candidate set in `study_items.csv`.
Its actions come from the current integrated policy. Rewrite text is replayed
from stored experiment output, so preparing the study makes no new API calls.
Recreate it from the stored project artifacts with:

```bash
python research/human_study/prepare_items.py
```

## Study design

- Use 30 items: 10 proposed `keep`, 10 `rewrite`, and 10 `clarify` actions.
- Each participant evaluates all 30 items in randomized order.
- Hide the dataset label and retrieval metric from participants.
- Show the original query and conversational context first.
- For rewrite items, show the proposed rewrite.
- For clarification items, show the proposed clarification question.
- Record answers in a copy of `responses_template.csv`.

## Questions for each item

1. Which system action is most appropriate: keep, rewrite, or clarify?
2. If a rewrite is shown, does it preserve the user's intent, entities, dates,
   locations, numbers, comparisons, and constraints?
3. If a clarification is shown, rate its usefulness from 1 (not useful) to 5
   (very useful).
4. Could you complete the search task after the system action? Record task
   success as 1 or 0.
5. Rate trust in the system action from 1 to 5.

## Metrics

- **Action Macro-F1:** compare `system_action` with each participant's
  `correct_action` label.
- **Harmful-rewrite rate:** fraction of evaluated rewrites marked as not
  preserving intent.
- **Clarification usefulness:** mean 1–5 usefulness score and fraction rated at
  least 4.
- **Task success:** fraction of clarification trials with `task_success = 1`.
- **Trust:** mean trust score.

Run the analysis after collecting responses:

```bash
python research/human_study/analyze_responses.py responses.csv
```

The pilot is complete only after real participant responses have been collected
and reported with the participant count and study limitations.
