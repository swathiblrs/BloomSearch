# Human Evaluation Results

## Scope

The BloomSearch pilot used a two-phase, within-subject design with two real
participants. Each participant evaluated the same 30 queries: 10 hidden system
`keep` decisions, 10 `rewrite` decisions, and 10 `clarify` decisions. Phase A
collected an independent action choice before Phase B revealed the proposed
rewrite or clarification. Both participants completed all items, producing 60
responses.

The study measures agreement with participant judgments and perceived output
quality. It does not establish population-level effectiveness or replace an
expert-adjudicated gold standard.

## Action-selection results

| Metric | Result |
|---|---:|
| Three-class Macro-F1 | 0.3818 |
| Keep F1 | 0.3043 |
| Rewrite F1 | **0.7805** |
| Clarify F1 | 0.0606 |
| Participant exact agreement | 56.67% |
| Cohen's kappa | 0.3500 |

The action confusion matrix counts participant selections for each hidden
system action:

| System action | Selected keep | Selected rewrite | Selected clarify |
|---|---:|---:|---:|
| Keep | 7 | 5 | 8 |
| Rewrite | 0 | 16 | 4 |
| Clarify | 19 | 0 | 1 |

The rewrite trigger aligns most strongly with participant judgments. The main
failure is over-triggering clarification: participants selected `keep` for 19
of the 20 system-clarify evaluations. The keep class is also unstable because
participants often preferred rewrite or clarification.

Per-participant Macro-F1 was 0.3761 for P01 and 0.3667 for P02. The similar
overall scores suggest the low combined score is not caused by only one
participant, although the class-level preferences differ.

## Output-quality results

| Metric | Result |
|---|---:|
| Rewrites evaluated | 20 |
| Harmful-rewrite rate | **45.00%** |
| Clarifications evaluated | 20 |
| Mean clarification usefulness | 3.55 / 5 |
| Clarifications rated at least 4 | 50.00% |
| Clarification perceived task success | **95.00%** |
| Overall perceived task success | 53.33% |
| Keep perceived task success | 30.00% |
| Rewrite perceived task success | 35.00% |
| Mean trust | 3.00 / 5 |

The current intent guard is not sufficient: participants marked 9 of 20 shown
rewrites as changing the user's intent. Clarification presents a more nuanced
result. Participants rarely agreed that the selected queries required
clarification, but once shown a clarification they usually considered it
sufficient to continue the task. This distinguishes a weak clarification
trigger from potentially useful clarification text.

## Research-question conclusions

1. **Joint action selection:** answered for the pilot. The current policy has
   low three-class agreement with participants, driven mainly by clarification
   over-selection. It should not be presented as a high-accuracy policy.
2. **Retrieval effectiveness:** answered by the QReCC and SciFact automatic
   experiments. Selective rewriting improves QReCC Hit@10 over original-query
   BM25, while selective reranking preserves rerank-all quality and reduces
   hosted calls.
3. **Intent preservation:** answered for the pilot. A 45% harmful-rewrite rate
   shows that the current rules require stronger semantic validation.
4. **Efficiency and user benefit:** the automatic evaluation answers calls,
   tokens, and latency. This study adds perceived task success and trust. The
   resulting trade-off is mixed rather than uniformly positive.

## Limitations

- The pilot has only two participants, as required by the course minimum.
- The participants are not an expert-adjudication panel.
- Repeated judgments from the same participants are not independent samples;
  therefore no population-level significance claim is made for human metrics.
- Perceived task success means that the displayed action seemed sufficient to
  continue. Participants did not execute the complete search task.
- The human study evaluates NLP decisions separately from the QReCC and
  SciFact retrieval benchmarks; it does not create one cross-dataset score.

Within those limits, the planned course-project evaluation is complete. The
results support a research conclusion that selective NLP can reduce hosted
reranking usage, but the current action policy and intent guard need further
work before deployment or a publication-level claim.
