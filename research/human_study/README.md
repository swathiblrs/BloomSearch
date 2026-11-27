# BloomSearch NLP Human Study

## Purpose

This within-subject pilot supplies the human metrics that retrieval logs cannot
provide honestly:

1. three-class `keep` / `rewrite` / `clarify` Macro-F1;
2. semantic harmful-rewrite rate;
3. clarification usefulness;
4. perceived task success, trust, and participant agreement.

Recruit at least two real participants. Do not invent or pre-fill responses.

## Freeze the study before P01

Run the item generator once before the first participant:

```bash
python research/human_study/prepare_items.py --seed 42
```

This writes `study_items.csv`, the **private study key**. It contains the hidden
`system_action`, dataset source, and record identifiers. Never send or show this
file to participants. Once P01 starts, do not regenerate or edit the key.

The fixed study contains 30 items: 10 system `keep`, 10 `rewrite`, and 10
`clarify` decisions. Keep and rewrite items are reproducibly sampled across
QReCC conversations. Clarification items are reproducibly sampled from
AmbigNQ-derived validation data, and their questions present concrete
interpretations recorded in that dataset instead of using a keyword template.

## Create a blinded participant packet

Create a separately randomized packet for each participant:

```bash
python research/human_study/prepare_participant_packet.py P01
python research/human_study/prepare_participant_packet.py P02
```

Each command creates three files under `participant_packets/`:

- `P01_phase_a_actions.csv`: queries and context only, for choosing the action;
- `P01_phase_b_outputs.csv`: the same items with the proposed system text;
- `P01_responses.csv`: the participant's response sheet.

Participant packets use opaque identifiers such as `Q01`. They never include
`system_action`, dataset labels, retrieval metrics, or private source IDs. Do
not give participants `study_items.csv`.

## Participant instructions

Complete the study in two phases. Do not open Phase B until every
`correct_action` response in Phase A has been recorded. This prevents the
presence of a rewrite or clarification from revealing the system's action.

### Phase A: action selection

Use only `P01_phase_a_actions.csv`. For every item, read the query and context,
then choose `keep`, `rewrite`, or `clarify` in the response sheet. Do not change
these choices after opening Phase B.

### Phase B: output evaluation

Open `P01_phase_b_outputs.csv` and evaluate the displayed output for the same
items. If both proposed-output fields are blank, the system kept the original
query unchanged.

1. If a proposed rewrite is shown, record whether it preserves the user's
   intent, entities, dates, locations, numbers, comparisons, and constraints.
2. If a clarification is shown, rate its usefulness from 1 (not useful) to 5
   (very useful).
3. Record whether the displayed action gives enough information to proceed
   with the search: 1 for yes or 0 for no.
4. Rate trust in the displayed action from 1 to 5.

The task-success field is collected for all 30 items. Because participants do
not execute a search and verify an answer, the report must call this metric
**perceived task success** or **action sufficiency**, not observed task success.

Participants may reasonably disagree with the system. Clear-looking ambiguity
items and context-dependent keep items must not be explained or relabeled
during a session; those disagreements are study findings.

## Analyze completed responses

After every participant completes the study, pass each response file to the
analysis command. Do not add `system_action`; the analysis joins it from the
private key.

Run:

```bash
python research/human_study/analyze_responses.py \
  research/human_study/participant_packets/P01_responses.csv \
  research/human_study/participant_packets/P02_responses.csv \
  --study-key research/human_study/study_items.csv \
  --output research/results/human_study.json
```

## Reported metrics

- **Action Macro-F1:** hidden system action versus each participant's selected
  correct action.
- **Harmful-rewrite rate:** fraction of displayed rewrites marked as not
  preserving intent.
- **Clarification usefulness:** mean 1--5 rating and fraction rated at least 4.
- **Primary perceived task success:** success rate for system clarification
  trials, matching the original study plan.
- **Secondary perceived task success:** overall and separate rates for keep,
  rewrite, and clarify trials.
- **Trust:** mean 1--5 rating.
- **Agreement:** fraction of items for which participants unanimously chose the
  same action.

## Completion status

P01 and P02 completed all 30 items. The final machine-readable metrics are in
`research/results/human_study.json`, and their interpretation, research-
question conclusions, and limitations are documented in
`research/HUMAN_STUDY_RESULTS.md`. The two-participant pilot is complete. No
P03 response is required by the study protocol.

## Optional paired-human RQ3 extension

The original pilot rated intent preservation only when the selective policy
showed a rewrite. The official RQ3 now has a negative automatic answer from
both retrieval harm and a common intent-guard comparison. For a stronger
paired-human replication, P01 and P02 may complete one short Phase C over the
20 QReCC study items. This is optional and is not a P03 study.

Generate the packets with:

```bash
python research/human_study/prepare_always_rewrite_baseline.py P01
python research/human_study/prepare_always_rewrite_baseline.py P02
```

Each participant rates `always_rewrite_preserves_intent` as `yes` or `no` for
every item without changing their earlier answers. After both response sheets
are returned, run:

```bash
python research/human_study/analyze_always_rewrite_baseline.py \
  research/human_study/participant_packets/P01_phase_c_always_rewrite_responses.csv \
  research/human_study/participant_packets/P02_phase_c_always_rewrite_responses.csv \
  --original-responses \
  research/human_study/participant_packets/P01_responses.csv \
  research/human_study/participant_packets/P02_responses.csv
```

This produces the paired selective-versus-always semantic harmful-rewrite
rates and an exact McNemar comparison. Do not invent or pre-fill Phase C
responses.
