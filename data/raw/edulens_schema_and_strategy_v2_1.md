# EduLens: schema v2.1 and revised generation strategy

Supersedes `edulens_inspection_schema_and_strategy.md` (kept for history). Nothing beyond the 20-session pilot has been generated.

## 1. Revised target and class counts

| | understood | partial | confused | total |
|---|---:|---:|---:|---:|
| Revised target (new synthetic data only) | 1,000 | 1,000 | 1,000 | **3,000** |
| Already generated (pilot, batch_000) | 6 | 7 | 7 | 20 |
| Still to generate | 994 | 993 | 993 | **2,980** |

- 30 batches: `edulens_expanded_batch_001` to `_029` with 100 sessions each, `_030` with 80. Pilot = batch_000 (counts toward the 3,000 once approved).
- The original 430 records (`data_source = original_legacy`) are **not** part of this target and are never merged automatically.
- The earlier plan (2,570 additional sessions to reach 1,000 per class with the legacy 430 included) is withdrawn.

### Implications of this decision

1. **No gold labels exist.** All 3,000 labels are LLM-authored and script-screened. Any model trained on them learns to reproduce our annotation guide, not to measure real students. Results must be reported as "agreement with synthetic labels" until a human-labelled or real test set exists.
2. **Classes are balanced (33/33/33).** The legacy prior was 45/36/19, and real classrooms are unlikely to be balanced. Use macro-F1 and per-class recall, and do not read accuracy as deployment accuracy; priors will need recalibration before real use.
3. **Cost.** 430 more sessions to generate than under the merge plan (about 17% more work), in exchange for removing unreliable labels.
4. **Single generator risk.** One LLM writes all text, so style homogeneity is likely. Mitigations: teacher styles, personas, scenario matrix, and (recommended) a small human-written seed set of 100–150 conversations as an additional test source.
5. **Legacy data stays usable**, see section 5.

## 2. Schema v2.1

Everything from schema 2.0 stays. **New in 2.1:** `data_source`, `self_correction_turns`, `generation_meta.teacher_style`, `generation_meta.qa_flag`, `quality_review.automated_semantic_screen`, `quality_review.human_review`.

Session level

| Field | Values | Notes |
|---|---|---|
| `chat_id` | `EXP_0001`… | prefix must match `data_source` (`EXP_` synthetic_llm, `CHAT_` original_legacy, `REAL_` real_student) |
| **`data_source`** | `synthetic_llm`, `original_legacy`, `real_student` | mandatory; one value per file; validator rejects missing or malformed values |
| `student_id` | `SYN_S001`–`SYN_S200` | random, never class-linked, group key for splits |
| `timestamp`, `llm_client` | as original | metadata only |
| `subject` | Python, C, C++, Java, DSA, DAA, DBMS, OS, CN, ML, AI, DL, NLP, DataSci, Maths, Cloud | original 11 codes + C, AI, DL, NLP, DataSci |
| `topic`, `main_concept`, `confusable_concept` | free text, consistent naming | as original |
| `student_persona` | beginner, skeptical, exam, casual, curious, debug, overconfident, hesitant, terse, explainer | |
| `language_style` | english_casual, english_standard, hinglish_light, hinglish_heavy | |
| `difficulty_level`, `task_type` | see validator enums | |
| `num_student_turns`, `confusion_points` | ints | points = sum of per-turn flags |
| `final_understanding` | understood, partial, confused | the only primary target; state at end of session |
| `overall_sentiment` | positive, mixed, negative | derived from turn sentiments; **not** an input |
| `self_correction` | yes, no | yes iff `self_correction_turns` is non-empty |
| **`self_correction_turns`** | list of turn numbers | turns where the student corrects their own earlier claim |
| `engagement_level`, `student_confidence`, `reasoning_quality` | enums as before | confidence only when explicitly expressed, else `unknown` |
| `misconception_present`, `misconception_category`, `misconception_resolved` | bool, snake_case or null, bool or null | |
| `turns[]` | turn, student_message, teacher_message, llm_message_context, question_type, cognitive_level, sentiment, confusion_flag | one exchange per turn |
| `quiz` | num_questions (10–15), correct, score_percent = round(100·correct/num_questions) | independent-ish of label, never a feature |
| `annotation_rationale` | string | QA only |
| `quality_review` | `automated_structural_check`, `automated_semantic_screen`, `checks_run`, `human_verified` (false), `human_review` (null) | `human_verified` may be true only with `human_review.reviewers >= 2` and `adjudicated: true` |
| `generation_meta` | `schema_version`, `batch_id`, `scenario_tag`, **`teacher_style`** (direct_corrector, socratic, lenient, minimal), **`qa_flag`**, `text_source`, `use_as_training_feature: false` | |

### Annotation rules that are now checkable

- **self_correction = yes**: the student states that an earlier claim of their own was wrong, and the immediately preceding teacher message neither states nor hints at the fix. Corrections made after a teacher correction or hint do not count and are listed as `no`.
- **student_confidence**: set only when the student explicitly expresses confidence or doubt ("not very confident", "obviously", "idk if that's right"); assertive wrong claims and questions such as "Am I right?" are `unknown`.
- **misconception_resolved**: true only if the student's last turn no longer shows that misconception.

## 3. Model input restrictions

The primary model predicts `final_understanding` from conversation text only. Two views are exported by `edulens_tools.py model-inputs`:

- `student_only`: student messages only (primary evidence)
- `full`: student and teacher messages

Excluded from the input in both views (also listed in every validator report): final_understanding (target), overall_sentiment, per-turn sentiment, confusion_flag, confusion_points, quiz, reasoning_quality, misconception_present / category / resolved, self_correction(_turns), engagement_level, student_confidence, annotation_rationale, quality_review, generation_meta, chat_id, student_id, timestamp, data_source, llm_client, and `llm_message_context` (teacher-derived). Rows are exported with shuffled opaque ids; `student_id` is exported only as `group_id` for grouped train/test splitting.

Design implications for generation: the label must be recoverable from what the **student** says (attempts, explanations, answers, challenges); at least 2 such turns per session; teacher text must not mirror the label (see section 4).

## 4. Generation strategy (changes after the pilot review)

1. Stratified quota grid, independent of class: subject, length, persona, language style, emoji rate, difficulty, task type, teacher style, last-message type. Targets per class are identical; natural noise of about ±10% is allowed.
2. Totals for 3,000 sessions: subject DSA 300, DAA 270, Python 240, ML 240, DBMS 210, OS 210, C++ 180, Java 180, CN 180, DL 180, Maths 180, C 150, NLP 150, DataSci 150, AI 120, Cloud 60. Length short 1,200 / medium 1,200 / long 600 (400/400/200 per class). Style english_casual 45%, english_standard 27%, hinglish_light 18%, hinglish_heavy 10%. About 30% of student messages carry an emoji.
3. **Teacher style** (new, independent of class): direct_corrector 45%, socratic 25%, lenient 20%, minimal 10%. Lenient teachers sometimes miss student errors; socratic ones ask instead of confirming. This removes the pilot's "praise = understood, correction = confused" cue.
4. **Last student message type** randomised across classes: question, statement, thanks, trailing off, no summary. No rule that confused students end on a wrong "so…" summary.
5. **Error patterns**: at least 40% of understood sessions contain a flagged slip mid-conversation; at least 25% of confused sessions contain correct turns; not every confused turn is wrong.
6. Teacher openers capped (no opener above 8% of turns dataset-wide); stock phrases not reused.
7. Annotation design targets (conditional on class, overlapping by design): reasoning strong/moderate/weak 60/36/4 (understood), 15/60/25 (partial), 3/27/70 (confused); misconception_present 25% / 70% / 80%, but dataset-wide about 55–60%, with confused or partial sessions that have **no** identifiable misconception; engagement high/moderate/low about 35/40/25 overall; self_correction yes 24/17/8%.
8. Quiz scores: overlapping class-conditional distributions (means 66/58/50, SD 17), independent of text.
9. Each batch is validated and the manifest is updated before the next batch starts; generation pauses for review after batch_001 and after any batch with a diagnostic warning.

## 5. Legacy 430 records

Plan, to run separately from generation:

1. Keep files untouched; tag as `original_legacy` in any converted copy; never merge into the synthetic file.
2. Blind audit: 2 independent reviewers label a stratified random sample of 90 legacy sessions (30 per legacy class) using the annotation guide, **without** seeing sentiment, confusion fields or labels.
3. Compute Cohen's kappa between reviewer consensus and the legacy labels (`edulens_tools.py agreement`).
4. Decision rule: kappa below 0.40 → discard legacy labels (keep texts only for style or unlabelled use); 0.40–0.60 → relabel all 430 by humans if they are wanted; above 0.60 → usable as a separately reported, weakly-labelled out-of-distribution set. Expect low agreement, because most legacy sessions contain only questions and no demonstrated understanding.
5. Report any model result on legacy data only as a separate line, never pooled.

## 6. Independent quality assurance (no human verification yet)

All current labels are synthetic and unverified. Proposed process:

1. **Calibration**: two reviewers label the 20 pilot sessions blind (`review-sheet`), discuss, and refine the annotation guide.
2. **Sampling**: 10 sessions per batch, stratified by class and over-sampling `qa_flag` and high-risk scenarios (fake_understanding, solves_but_cannot_explain, lenient teacher); about 300 sessions (10%) in total.
3. **Reviewers**: two people who did not write or prompt the data, blind to label, rationale and metadata; one subject-matter reviewer per subject checks `teacher_message` accuracy (about 5% per subject).
4. **Metrics**: Cohen's kappa for `final_understanding` ≥ 0.70 and ≥ 0.60 for each annotation field; reviewer-vs-dataset accuracy ≥ 85%. Below 0.60 on a batch → pause, revise the guide, regenerate the affected scenario/class cells, re-sample.
5. **Adjudication** by a third reviewer; only records reviewed by ≥ 2 humans and adjudicated may set `human_verified = true`. All others stay false.
6. **Gold subset**: adjudicated sessions form a held-out evaluation set, excluded from training. An LLM audit by a different model may be added as an extra signal but is recorded as `llm_audit`, never as human verification.
7. About 40 reviewer-hours for 300 items at roughly 4 minutes each, two reviewers.

## 7. Validation layers (see `validate_edulens.py`)

- **Structural** (errors, decide the exit code): required fields, enums, `data_source`, id formats and prefix rules, turn numbering, derived fields, annotation shape, self_correction ⇔ turns, quiz arithmetic, no class names in ids, `human_verified` rules.
- **Semantic screening** (heuristic warnings, never a verdict): self-correction markers and teacher-supplied corrections, misconception evidence, label vs annotation plausibility, explicit confidence cues, rationale grounding, language-style markers, evidence of student attempts, within-session repetition. These rules produce false positives and false negatives.
- **Leakage diagnostics** (descriptive): marked `not_interpretable` for n < 100 and `indicative_only` for n < 1,000. Small samples cannot establish reliable correlations, and absence of warnings is not evidence of absence of shortcuts.
- A passing run never means the data is educationally correct.
