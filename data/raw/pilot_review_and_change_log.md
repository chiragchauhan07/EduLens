# Pilot review (20 sessions) and change log

Source: `edulens_expanded_pilot_020.json` (v1, unchanged). Revised: `edulens_expanded_pilot_020_v2.json`. Every edit (20 entries, with old and new values and reason) is in `pilot_change_log.json`. This is a self-review by the generating model plus script checks. It is **not** an independent review and nothing here is human-verified.

## 1. Technically incorrect or imprecise teacher explanations (4 found, all fixed)

| Session | Problem | Fix |
|---|---|---|
| EXP_0009 t2 | Said duplicate ACKs are "for 2"; ACKs name the next expected segment | now "asking for segment 3 again" |
| EXP_0013 t8 | Reason why constructors cannot be virtual (vptr wording) was muddled | rewritten: construction sets up the vptr that dispatch depends on |
| EXP_0014 t2 | "A* would expand only nodes on an optimal path" is too strong (ties) | "essentially only" |
| EXP_0005 t3 | Teacher promised an example the student then supplied (incoherent) | promise removed |

EXP_0013 t6 (teacher says deletion through Base* "leaks") is an intentional error that the student challenges and the teacher concedes in t7; it is kept. All numeric examples were re-checked (knapsack table, F1 = 0.727, gradient 8 and w = 2.2, eigenpairs, loop counts 11 and n²·log n, T(n)=T(n/2)+n = Θ(n)). I found no other factual errors, but a subject expert should still check them.

## 2. Questionable labels (none changed, 5 flagged)

Flagged in `generation_meta.qa_flag` for priority human review:

- EXP_0001, EXP_0006, EXP_0009: partial vs understood. Final restatement or most reasoning is correct, but the student relied on teacher correction or slipped at the end.
- EXP_0011, EXP_0015: confused vs partial. The student derives some correct pieces (two-layer gradient; a working fix) but the final explanation is still conceptually wrong.

I kept these labels because the end-of-session student statements are the evidence, but they are exactly the cases where a reviewer may disagree.

## 3. Rationale and annotation mismatches (5 fixed)

- EXP_0009: `misconception_resolved = true` was too generous because the student ends with the Reno error; now false, rationale updated.
- EXP_0001: rationale was abstract; rewritten to quote the student's actual claims.
- EXP_0018: `student_confidence = high` had no explicit cue ("Am I right?" is a question); now `unknown`.
- EXP_0012: `moderate` changed to `low` (student says they have no intuition).
- EXP_0020: three turns re-tagged as `answer_attempt` / `explanation_attempt` (student claims were inside questions), giving the label enough visible student evidence.

## 4. Unrealistic or repetitive patterns (partly addressed, rest handled in the strategy)

- 18 of 20 sessions use the same teacher behaviour (direct confirm/correct). Now 2 are `lenient` (EXP_0005, EXP_0020). The full plan mandates a mix.
- Teacher openers across 206 turns: "Yes" 18, "Not" 15, "That's" 14, "No" 14, "Exactly" 9, "Right" 7, "Correct" 6; this is repetitive and verdict-driven.
- Confused students are wrong in almost every turn and end with a wrong "so…" summary (5 of 7 confused sessions' last message starts "ok/so/hmm"); understood students rarely slip mid-conversation.
- Misconceptions appear in 16 of 20 sessions (target about 55–60%); engagement is `high` in 12 of 20 (target about 35%). Confused and partial sessions without a specific misconception are missing.
- All sessions end tidily with a clear last turn.
- Not fixed in the pilot (would need new conversations): mid-conversation slips for understood students, correct turns inside confused sessions, sessions with no identifiable misconception. These are mandatory in the full generation plan.

## 4b. Possible shortcuts for a text classifier (pilot, n = 20; indicative only)

| Signal | understood | partial | confused |
|---|---:|---:|---:|
| Share of teacher messages with correction cues | 0.17 | 0.24 | 0.44 |
| Final teacher message: praise / correction / other | 2 / 1 / 3 | 1 / 3 / 3 | 1 / 5 / 1 |
| Last student turn flagged as confused | 0 of 6 | 5 of 7 | 7 of 7 |
| Last student message contains thanks | 3 | 0 | 4 |
| Student attempt turns (share) | 0.55 | 0.50 | 0.54 |

- The teacher-correction rate and the final teacher message are real shortcuts for the `full` view (the teacher reacts to the student's correctness). Mitigations: lenient, socratic and minimal teacher styles, and closing messages that do not mirror the label (done for 5 pilot sessions).
- The pilot's `confusion_flag` pattern (last turn flagged in every confused session) is the same artefact as in the legacy data. It is excluded from the model, and the full plan randomises when flags occur.
- `reasoning_quality` (Cramér's V 0.87), `misconception_present` (0.76) and `scenario_tag` (0.96, by design) are strongly associated with class in this sample. They are annotation or QA fields, not inputs, but the quotas in the strategy are designed to overlap them across classes.
- Sentiment is no longer deterministic: final-turn sentiment crosses all classes (confused: 1 positive, 2 neutral, 3 negative, 1 mixed).
- With 20 sessions none of these statistics is reliable; the validator labels them `not_interpretable`.
- No lexical probe finding: only 3 tokens ("code", "returns", "next") are concentrated in one class (all `understood`, topic words, n ≤ 4).

## 5. Validator results on the revised pilot

Structural: pass (20 of 20, no duplicates, `data_source` present everywhere, no missing or invalid fields). Semantic screen: 0 warnings after fixes. I verified the screen fires by injecting faults (missing or invalid `data_source`, self_correction without turns, a claimed self-correction on a teacher-corrected turn, mismatched confidence, misconception without evidence): each was caught. Zero warnings still do not mean the labels are right.
