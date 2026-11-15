#!/usr/bin/env python3
"""
EduLens dataset validator, schema v2.1.

Three clearly separated layers:

 1. STRUCTURAL validation  (errors; decides the exit code)
    JSON shape, required fields, enums, id formats, derived fields, quiz arithmetic.
    Passing it says ONLY that the file is well formed. It says nothing about educational correctness.

 2. SEMANTIC screening     (warnings; heuristic, never a pass/fail verdict)
    Keyword/rule-based consistency screens between annotations and conversation text
    (self-correction, confidence cues, misconception evidence, label-vs-annotation plausibility,
    language style, evidence of student reasoning). They produce false positives AND false negatives.
    Only an independent human review can verify that a label is educationally correct.

 3. LEAKAGE DIAGNOSTICS    (descriptive statistics)
    Association of class with metadata and surface features. With small n these are NOT reliable
    evidence of presence or absence of shortcuts; the report marks them 'not_interpretable'
    below n=100 and 'indicative' below n=1000.

Usage:
    python validate_edulens.py INPUT.json [--report OUT.json] [--expect U P C] [--fail-on-semantic]
"""
import argparse, json, math, re, statistics, sys
from collections import Counter, defaultdict
from itertools import combinations

SCHEMA_VERSION = "2.1"
CLASSES = ["understood", "partial", "confused"]
ENUMS = {
    "data_source": {"synthetic_llm", "original_legacy", "real_student"},
    "llm_client": {"Claude", "Gemini", "ChatGPT"},
    "final_understanding": set(CLASSES),
    "overall_sentiment": {"positive", "mixed", "negative"},
    "self_correction": {"yes", "no"},
    "engagement_level": {"high", "moderate", "low"},
    "student_confidence": {"high", "moderate", "low", "unknown"},
    "reasoning_quality": {"strong", "moderate", "weak"},
    "student_persona": {"beginner", "skeptical", "exam", "casual", "curious", "debug",
                        "overconfident", "hesitant", "terse", "explainer"},
    "language_style": {"english_casual", "english_standard", "hinglish_light", "hinglish_heavy"},
    "difficulty_level": {"basic", "intermediate", "advanced"},
    "task_type": {"concept_explanation", "programming_problem", "algorithmic_reasoning",
                  "mathematical_explanation", "practical_application", "debugging",
                  "conceptual_comparison"},
}
TEACHER_STYLES = {"direct_corrector", "socratic", "lenient", "minimal"}
TURN_ENUMS = {
    "question_type": {"definition", "application", "clarification", "comparison", "why", "feedback",
                      "evaluation", "quiz_request", "creation", "explanation_attempt",
                      "answer_attempt", "challenge", "debugging"},
    "cognitive_level": {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"},
    "sentiment": {"positive", "neutral", "negative", "mixed", "confused"},
}
SUBJECTS = {"Python", "C", "C++", "Java", "DSA", "DAA", "DBMS", "OS", "CN", "ML", "AI", "DL",
            "NLP", "DataSci", "Maths", "Cloud"}
REQUIRED = ["chat_id", "data_source", "student_id", "timestamp", "llm_client", "subject", "topic",
            "main_concept", "confusable_concept", "student_persona", "language_style",
            "difficulty_level", "task_type", "num_student_turns", "confusion_points",
            "final_understanding", "overall_sentiment", "self_correction", "self_correction_turns",
            "engagement_level", "misconception_present", "misconception_category",
            "misconception_resolved", "student_confidence", "reasoning_quality", "turns", "quiz",
            "annotation_rationale", "quality_review", "generation_meta"]
TURN_REQUIRED = ["turn", "student_message", "teacher_message", "llm_message_context", "question_type",
                 "cognitive_level", "sentiment", "confusion_flag"]
ID_PREFIX = {"synthetic_llm": "EXP_", "original_legacy": "CHAT_", "real_student": "REAL_"}
STUDENT_ID = re.compile(r"^SYN_S\d{3}$")
TS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")

# Fields that must never be inputs to the primary (text-only) understanding classifier.
FORBIDDEN_MODEL_FIELDS = [
    "final_understanding", "overall_sentiment", "turns[].sentiment", "turns[].confusion_flag",
    "confusion_points", "quiz", "reasoning_quality", "misconception_present",
    "misconception_category", "misconception_resolved", "self_correction", "self_correction_turns",
    "engagement_level", "student_confidence", "annotation_rationale", "quality_review",
    "generation_meta", "chat_id", "student_id", "timestamp", "data_source", "llm_client",
    "turns[].llm_message_context (teacher-derived; allowed only in the full-conversation view)"]


def length_bucket(n):
    return "short" if n <= 8 else "medium" if n <= 15 else "long"


def derive_overall_sentiment(turn_sents):
    n = len(turn_sents)
    score = (turn_sents.count("positive") - turn_sents.count("negative")) / n
    return "positive" if score >= 0.15 else "negative" if score <= -0.15 else "mixed"


# =============================================================== 1. STRUCTURAL
def validate_record(r):
    e = []
    for k in REQUIRED:
        if k not in r:
            e.append(f"missing field {k}")
    if e:
        return e
    ds = r["data_source"]
    if ds not in ENUMS["data_source"]:
        e.append(f"malformed data_source={ds!r}")
    elif not str(r["chat_id"]).startswith(ID_PREFIX[ds]):
        e.append(f"chat_id prefix does not match data_source {ds}")
    if not re.match(r"^[A-Z]+_\d{4,5}$", str(r["chat_id"])): e.append("bad chat_id format")
    if not STUDENT_ID.match(str(r["student_id"])): e.append("bad student_id format")
    if not TS.match(str(r["timestamp"])): e.append("bad timestamp format")
    if r["subject"] not in SUBJECTS: e.append(f"unknown subject {r['subject']}")
    for k, allowed in ENUMS.items():
        if r[k] not in allowed: e.append(f"invalid {k}={r[k]!r}")
    for k in ("topic", "main_concept", "confusable_concept", "annotation_rationale"):
        if not isinstance(r[k], str) or not r[k].strip(): e.append(f"empty {k}")
    turns = r["turns"]
    if not isinstance(turns, list) or not (5 <= len(turns) <= 25):
        e.append(f"turn count out of range: {len(turns) if isinstance(turns, list) else 'n/a'}")
        return e
    if r["num_student_turns"] != len(turns): e.append("num_student_turns != len(turns)")
    for i, t in enumerate(turns, 1):
        miss = [k for k in TURN_REQUIRED if k not in t]
        for k in miss: e.append(f"turn {i} missing {k}")
        if miss: continue
        if t["turn"] != i: e.append(f"turn numbering wrong at {i}")
        for k in ("student_message", "teacher_message"):
            if not isinstance(t[k], str) or not t[k].strip(): e.append(f"turn {i} empty {k}")
        if i == 1 and t["llm_message_context"] is not None: e.append("turn 1 llm_message_context must be null")
        if i > 1 and not str(t["llm_message_context"] or "").strip(): e.append(f"turn {i} empty llm_message_context")
        for k, allowed in TURN_ENUMS.items():
            if t[k] not in allowed: e.append(f"turn {i} invalid {k}={t[k]!r}")
        if t["confusion_flag"] not in (0, 1): e.append(f"turn {i} invalid confusion_flag")
    if any(k not in t for t in turns for k in TURN_REQUIRED):
        return e
    if r["confusion_points"] != sum(t["confusion_flag"] for t in turns):
        e.append("confusion_points != sum(confusion_flag)")
    if r["overall_sentiment"] != derive_overall_sentiment([t["sentiment"] for t in turns]):
        e.append("overall_sentiment does not match derivation rule")
    # annotation shape consistency
    if r["misconception_present"] is True:
        if not r["misconception_category"]: e.append("misconception_present true but no category")
        if r["misconception_resolved"] not in (True, False): e.append("misconception_resolved must be bool")
    elif r["misconception_present"] is False:
        if r["misconception_category"] is not None: e.append("category must be null when no misconception")
        if r["misconception_resolved"] is not None: e.append("misconception_resolved must be null")
    else:
        e.append("misconception_present must be bool")
    if r["misconception_category"] and not re.match(r"^[a-z0-9_]+$", r["misconception_category"]):
        e.append("misconception_category must be snake_case")
    sct = r["self_correction_turns"]
    if not (isinstance(sct, list) and all(isinstance(x, int) and 1 <= x <= len(turns) for x in sct)
            and len(set(sct)) == len(sct)):
        e.append("self_correction_turns must be a list of unique valid turn numbers")
    elif (r["self_correction"] == "yes") != bool(sct):
        e.append("self_correction must be 'yes' iff self_correction_turns is non-empty")
    q = r["quiz"]
    if not (isinstance(q, dict) and {"num_questions", "correct", "score_percent"} <= set(q)):
        e.append("bad quiz object")
    else:
        if not (10 <= q["num_questions"] <= 15): e.append("quiz num_questions out of range")
        if not (0 <= q["correct"] <= q["num_questions"]): e.append("quiz correct out of range")
        elif q["score_percent"] != int(100 * q["correct"] / q["num_questions"] + 0.5):
            e.append("quiz score_percent != round-half-up(100*correct/num_questions)")
    blob = (str(r["chat_id"]) + str(r["student_id"])).lower()
    if any(c in blob for c in CLASSES): e.append("class name leaked into ids")
    qr = r["quality_review"]
    if not (isinstance(qr, dict) and "automated_structural_check" in qr and "human_verified" in qr):
        e.append("quality_review malformed")
    elif qr["human_verified"] is True:
        hr = qr.get("human_review")
        if not (isinstance(hr, dict) and hr.get("reviewers", 0) >= 2 and hr.get("adjudicated") is True):
            e.append("human_verified=true requires human_review with >=2 reviewers and adjudication")
    gm = r["generation_meta"]
    if not isinstance(gm, dict): e.append("generation_meta malformed")
    else:
        if gm.get("schema_version") != SCHEMA_VERSION: e.append(f"schema_version must be {SCHEMA_VERSION}")
        if gm.get("teacher_style") not in TEACHER_STYLES: e.append("generation_meta.teacher_style invalid")
        if not gm.get("scenario_tag"): e.append("generation_meta.scenario_tag missing")
        if gm.get("use_as_training_feature") is not False: e.append("generation_meta.use_as_training_feature must be false")
    return e


# =============================================================== 2. SEMANTIC (heuristic)
STOP = set("""about above after again their there these those which while would could should where
    other being because between through during before under right think thing really still just""".split())
SELF_MARKERS = [r"\bwait\b", r"\bwas wrong\b", r"\bmy mistake\b", r"\bby mistake\b", r"\boops\b",
                r"\bi meant\b", r"\bgalat tha\b", r"\bshould(?:'ve| have) said\b", r"\bscratch that\b",
                r"\bi said always\b"]
def has_self_marker(msg):
    return any(re.search(m, msg) for m in SELF_MARKERS)
TEACHER_CORR = ["not quite", "no,", "no.", "incorrect", "wrong", "isn't", "aren't", "actually", "careful",
                "that's not", "check ", "again", "mistake", "ulta", "nahi", "not exactly", "close, but",
                "the catch", "try again", "instead"]
CONF_CUES = {
    "low": ["not sure", "not certain", "not very confident", "not fully confident", "idk", "might be wrong",
            "i'm lost", "still lost", "missing something", "don't have an intuition", "nahi jam raha",
            "not confident", "confusing"],
    "moderate": ["think i get", "got it mostly", "gaya shayad", "samajh aaya", "kind of clear",
                 "sort of"],
    "high": ["obviously", "easy hain", "i'm sure", "definitely", "got it!!", "makes sense now",
             "viva ready", "safe in exam", "yaad rakh liya"],
}
HINGLISH = set("""hai hain nahi samajh kya toh bhai yaar kaise mujhe bata batao kyun kyu matlab wala lag isme
    dikhao karna karun chahiye phir abhi bahut raha rahi theek sahi galat bas hoga hota karta daalu jata
    aata aaye aaya mera meri sirf zyada kahan kaam karo karke hoon jab agar woh yeh ye ka ki ke se mein
    par lekin isliye jo uska iska unka""".split())
HINGLISH_STRONG = set("""hai hain nahi samajh kya toh bhai yaar kaise mujhe bata batao kyun matlab isme
    dikhao karun chahiye lekin isliye woh yeh""".split())
PRAISE = ["exactly", "nicely", "perfect", "great", "well done", "spot on", "excellent", "good catch",
          "that's correct", "that's right", "you're right", "correct."]
ATTEMPT_TYPES = {"explanation_attempt", "answer_attempt", "challenge", "creation"}


def _words(t): return re.findall(r"[a-z']+", t.lower())


def teacher_has_correction(msg):
    m = msg.lower()
    return any(c in m for c in TEACHER_CORR)


def semantic_screen(r):
    """Return list of (code, message). Heuristic; may be wrong in either direction."""
    w = []
    turns = r["turns"]
    stu = [t["student_message"] for t in turns]
    stu_l = [s.lower() for s in stu]
    # -- self_correction: student-initiated, not teacher-supplied
    for k in r["self_correction_turns"]:
        msg = stu_l[k - 1]
        if not has_self_marker(msg):
            w.append(("SC1", f"self_correction_turns includes turn {k} but no correction marker found in the student message"))
        if k >= 2 and teacher_has_correction(turns[k - 2]["teacher_message"]):
            w.append(("SC2", f"turn {k}: preceding teacher message contains correction/hint cues, so correction may not be student-initiated"))
    if r["self_correction"] == "no":
        for i, msg in enumerate(stu_l, 1):
            if has_self_marker(msg):
                prev_corr = i >= 2 and teacher_has_correction(turns[i - 2]["teacher_message"])
                if not prev_corr:
                    w.append(("SC3", f"turn {i} has a correction marker with no teacher correction before it but self_correction='no'"))
    # -- misconception evidence
    flags = [t["confusion_flag"] for t in turns]
    if r["misconception_present"] and sum(flags) == 0:
        w.append(("MC1", "misconception_present=true but no turn carries confusion_flag=1"))
    if r["misconception_present"] is False and sum(flags) >= max(3, len(turns) // 2):
        w.append(("MC2", "misconception_present=false but half or more turns carry confusion_flag=1"))
    if r["misconception_resolved"] is True and flags[-1] == 1:
        w.append(("MC3", "misconception_resolved=true but the last student turn is flagged as confused"))
    # -- label vs annotation plausibility
    fu = r["final_understanding"]
    if fu == "understood" and r["reasoning_quality"] == "weak":
        w.append(("LB1", "understood with weak reasoning_quality"))
    if fu == "confused" and r["reasoning_quality"] == "strong":
        w.append(("LB2", "confused with strong reasoning_quality"))
    if fu == "understood" and flags[-1] == 1:
        w.append(("LB3", "understood but the last student turn is flagged as confused"))
    if fu == "understood" and r["misconception_present"] and r["misconception_resolved"] is False:
        w.append(("LB4", "understood with an unresolved misconception"))
    if fu == "confused" and r["misconception_resolved"] is True:
        w.append(("LB5", "confused with a resolved misconception"))
    # -- student evidence available for the label
    attempts = sum(t["question_type"] in ATTEMPT_TYPES for t in turns)
    if attempts < 2:
        w.append(("EV1", f"only {attempts} student turn(s) that attempt/explain/answer: thin evidence for the label"))
    # -- student_confidence must be explicitly expressed
    alltext = " ".join(stu_l)
    hits = {lvl for lvl, cues in CONF_CUES.items() if any(c in alltext for c in cues)}
    sc = r["student_confidence"]
    if sc == "unknown" and hits:
        w.append(("CF1", f"student_confidence='unknown' but explicit cues found: {sorted(hits)}"))
    if sc != "unknown" and not hits:
        w.append(("CF2", f"student_confidence='{sc}' but no explicit confidence cue found"))
    if sc != "unknown" and hits and sc not in hits:
        w.append(("CF3", f"student_confidence='{sc}' but cues found only for {sorted(hits)}"))
    # -- rationale grounded in conversation
    rat_words = {x for x in _words(r["annotation_rationale"]) if len(x) >= 5 and x not in STOP}
    conv_words = {x for x in _words(" ".join(stu) + " " + " ".join(t["teacher_message"] for t in turns))}
    if len(r["annotation_rationale"].split()) < 15:
        w.append(("RT1", "annotation_rationale shorter than 15 words"))
    if len(rat_words & conv_words) < 4:
        w.append(("RT2", "annotation_rationale shares fewer than 4 content words with the conversation"))
    # -- language style
    ws = [set(_words(s)) for s in stu]
    hing_msgs = sum(bool(x & HINGLISH_STRONG) for x in ws)
    first = [s[:1] for s in stu if s[:1].isalpha()]
    lower_share = sum(c.islower() for c in first) / max(1, len(first))
    ls = r["language_style"]
    if ls.startswith("hinglish") and hing_msgs < 2:
        w.append(("LS1", "hinglish style but fewer than 2 student messages contain Hinglish markers"))
    if ls == "hinglish_heavy" and hing_msgs < len(stu) * 0.5:
        w.append(("LS2", "hinglish_heavy but under half of the student messages contain Hinglish markers"))
    if ls.startswith("english") and hing_msgs >= 3:
        w.append(("LS3", "english style but 3+ student messages contain Hinglish markers"))
    if ls == "english_casual" and lower_share < 0.6:
        w.append(("LS4", "english_casual but under 60% of student messages start lowercase"))
    if ls == "english_standard" and lower_share > 0.4:
        w.append(("LS5", "english_standard but over 40% of student messages start lowercase"))
    # -- repetitiveness inside a session
    openers = Counter(" ".join(t["teacher_message"].split()[:2]).lower() for t in turns)
    top, c = openers.most_common(1)[0]
    if c >= 4 and c / len(turns) > 0.3:
        w.append(("RP1", f"teacher opener '{top}' used in {c} of {len(turns)} turns"))
    return w


# =============================================================== duplicates
def _shingles(text, n=3):
    t = re.findall(r"\w+", text.lower())
    return {" ".join(t[i:i + n]) for i in range(max(1, len(t) - n + 1))}


def find_duplicates(records, threshold=0.5):
    sh = [(r["chat_id"], _shingles(" ".join(t["student_message"] for t in r["turns"]))) for r in records]
    dups = []
    for (a, sa), (b, sb) in combinations(sh, 2):
        inter = len(sa & sb)
        if inter and inter / len(sa | sb) >= threshold:
            dups.append((a, b, round(inter / len(sa | sb), 3)))
    msgs = Counter(t["student_message"].strip().lower() for r in records for t in r["turns"])
    rep = {m: c for m, c in msgs.items() if c > 1}
    sess = Counter(json.dumps([t["student_message"] for t in r["turns"]]) for r in records)
    return {"exact_duplicate_sessions": sum(c - 1 for c in sess.values() if c > 1),
            "near_duplicate_pairs": dups,
            "student_messages_repeated_across_dataset": len(rep),
            "most_repeated_student_messages": sorted(rep.items(), key=lambda x: -x[1])[:5]}


# =============================================================== 3. DIAGNOSTICS
def cramers_v(xs, ys):
    n = len(xs); xv, yv = sorted(set(xs)), sorted(set(ys))
    if n == 0 or len(xv) < 2 or len(yv) < 2: return 0.0
    obs = Counter(zip(xs, ys)); rx, cy = Counter(xs), Counter(ys)
    chi = sum((obs.get((a, b), 0) - rx[a] * cy[b] / n) ** 2 / (rx[a] * cy[b] / n) for a in xv for b in yv)
    return math.sqrt(chi / (n * (min(len(xv), len(yv)) - 1)))


def eta_squared(values, groups):
    if len(values) < 3: return 0.0
    mu = statistics.mean(values); tot = sum((v - mu) ** 2 for v in values)
    if tot == 0: return 0.0
    by = defaultdict(list)
    for v, g in zip(values, groups): by[g].append(v)
    return sum(len(vs) * (statistics.mean(vs) - mu) ** 2 for vs in by.values()) / tot


def deterministic(xs, ys):
    m = defaultdict(set)
    for x, y in zip(xs, ys): m[x].add(y)
    return all(len(v) == 1 for v in m.values()) and len(m) > 1


def final_teacher_cue(r):
    m = r["turns"][-1]["teacher_message"].lower()
    if teacher_has_correction(m): return "correction"
    if any(p in m for p in PRAISE): return "praise"
    return "other"


def reliability(n):
    return ("not_interpretable" if n < 100 else "indicative_only" if n < 1000 else "usable")


def diagnostics(records):
    y = [r["final_understanding"] for r in records]; n = len(records)
    emo = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    cat = {
        "overall_sentiment": [r["overall_sentiment"] for r in records],
        "final_turn_sentiment": [r["turns"][-1]["sentiment"] for r in records],
        "student_confidence": [r["student_confidence"] for r in records],
        "engagement_level": [r["engagement_level"] for r in records],
        "reasoning_quality": [r["reasoning_quality"] for r in records],
        "self_correction": [r["self_correction"] for r in records],
        "misconception_present": [str(r["misconception_present"]) for r in records],
        "student_persona": [r["student_persona"] for r in records],
        "language_style": [r["language_style"] for r in records],
        "subject": [r["subject"] for r in records],
        "length_bucket": [length_bucket(len(r["turns"])) for r in records],
        "llm_client": [r["llm_client"] for r in records],
        "difficulty_level": [r["difficulty_level"] for r in records],
        "task_type": [r["task_type"] for r in records],
        "teacher_style": [r["generation_meta"]["teacher_style"] for r in records],
        "scenario_tag": [r["generation_meta"]["scenario_tag"] for r in records],
        "final_teacher_message_cue": [final_teacher_cue(r) for r in records],
        "last_student_msg_thanks": [str(bool(re.search(r"thank|thanks|shukriya", r["turns"][-1]["student_message"].lower()))) for r in records],
    }
    num = {
        "num_student_turns": [len(r["turns"]) for r in records],
        "student_question_count": [sum("?" in t["student_message"] for t in r["turns"]) for r in records],
        "mean_student_words": [statistics.mean(len(t["student_message"].split()) for t in r["turns"]) for r in records],
        "emoji_message_share": [sum(bool(emo.search(t["student_message"])) for t in r["turns"]) / len(r["turns"]) for r in records],
        "teacher_correction_share": [sum(teacher_has_correction(t["teacher_message"]) for t in r["turns"]) / len(r["turns"]) for r in records],
        "student_attempt_share": [sum(t["question_type"] in ATTEMPT_TYPES for t in r["turns"]) / len(r["turns"]) for r in records],
        "confusion_points": [r["confusion_points"] for r in records],
        "quiz_score_percent": [r["quiz"]["score_percent"] for r in records],
    }
    out = {"n": n, "reliability": reliability(n), "categorical_vs_class": {}, "numeric_vs_class": {},
           "lexical_probe_student_text": {}, "warnings": []}
    for k, xs in cat.items():
        v, det = cramers_v(xs, y), deterministic(xs, y)
        out["categorical_vs_class"][k] = {"cramers_v": round(v, 3), "deterministic_mapping": det}
        if n >= 100 and det: out["warnings"].append(f"{k} maps deterministically to final_understanding")
        elif n >= 100 and v >= 0.5: out["warnings"].append(f"{k} strongly associated with class (V={v:.2f})")
    for k, xs in num.items():
        e2 = eta_squared(xs, y)
        means = {c: round(statistics.mean([x for x, g in zip(xs, y) if g == c]), 3) for c in CLASSES if c in y}
        out["numeric_vs_class"][k] = {"eta_squared": round(e2, 3), "class_means": means}
        if n >= 100 and e2 >= 0.25: out["warnings"].append(f"{k} explains {e2:.0%} of variance across classes")
    # lexical probe: tokens used in >=max(3, 2% of sessions) sessions, concentrated in one class
    df, dfc = Counter(), defaultdict(Counter)
    for r in records:
        toks = {t for t in _words(" ".join(x["student_message"] for x in r["turns"])) if len(t) >= 3 and t not in STOP}
        for t in toks:
            df[t] += 1; dfc[t][r["final_understanding"]] += 1
    prior = Counter(y)
    thr = max(3, int(0.02 * n)); rows = []
    for t, c in df.items():
        if c >= thr:
            cls, k = dfc[t].most_common(1)[0]
            if k / c >= 0.9 and prior[cls] / n < 0.8:
                rows.append((t, cls, c))
    rows.sort(key=lambda x: -x[2])
    out["lexical_probe_student_text"] = {"min_sessions": thr, "n_class_concentrated_tokens": len(rows),
                                         "examples": rows[:12]}
    if n >= 100 and rows:
        out["warnings"].append(f"{len(rows)} student-text tokens are >=90% concentrated in one class")
    out["interpretation"] = (
        f"n={n}: statistics are '{reliability(n)}'. Small samples cannot establish or rule out correlations "
        "or shortcuts; empty warnings do not mean the data are free of leakage. Natural relationships between "
        "understanding and e.g. reasoning_quality are expected; only deterministic or implausibly strong "
        "associations indicate a generation artefact. Re-run on the full dataset and on a held-out, "
        "student-grouped classifier probe.")
    return out


def distributions(records):
    c = lambda f: dict(Counter(f(r) for r in records).most_common())
    return {"total": len(records),
            "data_source": c(lambda r: r["data_source"]),
            "final_understanding": c(lambda r: r["final_understanding"]),
            "subject": c(lambda r: r["subject"]),
            "topic": c(lambda r: f"{r['subject']}/{r['topic']}"),
            "length_bucket": c(lambda r: length_bucket(len(r["turns"]))),
            "student_persona": c(lambda r: r["student_persona"]),
            "language_style": c(lambda r: r["language_style"]),
            "teacher_style": c(lambda r: r["generation_meta"]["teacher_style"]),
            "self_correction": c(lambda r: r["self_correction"]),
            "engagement_level": c(lambda r: r["engagement_level"]),
            "misconception_present": c(lambda r: r["misconception_present"]),
            "misconception_category": c(lambda r: r["misconception_category"]),
            "student_confidence": c(lambda r: r["student_confidence"]),
            "reasoning_quality": c(lambda r: r["reasoning_quality"]),
            "llm_client": c(lambda r: r["llm_client"]),
            "difficulty_level": c(lambda r: r["difficulty_level"]),
            "task_type": c(lambda r: r["task_type"]),
            "unique_student_ids": len({r["student_id"] for r in records})}


def run(path, expect=None):
    data = json.load(open(path, encoding="utf-8"))
    errors, sem = {}, {}
    ids = Counter(r.get("chat_id") for r in data)
    for r in data:
        errs = validate_record(r)
        if errs: errors[r.get("chat_id", "?")] = errs
        else:
            s = semantic_screen(r)
            if s: sem[r["chat_id"]] = s
    dup_ids = [k for k, v in ids.items() if v > 1]
    dups = find_duplicates(data) if not errors else {"skipped": "fix structural errors first"}
    sources = {r.get("data_source") for r in data}
    structural_pass = not errors and not dup_ids and dups.get("exact_duplicate_sessions", 0) == 0 \
        and len(sources) == 1
    report = {
        "file": path, "records": len(data), "schema_version": SCHEMA_VERSION,
        "claim_discipline": ("structural_pass only means the JSON is well formed and internally consistent. "
                             "It does NOT mean labels or conversations are educationally correct. "
                             "All labels are LLM-authored and unverified by humans unless quality_review says otherwise."),
        "structural": {
            "structural_pass": structural_pass,
            "records_with_errors": len(errors), "errors": errors,
            "duplicate_chat_ids": dup_ids,
            "mixed_data_sources_in_file": sorted(map(str, sources)) if len(sources) > 1 else None,
            "missing_field_count": sum(1 for es in errors.values() for x in es if x.startswith("missing")),
            "malformed_data_source_count": sum(1 for es in errors.values() for x in es if "data_source" in x),
            "invalid_label_count": sum(1 for es in errors.values() for x in es if x.startswith("invalid")),
        },
        "semantic_screening": {
            "kind": "heuristic warnings, not a verdict",
            "records_with_warnings": len(sem),
            "warning_count": sum(len(v) for v in sem.values()),
            "by_code": dict(Counter(c for v in sem.values() for c, _ in v)),
            "warnings": {k: [f"{c}: {m}" for c, m in v] for k, v in sem.items()},
            "human_verified_records": sum(1 for r in data if r.get("quality_review", {}).get("human_verified") is True),
        },
        "duplicates": dups,
        "distributions": distributions(data) if not errors else None,
        "leakage_diagnostics": diagnostics(data) if not errors else None,
        "forbidden_model_input_fields": FORBIDDEN_MODEL_FIELDS,
    }
    if expect:
        got = Counter(r["final_understanding"] for r in data)
        report["class_counts_match_expected"] = all(got.get(k, 0) == v for k, v in zip(CLASSES, expect))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("--report")
    ap.add_argument("--expect", nargs=3, type=int, metavar=("U", "P", "C"))
    ap.add_argument("--fail-on-semantic", action="store_true")
    a = ap.parse_args()
    rep = run(a.input, a.expect)
    if a.report: json.dump(rep, open(a.report, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    s, m = rep["structural"], rep["semantic_screening"]
    print(f"STRUCTURAL: pass={s['structural_pass']} errors_in={s['records_with_errors']} records")
    print(f"SEMANTIC (heuristic): {m['warning_count']} warnings in {m['records_with_warnings']} records; "
          f"human-verified records: {m['human_verified_records']}")
    print("NOTE:", rep["claim_discipline"])
    for ce, es in list(s["errors"].items())[:20]: print("ERR", ce, es)
    for ce, ws in list(m["warnings"].items())[:40]:
        for x in ws: print("WARN", ce, x)
    if rep["leakage_diagnostics"]:
        print("DIAGNOSTICS reliability:", rep["leakage_diagnostics"]["reliability"])
        for x in rep["leakage_diagnostics"]["warnings"]: print("DIAG-WARN", x)
    sys.exit(0 if s["structural_pass"] and not (a.fail_on_semantic and m["warning_count"]) else 1)
