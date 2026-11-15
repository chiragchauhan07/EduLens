#!/usr/bin/env python3
"""
EduLens helper tools (schema v2.1).

  model-inputs  Export text-only classifier inputs. Two views for the planned experiment:
                  student_only : student messages only
                  full         : student + teacher messages
                Only text, an opaque row id, the label and a group id (student_id, for grouped
                train/test splitting ONLY, never as a feature) are written. All other fields are dropped.
  review-sheet  Export a BLIND review CSV for independent reviewers (no labels, rationale or metadata).
  agreement     Compare a completed reviewer CSV with dataset labels (accuracy + Cohen's kappa).
Usage:
  python edulens_tools.py model-inputs DATA.json OUT.jsonl --view student_only|full [--seed 7]
  python edulens_tools.py review-sheet DATA.json OUT.csv --n 100 [--seed 7] [--key KEY.json]
  python edulens_tools.py agreement DATA.json REVIEWED.csv [--key KEY.json]
"""
import argparse, csv, json, random
from collections import Counter

def conv_text(r, view):
    parts = []
    for t in r["turns"]:
        parts.append("Student: " + t["student_message"])
        if view == "full":
            parts.append("Teacher: " + t["teacher_message"])
    return "\n".join(parts)

def model_inputs(a):
    data = json.load(open(a.data, encoding="utf-8")); rng = random.Random(a.seed)
    order = list(range(len(data))); rng.shuffle(order)
    with open(a.out, "w", encoding="utf-8") as f:
        for k, i in enumerate(order):
            r = data[i]
            f.write(json.dumps({"row_id": f"R{k:05d}", "text": conv_text(r, a.view),
                                "label": r["final_understanding"], "group_id": r["student_id"]},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(data)} rows ({a.view}); row ids are shuffled and opaque")

def review_sheet(a):
    data = json.load(open(a.data, encoding="utf-8")); rng = random.Random(a.seed)
    # stratified by class so reviewers see all three classes; order shuffled
    by = {}
    for r in data: by.setdefault(r["final_understanding"], []).append(r)
    per = max(1, a.n // len(by)); pick = []
    for c, rs in by.items(): pick += rng.sample(rs, min(per, len(rs)))
    rng.shuffle(pick); key = {}
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["review_id", "conversation", "final_understanding(understood|partial|confused)",
                    "self_correction(yes|no)", "engagement(high|moderate|low)",
                    "misconception(none|<short description>)", "student_confidence(high|moderate|low|unknown)",
                    "reasoning_quality(strong|moderate|weak)", "technically_correct_teacher(yes|no)", "notes"])
        for k, r in enumerate(pick):
            rid = f"V{k:04d}"; key[rid] = r["chat_id"]
            w.writerow([rid, conv_text(r, "full")] + [""] * 8)
    if a.key: json.dump(key, open(a.key, "w"), indent=1)
    print(f"wrote {len(pick)} blind items; the id->chat_id key is stored separately and not shown to reviewers")

def kappa(x, y):
    n = len(x); labs = sorted(set(x) | set(y)); po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y); pe = sum(cx[l] * cy[l] for l in labs) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0

def agreement(a):
    data = {r["chat_id"]: r for r in json.load(open(a.data, encoding="utf-8"))}
    key = json.load(open(a.key)); rows = list(csv.DictReader(open(a.reviewed, encoding="utf-8")))
    cols = {"final_understanding": rows[0].keys()}
    fu = [k for k in rows[0].keys() if k.startswith("final_understanding")][0]
    x = [data[key[r["review_id"]]]["final_understanding"] for r in rows if r[fu]]
    y = [r[fu].strip() for r in rows if r[fu]]
    print(f"items={len(x)} accuracy={sum(a_==b for a_,b in zip(x,y))/len(x):.3f} cohen_kappa={kappa(x,y):.3f}")
    print("confusions (dataset -> reviewer):", dict(Counter((a_, b) for a_, b in zip(x, y) if a_ != b)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("model-inputs"); m.add_argument("data"); m.add_argument("out")
    m.add_argument("--view", choices=["student_only", "full"], required=True); m.add_argument("--seed", type=int, default=7)
    s = sp.add_parser("review-sheet"); s.add_argument("data"); s.add_argument("out")
    s.add_argument("--n", type=int, default=100); s.add_argument("--seed", type=int, default=7); s.add_argument("--key")
    g = sp.add_parser("agreement"); g.add_argument("data"); g.add_argument("reviewed"); g.add_argument("--key", required=True)
    a = ap.parse_args()
    {"model-inputs": model_inputs, "review-sheet": review_sheet, "agreement": agreement}[a.cmd](a)
