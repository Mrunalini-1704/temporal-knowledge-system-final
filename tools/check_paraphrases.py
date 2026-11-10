"""
Regression check beyond the public eval: reworded questions with the values a correct answer must contain.

    python tools/check_paraphrases.py [--submission submission]

Each line of paraphrase_checks.txt is `expected|question[@as_of]`, where expected is `;`-separated tokens
(compared with the scorer's own value-aware `contains`) or REFUSE for questions the site can't answer.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from run_eval import load_snapshots, load_submission  # noqa: E402
from scoring import REFUSAL_RE, _numbers, contains  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--submission", type=Path, default=ROOT / "submission")
args = ap.parse_args()

ks = load_submission(args.submission)
for snap in load_snapshots(ROOT / "snapshots"):
    ks.ingest(snap)

ok = total = 0
for line in (Path(__file__).parent / "paraphrase_checks.txt").read_text().splitlines():
    if "|" not in line:
        continue
    expected, q = line.split("|", 1)
    q, _, as_of = q.partition("@")
    a = ks.answer(q, as_of or None)
    if expected == "REFUSE":
        good = (not a.confident or REFUSAL_RE.search(a.text)) and not (_numbers(a.text) - _numbers(q))
    else:
        good = all(contains(a.text, t) for t in expected.split(";"))
    total += 1
    ok += bool(good)
    if not good:
        print(f"FAIL [{expected}] {q}\n     -> {a.text[:240]}")
print(f"{ok}/{total} paraphrase checks passed")
