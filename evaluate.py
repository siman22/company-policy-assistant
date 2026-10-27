"""
evaluate.py - run all three approaches on test_questions.csv and save results.csv.

The website reads results.csv to show the comparison, so run this once locally
(with GEMINI_API_KEY set) and commit results.csv to GitHub.

    python evaluate.py

Progress is saved after every question. If the free-tier daily limit runs out,
the script stops cleanly; run it again later and it continues where it stopped.

Optional settings (environment variables):
    GEMINI_MODEL        model used to answer (pick one with a high daily free limit)
    GEMINI_JUDGE_MODEL  different model for the LLM judge, so it uses its own daily quota
    EVAL_JUDGE=0        skip the LLM judge (halves the number of requests)
    EVAL_DELAY=6        seconds between requests
"""

import os
import sys
import time

import pandas as pd

import policy_engine as pe

DELAY = float(os.getenv("EVAL_DELAY", "6"))  # seconds between LLM calls (free tier)
USE_JUDGE = os.getenv("EVAL_JUDGE", "1") != "0"
OUT = pe.HERE / "results.csv"
PARTIAL = pe.HERE / "results_partial.csv"


def run_question(q):
    rows = []
    for key, fn in pe.METHODS.items():
        out = fn(q.question)
        predicted = out["policy_title"] or pe.NOT_FOUND
        reasons = [out["unsupported_reason"]] if out["unsupported"] else []

        # Out-of-scope question answered with a policy = unsupported answer
        if q.expected_policy == pe.NOT_FOUND and predicted != pe.NOT_FOUND:
            reasons.append("Answered a question the database does not cover")

        # LLM judge for the two LLM approaches (rules output is verbatim policy text)
        judge_reason = "verbatim policy text" if key == "rules" else "judge not run"
        if key != "rules":
            time.sleep(DELAY)
            if USE_JUDGE:
                judge_supported, judge_reason = pe.judge(q.question, out["answer"], out["policy_title"])
                if not judge_supported:
                    reasons.append("Judge: " + judge_reason)
                time.sleep(DELAY)

        rows.append(dict(
            question=q.question, type=q.type, expected_policy=q.expected_policy,
            method=out["method"], predicted_policy=predicted,
            correct_policy=predicted == q.expected_policy,
            answer=out["answer"], latency_s=round(out["latency_s"], 4),
            prompt_tokens=out["prompt_tokens"], output_tokens=out["output_tokens"],
            total_tokens=out["total_tokens"],
            unsupported=bool(reasons), unsupported_reason=" | ".join(reasons),
            judge_reason=judge_reason, retrieved="; ".join(out["retrieved"]),
            model=pe._active_model or "",
        ))
        print(f"   {out['method']:<26} -> {predicted}  ({out['latency_s']:.2f}s, {out['total_tokens']} tok)"
              + ("  UNSUPPORTED" if reasons else ""))
    return rows


def main():
    questions = pd.read_csv(pe.HERE / "test_questions.csv")
    done = pd.read_csv(PARTIAL) if PARTIAL.exists() else pd.DataFrame()
    done_q = set(done.question) if len(done) else set()
    if done_q:
        print(f"Resuming: {len(done_q)} of {len(questions)} questions already done.")

    print("Building vector index ...")
    pe.get_index()
    for i, q in questions.iterrows():
        if q.question in done_q:
            continue
        print(f"[{i + 1}/{len(questions)}] {q.question}")
        try:
            rows = run_question(q)
        except pe.DailyQuotaExhausted as e:
            print(f"\nSTOPPED: {e}\nProgress is saved in {PARTIAL.name}; run `python evaluate.py` again to continue.")
            sys.exit(1)
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(PARTIAL, index=False)

    done.to_csv(OUT, index=False)
    PARTIAL.unlink(missing_ok=True)
    print(f"\nSaved {OUT.name}\n")
    print(summarise(done).to_string())


def summarise(df):
    order = ["Rules-based search", "LLM without vector index", "LLM with vector index"]
    g = df.groupby("method")
    oos = df[df.type == "out_of_scope"].groupby("method")
    s = pd.DataFrame({
        "Correct policy": g.correct_policy.mean(),
        "Median response time (s)": g.latency_s.median(),
        "Avg. tokens / question": g.total_tokens.mean(),
        "Unsupported answers": g.unsupported.mean(),
        "Out-of-scope declined": oos.apply(lambda x: (x.predicted_policy == "NONE").mean()),
    }).reindex(order)
    return s


if __name__ == "__main__":
    main()
