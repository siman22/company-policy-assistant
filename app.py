"""
app.py - Streamlit website comparing three ways to answer company policy questions.
Run locally:  streamlit run app.py
"""

import pandas as pd
import streamlit as st

import policy_engine as pe

STUDENT = "Simon Devald"
COURSE = "AI in Business and Accounting (SDU)"

st.set_page_config(page_title="Company Policy Assistant", page_icon="📘", layout="wide")
st.title("📘 Company Policy Assistant")
st.caption(f"{STUDENT} · {COURSE} · Rules-based search vs. LLM without a vector index vs. LLM with a vector index")

has_key = bool(pe.get_api_key())


@st.cache_resource(show_spinner="Building the vector index (one-time)...")
def warm_index():
    return pe.get_index()


EXAMPLES = [
    "Can I work from home on Fridays?",
    "Do I need approval before spending $800 on a client dinner?",
    "How many unused vacation days can I roll over to next year?",
    "My grandmother passed away. How much time off can I take?",
    "Can I bring my dog to the office?",
]

tab_ask, tab_compare = st.tabs(["Ask a question (live)", "Comparison & conclusion"])

# --------------------------------------------------------------------------------------
# Live tab
# --------------------------------------------------------------------------------------
with tab_ask:
    st.write("Type a policy question. All three approaches answer it side by side, and each answer is "
             "checked for content that is not supported by the policy database.")
    example = st.selectbox("Example questions", ["(write your own)"] + EXAMPLES)
    question = st.text_input("Your question", value="" if example == "(write your own)" else example)
    go = st.button("Compare the three approaches", type="primary", disabled=not question.strip())

    if go:
        if has_key:
            warm_index()
        cols = st.columns(3)
        for col, key in zip(cols, ["rules", "llm", "rag"]):
            with col:
                try:
                    with st.spinner("Thinking..."):
                        out = pe.METHODS[key](question)
                except Exception as e:
                    st.subheader({"rules": "Rules-based search", "llm": "LLM without vector index",
                                  "rag": "LLM with vector index"}[key])
                    st.error(f"Could not get an answer: {e}")
                    continue
                st.subheader(out["method"])
                st.markdown(f"**Answer:** {out['answer']}")
                if out["policy_title"]:
                    p = pe.BY_TITLE[out["policy_title"]]
                    st.markdown(f"**Relevant policy:** {p['title']}  \n"
                                f"<small>{p['category']} · {p['department']} department</small>",
                                unsafe_allow_html=True)
                else:
                    st.markdown("**Relevant policy:** none found")
                m1, m2 = st.columns(2)
                t = out["latency_s"]
                m1.metric("Response time", f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s")
                m2.metric("Tokens used", f"{out['total_tokens']:,}")
                if out["unsupported"]:
                    st.warning(f"⚠️ Possibly unsupported: {out['unsupported_reason']}")
                else:
                    st.success("✅ No unsupported content detected")
                if out["retrieved"]:
                    with st.expander("Policies retrieved by vector search"):
                        st.write("\n".join(f"- {r}" for r in out["retrieved"]))
        st.caption("Unsupported-content check (live): flags answers that cite a policy not in the database, "
                   "state numbers that do not appear in the cited policy, or give specifics without citing a policy. "
                   "The benchmark tab additionally uses an LLM judge and out-of-scope test questions.")
    if not has_key:
        st.info("No Gemini API key configured: only the rules-based approach can run live. "
                "The comparison tab still shows the saved benchmark results.")

# --------------------------------------------------------------------------------------
# Comparison tab
# --------------------------------------------------------------------------------------


def pct(x):
    return f"{x * 100:.0f}%"


def paragraphs(df, s):
    n = df.question.nunique()
    types = df.drop_duplicates("question").type.value_counts()
    n_trap, n_oos = types.get("trap_detail", 0), types.get("out_of_scope", 0)
    r, l, v = (s.loc[m] for m in ["Rules-based search", "LLM without vector index", "LLM with vector index"])
    token_ratio = l["Avg. tokens / question"] / max(v["Avg. tokens / question"], 1)
    rules_oos_wrong = round((1 - r["Out-of-scope declined"]) * n_oos)

    p1 = (f"I tested the three approaches on the same {n} employee questions: {n - n_trap - n_oos} that a policy "
          f"answers (mostly phrased in everyday words rather than policy wording), {n_trap} \"trap\" questions that ask "
          f"for a detail the policy does not state (e.g. how many vacation days can roll over), and {n_oos} questions "
          f"the database does not cover at all. An answer counts as unsupported if it cites a policy that does not "
          f"exist, adds numbers or details that are not in the cited policy (checked by a rule and by an LLM judge), "
          f"or answers an out-of-scope question with a policy. Rules-based search was by far the fastest "
          f"({r['Median response time (s)'] * 1000:.1f} ms) and free (0 tokens), and because it only returns verbatim "
          f"policy text it cannot invent details; however, it found the right policy in only "
          f"{pct(r['Correct policy'])} of the questions, because keyword matching breaks when people use different "
          f"words, and it answered {rules_oos_wrong} of {n_oos} out-of-scope questions with an unrelated policy, giving an "
          f"unsupported-answer rate of {pct(r['Unsupported answers'])}. The LLM without a vector index read all "
          f"{len(pe.POLICIES)} policies on every question and identified the right policy in "
          f"{pct(l['Correct policy'])} of cases, but used about {l['Avg. tokens / question']:,.0f} tokens and a median "
          f"response time of {l['Median response time (s)']:.2f} s per question, with {pct(l['Unsupported answers'])} unsupported answers. "
          f"The LLM with a vector index only sent the three most similar policies to the model and reached "
          f"{pct(v['Correct policy'])} correct policies with about {v['Avg. tokens / question']:,.0f} tokens "
          f"({token_ratio:.0f}x fewer) and a median response time of {v['Median response time (s)']:.2f} s per question, with "
          f"{pct(v['Unsupported answers'])} unsupported answers.")

    diff = v["Correct policy"] - l["Correct policy"]
    if diff >= 0.02:
        acc = "it was also more accurate than the LLM that read the whole database"
    elif diff > -0.06:
        acc = "its accuracy was on par with the LLM that read the whole database"
    else:
        acc = ("it was somewhat less accurate than the LLM that read the whole database, because the right policy "
               "occasionally fell outside the three retrieved ones")
    p2 = (f"My preferred approach is the LLM with a vector index, which is also what the Slack bot uses. It "
          f"understands everyday phrasing the way the full-context LLM does, but {acc}, while costing roughly "
          f"{token_ratio:.0f} times fewer tokens per question. That difference matters more as the policy database "
          f"grows: the no-index approach has to send every policy with every question, so its cost and response time "
          f"rise in step with the database (and it eventually would not fit in the prompt), whereas the vector index "
          f"is built once and each question only sends a few policies. Showing the model only a few closely related "
          f"policies, together with an instruction to say when a detail is not specified, also keeps answers tied to "
          f"the source text, so unsupported answers stay rare. Rules-based search remains useful as a free, instant "
          f"fallback or for exact title lookups, but its weak accuracy on natural questions and its tendency to "
          f"return a confident but wrong policy make it unsuitable as the main assistant. The main remaining risk of "
          f"the vector approach is a retrieval miss, which could be reduced by retrieving more policies, adding a "
          f"similarity threshold, or combining keyword and vector search.")
    return p1, p2


with tab_compare:
    try:
        df = pd.read_csv(pe.HERE / "results.csv")
    except FileNotFoundError:
        st.info("No benchmark results yet. Run `python evaluate.py` locally and commit results.csv.")
        st.stop()

    from evaluate import summarise
    s = summarise(df)
    p1, p2 = paragraphs(df, s)

    st.header("Comparison and preferred approach")
    st.write(p1)
    st.write(p2)

    st.subheader("Table 1 – Summary across all test questions")
    t1 = s.copy()
    for c in ["Correct policy", "Unsupported answers", "Out-of-scope declined"]:
        t1[c] = t1[c].map(pct)
    t1["Median response time (s)"] = s["Median response time (s)"].map(lambda x: f"{x:.3f}")
    t1["Avg. tokens / question"] = s["Avg. tokens / question"].map(lambda x: f"{x:,.0f}")
    st.dataframe(t1, width="stretch")
    st.caption("Response time is the median per question, because free-tier Gemini latency varied widely "
               "(about 1-30 s for the same kind of request). Retry waits after rate limits are excluded.")

    st.subheader("Table 2 – Answer, relevant policy, time, tokens and support for every question")
    short = {"Rules-based search": "Rules", "LLM without vector index": "LLM (no index)",
             "LLM with vector index": "LLM + vector index"}
    method_filter = st.radio("Approach", list(short.values()), horizontal=True)
    full = {v: k for k, v in short.items()}[method_filter]
    t2 = df[df.method == full][["question", "type", "expected_policy", "predicted_policy", "answer",
                                "latency_s", "total_tokens", "unsupported", "unsupported_reason"]].copy()
    t2["correct"] = t2.expected_policy == t2.predicted_policy
    t2 = t2.rename(columns={"predicted_policy": "relevant policy (returned)", "latency_s": "time (s)",
                            "total_tokens": "tokens"})
    st.dataframe(t2, width="stretch", hide_index=True)
    st.caption(f"Model: {(df['model'].dropna().mode().iat[0] if 'model' in df and df['model'].notna().any() else pe.GEN_MODEL)} (temperature 0) · Embeddings: {pe.EMBED_MODEL} · "
               f"Top-k retrieval: {pe.TOP_K} · Test questions: test_questions.csv")
