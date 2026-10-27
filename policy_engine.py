"""
policy_engine.py
Shared logic for the Company Policy Assistant.

Three approaches answer the same question:
  1. rules  - rules-based keyword search (BM25 + synonyms), no LLM, zero tokens
  2. llm    - LLM WITHOUT a vector index: the whole policy database is put in the prompt
  3. rag    - LLM WITH a vector index: embed policies once, retrieve top-k, send only those to the LLM

Every approach returns the same dict, so the website, the evaluation script and the
Slack bot can all use it:
  answer, policy_title, latency_s, prompt_tokens, output_tokens, total_tokens,
  unsupported (bool), unsupported_reason, retrieved (list of titles, rag only)
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
POLICY_CSV = HERE / "company_policies.csv"
INDEX_FILE = HERE / "policy_index.npz"

# Model names can be overridden with environment variables / Streamlit secrets.
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODELS = ["gemini-3-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
JUDGE_MODEL = os.getenv("GEMINI_JUDGE_MODEL")  # optional: use a different model (own daily quota) for the judge
TOP_K = 3
NOT_FOUND = "NONE"

# --------------------------------------------------------------------------------------
# Policy database
# --------------------------------------------------------------------------------------

def load_policies(path=POLICY_CSV):
    df = pd.read_csv(path)
    df = df.rename(columns={"policy_text": "text"})
    df.insert(0, "id", [f"P{i:02d}" for i in range(1, len(df) + 1)])
    return df.to_dict("records")


POLICIES = load_policies()
BY_ID = {p["id"]: p for p in POLICIES}
BY_TITLE = {p["title"]: p for p in POLICIES}


def policy_line(p):
    return f"[{p['id']}] {p['title']} | category: {p['category']} | department: {p['department']} | {p['text']}"


# --------------------------------------------------------------------------------------
# Unsupported-answer check (deterministic, no API call)
# --------------------------------------------------------------------------------------

NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "fifteen": "15", "twenty": "20", "thirty": "30", "ninety": "90",
}


def _numbers(text):
    text = text.lower().replace(",", "")
    nums = set(re.findall(r"\d+(?:\.\d+)?", text))
    for word, digit in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", text):
            nums.add(digit)
    # quarter / Q4 style references
    nums |= {m for m in re.findall(r"q(\d)", text)}
    return nums


def check_supported(answer, policy_title, raw_id=None):
    """Return (unsupported: bool, reason: str).

    An answer is flagged as unsupported when
      * it cites a policy id/title that does not exist in the database, or
      * it states a number that does not appear in the cited policy text, or
      * it gives a substantive answer without citing any policy.
    """
    if raw_id is not None and raw_id != NOT_FOUND and raw_id not in BY_ID:
        return True, f"Cited policy id '{raw_id}' does not exist in the database"
    if policy_title is None:
        if _numbers(answer):
            return True, "Gives specific numbers but cites no policy"
        return False, ""
    if policy_title not in BY_TITLE:
        return True, f"Cited policy '{policy_title}' does not exist in the database"
    source = BY_TITLE[policy_title]["text"] + " " + policy_title
    extra = _numbers(answer) - _numbers(source)
    if extra:
        return True, "Numbers not found in the policy: " + ", ".join(sorted(extra))
    return False, ""


# --------------------------------------------------------------------------------------
# 1. Rules-based search (BM25 with synonym expansion)
# --------------------------------------------------------------------------------------

STOPWORDS = set("""a an the and or of to in on for is are be can i my me we our you your do does did
how what when where which who whom why many much may must should will would with without at by from
it its this that there their they them as if about any all am was were has have had get got not no
company employee employees policy policies rule rules allowed allow us so than then into up out
please tell know need want work""".split())

# Everyday words -> words used in the policy database
SYNONYMS = {
    "pto": ["vacation"], "holiday": ["vacation", "holiday"], "holidays": ["vacation", "holiday"],
    "leave": ["leave"], "off": ["vacation"], "annual": ["annually"],
    "wfh": ["remote"], "home": ["remote", "home"], "remotely": ["remote"],
    "ill": ["sick"], "illness": ["sick"], "doctor": ["sick"],
    "baby": ["parental"], "maternity": ["parental"], "paternity": ["parental"], "newborn": ["parental"],
    "funeral": ["bereavement"], "died": ["bereavement"], "death": ["bereavement"], "passed": ["bereavement"],
    "wear": ["dress", "attire"], "clothes": ["dress", "attire"], "outfit": ["dress", "attire"], "jeans": ["dress", "attire"],
    "car": ["vehicle"], "drive": ["vehicle"],
    "overseas": ["abroad"], "country": ["abroad"], "international": ["abroad"],
    "phone": ["mobile", "device"], "smartphone": ["mobile", "device"],
    "computer": ["laptop", "hardware"], "pc": ["laptop", "hardware"],
    "app": ["software"], "program": ["software"], "install": ["installation", "software"],
    "course": ["training", "tuition"], "class": ["training", "tuition"], "degree": ["tuition"],
    "certification": ["training"], "university": ["tuition"],
    "spend": ["expense"], "purchase": ["procurement", "expense"], "buy": ["procurement"],
    "cost": ["expense"], "costs": ["expense"], "money": ["expense"], "dollar": ["expense"],
    "receipt": ["receipts", "travel"], "trip": ["travel"], "flight": ["travel"], "hotel": ["travel"],
    "reimbursed": ["reimbursement"], "reimburse": ["reimbursement"], "refund": ["reimbursement"],
    "bully": ["harassment"], "bullying": ["harassment"], "harassed": ["harassment"],
    "hacked": ["security", "incident"], "breach": ["security", "incident"], "phishing": ["security", "incident"],
    "virus": ["security", "incident"],
    "login": ["password"], "credentials": ["password"],
    "badge": ["access", "badges"], "keycard": ["access", "badges"],
    "fired": ["exit"], "quit": ["exit", "departing"], "resign": ["exit", "departing"], "leaving": ["exit", "departing"],
    "hire": ["hires"], "hired": ["hires"], "start": ["onboarding"], "starting": ["onboarding"], "first": ["onboarding"],
    "trial": ["probation"],
    "raise": ["promotion"], "promoted": ["promotion"],
    "journalist": ["media"], "press": ["media", "press"], "reporter": ["media"],
    "twitter": ["social", "media"], "linkedin": ["social", "media"], "facebook": ["social", "media"],
    "report": ["report"], "anonymous": ["anonymously"], "fraud": ["unethical", "whistleblower"],
    "unethical": ["unethical"],
    "present": ["speaking"], "talk": ["speaking"], "speech": ["speaking"],
    "meeting": ["meeting", "meetings"], "room": ["room", "rooms"], "book": ["booked", "booking"],
    "zoom": ["video", "conferencing"], "teams": ["video", "conferencing"], "video": ["video", "conferencing"],
    "chair": ["ergonomic", "equipment"], "desk": ["ergonomic", "equipment"], "monitor": ["ergonomic", "equipment"],
    "vpn": ["vpn"], "wifi": ["internet"], "browse": ["internet"],
    "dropbox": ["cloud", "storage"], "gdrive": ["cloud", "storage"], "onedrive": ["cloud", "storage"],
    "broken": ["helpdesk", "ticket"], "help": ["helpdesk"],
    "card": ["card"], "credit": ["card", "corporate"],
    "cash": ["petty", "cash"],
    "conference": ["conferences", "conference"], "event": ["conferences"],
    "refer": ["refer", "referral"], "friend": ["refer", "referral", "candidates"],
    "mentor": ["mentorship"], "coach": ["mentorship"],
    "contract": ["contracts", "contract"], "sign": ["signing", "contracts"],
    "gift": ["gift", "gifts"], "gifts": ["gift", "gifts"],
    "disabled": ["accommodation", "accommodations"], "disability": ["accommodation", "accommodations"],
    "hours": ["hours"], "schedule": ["hours"], "flexible": ["flexible", "adjust"],
    "parking": ["parking"], "park": ["parking"],
    "clean": ["clean", "cleanliness"], "messy": ["clean", "cleanliness"],
    "hazard": ["hazards", "safety"], "injury": ["safety", "hazards"], "accident": ["safety", "hazards"],
    "review": ["review", "reviews"], "evaluation": ["review", "reviews", "evaluated"],
    "warranty": ["warranty"], "return": ["return", "returns"], "returns": ["return", "returns"],
}

WEIGHTS = {"title": 2.5, "text": 1.0, "category": 0.5}
RULES_MIN_SCORE = 3.0  # below this the rules engine says it found nothing


def _stem(w):
    for suf in ("ations", "ation", "ments", "ment", "ings", "ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def _tokens(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS]


def _query_tokens(q):
    raw = [w for w in re.findall(r"[a-z0-9$]+", q.lower()) if w not in STOPWORDS]
    out = []
    for w in raw:
        out.append(_stem(w))
        out += [_stem(s) for s in SYNONYMS.get(w, [])]
    return out


class RulesIndex:
    def __init__(self, policies, k1=1.5, b=0.75):
        self.policies = policies
        self.k1, self.b = k1, b
        self.fields = {f: [Counter(_tokens(str(p["text" if f == "text" else f]))) for p in policies]
                       for f in WEIGHTS}
        self.avglen = {f: np.mean([sum(c.values()) for c in docs]) for f, docs in self.fields.items()}
        n = len(policies)
        df = Counter()
        for i in range(n):
            terms = set()
            for f in WEIGHTS:
                terms |= set(self.fields[f][i])
            df.update(terms)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query):
        q = set(_query_tokens(query))
        out = []
        for i in range(len(self.policies)):
            s = 0.0
            for f, w in WEIGHTS.items():
                doc = self.fields[f][i]
                L = sum(doc.values())
                for t in q:
                    tf = doc.get(t, 0)
                    if tf:
                        s += w * self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * L / self.avglen[f]))
            out.append(s)
        return np.array(out)


RULES = RulesIndex(POLICIES)


def answer_rules(question):
    t0 = time.perf_counter()
    scores = RULES.scores(question)
    best = int(np.argmax(scores))
    if scores[best] < RULES_MIN_SCORE:
        answer, title = "No matching policy was found in the policy database.", None
    else:
        p = POLICIES[best]
        answer, title = f"According to the {p['title']}: {p['text']}", p["title"]
    latency = time.perf_counter() - t0
    unsupported, reason = check_supported(answer, title)
    return dict(method="Rules-based search", answer=answer, policy_title=title,
                latency_s=latency, prompt_tokens=0, output_tokens=0, total_tokens=0,
                unsupported=unsupported, unsupported_reason=reason,
                retrieved=[], score=float(scores[best]))


# --------------------------------------------------------------------------------------
# Gemini helpers
# --------------------------------------------------------------------------------------

_client = None
_active_model = None
_skip_thinking_cfg = set()  # models that reject thinking_budget=0


def get_api_key():
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key


def client():
    global _client
    if _client is None:
        from google import genai
        key = get_api_key()
        if not key:
            raise RuntimeError("No Gemini API key found. Set GEMINI_API_KEY (see README).")
        _client = genai.Client(api_key=key)
    return _client


class DailyQuotaExhausted(RuntimeError):
    """The free-tier requests-per-day limit for a model is used up (resets at midnight US Pacific time)."""


_wasted = 0.0  # seconds spent on failed attempts and retry waits (excluded from response time)


def _with_retry(fn, tries=5):
    global _wasted
    for attempt in range(tries):
        t = time.perf_counter()
        try:
            return fn()
        except Exception as e:  # rate limits on the free tier -> wait and retry
            _wasted += time.perf_counter() - t
            msg = str(e)
            if "PerDay" in msg or "per day" in msg.lower():
                raise DailyQuotaExhausted(
                    "The free-tier daily request limit for this model is used up. Wait until it resets, "
                    "or pick a model with a higher daily limit (see https://aistudio.google.com/rate-limit) "
                    "and set GEMINI_MODEL.") from e
            transient = ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "INTERNAL",
                         "Connection", "connection", "reset", "timed out", "Timeout")
            if (any(t in msg for t in transient) or type(e).__name__ in ("ReadError", "ConnectError",
                    "RemoteProtocolError", "ReadTimeout", "ConnectTimeout")) and attempt < tries - 1:
                ts = time.perf_counter()
                time.sleep(15 * (attempt + 1))
                _wasted += time.perf_counter() - ts
                continue
            raise


def generate_json(system, user, model=None):
    """Call Gemini, return (parsed_json, usage_dict). Falls back to older models if needed."""
    from google.genai import types
    global _active_model
    explicit = bool(model)
    if explicit:
        models = [model]
    else:
        models = [_active_model] if _active_model else [GEN_MODEL] + [m for m in FALLBACK_MODELS if m != GEN_MODEL]
    last_err = None
    for model in models:
        for use_thinking_cfg in ((False,) if model in _skip_thinking_cfg else (True, False)):
            cfg = dict(system_instruction=system, temperature=0,
                       response_mime_type="application/json")
            if use_thinking_cfg:
                cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            try:
                resp = _with_retry(lambda: client().models.generate_content(
                    model=model, contents=user, config=types.GenerateContentConfig(**cfg)))
                if not explicit:
                    _active_model = model
                u = resp.usage_metadata
                usage = dict(prompt_tokens=u.prompt_token_count or 0,
                             output_tokens=(u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0),
                             total_tokens=u.total_token_count or 0)
                return _parse_json(resp.text), usage
            except DailyQuotaExhausted:
                raise
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if use_thinking_cfg and ("think" in msg or "invalid_argument" in msg or "400" in msg):
                    _skip_thinking_cfg.add(model)
                    continue          # model does not accept thinking_budget=0 -> retry without
                if "404" in msg or "not found" in msg or "not supported" in msg:
                    break             # model name not available -> try next model
                raise
    raise last_err


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", re.S)
        return json.loads(m.group(0)) if m else {}


def embed(texts, task):
    from google.genai import types
    out = []
    for i in range(0, len(texts), 90):  # batch limit is 100
        res = _with_retry(lambda: client().models.embed_content(
            model=EMBED_MODEL, contents=texts[i:i + 90],
            config=types.EmbedContentConfig(task_type=task)))
        out += [e.values for e in res.embeddings]
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def estimate_tokens(text):
    return max(1, round(len(text) / 4))


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a company policy assistant. Answer the employee's question using ONLY the
policies provided. Rules:
- Pick the single most relevant policy and put its id (e.g. "P07") ONLY in the policy_id field,
  never in the answer text.
- Base the answer strictly on that policy's text. Do not add numbers, deadlines, conditions or
  details that are not written in the policy. If the policy does not specify a detail the user
  asks about, say that the policy does not specify it.
- If no provided policy covers the question, use policy_id "{NOT_FOUND}" and say that the policy
  database does not cover the question.
- Keep the answer to 1-3 sentences.
Respond as JSON: {{"policy_id": "<id or {NOT_FOUND}>", "answer": "<answer>"}}"""


def _finish(method, data, usage, latency, retrieved=None, extra_tokens=0):
    raw_id = str(data.get("policy_id", NOT_FOUND)).strip().strip("[]") or NOT_FOUND
    answer = str(data.get("answer", ""))
    answer = re.sub(r"\s*\(?\[?\bP\d{2}\b\]?\)?", "", answer).strip() or "(empty answer)"  # drop policy-id tags
    title = BY_ID[raw_id]["title"] if raw_id in BY_ID else None
    unsupported, reason = check_supported(answer, title, raw_id)
    return dict(method=method, answer=answer, policy_title=title, latency_s=latency,
                prompt_tokens=usage["prompt_tokens"] + extra_tokens,
                output_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"] + extra_tokens,
                unsupported=unsupported, unsupported_reason=reason,
                retrieved=retrieved or [])


# --------------------------------------------------------------------------------------
# 2. LLM without a vector index (full policy database in every prompt)
# --------------------------------------------------------------------------------------

FULL_CONTEXT = "\n".join(policy_line(p) for p in POLICIES)


def answer_llm(question):
    w0, t0 = _wasted, time.perf_counter()
    user = f"POLICY DATABASE ({len(POLICIES)} policies):\n{FULL_CONTEXT}\n\nQUESTION: {question}"
    data, usage = generate_json(SYSTEM_PROMPT, user)
    latency = time.perf_counter() - t0 - (_wasted - w0)
    return _finish("LLM without vector index", data, usage, latency)


# --------------------------------------------------------------------------------------
# 3. LLM with a vector index (retrieval-augmented generation)
# --------------------------------------------------------------------------------------

_index = None


def get_index():
    """Load the embedding matrix from disk, or build it once with the embedding model."""
    global _index
    if _index is not None:
        return _index
    if INDEX_FILE.exists():
        z = np.load(INDEX_FILE, allow_pickle=False)
        if list(z["ids"]) == [p["id"] for p in POLICIES] and str(z["model"]) == EMBED_MODEL:
            _index = z["vectors"]
            return _index
    docs = [f"{p['title']}. {p['text']} Category: {p['category']}." for p in POLICIES]
    _index = embed(docs, "RETRIEVAL_DOCUMENT")
    try:
        np.savez(INDEX_FILE, vectors=_index, ids=np.array([p["id"] for p in POLICIES]), model=EMBED_MODEL)
    except OSError:
        pass
    return _index


def retrieve(question, k=TOP_K):
    q = embed([question], "RETRIEVAL_QUERY")[0]
    sims = get_index() @ q
    top = np.argsort(-sims)[:k]
    return [(POLICIES[i], float(sims[i])) for i in top]


def answer_rag(question):
    get_index()  # make sure index is built before timing the query
    w0, t0 = _wasted, time.perf_counter()
    hits = retrieve(question)
    context = "\n".join(f"{policy_line(p)} (similarity {s:.2f})" for p, s in hits)
    user = f"RETRIEVED POLICIES (top {len(hits)} from vector search):\n{context}\n\nQUESTION: {question}"
    data, usage = generate_json(SYSTEM_PROMPT, user)
    latency = time.perf_counter() - t0 - (_wasted - w0)
    return _finish("LLM with vector index", data, usage, latency,
                   retrieved=[f"{p['title']} ({s:.2f})" for p, s in hits],
                   extra_tokens=estimate_tokens(question))


METHODS = {"rules": answer_rules, "llm": answer_llm, "rag": answer_rag}


# --------------------------------------------------------------------------------------
# LLM-as-judge (used by the evaluation script for a second unsupported-answer check)
# --------------------------------------------------------------------------------------

JUDGE_PROMPT = """You check whether a policy assistant's answer is supported by the policy database.
You get the question, the answer, and the text of the policy the assistant cited (or NONE).
The answer is SUPPORTED if every factual claim in it is stated in, or directly implied by, the cited
policy text. Saying a detail is not specified, or that the database does not cover the question,
is supported. Any added number, deadline, condition, benefit or procedure not in the policy text
makes it UNSUPPORTED. Respond as JSON: {"supported": true/false, "reason": "<short reason>"}"""


def judge(question, answer, policy_title):
    text = BY_TITLE[policy_title]["text"] if policy_title in BY_TITLE else NOT_FOUND
    user = f"QUESTION: {question}\nANSWER: {answer}\nCITED POLICY: {policy_title or NOT_FOUND}\nPOLICY TEXT: {text}"
    data, _ = generate_json(JUDGE_PROMPT, user, model=JUDGE_MODEL)
    return bool(data.get("supported", True)), str(data.get("reason", ""))
