# Company Policy Assistant

Simon Devald · AI in Business and Accounting (SDU)

A policy question bot that answers questions about the 98 policies in `company_policies.csv` and names the relevant policy. It compares three approaches:

| Approach | How it works |
|---|---|
| Rules-based search | BM25 keyword scoring with a synonym list. Returns the verbatim policy text. No LLM, 0 tokens. |
| LLM without vector index | Gemini gets **all 98 policies** in every prompt and picks the one that applies. |
| LLM with vector index | Policies are embedded once with `gemini-embedding-001`. Each question is embedded, the top 3 most similar policies are retrieved, and only those go to Gemini. |

Each answer is checked for **unsupported content**, meaning content that is not backed by the policy database:
- Live (website and Slack): the check flags a cited policy that doesn't exist, numbers in the answer that aren't in the cited policy, and specifics given without citing any policy.
- Benchmark (`evaluate.py`): the same checks, plus an LLM judge and 4 out-of-scope questions. Answering an out-of-scope question with a policy counts as unsupported.

## Files

| File | Purpose |
|---|---|
| `policy_engine.py` | All three approaches, the unsupported-answer checks, and the LLM judge |
| `app.py` | Streamlit website (live side-by-side comparison, plus the two-paragraph conclusion and two tables) |
| `evaluate.py` | Runs the 26 test questions through all three approaches and writes `results.csv` |
| `test_questions.csv` | Test set: 18 answerable, 4 "trap" (the detail isn't in the policy), 4 out-of-scope |
| `slack_bot.py` | Slack bot using the LLM with vector index. Runs locally in Socket Mode. |
| `slack_manifest.yml` | Paste-in configuration for creating the Slack app |

---

## 1. Get a Gemini API key
Go to https://aistudio.google.com/apikey and create a key. The free tier is enough. You can reuse your key from the P-Card assignment.

## 2. Run it locally

**Mac (Terminal)**
```bash
cd ~/Desktop/policy-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then open .env and paste your Gemini key
export GEMINI_API_KEY="paste-key-here"
python evaluate.py            # ~10-15 min, writes results.csv
streamlit run app.py
```

**Windows (PowerShell in VS Code: Terminal → New Terminal)**
```powershell
cd $HOME\Desktop\policy-assistant
py -m venv .venv
.venv\Scripts\Activate.ps1     # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
copy .env.example .env         # then open .env and paste your Gemini key
$env:GEMINI_API_KEY="paste-key-here"
python evaluate.py
streamlit run app.py
```

`evaluate.py` waits between calls to stay inside the free-tier limits. If you hit rate limits anyway, set `EVAL_DELAY=10` (Mac: `export EVAL_DELAY=10`, Windows: `$env:EVAL_DELAY="10"`). If `gemini-3.5-flash` isn't available on your key, the code falls back to older Flash models automatically. You can also set `GEMINI_MODEL` yourself.

## 3. Put it on GitHub
1. On github.com, create a new **public** repository, e.g. `Assignment-policy-bot`.
2. Click **uploading an existing file** and drag in every file from the folder **except** `.env` and `.venv`. Include `results.csv` and `policy_index.npz` (both are created by step 2).
3. Commit.

## 4. Deploy the website (Streamlit Community Cloud)
1. Go to https://share.streamlit.io, then **Create app**, then choose your repo, branch `main`, and file `app.py`.
2. Under **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "paste-key-here"
   ```
3. Click Deploy, then copy the URL. That's deliverable 1.

## 5. Slack workspace, channel and bot

**Workspace and channel (these must identify you)**
1. Go to https://slack.com/get-started#/createnew and create a workspace with your SDU or Gmail email.
2. Name the workspace something like **"Simon Devald – AI in Business"**.
3. Create a channel called **#policy-bot-simon-devald**.

**Create the bot**
1. Go to https://api.slack.com/apps, then **Create New App**, then **From an app manifest**, and pick your new workspace.
2. Paste the contents of `slack_manifest.yml` (choose the YAML tab), then click Next and Create.
3. Under **Basic Information → App-Level Tokens**, click **Generate Token and Scopes**. Name it `socket`, add the scope `connections:write`, and click Generate. Copy the `xapp-…` token into `.env` as `SLACK_APP_TOKEN`.
4. Under **Install App**, click **Install to Workspace** and then Allow. Copy the **Bot User OAuth Token** (`xoxb-…`) into `.env` as `SLACK_BOT_TOKEN`.
5. In Slack, open #policy-bot-simon-devald and type `/invite @Policy Bot`.

**Run the bot** (keep the terminal open while testing)
```bash
python slack_bot.py
```
In the channel, write: `@Policy Bot can I work from home on Fridays?`. You can also use `/policy can I work from home on Fridays?`. The bot replies in a thread with the answer, the relevant policy, the time and the token count. **Take a screenshot.** That's deliverable 4.

**Invite the teacher**
1. In Slack, click the workspace name, then **Invite people to …**, enter `raz@sdu.dk`, and send.
2. Open the channel, click its name, then **Add people**, and add raz@sdu.dk. On the free plan you can add him once he's a workspace member. Slack also lets you add an email address to the channel invite directly.

## Deliverables checklist
- [ ] Link to the deployed website (step 4)
- [ ] Two-paragraph comparison. It's shown on the website's "Comparison & conclusion" tab and is generated from your actual `results.csv` numbers, so copy it from there.
- [ ] Link to the GitHub repository (step 3)
- [ ] Screenshot of the bot answering in the channel (step 5)
- [ ] raz@sdu.dk invited to the workspace and channel
