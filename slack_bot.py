"""
slack_bot.py - Slack bot that answers company policy questions.

Uses the preferred approach (LLM with vector index). Runs locally in Socket Mode,
so no public server is needed.

    python slack_bot.py

Needs three values in a .env file (see .env.example):
    SLACK_BOT_TOKEN   xoxb-...   (OAuth & Permissions page)
    SLACK_APP_TOKEN   xapp-...   (Basic Information -> App-Level Tokens, scope connections:write)
    GEMINI_API_KEY
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from slack_bolt import App  # noqa: E402
from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: E402

import policy_engine as pe  # noqa: E402

app = App(token=os.environ["SLACK_BOT_TOKEN"])


def build_reply(question):
    try:
        out = pe.answer_rag(question)
    except Exception as e:  # fall back to rules-based search if the LLM is unavailable
        out = pe.answer_rules(question)
        out["answer"] += f"\n_(LLM unavailable: {type(e).__name__}; used rules-based search instead)_"
    if out["policy_title"]:
        p = pe.BY_TITLE[out["policy_title"]]
        policy = f"*Relevant policy:* {p['title']}  ({p['category']} · {p['department']} dept.)\n> {p['text']}"
    else:
        policy = "*Relevant policy:* none found in the policy database"
    footer = f"_{out['method']} · {out['latency_s']:.2f}s · {out['total_tokens']} tokens_"
    if out["unsupported"]:
        footer += f"\n:warning: _Check this answer: {out['unsupported_reason']}_"
    return f"{out['answer']}\n\n{policy}\n{footer}"


def clean(text):
    return re.sub(r"<@[A-Z0-9]+>", "", text or "").strip()


@app.event("app_mention")
def on_mention(event, say):
    question = clean(event.get("text"))
    thread = event.get("thread_ts") or event["ts"]
    if not question:
        say(text="Hi! Ask me a company policy question, e.g. `@Policy Bot can I work from home on Fridays?`",
            thread_ts=thread)
        return
    say(text=build_reply(question), thread_ts=thread)


@app.event("message")
def on_dm(event, say):
    # Direct messages to the bot (channel messages are handled by app_mention)
    if event.get("channel_type") == "im" and not event.get("bot_id") and not event.get("subtype"):
        say(text=build_reply(clean(event.get("text"))))


@app.command("/policy")
def on_slash(ack, command, respond):
    ack()
    question = command.get("text", "").strip()
    if not question:
        respond("Usage: `/policy <your question>`")
        return
    respond(response_type="in_channel", text=f"*Question:* {question}\n\n{build_reply(question)}")


if __name__ == "__main__":
    print("Building vector index ...")
    pe.get_index()
    print("Policy bot is running. Mention it in a channel or use /policy. Press Ctrl+C to stop.")
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
