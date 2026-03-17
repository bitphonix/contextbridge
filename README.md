# ContextBridge

**Context portability layer for AI conversations.**

When you hit a quota limit mid-session, everything you've built up — the decisions, the dead ends, the current state — disappears. You open a new model and start from scratch. ContextBridge fixes that.

Paste any AI conversation. Get back a structured **session brain** and a ready-to-paste briefing that lets any AI model pick up exactly where you left off.

> Live demo: [contextbridge.dev](https://contextbridge.dev) &nbsp;·&nbsp; Built with LangGraph + Google Gemini + MongoDB Atlas

[![Codecov](https://codecov.io/gh/YOUR_USERNAME/contextbridge/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/contextbridge)
[![CodeScene](https://codescene.io/projects/YOUR_PROJECT_ID/status-badges/code-health)](https://codescene.io/projects/YOUR_PROJECT_ID)
[![Datadog](https://img.shields.io/badge/observability-datadog-632CA6)](https://datadoghq.com)
[![Python](https://img.shields.io/badge/python-3.11-3776AB)](https://python.org)

---

## The problem

Every AI power user hits this daily:

```
2 hours into a Claude session. Context established. Decisions made.
Approaches tried and rejected. Clear next step identified.

→ Daily limit hit.
→ Open ChatGPT.
→ Stare at empty chat box.
```

You either re-explain everything from memory (losing nuance), paste the raw conversation (too long, model gets confused), or start over. **Nobody has solved cross-tool context portability** — until now.

---

## How it works

ContextBridge runs a **4-node LangGraph pipeline** with a self-correcting reflection loop:

```
User pastes conversation
        ↓
┌──────────────────────────────────────────────────────┐
│                  LangGraph Pipeline                  │
│                                                      │
│  [Classifier] → [Extractor] → [Compressor] → [Critic]│
│                      ↑__________________|            │
│                   reflection loop (max 2 retries)    │
└──────────────────────────────────────────────────────┘
        ↓
  Session Brain (5 fields) + Ready-to-paste Briefing
        ↓
  Saved to MongoDB Atlas
```

### The 5-field session brain

| Field | What it captures |
|---|---|
| **Goal** | What the user was trying to accomplish — one precise sentence |
| **Decisions** | Specific choices locked in, with reasoning ("Chose X over Y because Z") |
| **Dead ends** | Approaches tried and rejected — so the next model won't re-suggest them |
| **Current state** | Exactly where things stand — what works, what's broken, what's blocked |
| **Next steps** | Immediate actions, ordered by priority |

### The briefing output

A single dense paragraph (150–250 words) written for an AI model to read cold — specific enough that it can continue without asking clarifying questions, explicit about dead ends so it doesn't waste your time.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS)          served by FastAPI               │
│  ├── Paste input                                                 │
│  ├── Extraction result (5-field brain cards)                    │
│  ├── Ready-to-paste briefing (one-click copy)                   │
│  └── Past sessions dashboard                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │ POST /extract
┌────────────────────────▼────────────────────────────────────────┐
│  FastAPI Backend                                                 │
│  ├── Doppler      → secret management                           │
│  ├── Sentry       → error tracking                              │
│  └── Datadog APM  → LLM trace observability                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  LangGraph StateGraph                                           │
│                                                                  │
│  classifier_node  → detects domain (flash model, fast)          │
│  extractor_node   → extracts 5 brain fields (pro model)         │
│  compressor_node  → generates briefing paragraph (pro model)    │
│  critic_node      → scores briefing 0-10 (flash model)         │
│  save_node        → persists to MongoDB on passing score        │
│                                                                  │
│  Reflection loop: critic score < 7 → retry extractor (max 2x)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│  Storage                                                         │
│  ├── MongoDB Atlas  → session brain documents                   │
│  └── mem0           → cross-session user memory (roadmap)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Agent framework | LangGraph 0.2 | StateGraph with conditional edges for reflection loop |
| LLM | Google Gemini 2.5 Pro/Flash | Pro for extraction quality, Flash for fast classifier/critic |
| API | FastAPI | Async, auto-docs, Pydantic validation |
| Database | MongoDB Atlas | Document-shaped brain storage, free tier |
| Observability | Datadog APM | Full LLM trace visibility per pipeline run |
| Error tracking | Sentry | Real-time error capture with stack traces |
| Secret management | Doppler | Zero `.env` files, secrets injected at runtime |
| Deployment | DigitalOcean App Platform | Always-on, no cold starts |

---

## Agentic patterns demonstrated

This project is intentionally built to cover the most in-demand agentic AI paradigms:

- **Multi-node orchestration** — 5 specialized nodes, each with a single responsibility
- **Structured output extraction** — Pydantic models enforcing LLM output schema
- **Self-correction / reflection loop** — Critic scores output, failed attempts retry with feedback
- **Stateful pipeline** — Shared `TypedDict` state flows through every node
- **Production observability** — Every node execution traced in Datadog APM

---

## Project structure

```
contextbridge/
├── app/
│   ├── main.py              ← FastAPI app and routes
│   ├── graph/
│   │   ├── state.py         ← State TypedDict (shared across all nodes)
│   │   ├── nodes.py         ← All 5 agent functions
│   │   ├── edges.py         ← Conditional routing logic
│   │   └── pipeline.py      ← StateGraph assembly
│   ├── models/
│   │   └── brain.py         ← Pydantic models for structured output
│   └── db/
│       └── mongo.py         ← MongoDB connection and queries
├── frontend/
│   └── index.html           ← Single-file UI
└── requirements.txt
```

---

## Running locally

**Prerequisites:** Python 3.11+, [Doppler CLI](https://docs.doppler.com/docs/install-cli)

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/contextbridge
cd contextbridge
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure secrets via Doppler
doppler setup   # select project: contextbridge, config: dev

# Required secrets (set via: doppler secrets set KEY)
# GEMINI_API_KEY   → from aistudio.google.com
# MONGODB_URI      → from MongoDB Atlas
# SENTRY_DSN       → from sentry.io
# DD_API_KEY       → from datadoghq.com
# DD_SERVICE       → contextbridge
# DD_SITE          → us5.datadoghq.com

# Run with full observability
doppler run -- ddtrace-run uvicorn app.main:app --reload
```

Open `http://localhost:8000`

---

## Design decisions

**Why LangGraph over a simple chain?**
The reflection loop requires conditional routing — if the critic scores below 7, the graph routes back to the extractor with feedback. This isn't possible in a linear chain. LangGraph's `add_conditional_edges` handles this cleanly with a shared state object.

**Why JSON prompt instead of `.with_structured_output()` for the extractor?**
`langchain-google-genai` doesn't correctly serialize `List[str]` fields into Gemini's function declaration schema — it generates the list type but omits the required `items` field. The fix: prompt Gemini to return raw JSON directly, then validate through Pydantic. The model still enforces the schema; we just bypass the broken serialization layer.

**Why two Gemini models?**
Classifier and Critic are fast, cheap tasks — Flash is sufficient. Extractor and Compressor require reasoning quality — Pro is worth the extra latency. This keeps average pipeline cost low while maintaining extraction accuracy.

**Why Doppler over `.env`?**
No risk of accidentally committing secrets. Secrets are injected at runtime, synced automatically to production. Zero config difference between local and deployed environments.

---

## Roadmap

- [ ] File upload support (ChatGPT JSON export, Claude conversation export)
- [ ] Browser extension for one-click capture
- [ ] mem0 integration for cross-session user memory
- [ ] Model-specific briefing optimization (Claude vs GPT vs Gemini formatting)
- [ ] Project brain — link related sessions into a unified project context

---

## What makes this different from mem0?

[mem0](https://github.com/mem0ai/mem0) is a developer SDK — you integrate it into your own app. It works inside a single application's ecosystem and requires the app developer to implement it. You cannot use mem0 with Claude.ai, ChatGPT, or any existing AI tool.

ContextBridge is a user-facing tool that works *on top of* any AI conversation, regardless of what tool generated it. mem0 stores facts across sessions within one app. ContextBridge extracts decisions and reasoning across tools. They're complementary, not competing.

---

Built by [Tanishk](https://github.com/bitphonix) · [LinkedIn](https://linkedin.com/in/tanishk-soni-a94077239)