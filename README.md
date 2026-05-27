---
title: LLM Arena
emoji: robot
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Dual AI Assistant Comparison System

A production-quality evaluation platform that runs two AI assistants side by side — an open-source Qwen2.5-0.5B model and Groq-hosted Llama-3.3-70B — scoring each on hallucination, bias handling, and content safety using an LLM judge and Detoxify guardrails.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         Gradio UI (app.py)                       │
│          Side-by-side chat │ Safety accordion │ Clear/Send       │
└────────────────┬─────────────────────────┬───────────────────────┘
                 │ chat(user_input)         │ chat(user_input)
                 ▼                          ▼
     ┌───────────────────┐      ┌────────────────────────┐
     │   OSSAssistant    │      │   FrontierAssistant    │
     │  (BaseAssistant)  │      │   (BaseAssistant)      │
     │  ┌─────────────┐  │      │  ┌──────────────────┐  │
     │  │ConvMemory   │  │      │  │ ConvMemory       │  │
     │  └─────────────┘  │      │  └──────────────────┘  │
     └────────┬──────────┘      └───────────┬────────────┘
              │                              │
     USE_MODAL?                         Groq API
      Yes ──► Modal T4 GPU endpoint     llama-3.3-70b-versatile
      No  ──► Ollama localhost:11434
              qwen2.5:0.5b
                 │                              │
                 └──────────────┬───────────────┘
                                │ AssistantResponse
                                ▼
                     ┌──────────────────────┐
                     │    SafetyFilter      │
                     │    (Detoxify)        │
                     └──────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────┐
         │           Evaluation Engine              │
         │  HallucinationEvaluator (Groq judge)     │
         │  SafetyEvaluator (Groq judge + Detoxify) │
         │  → EvalResult list                       │
         └─────────────────┬────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   ReportGenerator     │
               │   Evidently HTML      │
               │   Rich console table  │
               │   Raw JSON dump       │
               └───────────────────────┘
```

---

## Tech Stack

| Library              | Version  | Role                                          | Rationale                                                    |
|----------------------|----------|-----------------------------------------------|--------------------------------------------------------------|
| Gradio               | 4.44.0   | Web UI                                        | Fastest path from Python functions to shareable web demo     |
| Groq Python SDK      | 0.11.0   | Frontier model API                            | Llama-3.3-70B free tier, ~200-400 ms latency                 |
| Modal                | 0.64.0   | Serverless GPU for OSS model                  | Scale-to-zero T4 GPU; only pay during inference              |
| Ollama               | 0.3.3    | Local OSS model for development               | Zero-cost local dev; drop-in swap via USE_MODAL flag         |
| Detoxify             | 0.5.2    | Real-time toxicity classifier                 | Runs locally, no API key, 6 toxicity dimensions              |
| Evidently            | 0.4.33   | Evaluation metrics + HTML reports             | Purpose-built for ML model comparison dashboards             |
| Pydantic / Settings  | 2.9.2    | Data validation and config management         | Type-safe config from env vars; validates at startup         |
| Rich                 | 13.9.2   | CLI logging and pretty console output         | Replaces bare print(); structured, coloured, traceback-aware |
| pytest               | 8.3.3    | Unit tests                                    | Industry standard; all tests run without live API keys       |

---

## Setup Instructions

### a. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd dual-ai-assistant
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### b. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch==2.4.1` is a large download. If you only need the Gradio UI and don't
> plan to run Detoxify locally, you can skip torch by editing requirements.txt first.

### c. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
- `GROQ_API_KEY` — get a free key at [console.groq.com](https://console.groq.com)
- Leave `USE_MODAL=False` for local development with Ollama

### d. Install Ollama and pull the OSS model (local dev)

1. Download Ollama from [ollama.com](https://ollama.com)
2. Start the Ollama server (it runs as a background service)
3. Pull the model:

```bash
ollama pull qwen2.5:0.5b
```

### e. Launch the chat UI

```bash
python main.py --mode chat
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`) and a public share link.

---

## Running Evaluation

```bash
python main.py --mode eval
```

This will:
1. Load all 39 prompts from `eval_data/`
2. Run each prompt through both assistants
3. Score each response with Groq-based LLM judges
4. Generate an HTML report in `outputs/eval_report_<timestamp>.html`
5. Print a summary table to the console

---

## Deploying to Modal

1. Authenticate with Modal:

```bash
modal setup
```

2. Deploy the serverless endpoint:

```bash
modal deploy deployment/modal_app.py
```

3. Copy the printed HTTPS endpoint URL into your `.env`:

```
MODAL_ENDPOINT=https://your-workspace--qwen-assistant-chat-endpoint.modal.run
USE_MODAL=True
```

The endpoint auto-scales to zero when idle and wakes on the first request (~10-15 s cold start for the 0.5B model on T4).

---

## Architecture Decisions

- **Identical system prompts across both models** — ensures any score difference is attributable to model capability, not prompt phrasing.
- **LLM-as-judge for hallucination and safety** — Groq free tier makes this zero-cost; a dedicated judge model is far more nuanced than keyword matching or ROUGE scores.
- **Detoxify loaded lazily, never blocking startup** — model is ~500 MB; lazy loading avoids slowing down chat mode if only the UI is needed.
- **SafetyFilter as a pure classifier (never blocking)** — keeping classification separate from policy enforcement makes the system testable and auditable; the UI decides what to show.
- **ConversationMemory as a sliding-window deque** — O(1) push and automatic eviction of oldest messages; no risk of unbounded memory growth in long sessions.

---

## Tradeoffs Made

- **No streaming** — both assistants return complete responses. Adding streaming would require `yield`-based generators and Gradio's streaming mode, adding significant complexity for limited user benefit at sub-1 s latency.
- **Groq as judge for evaluations** — this introduces a conflict of interest (Groq judges a Groq-hosted model). In production, a separate judge model (e.g. Claude via API) would be preferable.
- **Ollama for local OSS dev, not the full Transformers stack** — Ollama is more reliable across platforms than running Transformers + CUDA locally and much faster to set up for development.
- **Evidently drift metrics for the report** — Evidently is designed for data/concept drift, not LLM benchmarking. It provides a reasonable visual baseline; a dedicated LLM eval framework (e.g. LangSmith, Ragas) would give deeper metrics.
- **Single T4 GPU on Modal** — sufficient for a 0.5B model but limits throughput. A production deployment would use autoscaling with multiple replicas.

---

## What I Would Improve With More Time

- **Add RAG (Retrieval-Augmented Generation)** — inject a knowledge base so both models answer grounded questions, reducing hallucination and making the comparison more meaningful.
- **Replace the Groq judge with Claude Sonnet** — removes judge-model bias and gives access to constitutional AI scoring for more nuanced safety evaluation.
- **Streaming responses** — implement `yield`-based streaming for both backends so users see tokens as they arrive rather than waiting for the full response.
- **Persistent evaluation history in SQLite** — store all EvalResults in a local DB so reports can compare across sessions and track model improvement over time.
- **CI pipeline with automated eval regression tests** — run a subset of factual prompts on every commit and fail the build if mean hallucination score drops below a threshold.

---

## Cost & Latency Table

| Backend        | Avg Latency   | Cost per 1K tokens | Notes                              |
|----------------|---------------|--------------------|------------------------------------|
| Ollama (local) | ~2–4 s on CPU | $0                 | Dev only; no GPU needed for 0.5B   |
| Modal T4 GPU   | ~300–500 ms   | ~$0.0002           | Serverless; ~10–15 s cold start    |
| Groq API       | ~200–400 ms   | $0 (free tier)     | Rate limited; ~30 req/min free tier|
