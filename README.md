---
title: LLM Arena
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Dual AI Assistant Comparison System

A production-quality evaluation platform that runs two AI assistants side by side — an open-source Qwen2.5-0.5B deployed on Modal serverless GPU and Groq-hosted Llama-3.3-70B — with real-time async parallel token streaming, three-layer persistent memory backed by Pinecone, DuckDuckGo web search, LlamaGuard-style safety classification via Groq, and automated deployment to HuggingFace Spaces via GitHub Actions CI/CD. Assistants are scored on hallucination, bias handling, and content safety using a Groq LLM judge.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│              Gradio UI -- HuggingFace Spaces (app.py)                    │
│        Chat tab  |  Safety accordion  |  Memory Inspector  |  Clear/Send  │
└────────────────────────┬───────────────────────────┬─────────────────────┘
                         │ async stream               │ async stream
                         ▼                            ▼
         ┌──────────────────────┐      ┌──────────────────────────┐
         │    OSSAssistant      │      │    FrontierAssistant     │
         │  (BaseAssistant)     │      │    (BaseAssistant)       │
         │  ConversationMemory  │      │    ConversationMemory    │
         └──────────┬───────────┘      └────────────┬─────────────┘
                    │                               │
           USE_MODAL=True                      Groq API
                    ▼                               ▼
         Modal T4 GPU endpoint         llama-3.3-70b-versatile
         qwen2.5:0.5b                  ~200-400 ms latency
         ~300-500 ms (warm)
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         ToolRegistry           │
                    │   WebSearchTool (DuckDuckGo)   │
                    │   SEARCH[query] pattern match  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   StructuredMemoryManager      │
                    │  Layer 1: Working  (deque)     │
                    │  Layer 2: Episodic (Pinecone)  │
                    │  Layer 3: Semantic (Pinecone)  │
                    │  Embeddings: all-MiniLM-L6-v2  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         SafetyFilter           │
                    │  LlamaGuard via Groq (primary) │
                    │  Keyword hard filter (second)  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────┐
         │              Evaluation Engine               │
         │  HallucinationEvaluator (Groq LLM judge)     │
         │  SafetyEvaluator        (Groq LLM judge)     │
         │  -> EvalResult list                          │
         └──────────────────────────┬───────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │    ReportGenerator    │
                        │    Evidently HTML     │
                        │    Rich console table │
                        │    Raw JSON dump      │
                        └───────────────────────┘
```

---

## Tech Stack

| Library                   | Version      | Role                                              | Rationale                                                             |
|---------------------------|--------------|---------------------------------------------------|-----------------------------------------------------------------------|
| Gradio                    | 4.44.0       | Web UI                                            | Fastest path from Python functions to shareable web demo              |
| Groq Python SDK           | 0.11.0       | Frontier chat, LlamaGuard safety, LLM judge       | Single free API key covers all three uses; ~200-400 ms latency        |
| LlamaGuard via Groq SDK   | Groq 0.11.0  | Real-time safety classification                   | Context-aware, understands intent not just keywords, 14 harm categories, reuses existing Groq key |
| Modal                     | 0.64.0       | Serverless GPU for OSS model                      | Scale-to-zero T4 GPU; only pay during inference                       |
| Ollama                    | 0.3.3        | Local OSS model for development                   | Zero-cost local dev; drop-in swap via USE_MODAL flag                  |
| Pinecone                  | >=3.0.0      | Persistent vector memory storage                  | Free tier, serverless, survives redeployments; one index with namespace separation |
| sentence-transformers     | 3.1.1        | Local embeddings for memory retrieval             | Runs on CPU, no API key, all-MiniLM-L6-v2 at 384 dimensions          |
| duckduckgo-search         | 6.2.13       | Web search tool for both assistants               | Free, no API key, no rate limits                                      |
| Evidently                 | 0.4.33       | Evaluation metrics + HTML reports                 | Purpose-built for ML model comparison dashboards                      |
| Pydantic / Settings       | 2.9.2        | Data validation and config management             | Type-safe config from env vars; validates at startup                  |
| Rich                      | 13.9.2       | CLI logging and pretty console output             | Replaces bare print(); structured, coloured, traceback-aware          |
| pytest                    | 8.3.3        | Unit tests                                        | Industry standard; all tests run without live API keys                |

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

> **Note:** `torch` is a large download and is required by `sentence-transformers` for local memory embeddings. If you only need the Gradio UI without persistent memory, you can comment out `sentence-transformers` and `torch` in `requirements.txt` first.

### c. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in at minimum:
- `GROQ_API_KEY` — get a free key at [console.groq.com](https://console.groq.com)
- Leave `USE_MODAL=False` and `USE_PINECONE=False` for local development

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

### f. Running on HuggingFace Spaces

The live deployment is available at:
**https://huggingface.co/spaces/IntimateUser6969/llm-arena**

To use the hosted Space, `USE_MODAL=True` and `USE_PINECONE=True` must be set as Space secrets so the OSS model routes to the Modal endpoint and memory persists in Pinecone rather than local disk. All API keys must be added as Space secrets in the Space settings panel. See the Deployment section below for the full secrets checklist.

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

## Deployment

### Modal OSS Model Deployment

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

### HuggingFace Spaces Deployment

**Prerequisites:** Groq API key, Modal endpoint URL, and Pinecone API key must all be ready before deploying.

**Step 1: Push code to the HF remote**

```bash
git push hf main
```

**Step 2: Add all required secrets in Space settings**

In the Space settings panel under "Variables and secrets", add each of the following:

| Secret | Value |
|--------|-------|
| `GROQ_API_KEY` | your Groq API key |
| `MODAL_ENDPOINT` | your Modal endpoint URL |
| `PINECONE_API_KEY` | your Pinecone API key |
| `PINECONE_INDEX_NAME` | `llm-arena-memory` |
| `USE_PINECONE` | `True` |
| `USE_MODAL` | `True` |
| `OSS_MODEL_NAME` | `qwen2.5:0.5b` |
| `FRONTIER_MODEL_NAME` | `llama-3.3-70b-versatile` |
| `LLAMAGUARD_MODEL` | `llama-3.1-8b-instant` |
| `CONVERSATION_MAX_TURNS` | `10` |
| `TOXICITY_THRESHOLD` | `0.7` |

**Step 3: Restart the Space to apply secrets**

After adding all secrets, restart the Space from the settings panel. The build log will show the Gradio server starting on port 7860.

**Live URL:** https://huggingface.co/spaces/IntimateUser6969/llm-arena

### CI/CD via GitHub Actions

Every push to the `main` branch triggers the workflow in `.github/workflows/deploy.yml`:

1. `pytest` runs first against the full test suite with a dummy Groq key
2. If all tests pass, the workflow force-pushes the branch to the HF Spaces remote

To enable this, add `HF_TOKEN` as a secret in the GitHub repository settings under Settings > Secrets and variables > Actions.

---

## Architecture Decisions

- **Identical system prompts across both models** — ensures any score difference is attributable to model capability, not prompt phrasing. Both assistants receive the same instructions so comparisons are fair.

- **LlamaGuard via Groq for safety classification** — LlamaGuard reuses the existing Groq API key and understands context and intent rather than matching surface patterns. A keyword hard filter runs as a second layer to catch drug synthesis and weapons content that semantic classifiers may miss when content is framed as hypothetical or educational. Together the two layers cover both intent-based and pattern-based harmful content.

- **Three-layer memory architecture** — working memory in a deque handles recent turns without I/O overhead. Episodic and semantic layers use Pinecone for cross-session persistence that survives redeployments. Local sentence-transformers embeddings mean zero API calls for memory retrieval. Facts are extracted via `[REMEMBER: ...]` tags parsed by regex — deterministic and reproducible, unlike LLM-based extraction.

- **Tool registry pattern for web search** — a `ToolRegistry` with `BaseTool` abstraction means adding new tools requires zero changes to assistants or UI. DuckDuckGo search is free with no API key and no rate limits. Tool calls are detected via regex pattern matching in model output, which works with both OSS and frontier models without requiring native function-calling support.

- **Async parallel streaming** — both assistants stream tokens simultaneously using asyncio generators and queue-based concurrency. Perceived latency equals the time to first token of the slower model rather than the sum of both latencies. `trigger_mode="once"` on Gradio event handlers prevents duplicate submissions during streaming.

---

## Tradeoffs Made

- **LlamaGuard binary verdict vs float scores** — LlamaGuard returns `safe` or `unsafe`, not a confidence float. The `SafetyResult` interface maps `unsafe` to `toxicity_score=1.0` and `safe` to `0.0` for caller compatibility. This loses score granularity but gains semantic understanding of intent that float-based classifiers lack.

- **Groq as both chat model and judge** — using Groq for frontier chat, safety classification, and evaluation judging introduces potential bias since the judge and the judged share an API provider. A separate judge model would be more rigorous. Accepted because it keeps the entire stack to one free API key.

- **Pinecone free tier single index** — the free tier allows one index only. Episodes and facts are separated by namespace within that index. This works but means all users share index capacity on the free tier and a high-traffic deployment would require a paid plan.

- **Keyword hard filter for OSS safety** — the keyword filter is bypassable with paraphrasing and does not understand context. It catches obvious patterns but not sophisticated harmful requests. Accepted as a second layer behind LlamaGuard, not as a primary safety mechanism.

- **Modal cold starts for OSS model** — Qwen2.5-0.5B on Modal T4 has 2-3 second cold starts after idle periods. Setting `min_containers=1` would eliminate this but costs money even when idle. Accepted for demo purposes where occasional cold starts are tolerable.

---

## What I Would Improve With More Time

- **Replace custom memory with mem0** — the current three-layer memory system is hand-built and accumulates contradictions over time. mem0 with a ChromaDB backend would add automatic memory conflict resolution, updating stored facts when they change rather than appending duplicates.

- **Streaming tool use** — currently tool calls interrupt streaming and return a non-streaming response after the search completes. True streaming tool use would show the search query and results appearing inline within the response stream, giving users visibility into what was searched and why.

- **Memory utilization as an evaluation metric** — adding a fifth evaluation category measuring whether each model correctly uses injected memory context would surface a meaningful capability gap. Frontier models use retrieved facts reliably. OSS models at 0.5B often ignore injected context entirely. This difference is measurable and worth surfacing in the report.

- **Persistent evaluation history** — store all `EvalResult` records in a database so reports can compare across sessions and track whether model behaviour changes over time. Currently each eval run is independent and cannot be compared to previous runs.

- **Live evaluation dashboard** — a second Gradio tab showing rolling safety scores, latency percentiles, and memory retention rate updating in real time as users chat would turn the demo into a living experiment rather than a static one-off comparison.

---

## Cost & Latency Table

| Backend              | Avg Latency         | Cost                    | Notes                                                |
|----------------------|---------------------|-------------------------|------------------------------------------------------|
| Ollama (local)       | ~2-4 s on CPU       | $0                      | Dev only; no GPU needed for 0.5B                     |
| Modal T4 GPU         | ~300-500 ms (warm)  | ~$0.0002 per 1K tokens  | Serverless; ~10-15 s cold start after idle           |
| Groq API             | ~200-400 ms         | $0 (free tier)          | Rate limited; ~30 req/min free tier                  |
| Pinecone free tier   | ~50-100 ms retrieval| $0 (per vector op)      | 1 index free, 2 GB storage, persists across deploys  |
| LlamaGuard via Groq  | ~200-400 ms         | $0 (free tier)          | 1 extra Groq call per response, reuses existing key  |
| DuckDuckGo Search    | ~500 ms-2 s         | $0 (no API key)         | Triggered only when model outputs SEARCH pattern     |
| HuggingFace Spaces   | N/A (hosting)       | $0 (free CPU tier)      | Permanent URL; no forced sleep with active traffic   |
