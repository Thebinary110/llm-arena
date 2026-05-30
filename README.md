---
title: LLM Arena
emoji: 🤖
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# LLM Arena

Side-by-side comparison and evaluation of an open-source model (Qwen 2.5-0.5B on Modal GPU) against a frontier hosted model (Llama 3.3-70B via Groq), with real-time streaming, three-layer persistent memory, web search tool use, LlamaGuard-based safety classification, and automated evaluation across hallucination, bias, and content safety.

**GitHub:** https://github.com/Thebinary110/llm-arena/tree/Refractoring-Scale  
**Live Demo:** https://huggingface.co/spaces/IntimateUser6969/llm-arena

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HuggingFace Spaces                              │
│                     Gradio UI  (app.py)                             │
│          Chat Tab (side-by-side)  |  Memory Inspector Tab           │
└────────────────┬────────────────────────────┬───────────────────────┘
                 │                            │
    ┌────────────▼──────────┐    ┌────────────▼──────────────┐
    │     OSSAssistant      │    │    FrontierAssistant       │
    │   (BaseAssistant)     │    │    (BaseAssistant)         │
    │   USE_MODAL=True      │    │    Groq API                │
    └────────────┬──────────┘    └────────────┬──────────────┘
                 │                            │
    Modal T4 GPU endpoint           llama-3.3-70b-versatile
    qwen2.5:0.5b (Transformers)     ~200-400ms warm latency
    ~300-500ms warm latency
                 │                            │
                 └──────────────┬─────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │              ToolRegistry                   │
          │         WebSearchTool (DuckDuckGo)          │
          │    Triggered by SEARCH: pattern in output   │
          └─────────────────────┬──────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │         StructuredMemoryManager             │
          │  Layer 1: Working Memory (deque, in-proc)   │
          │  Layer 2: Episodic Memory (Pinecone)        │
          │  Layer 3: Semantic Memory (Pinecone)        │
          │  Embeddings: all-MiniLM-L6-v2 (local CPU)  │
          └─────────────────────┬──────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │              SafetyFilter                   │
          │  Primary:  GPT-OSS-Safeguard-20B via Groq  │
          │  Secondary: Keyword hard filter             │
          └─────────────────────┬──────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │           Evaluation Engine                 │
          │  HallucinationEvaluator (Groq LLM judge)   │
          │  SafetyEvaluator (Groq + keyword filter)    │
          │  BiasEvaluator (Groq LLM judge)             │
          │  ReportGenerator (Evidently + Rich)         │
          └─────────────────────────────────────────────┘
```

---

## Tech Stack

| Library | Version | Role | Rationale |
|---|---|---|---|
| Gradio | 5.9.1 | Web UI | Native HF Spaces support, built-in chat, shareable link |
| Groq Python SDK | 0.11.0 | Frontier chat + safety + eval judge | Single free API key for three roles |
| Modal | 0.64.0 | Serverless GPU for OSS model | Scale-to-zero T4, pay only during inference |
| Ollama | 0.3.3 | Local OSS model for development | Zero-cost local dev, drop-in swap via USE_MODAL flag |
| Pinecone | >=3.0.0 | Persistent vector memory | Free tier, serverless, survives redeployments |
| sentence-transformers | 3.1.1 | Local embeddings for memory | CPU, no API key, 384-dim all-MiniLM-L6-v2 |
| duckduckgo-search | 6.2.13 | Web search tool | Free, no API key, no rate limits locally |
| Evidently | 0.4.33 | Evaluation metrics and HTML reports | Purpose-built for ML model comparison |
| Pydantic Settings | 2.9.2 | Config from environment variables | Type-safe, validates at startup |
| Rich | 13.9.2 | CLI logging and console output | Structured, coloured, traceback-aware |
| pytest | 8.3.3 | Unit tests | Runs without live API keys via mocks |

---

## Local Development Setup

### a. Clone the repository

```bash
git clone https://github.com/Thebinary110/llm-arena.git
cd llm-arena
git checkout Refractoring-Scale
```

### b. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### c. Install dependencies

```bash
pip install -r requirements.txt
```

### d. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```
GROQ_API_KEY=your_groq_api_key_here
USE_MODAL=False
USE_PINECONE=False
```

Get a free Groq API key at https://console.groq.com

### e. Install Ollama for local OSS model development

1. Download Ollama from https://ollama.com
2. Start the Ollama server (runs as a background service after install)
3. Pull the model:

```bash
ollama pull qwen2.5:0.5b
```

### f. Run the app locally

```bash
python main.py --mode chat
```

Gradio prints a local URL at `http://127.0.0.1:7860`.

### g. Run tests

```bash
python main.py --mode test
```

Or directly:

```bash
pytest tests/ -v
```

### h. Run evaluation

```bash
python main.py --mode eval
```

This loads all 39 prompts from `eval_data/`, runs them through both assistants, scores with an LLM judge, and generates an HTML report in `outputs/`.

---

## Deploying the OSS Model to Modal

If you do not have a Modal account, create one at https://modal.com (free tier available, no credit card required for the initial credits).

### Step 1: Install Modal and authenticate

```bash
pip install modal
python -m modal setup
```

This opens a browser window to authenticate. Follow the prompts.

### Step 2: Deploy the endpoint

```bash
modal deploy deployment/modal_app.py
```

Modal will print an HTTPS endpoint URL after deployment. It looks like:

```
https://your-workspace--qwen-assistant-chat-endpoint.modal.run
```

Copy this URL.

### Step 3: Update your environment

Add to your `.env`:

```
MODAL_ENDPOINT=https://your-workspace--qwen-assistant-chat-endpoint.modal.run
USE_MODAL=True
```

### Step 4: Verify the endpoint

```bash
python -c "
import requests
resp = requests.post(
    'https://your-workspace--qwen-assistant-chat-endpoint.modal.run',
    json={'messages': [{'role': 'user', 'content': 'hello'}], 'max_tokens': 50}
)
print(resp.json())
"
```

### Modal Troubleshooting

**429 Too Many Requests:**  
Your Modal free tier has hit its concurrency limit. Either wait a few minutes or create a new Modal account for fresh credits. Update `MODAL_ENDPOINT` with the new deployment URL.

**Cold start taking 15-20 seconds:**  
This is normal on the free tier. The container is loading Qwen from scratch. Warm calls after the first one are 300-500ms. Set `min_containers=1` in `deployment/modal_app.py` to keep the container warm permanently (costs ~$0.28/day).

**Modal endpoint not available / account exhausted:**  
Switch to the HuggingFace Inference API as a fallback. Set `USE_MODAL=False` in your `.env` and `OLLAMA_BASE_URL` to the HF Inference API endpoint for Qwen2.5-0.5B. The OSS assistant will degrade gracefully.

---

## Deploying to HuggingFace Spaces

### Prerequisites

Have these API keys ready:
- Groq API key from https://console.groq.com
- Modal endpoint URL from the deployment above
- Pinecone API key from https://app.pinecone.io (free account, no billing)
- HuggingFace token with write permissions from https://huggingface.co/settings/tokens

### Step 1: Add the HF remote to your local repo

```bash
git remote add hf https://YOUR_HF_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/llm-arena
```

### Step 2: Push your code to HF Spaces

```bash
git push hf YOUR_BRANCH:main --force
```

If the push is rejected due to binary files in git history:

```bash
pip install git-filter-repo
git filter-repo --path data/ --invert-paths --force
git remote add hf https://YOUR_HF_USERNAME:YOUR_HF_TOKEN@huggingface.co/spaces/YOUR_USERNAME/llm-arena
git push hf YOUR_BRANCH:main --force
```

### Step 3: Add secrets in HF Space settings

Go to: `https://huggingface.co/spaces/YOUR_USERNAME/llm-arena/settings`

Add these as **Secrets** (private):

| Secret Name | Value |
|---|---|
| GROQ_API_KEY | your Groq API key |
| MODAL_ENDPOINT | your Modal HTTPS endpoint URL |
| PINECONE_API_KEY | your Pinecone API key |

Add these as **Variables** (public):

| Variable Name | Value |
|---|---|
| PINECONE_INDEX_NAME | llm-arena-memory |
| USE_PINECONE | True |
| USE_MODAL | True |
| OSS_MODEL_NAME | qwen2.5:0.5b |
| FRONTIER_MODEL_NAME | llama-3.3-70b-versatile |
| LLAMAGUARD_MODEL | openai/gpt-oss-safeguard-20b |
| CONVERSATION_MAX_TURNS | 10 |
| TOXICITY_THRESHOLD | 0.7 |
| LOG_LEVEL | INFO |

### Step 4: Restart the Space

Click **Factory reboot** in Space settings to pick up all secrets and rebuild from scratch.

### Step 5: Watch build logs

Go to the Space URL and click the **Logs** tab. The build takes 3-5 minutes. Common errors and fixes:

**`No module named 'pyaudioop'`:**  
Python version mismatch. Add a `.python-version` file at the root containing `3.10` and push again. Or update `sdk_version` in README.md frontmatter to `5.9.1`.

**Binary files rejected:**  
Run the `git filter-repo` command above to purge `data/` from history.

**`StructuredMemoryManager init failed`:**  
PINECONE_API_KEY secret is missing or incorrect. Verify it in Space settings.

---

## CI/CD Pipeline

Every push to the `main` branch on GitHub triggers an automated pipeline via GitHub Actions.

### Pipeline steps

```
Push to main branch
      |
      v
Run pytest tests/ -v
      |
      v (if tests pass)
Push to HuggingFace Spaces
      |
      v
HF Spaces rebuilds and redeploys
```

### Setup

1. Go to your GitHub repository Settings > Secrets and Variables > Actions
2. Add a new repository secret:
   - Name: `HF_TOKEN`
   - Value: your HuggingFace token with write permissions

The workflow file is at `.github/workflows/deploy.yml`.

### Workflow behaviour

- Tests run on every push to every branch
- Deployment to HF Spaces only happens on push to `main`
- If pytest fails, deployment is blocked
- Force push to HF remote handles diverged history automatically

---

## Architecture Decisions

**Identical system prompts across both models**  
Any score difference is attributable to model capability, not prompt phrasing. Both models receive the same instructions, the same memory context, and the same tool definitions.

**LlamaGuard and GPT-OSS-Safeguard for safety classification**  
Context-aware safety models understand intent rather than just surface keywords. A drug synthesis tutorial scores 0.000 on Detoxify (no profanity) but correctly triggers GPT-OSS-Safeguard. A keyword hard filter runs as a second layer for deterministic blocking of known harmful patterns.

**Three-layer memory with Pinecone**  
Working memory (deque) handles recent turns in-process. Episodic and semantic layers use Pinecone for cross-session persistence. Local sentence-transformers embeddings mean zero API calls for retrieval. Facts are extracted via REMEMBER tags parsed by regex — deterministic and reproducible unlike LLM-based extraction.

**Tool registry pattern for web search**  
A ToolRegistry with BaseTool abstraction means adding new tools requires zero changes to assistants or the UI. Tools are triggered by pattern matching in model output, not native function calling, so they work identically for both models regardless of their tool-calling support.

**Async parallel streaming**  
Both assistants stream tokens simultaneously using asyncio generators. Perceived latency equals the time-to-first-token of the slower model rather than the sum of both. `trigger_mode="once"` on Gradio event handlers prevents duplicate submissions from Enter and click both firing.

**USE_MODAL and USE_PINECONE flags**  
Both flags default to False so local development uses Ollama and ChromaDB with zero external dependencies. Setting them to True in HF Space secrets switches to production backends without code changes.

---

## Tradeoffs

**LlamaGuard binary verdict vs float scores**  
LlamaGuard returns `safe` or `unsafe`, not a confidence float. The interface maps unsafe to 1.0 and safe to 0.0 to maintain compatibility with SafetyResult. This loses score granularity but gains semantic understanding of intent over any float-based toxicity model.

**Groq as chat model, safety classifier, and eval judge**  
Using one API for all three roles introduces potential judge bias — the same provider evaluates its own model. Accepted because it keeps the stack to one free API key. A separate judge provider (Claude, GPT-4) would be more rigorous.

**Pinecone free tier single index with namespaces**  
Free tier allows one index. Episodes and facts are separated by namespace within one index. This works but means all users and both assistants share index capacity on the free tier.

**Keyword hard filter is bypassable**  
The keyword filter catches obvious patterns but not paraphrased harmful requests. It is a second layer behind the semantic classifier, not a primary safety mechanism. Accepted as defence-in-depth rather than sole protection.

**DuckDuckGo rate limited on HF Spaces cloud IPs**  
DuckDuckGo blocks requests from shared cloud infrastructure. The tool triggers but returns a rate limit error on the deployed Space. The fix (Brave Search API free tier) is straightforward but not yet implemented.

---

## What I Would Improve With More Time

**Replace custom memory with mem0**  
mem0 adds automatic memory conflict resolution — when the user says they switched from PostgreSQL to MongoDB, it updates the existing memory rather than storing a contradiction. The current three-layer system accumulates contradictions without reconciling them. The abstraction is already shaped for this migration.

**Streaming tool use**  
Currently a tool call interrupts streaming and returns a non-streaming response. True streaming tool use would show the SEARCH trigger and results inline in the token stream, matching how users expect it to work.

**Memory utilization as a fifth evaluation metric**  
Injecting memory context the model ignores is measurably different from the frontier model behavior. A test sequence (introduce yourself in turn 1, ask an unrelated question in turns 2-4, ask the model to reference you by name in turn 10) would surface this gap quantitatively.

**Brave Search API replacing DuckDuckGo**  
Free tier, 2000 queries per month, works from cloud IPs, no credit card. One import change in `src/tools/web_search.py`.

**Live evaluation dashboard**  
A second Gradio tab showing rolling safety scores, latency percentiles, and memory retention rate updating in real time as users chat would turn the demo into a living experiment. Every chat message generates evaluation data automatically.

**vLLM deployment for OSS model**  
Replacing HuggingFace Transformers with vLLM on Modal would enable continuous batching and PagedAttention, reducing cold-start latency and supporting concurrent users. Estimated improvement: 2-4x throughput on the same T4 GPU.

**Persistent evaluation history**  
Store all EvalResults in SQLite so reports compare across sessions and track whether model behavior changes over time. Currently each eval run is independent and results are not persisted.

---

## Cost and Latency Reference

| Backend | Avg Latency | Cost per 1K tokens | Notes |
|---|---|---|---|
| Ollama local | 2-4s on CPU | $0.00 | Dev only, no GPU needed for 0.5B |
| Modal T4 GPU | 300-500ms warm | ~$0.0002 | Cold start 2-3s, serverless, scale-to-zero |
| Groq API | 200-400ms | $0.00 free tier | Rate limited, ~30 req/min free tier |
| Pinecone free | 50-100ms retrieval | $0.00 free tier | 1 index, 2GB storage, persistent across deploys |
| GPT-OSS-Safeguard | 200-400ms | $0.00 free tier | 1 extra Groq call per response |
| DuckDuckGo Search | 500ms-2s | $0.00 | Rate limited on HF Spaces cloud IPs |
| HuggingFace Spaces | N/A | $0.00 free CPU tier | Permanent URL, no sleep with traffic |

---

## Project Structure

```
dual-ai-assistant/
├── src/
│   ├── assistants/
│   │   ├── base.py                  # Abstract BaseAssistant, AssistantResponse
│   │   ├── oss_assistant.py         # Qwen 2.5-0.5B via Ollama or Modal
│   │   └── frontier_assistant.py    # Llama 3.3-70B via Groq
│   ├── memory/
│   │   ├── conversation_memory.py   # Sliding window (used by eval pipeline)
│   │   └── structured_memory.py     # Three-layer ChromaDB/Pinecone memory
│   ├── guardrails/
│   │   └── safety_filter.py         # GPT-OSS-Safeguard + keyword filter
│   ├── tools/
│   │   ├── base_tool.py             # BaseTool abstract class, ToolResult
│   │   ├── web_search.py            # DuckDuckGo WebSearchTool
│   │   └── tool_registry.py         # Registry + pattern detection
│   ├── evaluation/
│   │   ├── evaluator.py             # BaseEvaluator, EvalResult dataclass
│   │   ├── hallucination.py         # LLM-as-judge hallucination scorer
│   │   ├── bias_safety.py           # LLM-as-judge safety scorer
│   │   └── report_generator.py      # Evidently HTML + Rich console table
│   └── ui/
│       └── app.py                   # Gradio Blocks layout, async streaming
├── eval_data/
│   ├── factual_prompts.json         # 15 factual prompts with ground truth
│   ├── adversarial_prompts.json     # 12 jailbreak and harmful request prompts
│   └── bias_prompts.json            # 12 stereotype and discrimination prompts
├── deployment/
│   └── modal_app.py                 # Modal serverless GPU deployment
├── tests/
│   ├── test_memory.py
│   ├── test_guardrails.py
│   └── test_evaluators.py
├── .github/
│   └── workflows/
│       └── deploy.yml               # GitHub Actions CI/CD
├── app.py                           # HuggingFace Spaces entry point
├── config.py                        # Pydantic Settings from env vars
├── main.py                          # CLI: --mode chat | eval | test
├── requirements.txt
├── .env.example
├── .python-version                  # Pins Python 3.10 for HF Spaces
└── README.md
```