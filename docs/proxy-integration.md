# Proxy Integration Guide

How to run this RAG system against any LLM backend — Anthropic Claude, Qwen, Mistral, local models — through an OpenAI-compatible proxy.

## The Pattern

This system uses the OpenAI Python SDK for all LLM calls (classification + synthesis). The SDK supports a `base_url` parameter, which means **any service that speaks the OpenAI chat completions protocol works as a drop-in backend**.

```
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  RAG System  │────▶│  Proxy / Gateway  │────▶│  LLM Backend    │
│  (OpenAI SDK)│     │  (OpenAI-compat)  │     │                 │
│              │     │                   │     │  • OpenAI       │
│  base_url ───┘     │  • LiteLLM        │     │  • Anthropic    │
│              │     │  • OpenRouter     │     │  • Qwen         │
│              │     │  • vLLM           │     │  • Mistral      │
│              │     │  • Ollama         │     │  • Local (GGUF) │
└──────────────┘     └───────────────────┘     └─────────────────┘
```

## Configuration

Two environment variables control the connection:

| Variable | Required | Example |
|----------|----------|---------|
| `OPENAI_API_KEY` | Yes | Your proxy's API key (or the upstream provider's key) |
| `OPENAI_BASE_URL` | No | `http://localhost:4000/v1` — omit to hit OpenAI directly |

Set both in `.env` or export them:

```bash
# Route through a local LiteLLM proxy
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=sk-litellm-key

# Route through OpenRouter
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-v1-...

# Direct OpenAI (default — no base_url needed)
export OPENAI_API_KEY=sk-...
```

## Supported Proxies

### LiteLLM

The most common choice for multi-provider routing. Supports 100+ LLM providers behind a single OpenAI-compatible API.

```bash
pip install litellm
litellm --model claude-sonnet-4-20250514 --port 4000
```

```env
OPENAI_BASE_URL=http://localhost:4000/v1
OPENAI_API_KEY=sk-1234
SYNTHESIS_MODEL=claude-sonnet-4-20250514
CLASSIFICATION_MODEL=claude-sonnet-4-20250514
```

### OpenRouter

Cloud-hosted proxy with access to Anthropic, Google, Meta, Qwen, and more.

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-your-key
SYNTHESIS_MODEL=anthropic/claude-sonnet-4-20250514
CLASSIFICATION_MODEL=qwen/qwen-2.5-72b-instruct
```

### Ollama (Local Models)

Run open-weight models locally with zero API costs.

```bash
ollama serve
ollama pull llama3.1:8b
```

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
SYNTHESIS_MODEL=llama3.1:8b
CLASSIFICATION_MODEL=llama3.1:8b
```

### vLLM

High-throughput serving for production deployments.

```bash
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8001
```

```env
OPENAI_BASE_URL=http://localhost:8001/v1
OPENAI_API_KEY=unused
SYNTHESIS_MODEL=Qwen/Qwen2.5-72B-Instruct
```

## Model Selection Tips

The RAG system uses two model slots — pick models that match each task:

| Slot | Purpose | Recommendation |
|------|---------|----------------|
| `CLASSIFICATION_MODEL` | Categorize queries as factual/exploratory/ambiguous | Fast + cheap. Qwen 2.5 7B or GPT-4o-mini work well |
| `SYNTHESIS_MODEL` | Generate cited answers from retrieved context | Quality matters here. Claude Sonnet, GPT-4o, or Qwen 72B |

## Verifying the Connection

After configuring your proxy, verify it works:

```bash
# Health check (doesn't need LLM)
curl http://localhost:8000/health

# Real query (exercises the full pipeline)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "k": 1}'
```

If the proxy is down or misconfigured, the API returns `confidence: 0.0` with an error message — it won't crash.

## How It Works Internally

The `_client_kwargs()` function in `src/llm/synthesize.py` builds the OpenAI client configuration:

```python
def _client_kwargs() -> dict:
    kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs
```

Both the sync and async clients share this configuration, so proxy routing applies to all LLM calls — classification, synthesis, and any future endpoints.
