# Routellect

Open-source LLM model selector and routing proxy. Routellect sits between your application and LLM providers, making intelligent model routing decisions and learning from outcomes.

## What It Does

Routellect intercepts LLM API calls by posing as an OpenAI-compatible proxy. Your application sends requests to routellect; routellect picks the best model, forwards to the real provider, grades the result, and learns over time.

```
Your app  →  routellect proxy (:11411/v1)  →  real LLM provider
                    │
                    ├─ selects model (graduated demotion)
                    ├─ forwards via litellm
                    ├─ grades conversation quality
                    └─ records outcomes to local DB
```

## Quick Start

### 1. Install

```bash
# From PyPI (when published):
pip install routellect[proxy]

# From source:
git clone https://github.com/soverton/routellect.git
cd routellect
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[proxy]'
```

### 2. Run

```bash
python -m routellect.proxy
```

On first run, an interactive setup wizard prompts for your provider API keys:

```
  routellect proxy — first-time setup
  ────────────────────────────────────

  Paste your API keys below (Enter to skip a provider):

  OpenAI API key: sk-...         ✔ verified
  Anthropic API key: sk-ant-...  ✔ verified
  Google API key: ↵              (skipped)
  Groq API key: ↵                (skipped)

  Keys encrypted and saved to ~/.routellect/credentials

  ✔ Proxy running on http://127.0.0.1:11411

  Add this to your app's environment:
    OPENAI_BASE_URL=http://127.0.0.1:11411/v1
```

### 3. Point Your App at It

```python
import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:11411/v1",
    api_key="unused",  # or your ROUTELLECT_PROXY_TOKEN if auth is enabled
)

response = client.chat.completions.create(
    model="gpt-4o",  # hint only — routellect decides the actual model
    messages=[{"role": "user", "content": "Hello"}],
)

# Check which model actually served the request:
print(response.headers.get("x-routellect-model"))
```

That's it. Your existing code is unchanged — just one `base_url`.

## How Model Selection Works

### Graduated Demotion

Routellect starts with the best available models and cautiously steps down to cheaper tiers only when quality data confirms it's safe. It never wastes money on models that produce bad results.

```
Tier 1 (flagship):  claude-opus-4-6, claude-sonnet-4-6
Tier 2 (premium):   claude-opus-4-5, claude-sonnet-4-5
Tier 3 (standard):  claude-opus-4-1, claude-sonnet-4, claude-opus-4, gpt-4o
Tier 4 (efficient):  claude-haiku-4-5, gemini-2.5-flash, o3-mini
Tier 5 (budget):    gpt-4o-mini, llama-3.1-8b
```

**Rules:**
- Starts at tier 1 and serves 100% from that tier while building confidence
- After 10 graded passes, begins trialing the next tier at 15% of traffic
- If the trial tier's pass rate stays above 70%, it becomes the new default
- If pass rate drops below 70%, demotion stops permanently — locked at the last good tier
- Only counts quality grades from the grader, not raw HTTP success

### Session Grading

The proxy buffers conversation exchanges per session. When a session goes idle or hits 10 exchanges, the batch is sent to a cheap grading model (Haiku) that evaluates each assistant response:

- **pass**: user continued productively, said thanks, moved on
- **fail**: user corrected, retried, expressed frustration
- **mixed**: user partially accepted but corrected part of the response

Grades are stored in a local SQLite database at `~/.routellect/grades.db` and fed back to the model selector.

## CLI Reference

```bash
# Start the proxy (interactive setup on first run)
python -m routellect.proxy

# Re-run credential setup
python -m routellect.proxy --setup

# Custom host/port
python -m routellect.proxy --host 0.0.0.0 --port 9100

# View recent grades and model performance
python -m routellect.proxy --grades

# Export all data as a ZIP (for sharing/analysis)
python -m routellect.proxy --export
python -m routellect.proxy --export my-data.zip
```

The `routellect-proxy` command is also available as a script entry point after installation.

## Proxy Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/v1/chat/completions` | POST | Core proxy. Streaming and non-streaming. |
| `/v1/models` | GET | List available models in OpenAI format. |
| `/health` | GET | Proxy status, discovered providers, model count. |

## Configuration

### Provider API Keys

Stored encrypted in `~/.routellect/credentials` (Fernet encryption, file mode 600). Set during first-run setup or with `--setup`. Keys are verified against each provider's API before saving.

Supported providers:
- **OpenAI** (`OPENAI_API_KEY`)
- **Anthropic** (`ANTHROPIC_API_KEY`)
- **Google** (`GOOGLE_API_KEY` / `GEMINI_API_KEY`)
- **Groq** (`GROQ_API_KEY`)

### Proxy Settings (Environment Variables)

| Variable | Purpose | Default |
|----------|---------|---------|
| `ROUTELLECT_PROXY_HOST` | Bind address | `127.0.0.1` |
| `ROUTELLECT_PROXY_PORT` | Bind port | `11411` |
| `ROUTELLECT_PROXY_TOKEN` | Bearer token for proxy auth | None (no auth) |
| `ROUTELLECT_LOG_BODIES` | Log request/response bodies | `false` |

### Security

- **Localhost only** by default. Binding to `0.0.0.0` logs a warning.
- **Credentials encrypted at rest** using Fernet with a machine-derived key.
- **Optional proxy auth** via `ROUTELLECT_PROXY_TOKEN`.
- **No body logging** by default — message content is never logged.
- **Key scrubbing** — error responses are stripped of API key patterns.
- **Input masking** — keys are hidden during terminal setup.

## Data Export

```bash
python -m routellect.proxy --export
```

Produces a ZIP containing:
- `sessions.csv` — session metadata (start/end, message count, grading cost)
- `grades.csv` — individual grades per assistant response (model, grade, confidence, reason)
- `routing_log.csv` — every routed request (model, latency, tokens, exploration flag)
- `model_summary.csv` — aggregate pass/fail/mixed rates per model

## Programmatic API

For embedding the proxy in your own application:

```python
from routellect.proxy import create_app, serve

# Get a Starlette ASGI app
app = create_app()

# Or start blocking server
serve(port=9100)
```

## Canonical Interface (Library Use)

Routellect can also be used as a standalone library without the proxy:

```python
from routellect import ModelCapability, ModelSelectorProtocol, RoutingDecision, RoutingOutcome

# 1. Tell routellect what models are available
selector.set_model_universe(models: list[ModelCapability])

# 2. Ask for a routing decision
decision: RoutingDecision = selector.select_model(task_fingerprint, constraints)

# 3. Report what happened
selector.record_outcome(decision, outcome_metrics)
```

## Scope

**Routellect owns:**
- Model selection contracts and routing decisions
- The OpenAI-compatible proxy server
- Session grading and quality feedback
- Local execution telemetry and cost helpers
- Encrypted credential management

**Routellect does not own:**
- Model discovery beyond its static catalog
- Workflow control-plane concerns (retries, promotion, branching)
- Business-specific orchestration

## Harness Integration

Routellect ships an optional harness plugin module at `routellect.harness_plugins` for use with `accruvia-harness`:

```bash
export ACCRUVIA_PROJECT_ADAPTER_MODULES=routellect.harness_plugins
export ACCRUVIA_COGNITION_MODULES=routellect.harness_plugins
export ROUTELLECT_REPO_ROOT=/path/to/routellect
```

The plugin provides disposable per-run git worktrees so blocked or failed runs never dirty the source repo.

## Development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[proxy,dev]'
python -m pytest tests -q
```

## Migration Debt

`runner.py` and `issues.py` contain legacy issue-runner execution logic. That code is migration debt — the durable product value is the routing/selection behavior and the telemetry generated from real usage.
