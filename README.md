<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Restaurant Voice Agent — LiveKit Multi-Agent System

A voice AI system for a restaurant front desk, built with [LiveKit Agents for Python](https://github.com/livekit/agents). A caller talks to a receptionist that hands them off to specialist agents for **reservations**, **takeaway orders**, and **checkout** — each with its own scoped instructions, tools, and voice — while shared customer data (name, phone, order, payment) follows them across the conversation.

The project can run fully self-hosted via Docker Compose (LiveKit server, Redis, Postgres/pgvector, Qdrant, SIP, and a token/webhook API) and supports both web frontends and inbound telephony (SIP / Twilio).

## How it works

The conversation is modeled as a **handoff workflow** rather than one long prompt. This keeps each LLM request small and focused, which matters for voice latency.

```
                       ┌─────────────┐
              ┌───────▶│   Greeter   │◀────────┐
              │        └──────┬──────┘         │
              │       reserve │ takeaway       │  to_greeter
              │               ▼                │  (any agent can
       ┌──────┴──────┐ ┌─────────────┐         │   return here)
       │ Reservation │ │  Takeaway   │─────────┘
       └─────────────┘ └──────┬──────┘
                              │ to_checkout
                              ▼
                       ┌─────────────┐
                       │  Checkout   │
                       └─────────────┘
```

- **Greeter** — greets the caller, reads the menu, and routes to Reservation or Takeaway.
- **Reservation** — collects reservation time, name, and phone, then confirms.
- **Takeaway** — takes and updates the order, then sends the caller to checkout.
- **Checkout** — confirms the expense and collects payment details, then completes the order.

All agents extend a `BaseAgent` that, on each handoff:

- copies the last few turns from the previous agent (`truncate(max_items=6)`) so context growth stays bounded, and
- injects the current `UserData` (name, phone, reservation time, order, payment, totals) as a system message so the new agent has everything it needs.

Each specialist registers its own `@function_tool`s and a distinct [Cartesia](https://docs.livekit.io/agents/models/) voice, so the caller hears a different voice per role.

### Voice pipeline

| Stage | Provider / model |
|-------|------------------|
| STT   | OpenAI `gpt-4o-mini-transcribe` |
| LLM   | OpenAI `gpt-4o-mini` |
| TTS   | Cartesia `sonic-3` (per-agent voice) |
| VAD   | [Silero](https://docs.livekit.io/agents/logic/turns/vad/) |
| Turn detection | [LiveKit multilingual turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/) |

## Project layout

```
src/
  agent.py          # Multi-agent workflow: Greeter → Reservation / Takeaway → Checkout (entrypoint)
  api_server.py     # Flask: /getToken (frontend), /incoming-call (Twilio webhook), /health
  db_utils.py       # pgvector helpers (embeddings + meeting_recording inserts)
  vectordb.py       # pgvector schema bootstrap / smoke test
init_services.py    # Startup: wait for Postgres + Qdrant, create DB/tables, ingest knowledge base
setup_sip.py        # Create LiveKit SIP inbound trunk + dispatch rule
knowledge_base/     # SureFlow company profile (Markdown) + Qdrant ingest script
db_src/             # Standalone DB experiments
tests/              # Agent behavior tests (pytest)
docker-compose.yml  # Full self-hosted stack
Dockerfile.agent    # Agent image      Dockerfile.api  # API server image
```

> **Note:** `agent.py` must remain the entrypoint — the `Dockerfile.agent` deploys it directly.

## Prerequisites

- Python 3.11+ and the [`uv`](https://docs.astral.sh/uv/) package manager
- An `OPENAI_API_KEY` (used for STT/LLM/TTS routing and for RAG embeddings)
- For self-hosting: Docker + Docker Compose
- LiveKit credentials — either [LiveKit Cloud](https://cloud.livekit.io/) or the self-hosted server below

## Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

The agent and supporting services read from `.env`. Common variables:

| Variable | Purpose |
|----------|---------|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Connect to LiveKit (Cloud or self-hosted) |
| `OPENAI_API_KEY` | STT / LLM / TTS routing and embeddings |
| `OPENAI_MODEL` | Greeter LLM model override |
| `ELEVENLABS_API_KEY` | Optional ElevenLabs TTS |
| `QDRANT_URL`, `PG_HOST`, `PG_PORT` | RAG datastores (defaults target the Compose services) |

## Run locally (LiveKit Cloud or remote server)

Install dependencies and download the local models (Silero VAD, turn detector):

```bash
uv sync
uv run python src/agent.py download-files
```

Talk to the agent directly in your terminal:

```bash
uv run python src/agent.py console
```

Run it for a frontend or telephony:

```bash
uv run python src/agent.py dev     # development
uv run python src/agent.py start   # production
```

## Run the full stack with Docker Compose

The Compose file brings up a complete self-hosted environment — no LiveKit Cloud required:

| Service | Port(s) | Role |
|---------|---------|------|
| `livekit-server` | 7880, 7881, 7882–7892/udp | Self-hosted LiveKit server |
| `agent` | — | The voice agent (runs `init_services.py` then `agent.py start`) |
| `api-server` | 5001 | Token endpoint + Twilio webhook |
| `pgvector` | 5435→5432 | Postgres + pgvector for RAG |
| `qdrant` | 6333 / 6334 | Vector DB for the knowledge base |
| `redis` | 6379 | LiveKit / SIP coordination |
| `pgadmin` | 5050 | Postgres admin UI |
| `sip` | host network | SIP bridge for telephony |

```bash
docker compose up --build
```

On startup the `agent` container runs `init_services.py`, which waits for Postgres and Qdrant, creates the `rag` database and `meeting_recording` table, and ingests the company-profile knowledge base into the `sureflow` Qdrant collection. To rebuild everything from scratch:

```bash
docker compose run --rm agent uv run python init_services.py --force
```

## Telephony (inbound calls)

The agent can answer phone calls two ways:

- **SIP trunk** — register a SIP inbound trunk and dispatch rule with LiveKit:

  ```bash
  uv run python setup_sip.py
  ```

  This creates a trunk and an `Agent-Dispatch` rule that routes incoming calls into `call-*` rooms. Adjust the trunk numbers/addresses in `setup_sip.py` for your provider (the example targets a 3CX PBX).

- **Twilio** — point a Twilio number's webhook at `POST /incoming-call` on the API server. It returns TwiML that streams the call into a LiveKit room with the agent dispatched.

## Frontend

Use any LiveKit frontend with the token endpoint at `GET /getToken?room=<room>&identity=<id>`:

| Platform | Repo |
|----------|------|
| **Web (React/Next.js)** | [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react) |
| **iOS / macOS** | [`livekit-examples/agent-starter-swift`](https://github.com/livekit-examples/agent-starter-swift) |
| **Flutter** | [`livekit-examples/agent-starter-flutter`](https://github.com/livekit-examples/agent-starter-flutter) |
| **React Native** | [`livekit-examples/voice-assistant-react-native`](https://github.com/livekit-examples/voice-assistant-react-native) |
| **Android** | [`livekit-examples/agent-starter-android`](https://github.com/livekit-examples/agent-starter-android) |
| **Web Embed** | [`livekit-examples/agent-starter-embed`](https://github.com/livekit-examples/agent-starter-embed) |

See the [frontend guide](https://docs.livekit.io/frontends/) for details.

## Tests

```bash
uv run pytest
```

When changing agent behavior — instructions, tool descriptions, handoffs — write or update tests first. See `tests/test_agent.py` and the [testing & evaluation guide](https://docs.livekit.io/agents/start/testing/).

## Formatting & linting

```bash
uv run ruff format
uv run ruff check
```

## Documentation

This is a fast-moving SDK; prefer the latest docs. With the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/) (`lk` 2.15.0+) installed:

```bash
lk docs search "handoffs and workflows"
lk docs get-page /agents/build/workflows
```

LiveKit also offers a [docs MCP server](https://docs.livekit.io/reference/developer-tools/docs-mcp/). See `AGENTS.md` for the full coding-agent guide for this repo.

## Deploying to production

The repo includes production-ready Dockerfiles. To deploy the agent to LiveKit Cloud or your own infrastructure, see the [deploying to production](https://docs.livekit.io/deploy/agents/) guide.

## License

Licensed under the MIT License — see [LICENSE](LICENSE).
