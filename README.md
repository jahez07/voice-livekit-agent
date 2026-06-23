<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# SureFlow Voice Agent

A voice AI assistant for **meeting management** and **company knowledge retrieval**, built on [LiveKit Agents for Python](https://github.com/livekit/agents).

The agent listens in on meetings, captures and stores transcripts for later semantic search, and answers questions about SureFlow from a curated knowledge base. It speaks multiple languages, switches its voice to match the language being spoken, and can be reached over the web or the phone.

## What it does

- **Active / Passive modes** — A spoken state machine driven by voice commands:
  - Say **"vision passive mode"** and the agent goes silent, widens its voice-activity detection so every speaker in the room is transcribed, and quietly captures the meeting transcript in the background.
  - Say **"vision active mode"** and it tightens VAD to filter room noise, comes back to life, and offers to store the meeting it just captured.
- **Meeting capture & storage** — Transcripts are embedded with OpenAI and stored in **Postgres + pgvector** (`meeting_recording` table) for semantic retrieval, via the `insert_meeting_recording` tool.
- **SureFlow knowledge base (RAG)** — Questions about SureFlow are answered from a company profile indexed in **Qdrant**, via the `sureflow_information_retrieval` tool.
- **Multilingual voice** — STT runs on `gpt-4o-mini-transcribe`; the agent detects the spoken language and swaps the [Cartesia](https://cartesia.ai/) `sonic-3` voice between English and French on the fly. ([ElevenLabs](https://elevenlabs.io/) TTS is wired up as an alternative.)
- **Turn detection & noise handling** — Uses the [LiveKit multilingual turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/) plus tunable [Silero VAD](https://docs.livekit.io/agents/logic/turns/vad/) profiles for active vs. passive listening.
- **Web + telephony entry points** — A Flask API mints LiveKit tokens for a web frontend and handles inbound Twilio calls; a SIP trunk (3CX) supports inbound phone calls.

## Architecture

```
┌──────────────┐     web token / Twilio webhook      ┌──────────────────┐
│  Frontend /  │ ─────────────────────────────────►  │  api_server.py   │
│  Phone (SIP) │                                     │  (Flask, :5001)  │
└──────────────┘                                     └──────────────────┘
        │                                                     │ dispatch "my-agent"
        ▼                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       LiveKit Server (rtc)                          │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────┐   embeddings    ┌───────────────────┐
│   agent.py       │ ──────────────► │pgvector (Postgres)│  meeting transcripts
│   (Assistant)    │                 └───────────────────┘
│  active/passive  │   embeddings    ┌───────────────────┐
│  STT/LLM/TTS     │ ──────────────► │  Qdrant           │  SureFlow knowledge base
└──────────────────┘                 └───────────────────┘
```

| File | Role |
|------|------|
| `src/agent.py` | The voice agent: `Assistant` agent, mode state machine, VAD profiles, language-switching TTS, and the two function tools. **Entrypoint** (`my-agent`). |
| `src/api_server.py` | Flask service: `/getToken` for the web frontend, `/incoming-call` Twilio webhook, `/health`. |
| `src/db_utils.py` | Standalone Postgres/pgvector helpers (embedding + insert) used outside the agent. |
| `init_services.py` | Startup bootstrap: waits for Postgres & Qdrant, creates the `rag` DB and `meeting_recording` table, then creates the `sureflow` Qdrant collection and ingests the knowledge base. |
| `setup_sip.py` | Creates the inbound SIP trunk and dispatch rule for telephony. |
| `knowledge_base/` | `SureFlow_Company_Profile.md` (the RAG source) and ingestion helper. |
| `db_src/` | Database/vector-store scratch scripts. |
| `docker-compose.yml` | Self-hosted stack: LiveKit server, Redis, pgvector, Qdrant, pgAdmin, SIP, the API server, and the agent. |

## Prerequisites

- Python 3.11+ and the [`uv`](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for the self-hosted stack)
- An OpenAI API key (used for STT, the LLM, and embeddings)
- A Cartesia API key (for TTS) — and optionally ElevenLabs
- LiveKit credentials (either [LiveKit Cloud](https://cloud.livekit.io/) or the bundled self-hosted server)

## Configuration

Create a `.env` file in the project root. The keys referenced across the code are:

```bash
# LiveKit
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# Models
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o            # the agent LLM
EMBEDDING_MODEL=text-embedding-3-small
ELEVENLABS_API_KEY=           # optional, only if you switch to ElevenLabs TTS

# Vector / relational stores
QDRANT_URL=http://localhost:6333
PG_HOST=localhost
PG_PORT=5435
PG_USER=postgres
PG_PASSWORD=postgres
PG_DATABASE=rag

# Telephony (setup_sip.py)
IP_ADDRESS=                   # host IP the SIP service binds to
```

> The Cartesia voice IDs are set in `src/agent.py` (`voice_by_lang`). Swap them for the voices you prefer.

## Running locally (without Docker)

Install dependencies:

```bash
uv sync
```

Download the local models (Silero VAD + turn detector) the agent needs on first run:

```bash
uv run python src/agent.py download-files
```

Bring up the data stores (or run your own Postgres/pgvector + Qdrant) and bootstrap them:

```bash
uv run python init_services.py          # create DB, table, and ingest the knowledge base
# uv run python init_services.py --force  # drop & recreate everything
```

Talk to the agent in your terminal:

```bash
uv run python src/agent.py console
```

Run it for a frontend or telephony:

```bash
uv run python src/agent.py dev     # development
uv run python src/agent.py start   # production
```

Run the token / telephony API:

```bash
uv run python src/api_server.py    # serves on :5001
```

## Running the full stack with Docker

`docker-compose.yml` brings up everything — LiveKit server, Redis, pgvector, Qdrant, pgAdmin, the SIP service, the Flask API, and the agent. The agent container runs `entrypoint.sh`, which executes `init_services.py` (DB + knowledge-base bootstrap) before starting the agent.

```bash
docker compose up --build
```

Exposed ports:

| Service | Port |
|---------|------|
| LiveKit server | `7880` (ws), `7881` (tcp), `7882-7892/udp` |
| API server | `5001` |
| pgvector (Postgres) | `5435` → `5432` |
| Qdrant | `6333` (REST), `6334` (gRPC) |
| pgAdmin | `5050` |
| Redis | `6379` |
| SIP | host network (`5060`, RTP `10000-20000`) |

## Telephony (SIP)

After the stack is up, create the inbound trunk and dispatch rule:

```bash
uv run python setup_sip.py
```

It registers a 3CX inbound trunk and a dispatch rule that routes calls into `call-*` rooms where `my-agent` is dispatched. The script is idempotent — it prints and skips if a trunk/rule already exists. Inbound Twilio calls are handled separately by the `/incoming-call` webhook in `api_server.py`.

## Tests

Behavioral evals built on the LiveKit [testing & evaluation framework](https://docs.livekit.io/agents/start/testing/):

```bash
uv run pytest
```

When changing agent behavior (instructions, tools, modes), follow the TDD guidance in [`AGENTS.md`](AGENTS.md): write or update a test first, then iterate until it passes.

## Code style

```bash
uv run ruff format
uv run ruff check
```

## Frontend

This agent works with any LiveKit frontend. The API server's `/getToken` endpoint is designed for the React starter:

- Web: [`livekit-examples/agent-starter-react`](https://github.com/livekit-examples/agent-starter-react)

See the [frontend guide](https://docs.livekit.io/frontends/) for other platforms.

## Working with this project & LiveKit docs

See [`AGENTS.md`](AGENTS.md) for project conventions and for using the LiveKit CLI (`lk docs`) and MCP server to browse the latest documentation.

## License

MIT — see [LICENSE](LICENSE).
