"""
init_services.py — Run on container startup before the agent.

1. Waits for Postgres and Qdrant to be reachable
2. Creates the 'rag' database if it doesn't exist
3. Creates the 'meeting_recording' table + pgvector extension if missing
4. Creates the 'sureflow' Qdrant collection and ingests the markdown
   company profile if the collection doesn't exist or is empty

Usage:
    python init_services.py            # run checks + ingest
    python init_services.py --force    # drop & recreate everything
"""

import os
import sys
import time
import logging
import argparse
import re
from pathlib import Path
from uuid import uuid4

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env")

# ─── Config ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # dimension for text-embedding-3-small

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = "sureflow"

PG_HOST = os.getenv("PG_HOST", "pgvector")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DATABASE = os.getenv("PG_DATABASE", "rag")

MARKDOWN_PATH = os.getenv(
    "SUREFLOW_MD_PATH",
    str(Path(__file__).parent / "knowledge_base" / "data" / "SureFlow_Company_Profile.md"),
)

MAX_RETRIES = 30  # seconds to wait for each service
RETRY_INTERVAL = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s [init] %(message)s",
)
logger = logging.getLogger("init_services")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_embeddings(text: str) -> list[float]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def wait_for_postgres():
    """Block until Postgres accepts connections."""
    logger.info("Waiting for Postgres at %s:%s …", PG_HOST, PG_PORT)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT,
                user=PG_USER, password=PG_PASSWORD,
                dbname="postgres",
            )
            conn.close()
            logger.info("Postgres is ready.")
            return
        except psycopg2.OperationalError:
            if attempt % 5 == 0:
                logger.info("  … still waiting (%d/%d)", attempt, MAX_RETRIES)
            time.sleep(RETRY_INTERVAL)
    logger.error("Postgres did not become ready in time.")
    sys.exit(1)


def wait_for_qdrant():
    """Block until Qdrant responds to health check."""
    logger.info("Waiting for Qdrant at %s …", QDRANT_URL)
    client = QdrantClient(url=QDRANT_URL)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client.get_collections()
            logger.info("Qdrant is ready.")
            return
        except Exception:
            if attempt % 5 == 0:
                logger.info("  … still waiting (%d/%d)", attempt, MAX_RETRIES)
            time.sleep(RETRY_INTERVAL)
    logger.error("Qdrant did not become ready in time.")
    sys.exit(1)


# ─── Postgres setup ─────────────────────────────────────────────────────────

def ensure_database():
    """Create the database if it doesn't exist."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=PG_USER, password=PG_PASSWORD,
        dbname="postgres",
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,)
    )
    if cur.fetchone():
        logger.info("Database '%s' already exists.", PG_DATABASE)
    else:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(PG_DATABASE)))
        logger.info("Created database '%s'.", PG_DATABASE)

    cur.close()
    conn.close()


def ensure_tables():
    """Create pgvector extension and meeting_recording table if missing."""
    conn = psycopg2.connect(
        host=PG_HOST, port=PG_PORT,
        user=PG_USER, password=PG_PASSWORD,
        dbname=PG_DATABASE,
    )
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    logger.info("pgvector extension ready.")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS meeting_recording (
            id              SERIAL PRIMARY KEY,
            meeting_title   TEXT NOT NULL,
            context         TEXT NOT NULL,
            meeting_title_emb  vector(1536),
            embedding       vector(1536),
            created_at      TIMESTAMP DEFAULT NOW()
        );
    """)
    logger.info("Table 'meeting_recording' ready.")

    conn.commit()
    cur.close()
    conn.close()


# ─── Qdrant / Sureflow setup ────────────────────────────────────────────────

def parse_markdown_chunks(file_path: str) -> list[dict]:
    """Split a markdown file into section-based chunks (same logic as ingest script)."""
    path = Path(file_path)
    if not path.exists():
        logger.error("Markdown file not found: %s", file_path)
        sys.exit(1)

    raw = path.read_text(encoding="utf-8")
    source = path.name
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(raw))

    if not matches:
        return [{"section": path.stem, "heading_level": 0, "content": raw.strip(), "source": source}]

    raw_chunks: list[dict] = []

    preamble = raw[: matches[0].start()].strip()
    if preamble:
        raw_chunks.append({"section": "Introduction", "heading_level": 0, "content": preamble, "source": source})

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()
        body = re.sub(r"^---+$", "", body, flags=re.MULTILINE).strip()
        raw_chunks.append({"section": title, "heading_level": level, "content": body, "source": source})

    # Merge tiny chunks
    MIN_CONTENT_LENGTH = 30
    merged: list[dict] = []
    for chunk in raw_chunks:
        chunk_content = str(chunk.get("content", ""))
        if merged:
            prev_content = str(merged[-1].get("content", ""))
            if len(prev_content) < MIN_CONTENT_LENGTH:
                merged[-1]["content"] = prev_content + "\n\n" + f"## {chunk['section']}\n\n" + chunk_content
                merged[-1]["section"] += f" / {chunk['section']}"
                continue
        merged.append({
            "section": str(chunk["section"]),
            "heading_level": chunk["heading_level"],
            "content": chunk_content,
            "source": str(chunk["source"]),
        })

    return [c for c in merged if len(c.get("content", "").strip()) > 0]


def ensure_sureflow_collection(force: bool = False):
    """Create the sureflow collection and ingest the markdown if needed."""
    client = QdrantClient(url=QDRANT_URL)
    collection = QDRANT_COLLECTION

    # Check if collection exists and has data
    collection_exists = False
    point_count = 0
    try:
        info = client.get_collection(collection)
        collection_exists = True
        point_count = info.points_count
    except Exception:
        pass

    if collection_exists and point_count > 0 and not force:
        logger.info(
            "Qdrant collection '%s' exists with %d points — skipping ingestion.",
            collection, point_count,
        )
        return

    # Delete if forcing or if it exists but is empty
    if collection_exists:
        client.delete_collection(collection)
        logger.info("Deleted existing collection '%s'.", collection)

    # Create collection
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s'.", collection)

    # Parse markdown
    logger.info("Parsing markdown from %s …", MARKDOWN_PATH)
    chunks = parse_markdown_chunks(MARKDOWN_PATH)
    logger.info("Parsed %d chunks.", len(chunks))

    # Embed and upload
    points = []
    for chunk in chunks:
        text = f"Section: {chunk['section']} | Content: {chunk['content']}"
        vector = get_embeddings(text)
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "section": chunk["section"],
                    "heading_level": chunk["heading_level"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "text": text,
                    "type": "company_profile",
                },
            )
        )
        logger.info("  Embedded [H%d] %s", chunk["heading_level"], chunk["section"])

    client.upsert(collection_name=collection, points=points)
    logger.info("Uploaded %d points to '%s'.", len(points), collection)

    info = client.get_collection(collection)
    logger.info("Collection '%s' now has %d points.", collection, info.points_count)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Initialize Postgres + Qdrant on startup")
    parser.add_argument("--force", action="store_true", help="Drop and recreate everything")
    args = parser.parse_args()

    logger.info("=== Starting service initialization ===")

    # 1. Wait for services
    wait_for_postgres()
    wait_for_qdrant()

    # 2. Postgres: database + tables
    logger.info("--- Postgres setup ---")
    ensure_database()
    ensure_tables()

    # 3. Qdrant: sureflow collection + ingestion
    logger.info("--- Qdrant setup ---")
    ensure_sureflow_collection(force=args.force)

    logger.info("=== Initialization complete ===")


if __name__ == "__main__":
    main()