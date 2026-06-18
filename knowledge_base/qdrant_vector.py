"""
knowledge_base/ingest_company_profile.py - Ingest a Markdown company profile into Qdrant.

Reads a Markdown file (e.g. SureFlow_Company_Profile.md) and splits it
into semantic chunks based on headings. Each chunk becomes one document
in the 'sureflow' Qdrant collection with this structure:

    text:       "Section: Products & Devices | Content: SureFlow offers a range..."
    metadata:   {section: "Products & Devices", heading_level: 2, source: "SureFlow_Company_Profile.md"}

This means when an agent searches "water leak detection sensor",
Qdrant returns the most relevant section(s) of the company profile.

Run:
    python -m knowledge_base.ingest_company_profile --file knowledge_base/data/SureFlow_Company_Profile.md
    python -m knowledge_base.ingest_company_profile --file knowledge_base/data/SureFlow_Company_Profile.md --recreate
"""

import argparse
import asyncio
import logging
import re
import sys
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def embed_batch(
        texts: list[str],
        prefix: str = "search_document",
) -> list[list[float]]:
    
    prefixed = [f"{prefix}:{t}" for t in texts]

    response = await client.embeddings.create(
        input=prefixed,
        model="text-embedding-3-small"
    )

    return [item.embedding for item in response.data]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)
logger = logging.getLogger("ingest_company_profile")

# ── Hard-coded collection name ──────────────────────────────────────────────
COLLECTION_NAME = "sureflow"


# ─── Markdown parsing ───────────────────────────────────────────────────────

def load_markdown_chunks(file_path: str) -> list[dict]:
    """
    Parse a Markdown file into section-based chunks.

    Strategy:
        1. Split on heading lines (# / ## / ### etc.).
        2. Each heading + the body text beneath it becomes one chunk.
        3. Very small chunks (< 30 chars of body) are merged into the
           next chunk so we don't create near-empty vectors.
        4. Metadata records the section title, heading level, and source
           filename for filtering later.

    Returns a list of dicts:
        [
            {
                "section":       "About SureFlow",
                "heading_level": 2,
                "content":       "SureFlow is a technology company ...",
                "source":        "SureFlow_Company_Profile.md",
            },
            ...
        ]
    """
    path = Path(file_path)
    if path.suffix.lower() not in (".md", ".markdown", ".txt"):
        raise ValueError(
            f"Unsupported file type: {path.suffix}. Use .md, .markdown, or .txt"
        )

    raw = path.read_text(encoding="utf-8")
    source = path.name

    # ── Split on markdown headings ──
    # Regex captures: (heading_level_hashes, heading_text, body_until_next_heading)
    heading_pattern = re.compile(
        r"^(#{1,6})\s+(.+?)$",  # match heading lines
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(raw))

    if not matches:
        # No headings found — treat whole file as one chunk
        logger.warning("No headings found in %s; ingesting as a single chunk", file_path)
        return [
            {
                "section": path.stem,
                "heading_level": 0,
                "content": raw.strip(),
                "source": source,
            }
        ]

    raw_chunks: list[dict] = []

    # Text before the first heading (if any)
    preamble = raw[: matches[0].start()].strip()
    if preamble:
        raw_chunks.append(
            {
                "section": "Introduction",
                "heading_level": 0,
                "content": preamble,
                "source": source,
            }
        )

    for i, match in enumerate(matches):
        level = len(match.group(1))          # number of '#' chars
        title = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[body_start:body_end].strip()

        # Strip sub-headings' leading '#' symbols that may bleed in
        # (we keep the text itself; just clean decoration)
        body = re.sub(r"^---+$", "", body, flags=re.MULTILINE).strip()

        raw_chunks.append(
            {
                "section": title,
                "heading_level": level,
                "content": body,
                "source": source,
            }
        )

    # ── Merge tiny chunks into the next sibling ──
    MIN_CONTENT_LENGTH = 30
    merged: list[dict] = []

    for chunk in raw_chunks:
        if (
            merged
            and len(merged[-1]["content"]) < MIN_CONTENT_LENGTH
        ):
            # Append current chunk's content to previous small chunk
            merged[-1]["content"] += "\n\n" + f"## {chunk['section']}\n\n" + chunk["content"]
            merged[-1]["section"] += f" / {chunk['section']}"
        else:
            merged.append(dict(chunk))  # shallow copy

    # Drop any remaining chunks that are effectively empty
    merged = [c for c in merged if len(c["content"].strip()) > 0]

    return merged


def build_document_text(chunk: dict) -> str:
    """
    Build the text that gets embedded.

    Format: "Section: <title> | Content: <body>"

    Prefixing with the section title biases the embedding towards the
    topic of each chunk, improving retrieval when the user query aligns
    with a section's theme (e.g. "R&D capabilities" → the R&D section).
    """
    return f"Section: {chunk['section']} | Content: {chunk['content']}"


# ─── Ingestion pipeline ─────────────────────────────────────────────────────

async def ingest(file_path: str, recreate: bool = False):
    """
    Main ingestion pipeline:
        1. Load and chunk the Markdown file
        2. Build document texts
        3. Embed all documents via Ollama
        4. Create / recreate Qdrant collection
        5. Upload vectors with metadata
    """
    collection = COLLECTION_NAME

    # ── Step 1: Load & chunk ──
    logger.info("Loading markdown from %s", file_path)
    chunks = load_markdown_chunks(file_path)
    logger.info("Parsed %d chunks from the markdown file", len(chunks))

    for chunk in chunks:
        preview = chunk["content"][:80].replace("\n", " ")
        logger.info(
            "   [H%d] %s  (%d chars)  %.80s…",
            chunk["heading_level"],
            chunk["section"],
            len(chunk["content"]),
            preview,
        )

    # ── Step 2: Build document texts ──
    texts = [build_document_text(c) for c in chunks]

    # ── Step 3: Embed ──
    logger.info(
        "Embedding %d chunks ", len(texts)
    )
    vectors = await embed_batch(texts, prefix="search_document")
    logger.info("Embedding complete (dimension=%d)", len(vectors[0]))

    # ── Step 4: Create Qdrant collection ──
    client = QdrantClient(url='http://localhost:6333')

    if recreate:
        try:
            client.delete_collection(collection)
            logger.info("Deleted existing collection '%s'", collection)
        except Exception:
            pass

    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created collection '%s'", collection)
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Collection '%s' already exists", collection)
        else:
            raise

    # ── Step 5: Upload points ──
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "section": chunk["section"],
                    "heading_level": chunk["heading_level"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "text": texts[i],
                    "type": "company_profile",
                },
            )
        )

    client.upsert(collection_name=collection, points=points)
    logger.info("Uploaded %d points to '%s'", len(points), collection)

    # ── Verify ──
    info = client.get_collection(collection)
    logger.info("Collection '%s' now has %d points", collection, info.points_count)

    logger.info("Company profile ingestion complete!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ingest a Markdown company profile into Qdrant"
    )
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="Path to the Markdown file (.md)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (fresh start)",
    )
    args = parser.parse_args()

    asyncio.run(ingest(args.file, args.recreate))


if __name__ == "__main__":
    main()