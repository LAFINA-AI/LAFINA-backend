"""Indexes the USTP Student Handbook into Pinecone for the online assistant.

Run from the repository root, with PINECONE_API_KEY and PINECONE_INDEX_NAME
in backend/.env:

    python -m scripts.ingest_handbook --pdf path/to/ustp_handbook.pdf --replace
    python -m scripts.ingest_handbook --url https://.../ustp_handbook.pdf --replace
    python -m scripts.ingest_handbook --pdf handbook.pdf --dry-run

When `--replace` is set, new embeddings are computed and validated BEFORE the
live namespace is cleared, preventing a mid-run failure from leaving RAG empty.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

# Ensure repository root is on sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx

from app.clients.pinecone_index import EMBED_BATCH, INPUT_PASSAGE, PineconeIndexClient
from app.config import get_settings
from app.services.handbook_rag import build_chunks
from app.services.pdf_text import extract_pdf_text

MAX_HANDBOOK_PAGES = 1000
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MiB ceiling


async def _read_pdf(path: str | None, url: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    assert url is not None
    parts = urlsplit(url)
    if parts.scheme not in ("https", "http"):
        raise ValueError("Handbook --url must be an http(s) URL.")
    buf = bytearray()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                buf.extend(chunk)
                if len(buf) > MAX_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Downloaded PDF exceeds {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MiB limit."
                    )
    return bytes(buf)


async def ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    namespace = args.namespace or settings.HANDBOOK_NAMESPACE

    data = await _read_pdf(args.pdf, args.url)
    document = extract_pdf_text(data, max_pages=MAX_HANDBOOK_PAGES, allow_ocr=False)
    chunks = build_chunks(document.pages)
    print(f"Read {document.pages_read} pages into {len(chunks)} passages.")
    if not chunks:
        print("No passages: is this the handbook? Every page should end with its edition footer.")
        return 1
    if args.dry_run:
        for chunk in chunks[:3]:
            print(f"- {chunk.id} p.{chunk.page}-{chunk.page_end} [{chunk.section}] {chunk.text[:120]}...")
        print("Dry run: nothing was embedded or written.")
        return 0

    if not settings.is_handbook_configured():
        print(f"Handbook RAG is not configured: {settings.get_pinecone_invalid_reason()}")
        return 1

    index = PineconeIndexClient(settings)
    try:
        # Embed ALL chunks first (and verify index host/dimension) BEFORE clearing
        # the live namespace, so a rate limit or embedding error never wipes RAG.
        await index.host()
        print(f"Embedding {len(chunks)} passages with {settings.HANDBOOK_EMBED_MODEL}...")
        vectors: list[dict] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            embedded = await index.embed(
                [chunk.embedding_text for chunk in batch], input_type=INPUT_PASSAGE
            )
            for chunk, values in zip(batch, embedded):
                vectors.append(
                    {
                        "id": chunk.id,
                        "values": values,
                        "metadata": chunk.metadata(settings.HANDBOOK_TITLE),
                    }
                )
            print(f"  embedded {min(start + EMBED_BATCH, len(chunks))}/{len(chunks)}")

        if args.replace:
            print(f"Clearing namespace {namespace!r} prior to upsert...")
            await index.delete_namespace(namespace)

        print(f"Upserting {len(vectors)} vectors to {namespace!r}...")
        written = await index.upsert(vectors, namespace=namespace)
        print(f"Wrote {written} passages to {settings.PINECONE_INDEX_NAME}/{namespace}.")
    finally:
        await index.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf", help="Path to the handbook PDF")
    source.add_argument("--url", help="URL to download the handbook PDF from")
    parser.add_argument("--namespace", help="Pinecone namespace (default: HANDBOOK_NAMESPACE)")
    parser.add_argument("--replace", action="store_true", help="Clear the namespace after embedding succeeds")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only")
    sys.exit(asyncio.run(ingest(parser.parse_args())))


if __name__ == "__main__":
    main()
