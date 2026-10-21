"""Indexes the USTP Student Handbook into Pinecone for the online assistant.

Run from the repository root, with PINECONE_API_KEY and PINECONE_INDEX_NAME
in backend/.env. Pinecone embeds the passages too, so nothing else is needed:

    python -m scripts.ingest_handbook --pdf path/to/ustp_handbook.pdf --replace
    python -m scripts.ingest_handbook --url https://.../ustp_handbook.pdf --replace
    python -m scripts.ingest_handbook --pdf handbook.pdf --dry-run

`--replace` clears the target namespace first, so a new edition never leaves
passages from the old one behind. Only that namespace is touched. The
namespace defaults to HANDBOOK_NAMESPACE (`ustp-handbook-2023`), which is the
one the chat endpoint searches.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

from app.clients.pinecone_index import PineconeIndexClient
from app.config import get_settings
from app.services.handbook_rag import build_chunks, index_handbook
from app.services.pdf_text import extract_pdf_text

MAX_HANDBOOK_PAGES = 1000


async def _read_pdf(path: str | None, url: str | None) -> bytes:
    if path:
        return Path(path).read_bytes()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


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
        if args.replace:
            print(f"Clearing namespace {namespace!r}...")
            await index.delete_namespace(namespace)
        print(f"Embedding with {settings.HANDBOOK_EMBED_MODEL} and writing to {namespace!r}...")
        written = await index_handbook(
            chunks,
            index=index,
            settings=settings,
            namespace=namespace,
            on_progress=print,
        )
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
    parser.add_argument("--replace", action="store_true", help="Clear the namespace first")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk only")
    sys.exit(asyncio.run(ingest(parser.parse_args())))


if __name__ == "__main__":
    main()
