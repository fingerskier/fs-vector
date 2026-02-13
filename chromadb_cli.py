#!/usr/bin/env python3
"""CLI tool to index files into ChromaDB and query them by semantic similarity."""

import argparse
import hashlib
import os
import sys

import chromadb
from chromadb.config import Settings


DEFAULT_DB_DIR = ".chromadb"
COLLECTION_NAME = "files"

# Skip binary / non-text extensions
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".wasm",
    ".bin", ".dat", ".db", ".sqlite",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "node_modules", ".tox", ".venv", "venv", "env",
    DEFAULT_DB_DIR,
}

MAX_FILE_SIZE = 1_000_000  # 1 MB – skip very large files


def get_db_path(directory: str) -> str:
    """Return the path to the ChromaDB persistence directory."""
    return os.path.join(directory, DEFAULT_DB_DIR)


def get_client(directory: str) -> chromadb.ClientAPI:
    """Create (or open) a persistent ChromaDB client rooted in *directory*."""
    db_path = get_db_path(directory)
    return chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))


def get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """Get or create the file-content collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _file_id(filepath: str) -> str:
    """Deterministic document id derived from the file path."""
    return hashlib.sha256(filepath.encode()).hexdigest()


def _should_skip_file(filepath: str) -> bool:
    _, ext = os.path.splitext(filepath)
    if ext.lower() in BINARY_EXTENSIONS:
        return True
    try:
        size = os.path.getsize(filepath)
    except OSError:
        return True
    return size == 0 or size > MAX_FILE_SIZE


def _read_text(filepath: str) -> str | None:
    """Try to read a file as UTF-8 text.  Return None on failure."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except (UnicodeDecodeError, OSError):
        return None


def walk_and_index(directory: str, verbose: bool = False) -> None:
    """Recursively walk *directory* and upsert every text file into ChromaDB."""
    directory = os.path.abspath(directory)
    client = get_client(directory)
    collection = get_collection(client)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for root, dirs, files in os.walk(directory):
        # Prune directories we never want to enter
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            filepath = os.path.join(root, fname)

            if _should_skip_file(filepath):
                if verbose:
                    print(f"  skip  {filepath}")
                continue

            content = _read_text(filepath)
            if content is None:
                if verbose:
                    print(f"  skip  {filepath} (not utf-8)")
                continue

            relpath = os.path.relpath(filepath, directory)
            doc_id = _file_id(relpath)

            ids.append(doc_id)
            documents.append(content)
            metadatas.append({"path": relpath})

            if verbose:
                print(f"  index {relpath}")

            # Batch upsert every 50 files to keep memory bounded
            if len(ids) >= 50:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
                ids, documents, metadatas = [], [], []

    # Flush remaining
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    total = collection.count()
    print(f"Collection '{COLLECTION_NAME}' now contains {total} documents.")


def query_collection(directory: str, query_text: str, n_results: int = 5) -> None:
    """Query the collection and print the most relevant file paths."""
    directory = os.path.abspath(directory)
    client = get_client(directory)
    collection = get_collection(client)

    if collection.count() == 0:
        print("Collection is empty. Run --index first.", file=sys.stderr)
        sys.exit(1)

    results = collection.query(query_texts=[query_text], n_results=n_results)

    paths = results["metadatas"][0]  # type: ignore[index]
    distances = results["distances"][0]  # type: ignore[index]

    for meta, dist in zip(paths, distances):
        score = 1 - dist  # cosine similarity
        print(f"{score:+.4f}  {meta['path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index files into ChromaDB and query by semantic similarity.",
    )
    parser.add_argument(
        "directory",
        help="Target sub-directory to index / query.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index",
        action="store_true",
        help="Walk the directory and add/update every text file in the collection.",
    )
    group.add_argument(
        "--query",
        metavar="TEXT",
        help="Semantic query — returns the most related file paths.",
    )
    parser.add_argument(
        "-n", "--num-results",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print each file as it is processed.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        parser.error(f"Not a directory: {directory}")

    if args.index:
        print(f"Indexing {directory} ...")
        walk_and_index(directory, verbose=args.verbose)
    elif args.query:
        query_collection(directory, args.query, n_results=args.num_results)


if __name__ == "__main__":
    main()
