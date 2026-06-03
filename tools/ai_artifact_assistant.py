#!/usr/bin/env python3
"""Local AI assistant for project artifacts.

The tool is intentionally dependency-free. It can use a local Ollama server for
embeddings and text generation, and falls back to lexical search when Ollama is
not available.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib import error, request


DEFAULT_DB = ".ai_artifacts/index.json"
DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EXTENSIONS = {
    ".bash",
    ".bat",
    ".conf",
    ".csv",
    ".env",
    ".html",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".jsp",
    ".log",
    ".md",
    ".properties",
    ".ps1",
    ".py",
    ".sql",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
SKIP_DIRS = {
    ".ai_artifacts",
    ".git",
    ".gradle",
    ".idea",
    ".mvn",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
}
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_./:-]+")


class OllamaError(RuntimeError):
    """Raised when the local Ollama service is not reachable or fails."""


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _json_request(self, method: str, path: str, payload: Optional[dict] = None) -> dict:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
        except (error.URLError, TimeoutError) as exc:
            raise OllamaError(f"Ollama is not available at {self.base_url}: {exc}") from exc

        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned invalid JSON for {path}") from exc

    def tags(self) -> dict:
        return self._json_request("GET", "/api/tags")

    def embed(self, model: str, text: str) -> List[float]:
        payload = {"model": model, "prompt": text}
        result = self._json_request("POST", "/api/embeddings", payload)
        embedding = result.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaError(f"Model {model!r} did not return an embedding")
        return [float(value) for value in embedding]

    def generate(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 8192,
            },
        }
        result = self._json_request("POST", "/api/generate", payload)
        response = result.get("response")
        if not isinstance(response, str):
            raise OllamaError(f"Model {model!r} did not return text")
        return response.strip()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_extensions(values: Optional[Sequence[str]]) -> set:
    if not values:
        return set(DEFAULT_EXTENSIONS)
    normalized = set()
    for value in values:
        for item in value.split(","):
            item = item.strip().lower()
            if not item:
                continue
            normalized.add(item if item.startswith(".") else f".{item}")
    return normalized


def looks_binary(path: Path, sample_size: int = 2048) -> bool:
    try:
        sample = path.read_bytes()[:sample_size]
    except OSError:
        return True
    return b"\x00" in sample


def read_text_file(path: Path, max_bytes: int) -> Optional[str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"skip {path}: {exc}", file=sys.stderr)
        return None

    if len(data) > max_bytes:
        print(f"skip {path}: file is larger than {max_bytes} bytes", file=sys.stderr)
        return None
    if b"\x00" in data[:2048]:
        return None

    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def iter_source_files(paths: Sequence[Path], extensions: set) -> Iterator[Path]:
    seen = set()
    for root in paths:
        if not root.exists():
            print(f"skip {root}: path does not exist", file=sys.stderr)
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = (
                item
                for item in root.rglob("*")
                if item.is_file() and not any(part in SKIP_DIRS for part in item.parts)
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            if candidate.suffix.lower() not in extensions:
                continue
            if looks_binary(candidate):
                continue
            seen.add(resolved)
            yield candidate


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_text(text: str, max_chars: int = 3500, overlap: int = 350) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary <= start + max_chars // 2:
                boundary = text.rfind("\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0
    counts: Dict[str, int] = {}
    for token in text_tokens:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    for token in query_tokens:
        if token in counts:
            score += 1.0 + math.log(1 + counts[token])
    return score / math.sqrt(len(text_tokens))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def load_index(db_path: Path) -> dict:
    if not db_path.exists():
        raise SystemExit(f"Index not found: {db_path}. Run the 'index' command first.")
    with db_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_index(db_path: Path, index: dict) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def build_index(args: argparse.Namespace) -> int:
    sources = [Path(item) for item in args.sources]
    extensions = normalize_extensions(args.extensions)
    client = OllamaClient(args.ollama_url, timeout=args.timeout)
    use_embeddings = not args.no_embeddings

    if use_embeddings:
        try:
            client.tags()
        except OllamaError as exc:
            print(f"warning: {exc}", file=sys.stderr)
            print("warning: continuing without neural embeddings", file=sys.stderr)
            use_embeddings = False

    chunks = []
    files = []
    total_chunks = 0
    for path in iter_source_files(sources, extensions):
        text = read_text_file(path, args.max_file_bytes)
        if text is None:
            continue
        parts = chunk_text(text, max_chars=args.chunk_chars, overlap=args.overlap)
        rel_path = os.path.relpath(path, Path.cwd())
        files.append(
            {
                "path": rel_path,
                "sha256": file_hash(text),
                "bytes": len(text.encode("utf-8", errors="replace")),
                "chunks": len(parts),
            }
        )
        for chunk_index, chunk in enumerate(parts):
            embedding = None
            if use_embeddings:
                try:
                    embedding = client.embed(args.embed_model, chunk)
                except OllamaError as exc:
                    print(f"warning: embedding failed for {rel_path}: {exc}", file=sys.stderr)
                    print("warning: continuing without neural embeddings", file=sys.stderr)
                    use_embeddings = False
            chunk_id = hashlib.sha256(f"{rel_path}:{chunk_index}:{chunk}".encode("utf-8")).hexdigest()[:16]
            chunks.append(
                {
                    "id": chunk_id,
                    "source": rel_path,
                    "chunk_index": chunk_index,
                    "text": chunk,
                    "embedding": embedding,
                }
            )
            total_chunks += 1

    index = {
        "version": 1,
        "created_at": utc_now(),
        "source_roots": [str(path) for path in sources],
        "embedding_model": args.embed_model if any(chunk.get("embedding") for chunk in chunks) else None,
        "chunk_chars": args.chunk_chars,
        "overlap": args.overlap,
        "files": files,
        "chunks": chunks,
    }
    save_index(Path(args.db), index)
    print(f"indexed {len(files)} files, {total_chunks} chunks -> {args.db}")
    if not index["embedding_model"]:
        print("note: index was created without neural embeddings; search will use lexical ranking")
    return 0


def rank_chunks(
    index: dict,
    query: str,
    top_k: int,
    client: Optional[OllamaClient] = None,
    embed_model: Optional[str] = None,
) -> List[Tuple[float, dict, str]]:
    chunks = index.get("chunks", [])
    query_embedding = None
    used_mode = "lexical"
    if client and embed_model and any(chunk.get("embedding") for chunk in chunks):
        try:
            query_embedding = client.embed(embed_model, query)
            used_mode = "embedding"
        except OllamaError as exc:
            print(f"warning: query embedding failed, using lexical search: {exc}", file=sys.stderr)

    ranked: List[Tuple[float, dict, str]] = []
    for chunk in chunks:
        embedding = chunk.get("embedding")
        if query_embedding and embedding:
            score = cosine_similarity(query_embedding, embedding)
            mode = used_mode
        else:
            score = lexical_score(query, f"{chunk.get('source', '')}\n{chunk.get('text', '')}")
            mode = "lexical"
        if score > 0:
            ranked.append((score, chunk, mode))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def format_search_result(score: float, chunk: dict, mode: str) -> str:
    excerpt = " ".join(chunk.get("text", "").split())
    if len(excerpt) > 420:
        excerpt = excerpt[:417] + "..."
    return (
        f"[{mode} score={score:.4f}] {chunk.get('source')}#chunk-{chunk.get('chunk_index')}\n"
        f"  {excerpt}"
    )


def search_index(args: argparse.Namespace) -> int:
    index = load_index(Path(args.db))
    client = OllamaClient(args.ollama_url, timeout=args.timeout)
    query = " ".join(args.query)
    ranked = rank_chunks(index, query, args.top_k, client, index.get("embedding_model"))
    if not ranked:
        print("No matching chunks found.")
        return 1
    for score, chunk, mode in ranked:
        print(format_search_result(score, chunk, mode))
        print()
    return 0


def build_context(ranked: Sequence[Tuple[float, dict, str]], max_chars: int) -> str:
    blocks = []
    used = 0
    for _, chunk, _ in ranked:
        header = f"Source: {chunk.get('source')}#chunk-{chunk.get('chunk_index')}"
        body = chunk.get("text", "")
        block = f"{header}\n{body}".strip()
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining <= 200:
                break
            block = block[:remaining]
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def ask_index(args: argparse.Namespace) -> int:
    index = load_index(Path(args.db))
    question = " ".join(args.question)
    client = OllamaClient(args.ollama_url, timeout=args.timeout)
    ranked = rank_chunks(index, question, args.top_k, client, index.get("embedding_model"))
    if not ranked:
        print("No context found in the index for this question.")
        return 1

    context = build_context(ranked, args.context_chars)
    prompt = textwrap.dedent(
        f"""
        You are a local assistant for an MES implementation engineer working with enterprise clients.
        Answer strictly from the project artifacts in the context. If the context is insufficient,
        say what is missing. Include source references using the provided Source labels.

        Context:
        {context}

        Question:
        {question}
        """
    ).strip()

    try:
        answer = client.generate(args.model, prompt)
    except OllamaError as exc:
        print(f"Could not call the local neural model: {exc}", file=sys.stderr)
        print("Relevant context:")
        print(context)
        return 2

    print(answer)
    return 0


def summarize(args: argparse.Namespace) -> int:
    client = OllamaClient(args.ollama_url, timeout=args.timeout)
    if args.sources:
        texts = []
        for path in iter_source_files([Path(item) for item in args.sources], normalize_extensions(args.extensions)):
            text = read_text_file(path, args.max_file_bytes)
            if text:
                rel_path = os.path.relpath(path, Path.cwd())
                texts.append(f"Source: {rel_path}\n{text[: args.per_file_chars]}")
        context = "\n\n---\n\n".join(texts)
    else:
        index = load_index(Path(args.db))
        ranked = [
            (1.0, chunk, "index")
            for chunk in index.get("chunks", [])[: args.top_k]
        ]
        context = build_context(ranked, args.context_chars)

    if not context.strip():
        print("No text artifacts found to summarize.")
        return 1

    prompt = textwrap.dedent(
        f"""
        You are preparing a concise engineering summary of MES project artifacts.
        Extract:
        - business/process scope
        - integrations and systems mentioned
        - risks, open questions, and missing inputs
        - next engineering actions

        Use only this context:
        {context[: args.context_chars]}
        """
    ).strip()

    try:
        print(client.generate(args.model, prompt))
    except OllamaError as exc:
        print(f"Could not call the local neural model: {exc}", file=sys.stderr)
        print("Extractive fallback:")
        remaining_lines = args.fallback_lines
        for line in context.splitlines():
            stripped = line.strip()
            if stripped:
                print(f"- {stripped[:220]}")
                remaining_lines -= 1
            if remaining_lines <= 0:
                break
        return 2
    return 0


def doctor(args: argparse.Namespace) -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Index path: {args.db}")
    client = OllamaClient(args.ollama_url, timeout=args.timeout)
    try:
        tags = client.tags()
    except OllamaError as exc:
        print(f"Ollama: unavailable ({exc})")
        print("Install Ollama and pull models, for example:")
        print(f"  ollama pull {DEFAULT_MODEL}")
        print(f"  ollama pull {DEFAULT_EMBED_MODEL}")
        return 1

    models = [item.get("name", "") for item in tags.get("models", [])]
    print(f"Ollama: available at {args.ollama_url}")
    print("Models:")
    for name in models:
        print(f"  - {name}")
    for expected in (args.model, args.embed_model):
        if expected not in models:
            print(f"warning: model {expected!r} is not installed")
    return 0


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama base URL")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama generation model")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local AI assistant for project files and implementation artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index project files and artifacts")
    index_parser.add_argument("sources", nargs="+", help="Files or directories to index")
    index_parser.add_argument("--db", default=DEFAULT_DB, help="Index JSON path")
    index_parser.add_argument("--extensions", nargs="*", help="Comma-separated or repeated extensions")
    index_parser.add_argument("--chunk-chars", type=int, default=3500)
    index_parser.add_argument("--overlap", type=int, default=350)
    index_parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    index_parser.add_argument("--no-embeddings", action="store_true", help="Skip Ollama embeddings")
    add_common_model_args(index_parser)
    index_parser.set_defaults(func=build_index)

    search_parser = subparsers.add_parser("search", help="Search the local index")
    search_parser.add_argument("query", nargs="+")
    search_parser.add_argument("--db", default=DEFAULT_DB)
    search_parser.add_argument("--top-k", type=int, default=5)
    add_common_model_args(search_parser)
    search_parser.set_defaults(func=search_index)

    ask_parser = subparsers.add_parser("ask", help="Ask a question using indexed context")
    ask_parser.add_argument("question", nargs="+")
    ask_parser.add_argument("--db", default=DEFAULT_DB)
    ask_parser.add_argument("--top-k", type=int, default=6)
    ask_parser.add_argument("--context-chars", type=int, default=12000)
    add_common_model_args(ask_parser)
    ask_parser.set_defaults(func=ask_index)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize files or indexed chunks")
    summarize_parser.add_argument("sources", nargs="*", help="Optional files or directories")
    summarize_parser.add_argument("--db", default=DEFAULT_DB)
    summarize_parser.add_argument("--extensions", nargs="*")
    summarize_parser.add_argument("--top-k", type=int, default=20)
    summarize_parser.add_argument("--context-chars", type=int, default=16000)
    summarize_parser.add_argument("--per-file-chars", type=int, default=4000)
    summarize_parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    summarize_parser.add_argument("--fallback-lines", type=int, default=20)
    add_common_model_args(summarize_parser)
    summarize_parser.set_defaults(func=summarize)

    doctor_parser = subparsers.add_parser("doctor", help="Check local AI runtime")
    doctor_parser.add_argument("--db", default=DEFAULT_DB)
    add_common_model_args(doctor_parser)
    doctor_parser.set_defaults(func=doctor)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
