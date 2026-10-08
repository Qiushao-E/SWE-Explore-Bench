"""Shared chunking utilities for RAG-based explorers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

EXTS = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".rst"}


@dataclass
class Chunk:
    """A contiguous block of lines from a source file."""

    path: str  # relative path within repo
    start: int  # 1-based start line
    end: int  # 1-based end line (inclusive)
    content: str


def iter_source_files(repo_root: Path) -> Iterable[Path]:
    """Yield source files matching common extensions."""
    for p in repo_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def chunk_repo(
    repo_root: Path,
    *,
    chunk_size: int = 80,
    chunk_overlap: int = 20,
    max_chunks: int = 3000,
) -> list[Chunk]:
    """Chunk all source files in a repo into overlapping line windows.

    Returns at most *max_chunks* chunks (truncated if exceeded).
    """
    step = max(1, chunk_size - chunk_overlap)
    chunks: list[Chunk] = []

    for p in iter_source_files(repo_root):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        if not lines:
            continue
        rel = p.relative_to(repo_root).as_posix()
        for i in range(0, len(lines), step):
            end_idx = min(i + chunk_size, len(lines))
            content = "\n".join(lines[i:end_idx])
            if content.strip():
                chunks.append(Chunk(path=rel, start=i + 1, end=end_idx, content=content))
            if len(chunks) >= max_chunks:
                return chunks
            if end_idx >= len(lines):
                break

    return chunks
