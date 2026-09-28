"""
chunk_docs.py
Reads markdown/text files from DOCS_PATH, splits them by heading and then into
~MAX_WORDS blocks, and writes data/chunks.jsonl.

Each chunk keeps: id, source (relative file path), section (heading), text.

Run from the project root:
    python scripts/chunk_docs.py
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = Path(os.getenv("DOCS_PATH", "./source-docs"))
OUT_FILE = Path("data/chunks.jsonl")
MAX_WORDS = 500          # target upper size of one chunk
MIN_WORDS = 30           # drop tiny fragments (e.g. a heading with no content)
EXTENSIONS = {".md", ".markdown", ".txt"}
SKIP_DIRS = {".git", "node_modules", ".github"}

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            yield path


def split_by_heading(text: str):
    """Yield (section_title, section_text). Ignores '#' inside fenced code blocks."""
    sections = []
    title = "Introduction"
    buf = []
    in_fence = False

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING_RE.match(line)
        if match:
            if buf:
                sections.append((title, "\n".join(buf).strip()))
            title = match.group(2).strip()
            buf = []
        else:
            buf.append(line)

    if buf:
        sections.append((title, "\n".join(buf).strip()))
    return sections


def split_long(text: str, max_words: int):
    """Split a long section on blank lines into blocks of <= max_words."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    blocks, current, count = [], [], 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > max_words:
            blocks.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        blocks.append("\n\n".join(current))
    return blocks


def main():
    if not DOCS_PATH.exists():
        raise SystemExit(f"DOCS_PATH not found: {DOCS_PATH.resolve()}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with OUT_FILE.open("w", encoding="utf-8") as out:
        for path in iter_files(DOCS_PATH):
            rel = path.relative_to(DOCS_PATH).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")

            # Section index keeps ids unique when a heading repeats within a file
            for s_idx, (section, body) in enumerate(split_by_heading(text)):
                for i, block in enumerate(split_long(body, MAX_WORDS)):
                    if len(block.split()) < MIN_WORDS:
                        continue
                    chunk = {
                        "id": f"{rel}::{s_idx}::{section}::{i}",
                        "source": rel,
                        "section": section,
                        # Prefix with file + section so the embedding carries context
                        "text": f"[{rel} > {section}]\n{block}",
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Wrote {total} chunks to {OUT_FILE}")


if __name__ == "__main__":
    main()