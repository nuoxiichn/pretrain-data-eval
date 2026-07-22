"""HuggingFace Tokenizer loading shared across stages.

stage1 source_audit (token counting for char/token stats) and stage10
tokenization (fertility/UNK) both need to load a HF Tokenizer. Keep the
loader in one place so the path semantics (local file / local dir / hub
name) stay consistent.
"""

from __future__ import annotations

from pathlib import Path


def load_tokenizer(path: str):
    """Load a HF Tokenizer from a local tokenizer.json, a model dir, or a hub name."""
    from tokenizers import Tokenizer

    p = Path(path)
    if p.is_file():
        return Tokenizer.from_file(str(p))
    if p.is_dir():
        tok_file = p / "tokenizer.json"
        if tok_file.exists():
            return Tokenizer.from_file(str(tok_file))
    return Tokenizer.from_pretrained(path)


def find_unk_id(tokenizer) -> int | None:
    for unk_str in ("[UNK]", "<unk>", "<UNK>"):
        uid = tokenizer.token_to_id(unk_str)
        if uid is not None:
            return uid
    return None
