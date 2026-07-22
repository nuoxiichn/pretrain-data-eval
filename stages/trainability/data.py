"""Data preparation for the tiny-proxy trainability stage."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain_data_eval.reader import Document, read_documents
from pretrain_data_eval.sampling import sample_documents


EOD_TOKEN = "<eod>"
UNK_TOKEN = "<unk>"
_CORPUS_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class EncodedDocument:
    """One validation document encoded with the shared tokenizer."""

    doc_id: str
    token_ids: tuple[int, ...]
    original_token_count: int
    language: str | None = None
    source: str | None = None
    dedup_cluster_id: str | None = None
    comparison_groups: tuple[str, ...] = ()
    anchor_id: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class EncodedTrainDocument:
    """One train document's location in its corpus token stream."""

    doc_id: str
    dedup_cluster_id: str
    token_start: int
    token_count: int


class MemmapTokenStream:
    """Tensor-like random access over an integer token file without materializing it."""

    def __init__(self, path: Path, dtype: np.dtype, length: int):
        self.path = path
        self.dtype = dtype
        self.length = length
        self._values = np.memmap(path, mode="c", dtype=dtype, shape=(length,))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: slice | torch.Tensor) -> torch.Tensor:
        if isinstance(index, torch.Tensor):
            index = index.detach().cpu().numpy()
        values = np.asarray(self._values[index], dtype=np.int64)
        return torch.from_numpy(values)


class SegmentedTokenStream:
    """Lazy concatenation of auditable source-stream segments."""

    def __init__(
        self,
        streams: Sequence[torch.Tensor | MemmapTokenStream],
        segments: Sequence[tuple[int, int, int]],
    ) -> None:
        if not streams or not segments:
            raise ValueError("segmented token stream requires streams and segments")
        self.streams = tuple(streams)
        self.segments = tuple(segments)
        boundaries = [0]
        for stream_index, start, length in self.segments:
            if not 0 <= stream_index < len(self.streams):
                raise ValueError("segment references an unknown stream")
            if start < 0 or length <= 0 or start + length > len(self.streams[stream_index]):
                raise ValueError("segment lies outside its source stream")
            boundaries.append(boundaries[-1] + length)
        self._boundaries = np.asarray(boundaries, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._boundaries[-1])

    def __getitem__(self, index: slice | torch.Tensor) -> torch.Tensor:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            flat = np.arange(start, stop, step, dtype=np.int64)
            shape = flat.shape
        else:
            values = index.detach().cpu().numpy().astype(np.int64, copy=False)
            shape = values.shape
            flat = values.reshape(-1)
        if flat.size == 0:
            return torch.empty(shape, dtype=torch.long)
        if int(flat.min()) < 0 or int(flat.max()) >= len(self):
            raise IndexError("segmented stream index out of range")
        segment_indices = np.searchsorted(self._boundaries[1:], flat, side="right")
        output = torch.empty(flat.size, dtype=torch.long)
        for segment_index in np.unique(segment_indices):
            mask = segment_indices == segment_index
            stream_index, source_start, _ = self.segments[int(segment_index)]
            source_indices = (
                flat[mask] - self._boundaries[int(segment_index)] + source_start
            )
            source_tensor_indices = torch.from_numpy(source_indices.astype(np.int64, copy=False))
            output[torch.from_numpy(np.flatnonzero(mask))] = self.streams[stream_index][
                source_tensor_indices
            ]
        return output.reshape(shape)


@dataclass
class CorpusData:
    """Prepared train and validation data for one named corpus."""

    name: str
    path: Path
    sampled_docs: int
    train_docs: list[Document]
    validation_docs: list[Document]
    train_tokens: torch.Tensor | MemmapTokenStream | None = None
    encoded_validation: list[EncodedDocument] | None = None
    fingerprint: str = ""
    source_format: str = "documents"
    validation_region_start: int | None = None
    split_audit: dict | None = None
    encoded_train_documents: list[EncodedTrainDocument] | None = None


def parse_corpus_specs(specs: Sequence[str]) -> list[tuple[str, Path]]:
    """Parse repeated ``NAME=PATH`` CLI values and validate unique names."""
    parsed: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid corpus {spec!r}; expected NAME=PATH")
        name, raw_path = spec.split("=", 1)
        if not _CORPUS_NAME.fullmatch(name):
            raise ValueError(
                f"invalid corpus name {name!r}; use letters, digits, '.', '_' or '-'"
            )
        if name in seen:
            raise ValueError(f"duplicate corpus name {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((name, path))
        seen.add(name)
    if len(parsed) < 2:
        raise ValueError("trainability probe requires at least two --corpus NAME=PATH inputs")
    return parsed


def _split_value(doc_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def normalized_text_sha256(text: str) -> str:
    """Hash normalized text for exact-overlap auditing."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def document_cluster_id(document: Document) -> str:
    """Return the frozen near-dedup cluster, falling back to exact content."""
    meta = document.get("meta") or {}
    explicit = meta.get("dedup_cluster_id") or meta.get("near_cluster_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    return f"exact:{normalized_text_sha256(document['text'])}"


def _frozen_split(document: Document) -> str | None:
    value = (document.get("meta") or {}).get("split")
    if value is None:
        return None
    value = str(value)
    if value not in {"train", "validation"}:
        raise ValueError(
            f"document {document['doc_id']!r} has invalid meta.split {value!r}"
        )
    return value


def split_documents(
    documents: Sequence[Document], validation_fraction: float, seed: int
) -> tuple[list[Document], list[Document]]:
    """Split connected declared-near and current exact-text components."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(documents) < 2:
        raise ValueError("each corpus needs at least two documents")

    frozen = [_frozen_split(document) for document in documents]
    if any(value is not None for value in frozen) and not all(
        value is not None for value in frozen
    ):
        raise ValueError("meta.split must be present on every document or none")

    parent = list(range(len(documents)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    constraint_owner: dict[str, int] = {}
    document_constraints: list[tuple[str, str]] = []
    for index, document in enumerate(documents):
        constraints = (
            f"declared:{document_cluster_id(document)}",
            f"exact:{normalized_text_sha256(document['text'])}",
        )
        document_constraints.append(constraints)
        for constraint in constraints:
            previous = constraint_owner.setdefault(constraint, index)
            union(index, previous)

    component_constraints: dict[int, set[str]] = {}
    component_frozen_targets: dict[int, set[str]] = {}
    for index, constraints in enumerate(document_constraints):
        root = find(index)
        component_constraints.setdefault(root, set()).update(constraints)
        if frozen[index] is not None:
            component_frozen_targets.setdefault(root, set()).add(frozen[index])

    component_targets: dict[int, str] = {}
    for root, constraints in component_constraints.items():
        frozen_targets = component_frozen_targets.get(root, set())
        identity = min(constraints)
        if len(frozen_targets) > 1:
            raise ValueError(
                f"dedup/exact component {identity!r} is assigned to both train and validation"
            )
        component_targets[root] = (
            next(iter(frozen_targets))
            if frozen_targets
            else (
                "validation"
                if _split_value(identity, seed) < validation_fraction
                else "train"
            )
        )

    train: list[Document] = []
    validation: list[Document] = []
    for index, document in enumerate(documents):
        target = validation if component_targets[find(index)] == "validation" else train
        target.append(document)

    # Small smoke datasets can hash entirely to one side. Move the closest
    # boundary document deterministically rather than silently producing no data.
    if not validation or not train:
        if all(value is not None for value in frozen):
            raise ValueError("frozen meta.split must contain both train and validation documents")
        source = train if train else validation
        target = validation if train else train
        source_ids = {id(document) for document in source}
        source_indices = [
            index
            for index, document in enumerate(documents)
            if id(document) in source_ids
        ]
        component = min(
            {find(index) for index in source_indices},
            key=lambda value: abs(
                _split_value(min(component_constraints[value]), seed)
                - validation_fraction
            ),
        )
        moved = [
            document
            for index, document in enumerate(documents)
            if find(index) == component
        ]
        moved_ids = {id(document) for document in moved}
        source[:] = [document for document in source if id(document) not in moved_ids]
        target.extend(moved)
        if not source:
            raise ValueError("dedup/exact constraints collapse corpus to one split component")
    return train, validation


def audit_document_overlaps(corpora: Sequence[CorpusData]) -> dict:
    """Audit exact and declared near-dedup overlap across splits and corpora."""
    corpus_sets: dict[str, dict[str, set[str]]] = {}
    global_near_splits: dict[str, set[str]] = {}
    global_exact_splits: dict[str, set[str]] = {}
    by_corpus: dict[str, dict] = {}
    for corpus in corpora:
        split_sets: dict[str, dict[str, set[str]]] = {}
        for split, documents in (
            ("train", corpus.train_docs),
            ("validation", corpus.validation_docs),
        ):
            exact = {normalized_text_sha256(document["text"]) for document in documents}
            near = {document_cluster_id(document) for document in documents}
            doc_ids = {document["doc_id"] for document in documents}
            split_sets[split] = {"exact": exact, "near": near, "doc_ids": doc_ids}
            for cluster_id in near:
                global_near_splits.setdefault(cluster_id, set()).add(split)
            for exact_hash in exact:
                global_exact_splits.setdefault(exact_hash, set()).add(split)
        corpus_sets[corpus.name] = {
            key: split_sets["train"][key] | split_sets["validation"][key]
            for key in ("exact", "near", "doc_ids")
        }
        by_corpus[corpus.name] = {
            "train_validation_exact_hashes": len(
                split_sets["train"]["exact"] & split_sets["validation"]["exact"]
            ),
            "train_validation_near_clusters": len(
                split_sets["train"]["near"] & split_sets["validation"]["near"]
            ),
            "train_validation_doc_ids": len(
                split_sets["train"]["doc_ids"] & split_sets["validation"]["doc_ids"]
            ),
        }

    leaking_near = sorted(
        cluster_id for cluster_id, splits in global_near_splits.items() if len(splits) > 1
    )
    leaking_exact = sorted(
        exact_hash for exact_hash, splits in global_exact_splits.items() if len(splits) > 1
    )
    leaking_clusters = [*(f"near:{value}" for value in leaking_near)]
    leaking_clusters.extend(f"exact:{value}" for value in leaking_exact)
    cross_corpus: dict[str, dict] = {}
    for left_index, left in enumerate(corpora):
        for right in corpora[left_index + 1 :]:
            cross_corpus[f"{left.name}<->{right.name}"] = {
                "exact_hashes": len(
                    corpus_sets[left.name]["exact"] & corpus_sets[right.name]["exact"]
                ),
                "near_clusters": len(
                    corpus_sets[left.name]["near"] & corpus_sets[right.name]["near"]
                ),
                "doc_ids": len(
                    corpus_sets[left.name]["doc_ids"] & corpus_sets[right.name]["doc_ids"]
                ),
            }
    return {
        "split_key": "connected meta.dedup_cluster_id/meta.near_cluster_id and normalized exact text",
        "by_corpus": by_corpus,
        "cross_corpus": cross_corpus,
        "global_train_validation_leaking_clusters": len(leaking_clusters),
        "global_train_validation_leaking_near_clusters": len(leaking_near),
        "global_train_validation_leaking_exact_hashes": len(leaking_exact),
        "leaking_cluster_examples": leaking_clusters[:20],
    }


def corpus_fingerprint(documents: Iterable[Document]) -> str:
    """Return a content fingerprint for the sampled, pre-split corpus."""
    digest = hashlib.sha256()
    for document in documents:
        digest.update(document["doc_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(document["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_corpora(
    specs: Sequence[tuple[str, Path]],
    input_config: dict,
    max_docs: int | None,
    sample_mode: str,
    sample_seed: int,
    validation_fraction: float,
    split_seed: int,
) -> list[CorpusData]:
    """Read, sample and document-split all corpora."""
    corpora: list[CorpusData] = []
    for index, (name, path) in enumerate(specs):
        documents = sample_documents(
            read_documents(path, input_config),
            max_docs,
            mode=sample_mode,
            seed=sample_seed + index,
        )
        documents = [document for document in documents if document["text"].strip()]
        train, validation = split_documents(documents, validation_fraction, split_seed)
        corpora.append(
            CorpusData(
                name=name,
                path=path,
                sampled_docs=len(documents),
                train_docs=train,
                validation_docs=validation,
                fingerprint=corpus_fingerprint(documents),
            )
        )
    return corpora


def _balanced_text_iterator(
    corpora: Sequence[CorpusData], max_chars_per_corpus: int
) -> tuple[Iterable[str], int]:
    totals = [sum(len(document["text"]) for document in corpus.train_docs) for corpus in corpora]
    char_budget = min(max_chars_per_corpus, min(totals))
    if char_budget <= 0:
        raise ValueError("tokenizer training needs non-empty train text in every corpus")

    def iterator() -> Iterable[str]:
        for corpus in corpora:
            remaining = char_budget
            for document in corpus.train_docs:
                if remaining <= 0:
                    break
                text = document["text"][:remaining]
                if text:
                    yield text
                    remaining -= len(text)

    return iterator(), char_budget


def train_shared_tokenizer(
    corpora: Sequence[CorpusData],
    vocab_size: int,
    min_frequency: int,
    max_chars_per_corpus: int,
    output_path: Path,
) -> tuple[Tokenizer, dict]:
    """Train one balanced byte-level BPE shared by every compared corpus."""
    if vocab_size < 512:
        raise ValueError("vocab_size must be at least 512 for byte-level BPE")
    iterator, chars_per_corpus = _balanced_text_iterator(corpora, max_chars_per_corpus)
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN, byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=[EOD_TOKEN, UNK_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer, {
        "type": "byte_level_bpe",
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": min_frequency,
        "training_chars_per_corpus": chars_per_corpus,
        "path": str(output_path),
        "source": "trained_for_run",
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }


def load_shared_tokenizer(
    source_path: Path, output_path: Path, eod_token: str = EOD_TOKEN
) -> tuple[Tokenizer, dict]:
    """Load and snapshot a frozen tokenizer for comparable repeated runs."""
    source_path = source_path.expanduser().resolve()
    tokenizer = Tokenizer.from_file(str(source_path))
    eod_id = tokenizer.token_to_id(eod_token)
    if eod_id is None:
        raise ValueError(f"frozen tokenizer is missing required token {eod_token!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer, {
        "type": "frozen_tokenizer",
        "vocab_size": tokenizer.get_vocab_size(),
        "path": str(output_path),
        "source": str(source_path),
        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "eod_token": eod_token,
        "eod_token_id": eod_id,
    }


def _encode_train_stream(
    tokenizer: Tokenizer, documents: Sequence[Document], max_tokens: int
) -> tuple[torch.Tensor, list[EncodedTrainDocument]]:
    eod_id = tokenizer.token_to_id(EOD_TOKEN)
    if eod_id is None:
        raise ValueError(f"tokenizer is missing {EOD_TOKEN}")
    tokens: list[int] = []
    encoded_documents: list[EncodedTrainDocument] = []
    for document in documents:
        if len(tokens) >= max_tokens:
            break
        token_start = len(tokens)
        document_tokens = [eod_id, *tokenizer.encode(document["text"]).ids]
        document_tokens = document_tokens[: max_tokens - token_start]
        tokens.extend(document_tokens)
        encoded_documents.append(
            EncodedTrainDocument(
                doc_id=document["doc_id"],
                dedup_cluster_id=document_cluster_id(document),
                token_start=token_start,
                token_count=len(document_tokens),
            )
        )
    if len(tokens) < 2:
        raise ValueError("encoded train split has fewer than two tokens")
    return torch.tensor(tokens, dtype=torch.long), encoded_documents


def _encode_validation(
    tokenizer: Tokenizer, documents: Sequence[Document], max_tokens_per_doc: int
) -> list[EncodedDocument]:
    eod_id = tokenizer.token_to_id(EOD_TOKEN)
    if eod_id is None:
        raise ValueError(f"tokenizer is missing {EOD_TOKEN}")
    encoded: list[EncodedDocument] = []
    for document in documents:
        all_tokens = tokenizer.encode(document["text"]).ids
        kept = all_tokens[:max_tokens_per_doc]
        if not kept:
            continue
        encoded.append(
            EncodedDocument(
                doc_id=document["doc_id"],
                token_ids=tuple([eod_id, *kept]),
                original_token_count=len(all_tokens),
                language=document.get("language"),
                source=document.get("source"),
                dedup_cluster_id=document_cluster_id(document),
                comparison_groups=tuple(
                    str(value)
                    for value in (document.get("meta") or {}).get(
                        "comparison_groups", []
                    )
                ),
                anchor_id=(document.get("meta") or {}).get("anchor_id"),
                domain=(document.get("meta") or {}).get("domain"),
            )
        )
    if not encoded:
        raise ValueError("validation split has no tokenizable documents")
    return encoded


def encode_corpora(
    corpora: Sequence[CorpusData],
    tokenizer: Tokenizer,
    max_train_tokens_per_corpus: int,
    max_eval_tokens_per_doc: int,
) -> None:
    """Populate train streams and per-document validation tokens in place."""
    for corpus in corpora:
        corpus.train_tokens, corpus.encoded_train_documents = _encode_train_stream(
            tokenizer, corpus.train_docs, max_train_tokens_per_corpus
        )
        corpus.encoded_validation = _encode_validation(
            tokenizer, corpus.validation_docs, max_eval_tokens_per_doc
        )


def _hash_order(namespace: str, seed: int, value: str) -> bytes:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).digest()


def _round_robin(groups: Sequence[Sequence[tuple[int, int, int]]]) -> list[tuple[int, int, int]]:
    ordered: list[tuple[int, int, int]] = []
    for offset in range(max(len(group) for group in groups)):
        for group in groups:
            if offset < len(group):
                ordered.append(group[offset])
    return ordered


def _recipe_segments(
    corpora: Sequence[CorpusData], tokens_per_corpus: int, seed: int, chunk_tokens: int
) -> list[list[tuple[int, int, int]]]:
    groups: list[list[tuple[int, int, int]]] = []
    for corpus_index, corpus in enumerate(corpora):
        documents = corpus.encoded_train_documents
        segments: list[tuple[int, int, int]] = []
        remaining = tokens_per_corpus
        if documents:
            ordered_documents = sorted(
                documents,
                key=lambda document: _hash_order(
                    f"balanced:{corpus.name}", seed, document.dedup_cluster_id
                ),
            )
            for document in ordered_documents:
                if remaining <= 0:
                    break
                length = min(document.token_count, remaining)
                segments.append((corpus_index, document.token_start, length))
                remaining -= length
        else:
            stream = corpus.train_tokens
            if stream is None:
                raise ValueError("corpora must be encoded before pooling")
            raw_segments = [
                (corpus_index, start, min(chunk_tokens, len(stream) - start))
                for start in range(0, len(stream), chunk_tokens)
            ]
            raw_segments.sort(
                key=lambda segment: _hash_order(
                    f"balanced:{corpus.name}", seed, str(segment[1])
                )
            )
            for stream_index, start, length in raw_segments:
                if remaining <= 0:
                    break
                kept = min(length, remaining)
                segments.append((stream_index, start, kept))
                remaining -= kept
        if remaining:
            raise ValueError(f"could not allocate balanced tokens for corpus {corpus.name}")
        groups.append(segments)
    return groups


def _unique_cluster_segments(
    corpora: Sequence[CorpusData], seed: int
) -> tuple[list[tuple[int, int, int]], int]:
    representatives: dict[str, tuple[str, str, int, int, int]] = {}
    memberships = 0
    for corpus_index, corpus in enumerate(corpora):
        if not corpus.encoded_train_documents:
            raise ValueError(
                "unique_cluster_union requires document inputs with auditable train clusters"
            )
        for document in corpus.encoded_train_documents:
            memberships += 1
            candidate = (
                corpus.name,
                document.doc_id,
                corpus_index,
                document.token_start,
                document.token_count,
            )
            previous = representatives.get(document.dedup_cluster_id)
            if previous is None or candidate[:2] < previous[:2]:
                representatives[document.dedup_cluster_id] = candidate
    ordered = sorted(
        representatives.items(),
        key=lambda item: _hash_order("unique-cluster", seed, item[0]),
    )
    segments = [(value[2], value[3], value[4]) for _, value in ordered]
    return segments, memberships - len(representatives)


def make_balanced_pool(
    corpora: Sequence[CorpusData],
    *,
    seed: int = 0,
    chunk_tokens: int = 65536,
    policy: str = "recipe_multiset",
) -> tuple[SegmentedTokenStream, dict]:
    """Build a deterministic, auditable scale-pair training pool."""
    if chunk_tokens <= 0:
        raise ValueError("balanced pool chunk_tokens must be positive")
    streams = [corpus.train_tokens for corpus in corpora]
    if any(stream is None for stream in streams):
        raise ValueError("corpora must be encoded before pooling")
    typed_streams = [stream for stream in streams if stream is not None]
    overlap_clusters_removed = 0
    if policy == "recipe_multiset":
        tokens_per_corpus = min(len(stream) for stream in typed_streams)
        segment_groups = _recipe_segments(corpora, tokens_per_corpus, seed, chunk_tokens)
        segments = _round_robin(segment_groups)
        selected_tokens = {corpus.name: tokens_per_corpus for corpus in corpora}
    elif policy == "unique_cluster_union":
        segments, overlap_clusters_removed = _unique_cluster_segments(corpora, seed)
        tokens_per_corpus = None
        selected_tokens = {corpus.name: 0 for corpus in corpora}
        for stream_index, _, length in segments:
            selected_tokens[corpora[stream_index].name] += length
    else:
        raise ValueError(
            "balanced pool policy must be 'recipe_multiset' or 'unique_cluster_union'"
        )
    pool = SegmentedTokenStream(typed_streams, segments)
    digest = hashlib.sha256()
    for stream_index, start, length in segments:
        digest.update(f"{corpora[stream_index].name}:{start}:{length}\n".encode("utf-8"))
    manifest = {
        "policy": policy,
        "seed": seed,
        "chunk_tokens": chunk_tokens,
        "tokens_per_corpus": tokens_per_corpus,
        "selected_tokens_by_corpus": selected_tokens,
        "total_tokens": len(pool),
        "segment_count": len(segments),
        "overlap_clusters_removed": overlap_clusters_removed,
        "membership_sha256": digest.hexdigest(),
        "segments": [
            {
                "corpus": corpora[stream_index].name,
                "source_token_start": start,
                "token_count": length,
            }
            for stream_index, start, length in segments
        ],
    }
    return pool, manifest


def _stream_fingerprint(path: Path, sample_bytes: int = 65536) -> str:
    """Fingerprint a large stream without reading the entire multi-GB file."""
    size = path.stat().st_size
    digest = hashlib.sha256(f"{size}:".encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _validation_documents_from_stream(
    values: np.ndarray,
    corpus_name: str,
    absolute_start: int,
    eod_id: int,
    max_docs: int,
    max_tokens_per_doc: int,
) -> list[EncodedDocument]:
    """Extract complete EOS-delimited validation documents from a token region."""
    eod_positions = np.flatnonzero(values == eod_id)
    if len(eod_positions) < 2:
        raise ValueError(
            f"pretokenized corpus {corpus_name!r} validation region has fewer than two EOS tokens"
        )

    documents: list[EncodedDocument] = []
    for left, right in zip(eod_positions, eod_positions[1:], strict=False):
        original_length = int(right - left - 1)
        if original_length <= 0:
            continue
        kept = values[left + 1 : min(right, left + 1 + max_tokens_per_doc)]
        documents.append(
            EncodedDocument(
                doc_id=f"stream-{absolute_start + int(left) + 1}",
                token_ids=tuple([eod_id, *map(int, kept)]),
                original_token_count=original_length,
            )
        )
        if len(documents) >= max_docs:
            break
    if not documents:
        raise ValueError(f"pretokenized corpus {corpus_name!r} has no validation documents")
    return documents


def load_pretokenized_corpora(
    specs: Sequence[tuple[str, Path]],
    *,
    dtype: str,
    vocab_size: int,
    eod_id: int,
    max_train_tokens_per_corpus: int,
    validation_reserve_tokens: int,
    max_validation_docs: int,
    max_eval_tokens_per_doc: int,
) -> list[CorpusData]:
    """Memory-map raw token streams with disjoint prefix train and suffix validation regions."""
    dtype_map = {"uint16_le": np.dtype("<u2"), "uint32_le": np.dtype("<u4")}
    if dtype not in dtype_map:
        raise ValueError(f"unsupported pretokenized dtype {dtype!r}")
    if validation_reserve_tokens < 2:
        raise ValueError("validation_reserve_tokens must be at least 2")
    if max_validation_docs <= 0:
        raise ValueError("max validation documents must be positive")

    corpora: list[CorpusData] = []
    numpy_dtype = dtype_map[dtype]
    for name, path in specs:
        if path.stat().st_size % numpy_dtype.itemsize:
            raise ValueError(f"pretokenized stream {path} is not aligned to {dtype}")
        stream = np.memmap(path, mode="c", dtype=numpy_dtype)
        total_tokens = len(stream)
        if total_tokens <= validation_reserve_tokens + 2:
            raise ValueError(f"pretokenized stream {path} is too short for validation reserve")
        train_tokens = min(max_train_tokens_per_corpus, total_tokens - validation_reserve_tokens)
        if train_tokens < 2:
            raise ValueError(f"pretokenized stream {path} has fewer than two train tokens")

        validation_start = total_tokens - validation_reserve_tokens
        validation_values = np.asarray(stream[validation_start:])
        encoded_validation = _validation_documents_from_stream(
            validation_values,
            name,
            validation_start,
            eod_id,
            max_validation_docs,
            max_eval_tokens_per_doc,
        )
        observed_max = max(
            int(np.max(stream[:train_tokens])), int(np.max(validation_values))
        )
        if observed_max >= vocab_size:
            raise ValueError(
                f"pretokenized corpus {name!r} contains token {observed_max}, "
                f"outside tokenizer vocabulary {vocab_size}"
            )
        corpora.append(
            CorpusData(
                name=name,
                path=path,
                sampled_docs=len(encoded_validation),
                train_docs=[],
                validation_docs=[],
                train_tokens=MemmapTokenStream(path, numpy_dtype, train_tokens),
                encoded_validation=encoded_validation,
                fingerprint=_stream_fingerprint(path),
                source_format=dtype,
                validation_region_start=validation_start,
            )
        )
    return corpora
