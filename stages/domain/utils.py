"""Stage 8: 专项能力 — compute functions.

Subcommands:
  parsability — code parsability via tree-sitter (ERROR node counting)
  stem        — STEM subject classification via keyword density
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Callable, Iterable

import numpy as np

from src.reader import Document
from src.schema import DocResult


# ── Distribution stats ───────────────────────────────────────────────────────

def _dist_stats(values: list[float], pcts: tuple = (5, 25, 50, 75, 95)) -> dict:
    if not values:
        return {}
    a = np.array(values, dtype=float)
    out: dict = {
        "count": len(values),
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std()), 6),
        "min": round(float(a.min()), 6),
        "max": round(float(a.max()), 6),
    }
    for p in pcts:
        out[f"p{p}"] = round(float(np.percentile(a, p)), 6)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Parsability (tree-sitter)
# ══════════════════════════════════════════════════════════════════════════════

_PARSER_CACHE: dict[str, tuple] = {}


def _get_parser(language: str):
    """Get a (Parser, Language) tuple for the given language. Returns None if unsupported."""
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language]

    try:
        from tree_sitter import Language, Parser
    except ImportError:
        return None

    if language == "python":
        try:
            import tree_sitter_python as tspython
            lang = Language(tspython.language())
        except ImportError:
            return None
    else:
        return None

    parser = Parser(lang)
    _PARSER_CACHE[language] = (parser, lang)
    return parser, lang


def _count_errors(node) -> tuple[int, int]:
    """Recursively count (error_nodes, total_nodes) in a tree-sitter AST."""
    errors = 1 if (node.type == "ERROR" or node.is_missing) else 0
    total = 1
    for child in node.children:
        ce, ct = _count_errors(child)
        errors += ce
        total += ct
    return errors, total


def compute_parsability(
    docs: Iterable[Document],
    language: str = "python",
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Check code parsability using tree-sitter.

    Returns (per_doc_results, summary_dict).
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs": 0}

    result_pair = _get_parser(language)
    if result_pair is None:
        raise RuntimeError(
            f"tree-sitter 不支持语言 '{language}'。"
            f"确保已安装 tree-sitter 和对应语法包（如 tree-sitter-python）。"
        )
    parser, _ = result_pair

    per_doc: list[DocResult] = []
    error_ratios: list[float] = []
    error_counts: list[int] = []
    has_error_count = 0
    parsed_count = 0
    unparsable_count = 0

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")

        if not text.strip():
            result = DocResult(
                doc_id=doc_id,
                scores={"error_node_count": 0, "total_node_count": 0, "error_ratio": 0.0},
                flags={"has_error": False},
            )
            unparsable_count += 1
        else:
            tree = parser.parse(text.encode("utf-8", errors="replace"))
            errors, total = _count_errors(tree.root_node)
            ratio = errors / total if total > 0 else 0.0
            has_err = errors > 0

            result = DocResult(
                doc_id=doc_id,
                scores={
                    "error_node_count": errors,
                    "total_node_count": total,
                    "error_ratio": round(ratio, 6),
                },
                flags={"has_error": has_err},
            )
            parsed_count += 1
            error_ratios.append(ratio)
            error_counts.append(errors)
            if has_err:
                has_error_count += 1

        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

    total = len(doc_list)
    summary = {
        "total_docs": total,
        "parsed_docs": parsed_count,
        "unparsable_docs": unparsable_count,
        "has_error_docs": has_error_count,
        "has_error_pct": round(has_error_count / parsed_count, 4) if parsed_count else 0.0,
        "language": language,
        "error_ratio_stats": _dist_stats(error_ratios),
        "error_count_stats": _dist_stats([float(c) for c in error_counts]),
    }
    return per_doc, summary


# ══════════════════════════════════════════════════════════════════════════════
# STEM classification (keyword density)
# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_STEM_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "cs": {
        "machine_learning": [
            "neural network", "deep learning", "gradient descent", "backpropagation",
            "classifier", "regression", "overfitting", "transformer", "attention mechanism",
            "fine-tuning", "reinforcement learning", "convolutional",
        ],
        "algorithms": [
            "algorithm", "complexity", "sorting", "graph theory", "dynamic programming",
            "recursion", "binary search", "hash table", "data structure",
        ],
        "systems": [
            "operating system", "distributed system", "concurrency", "cache",
            "memory management", "kernel", "microservice",
        ],
        "nlp": [
            "tokenization", "embedding", "language model", "parsing", "named entity",
            "sentiment analysis", "machine translation", "text classification",
        ],
        "cv": [
            "convolution", "image classification", "object detection", "segmentation",
            "generative adversarial", "image recognition",
        ],
    },
    "math": {
        "algebra": [
            "polynomial", "matrix", "eigenvalue", "linear algebra", "vector space",
            "determinant", "eigenvector", "tensor",
        ],
        "calculus": [
            "derivative", "integral", "differential equation", "limit", "convergence",
            "partial derivative", "gradient",
        ],
        "statistics": [
            "probability", "distribution", "hypothesis test", "bayesian", "variance",
            "standard deviation", "confidence interval", "p-value", "regression analysis",
        ],
        "discrete": [
            "combinatorics", "permutation", "set theory", "boolean algebra",
            "graph coloring", "number theory",
        ],
    },
    "physics": {
        "mechanics": [
            "force", "momentum", "velocity", "acceleration", "kinetic energy",
            "potential energy", "angular momentum",
        ],
        "quantum": [
            "quantum mechanics", "wavefunction", "superposition", "entanglement",
            "quantum computing", "qubit",
        ],
        "thermodynamics": [
            "entropy", "thermodynamic", "heat transfer", "boltzmann",
            "thermal equilibrium",
        ],
        "electromagnetism": [
            "electromagnetic", "maxwell", "electric field", "magnetic field",
            "electromagnetic wave",
        ],
    },
    "chemistry": {
        "organic": [
            "organic compound", "polymer", "reaction mechanism", "catalyst",
            "functional group", "hydrocarbon",
        ],
        "inorganic": [
            "crystal structure", "ionic bond", "oxidation", "coordination compound",
            "electrochemistry",
        ],
        "biochemistry": [
            "protein", "enzyme", "amino acid", "dna", "rna", "metabolism",
            "molecular biology", "gene expression",
        ],
    },
    "biology": {
        "genetics": [
            "gene", "genome", "mutation", "crispr", "heritability", "allele",
            "chromosome", "dna sequencing",
        ],
        "ecology": [
            "ecosystem", "biodiversity", "population dynamics", "species",
            "ecological niche", "food chain",
        ],
        "cell_biology": [
            "cell membrane", "mitosis", "organelle", "cytoplasm", "apoptosis",
        ],
    },
    "engineering": {
        "electrical": [
            "circuit", "semiconductor", "transistor", "signal processing", "fpga",
            "integrated circuit",
        ],
        "mechanical": [
            "stress analysis", "strain", "fluid dynamics", "turbulence",
            "finite element",
        ],
        "materials": [
            "alloy", "composite material", "crystallography", "tensile strength",
            "nanomaterial",
        ],
    },
    "medicine": {
        "clinical": [
            "diagnosis", "treatment", "clinical trial", "prognosis", "pathology",
            "epidemiology",
        ],
        "pharmacology": [
            "drug", "dosage", "pharmacokinetics", "receptor", "pharmacodynamics",
        ],
    },
}

_DEFAULT_DIFFICULTY_KEYWORDS: dict[str, list[str]] = {
    "advanced": [
        "theorem", "proof", "lemma", "corollary", "conjecture", "non-trivial",
        "asymptotic", "intractable", "np-hard", "lagrangian", "hamiltonian",
        "perturbation theory", "renormalization", "homomorphism", "isomorphism",
        "stochastic", "manifold", "hilbert space", "variational",
    ],
    "intermediate": [
        "equation", "derivation", "formulation", "optimization", "constraint",
        "objective function", "loss function", "convergence rate",
        "complexity analysis", "approximation", "numerical method",
    ],
    "basic": [
        "introduction", "fundamental", "basic", "definition", "example",
        "tutorial", "beginner", "overview", "primer",
    ],
}


def _compile_taxonomy(taxonomy: dict[str, dict[str, list[str]]]) -> dict[str, list[re.Pattern]]:
    """Compile taxonomy keywords into regex patterns for efficient matching."""
    compiled: dict[str, list[re.Pattern]] = {}
    for cat, subcats in taxonomy.items():
        patterns = []
        for keywords in subcats.values():
            for kw in keywords:
                patterns.append(re.compile(r"\b" + re.escape(kw.lower()) + r"\b"))
        compiled[cat] = patterns
    return compiled


def _compile_difficulty(keywords: dict[str, list[str]]) -> dict[str, list[re.Pattern]]:
    compiled: dict[str, list[re.Pattern]] = {}
    for level, kws in keywords.items():
        compiled[level] = [re.compile(r"\b" + re.escape(kw.lower()) + r"\b") for kw in kws]
    return compiled


def compute_stem(
    docs: Iterable[Document],
    taxonomy: dict[str, dict[str, list[str]]] | None = None,
    difficulty_keywords: dict[str, list[str]] | None = None,
    min_density: float = 0.001,
    on_doc: Callable[[DocResult], None] | None = None,
) -> tuple[list[DocResult], dict]:
    """Classify documents by STEM subject and difficulty via keyword density.

    Returns (per_doc_results, summary_dict).
    """
    doc_list = list(docs)
    if not doc_list:
        return [], {"total_docs": 0}

    tax = taxonomy or _DEFAULT_STEM_TAXONOMY
    diff_kw = difficulty_keywords or _DEFAULT_DIFFICULTY_KEYWORDS
    compiled_tax = _compile_taxonomy(tax)
    compiled_diff = _compile_difficulty(diff_kw)

    per_doc: list[DocResult] = []
    stem_count = 0
    subject_counter: Counter = Counter()
    difficulty_counter: Counter = Counter()

    for doc in doc_list:
        doc_id = str(doc["doc_id"])
        text = str(doc.get("text") or "")
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)

        subject_scores: dict[str, float] = {}
        total_hits = 0
        for cat, patterns in compiled_tax.items():
            hits = sum(len(p.findall(text_lower)) for p in patterns)
            density = hits / word_count if word_count > 0 else 0.0
            subject_scores[cat] = round(density, 6)
            total_hits += hits

        primary = None
        max_density = 0.0
        for cat, density in subject_scores.items():
            if density >= min_density and density > max_density:
                max_density = density
                primary = cat

        difficulty = "unknown"
        for level in ("advanced", "intermediate", "basic"):
            if any(p.search(text_lower) for p in compiled_diff[level]):
                difficulty = level
                break

        is_stem = primary is not None

        result = DocResult(
            doc_id=doc_id,
            scores={
                "subject_scores": subject_scores,
                "primary_subject": primary,
                "difficulty_level": difficulty,
                "word_count": word_count,
                "stem_keyword_hits": total_hits,
            },
            flags={"is_stem": is_stem},
        )

        if on_doc is not None:
            on_doc(result)
        else:
            per_doc.append(result)

        if is_stem:
            stem_count += 1
            subject_counter[primary] += 1
        difficulty_counter[difficulty] += 1

    total = len(doc_list)
    subject_dist = {
        cat: {"docs": subject_counter.get(cat, 0),
              "pct": round(subject_counter.get(cat, 0) / total, 4) if total else 0.0}
        for cat in sorted(tax.keys())
    }

    summary = {
        "total_docs": total,
        "stem_docs": stem_count,
        "stem_pct": round(stem_count / total, 4) if total else 0.0,
        "min_keyword_density": min_density,
        "subject_distribution": subject_dist,
        "difficulty_distribution": dict(difficulty_counter.most_common()),
        "primary_subject_top10": dict(subject_counter.most_common(10)),
    }
    return per_doc, summary
