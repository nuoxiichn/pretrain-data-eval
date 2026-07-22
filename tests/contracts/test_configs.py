from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_all_public_stage_configs_are_valid_yaml_and_have_output():
    expected = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
    for stage in expected:
        path = ROOT / "configs" / f"stage{stage}.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(config, dict), path
        assert config.get("output", {}).get("base_dir") == f"outputs/stage{stage}", path
        if stage != 9:
            input_config = config.get("input", {})
            assert input_config.get("format") in {"auto", "jsonl", "parquet"}, path
            assert isinstance(input_config.get("field_map", {}), dict), path

    control = yaml.safe_load((ROOT / "configs" / "control.yaml").read_text(encoding="utf-8"))
    assert control["input"]["format"] == "jsonl"
    assert {"quality", "extraction", "exact", "minhash", "ngram"} <= control.keys()


def test_no_public_document_references_removed_user_guide_or_pipeline_overview():
    roots = [ROOT / "README.md", ROOT / "docs"]
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*.md")
        for path in paths:
            if "archive" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            assert "USER_GUIDE" not in text, path
            assert "pipeline_overview" not in text, path


def test_local_markdown_links_resolve():
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    markdown_files.extend(sorted((ROOT / "stages").glob("*/README.md")))
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    missing = []
    for path in markdown_files:
        for raw_target in link_pattern.findall(path.read_text(encoding="utf-8")):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not missing, "broken local links:\n" + "\n".join(missing)
