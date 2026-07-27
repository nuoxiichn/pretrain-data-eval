import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_retired_methods_have_archive_but_no_active_entrypoint() -> None:
    registry = yaml.safe_load((ROOT / "docs/method_registry.yaml").read_text(encoding="utf-8"))
    assert registry["policy"]["stages"] == "validated_and_supported_only"
    for method in registry["methods"].values():
        assert method["active_entrypoint"] is None
        assert (ROOT / method["archive"]).is_file()
        if research_path := method.get("reusable_research"):
            assert (ROOT / research_path).is_dir()


def test_retired_training_proxy_stage_directories_are_absent() -> None:
    assert not (ROOT / "stages/trainability").exists()
    assert not (ROOT / "stages/datadecide").exists()
    assert not (ROOT / "stages/production_alignment").exists()


def test_frozen_training_proxy_protocols_match_retention_index() -> None:
    archive = ROOT / "docs/archive/training_proxies"
    retention = yaml.safe_load((archive / "artifact_retention.yaml").read_text(encoding="utf-8"))
    assert retention["destructive_cleanup_executed"] is False

    protocols = archive / "protocols"
    for filename, expected_sha256 in retention["frozen_protocol_sha256"].items():
        matches = list(protocols.rglob(filename))
        assert len(matches) == 1, filename
        actual_sha256 = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        assert actual_sha256 == expected_sha256, filename
