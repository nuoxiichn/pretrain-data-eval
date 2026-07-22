from __future__ import annotations

import json

from click.testing import CliRunner

from pretrain_data_eval.schema import write_summary
from scripts import aggregate_batch
from stages.cleaning.run import cli as cleaning_cli
from stages.contamination.run import cli as contamination_cli
from stages.dedup.run import cli as dedup_cli
from stages.longctx.run import cli as longctx_cli
from stages.source_audit.run import cli as source_cli


def _write_input(path):
    rows = [
        {"doc_id": "a", "text": "What is two plus two?"},
        {"doc_id": "b", "text": "What is two plus two?"},
        {"doc_id": "c", "text": "<div>x</div><p>y</p><footer>z</footer>"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _assert_contract(output, schema_validator):
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    per_doc = [
        json.loads(line)
        for line in (output / "per_doc.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    schema_validator("summary.schema.json").validate(summary)
    for result in per_doc:
        schema_validator("per_doc.schema.json").validate(result)
    return summary, per_doc


def test_source_cleaning_and_dedup_clis(tmp_path, schema_validator):
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    config = tmp_path / "config.yaml"
    config.write_text(
        "input:\n  format: jsonl\n"
        "tokenizer:\n  backend: words\n"
        "stats:\n  percentiles: [50, 95]\n"
        "extraction:\n  short_stub_chars: 10\n"
        "exact:\n  min_para_chars: 10\n",
        encoding="utf-8",
    )
    runner = CliRunner()

    source_out = tmp_path / "source"
    result = runner.invoke(
        source_cli,
        ["stats", "--input", str(input_path), "--dataset", "test", "--config", str(config),
         "--output-dir", str(source_out)],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(source_out, schema_validator)
    assert summary["total_docs"] == 3

    cleaning_out = tmp_path / "cleaning"
    result = runner.invoke(
        cleaning_cli,
        ["extraction", "--input", str(input_path), "--dataset", "test",
         "--config", str(config), "--output-dir", str(cleaning_out)],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(cleaning_out, schema_validator)
    assert summary["html_residue_docs"] == 1

    dedup_out = tmp_path / "dedup"
    result = runner.invoke(
        dedup_cli,
        ["exact", "--input", str(input_path), "--dataset", "test", "--config", str(config),
         "--output-dir", str(dedup_out)],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(dedup_out, schema_validator)
    assert summary["exact_dup_docs"] == 2

    repetition_out = tmp_path / "repetition"
    result = runner.invoke(
        dedup_cli,
        [
            "repetition",
            "--input",
            str(input_path),
            "--dataset",
            "test",
            "--config",
            str(config),
            "--output-dir",
            str(repetition_out),
        ],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(repetition_out, schema_validator)
    assert summary["method"] == "gopher_repetition_v1"


def test_contamination_and_config_audit_clis(tmp_path, schema_validator):
    input_path = tmp_path / "input.jsonl"
    _write_input(input_path)
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(json.dumps({"question": "What is two plus two?"}) + "\n", encoding="utf-8")
    config = tmp_path / "contamination.yaml"
    config.write_text(
        "input:\n  format: jsonl\n"
        "benchmarks:\n  datasets:\n"
        f"    - {{path: {benchmark}, text_field: question, label: toy}}\n"
        "exact:\n  min_para_chars: 10\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    contamination_out = tmp_path / "contamination"
    result = runner.invoke(
        contamination_cli,
        ["exact", "--input", str(input_path), "--dataset", "test", "--config", str(config),
         "--output-dir", str(contamination_out)],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(contamination_out, schema_validator)
    assert summary["contaminated_docs"] == 2

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        "reset_position_ids: true\nreset_attention_mask: true\neod_mask_loss: true\n",
        encoding="utf-8",
    )
    longctx_config = tmp_path / "stage9.yaml"
    longctx_config.write_text("output:\n  base_dir: ignored\n", encoding="utf-8")
    audit_out = tmp_path / "audit"
    result = runner.invoke(
        longctx_cli,
        ["config-audit", "--config-file", str(train_config), "--dataset", "test",
         "--config", str(longctx_config), "--output-dir", str(audit_out)],
    )
    assert result.exit_code == 0, result.output
    summary, _ = _assert_contract(audit_out, schema_validator)
    assert summary["config_valid"] is True


def test_batch_aggregation_preserves_summary_contract(tmp_path, schema_validator, monkeypatch):
    base = tmp_path / "stage4" / "dataset" / "exact"
    first = base / "part-a"
    second = base / "part-b"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    write_summary(
        {
            "total_docs": 2,
            "exact_dup_docs": 0,
            "para_dup_docs": 0,
            "unique_doc_hashes": 2,
            "unique_para_hashes": 2,
        },
        first,
    )
    write_summary(
        {
            "total_docs": 3,
            "exact_dup_docs": 1,
            "para_dup_docs": 1,
            "unique_doc_hashes": 2,
            "unique_para_hashes": 2,
        },
        second,
    )
    monkeypatch.setattr(aggregate_batch.sys, "argv", ["aggregate_batch.py", str(base)])
    aggregate_batch.main()
    aggregated = json.loads((base / "aggregated_summary.json").read_text(encoding="utf-8"))
    schema_validator("summary.schema.json").validate(aggregated)
    assert aggregated["total_docs"] == 5
    assert aggregated["exact_dup_docs"] == 1
