# Training Proxy Archive

This directory is the canonical historical record for the retired Stage 11-13 training-proxy
work. Archived commands and configuration paths are evidence, not supported entrypoints.

| Experiment | Final decision | Retained record |
|---|---|---|
| Stage 11 trainability | `production_no_go` | [Final report](stage11_final_report.md) |
| Stage 12 DataDecide | `capability_specific_early_screen_only` | [Final report](stage12_datadecide_report.md) and [protocol](stage12_benchmark_proxy_protocol.md) |
| Stage 13 Qwen-like screen | `insufficient_evidence_screen_only` | [Final report](stage13_qwen_screen_report.md) and [protocol](stage13_production_alignment_protocol.md) |

Stage 11's anchor-relative gain, balanced-pool gain, and cross-corpus conditioning methods were
rejected. Stage 12 retained limited evidence for continuous likelihood as a calibrated knowledge
screen, but did not establish a general quality score. Stage 13 completed all eight runs but did
not satisfy the preregistered transfer gate.

Exact frozen YAML files remain under `protocols/`. Generic pairwise statistics, multiple-choice
evaluation, deterministic token streams, and run recording were extracted to
`research/data_advisor/`; the original experiment CLIs were retired.

Ignored outputs remain in place pending an explicit destructive-cleanup approval. Their current
sizes, checksums, and proposed disposition are recorded in [artifact retention](artifact_retention.yaml).
