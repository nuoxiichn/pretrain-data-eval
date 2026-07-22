# Stage 7: 合成数据检测

只读审计阶段，检测数据集中 AI 生成文本的比例。

## 子命令

| 子命令 | 说明 | 工具 | 依赖 |
|--------|------|------|------|
| `binoculars` | AI 生成文本检测（observer/performer 双模型对比打分） | 自实现（Binoculars, ICML 2024） | torch, transformers, GPU（两个 7B fp16 ≈ 28 GB） |

安装推理依赖：`python -m pip install -e '.[gpu]'`。

## 原理

Binoculars 用两个共享 tokenizer 的语言模型：
- **Observer**（base 模型）：计算文本的 next-token NLL
- **Performer**（instruct 模型）：提供参考分布

分数 = observer NLL / cross-entropy(performer_softmax, observer_log_softmax)

- **人类文本**：两模型差异大 → 分数高
- **AI 生成文本**：两模型行为趋同 → 分数低（接近 1）

分数 < 阈值 → 判定为 AI 生成。

## 模型对选择

| 模型对 | 适用语种 | 路径 |
|--------|----------|------|
| Llama-2-7b-hf + Llama-2-7b-chat-hf | 英文 | `/mnt/public/model/huggingface/Llama-2-*` |
| Qwen2.5-7B + Qwen2.5-7B-Instruct | 中英文 | `/mnt/public/model/huggingface/Qwen2.5-7B*` |

阈值 0.8536 来自原论文 Falcon 对（FPR ≤ 0.01%），换模型后需通过分数分布重新校准。

## 输入 / 输出

- **输入**：标准 `read_documents` 接口（Parquet/JSONL）
- **输出**：
  - `per_doc.jsonl`：公共 schema 头 + `doc_id` + `scores.binoculars` + `flags.is_ai_generated`
  - `summary.json`：AI 生成文档数/比例 + 分数分布统计（mean/std/分位数）

## 运行示例

```bash
# Mock 数据快速验证
PYTHONPATH=. python stages/synthetic/run.py binoculars \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --observer-model /mnt/public/model/huggingface/Llama-2-7b-hf \
  --performer-model /mnt/public/model/huggingface/Llama-2-7b-chat-hf

# 真实英文样本（限量）
PYTHONPATH=. python stages/synthetic/run.py binoculars \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style/ \
  --dataset ufw_en_l3 --config configs/stage7.yaml --max-docs 500

# 中文样本（切换 Qwen 模型对）
PYTHONPATH=. python stages/synthetic/run.py binoculars \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3/multi_style/ \
  --dataset ufw_zh_l3 --config configs/stage7.yaml \
  --observer-model /mnt/public/model/huggingface/Qwen2.5-7B \
  --performer-model /mnt/public/model/huggingface/Qwen2.5-7B-Instruct \
  --max-docs 500
```

## 注意事项

- GPU 必须能同时放下两个模型（7B fp16 ≈ 14 GB × 2 = 28 GB）
- `max_length` 默认 512 token，长文档只取前段；如需全文可调大但会增加显存和耗时
- Llama-2 的 tokenizer 对 CJK 支持弱，中文数据务必切 Qwen 对
- 阈值需要校准：先跑小样本看分数分布，再定阈值

## 性能参考

| 配置 | 数据量 | 耗时 | 吞吐 | 硬件 |
|------|--------|------|------|------|
| Llama-2-7b pair, batch_size=8, max_length=512, fp16 | 5000 docs (en) | ~8 min（含模型加载） | ~10 docs/s | MetaX C500 64 GB |

`summary.json` 中 `elapsed_seconds` / `docs_per_second` 字段记录实际耗时。模型加载约 2–3s，推理主导总耗时。
