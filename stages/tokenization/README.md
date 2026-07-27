# Stage 10: Tokenization 分析

## 评估目标

分析预训练数据在目标 tokenizer 下的分词特性：

- **Token/char 比（fertility）**：衡量 tokenizer 对不同语种/内容的效率
- **UNK 率**：检测 tokenizer 词表覆盖不足的内容
- **语种 fertility 对比**：不同语言的分词效率差异（CJK 通常 fertility 更高）
- **代码/LaTeX token 膨胀率**：专有内容的 token 膨胀程度

## 子命令

| 子命令 | 功能 | 输出 |
|--------|------|------|
| `tokenize` | 全维度 tokenization 分析 | per_doc.jsonl + summary.json |

## 输入输出

**输入**：标准 Document（通过 `src/reader.py`）

**Per-doc scores**：
- `token_count` — 文档 token 数
- `char_count` — 文档字符数
- `fertility` — token_count / char_count
- `unk_count` — UNK token 数
- `unk_rate` — unk_count / token_count
- `code_char_count/code_token_count/code_fertility` — 代码块分词统计
- `latex_char_count/latex_token_count/latex_fertility` — LaTeX 块分词统计

**Per-doc flags**：
- `high_unk_rate` — UNK 率超阈值（默认 0.01）
- `high_fertility` — fertility 超阈值（默认 5.0）

**Summary**：fertility 分布统计、UNK 统计、per-language fertility、代码/LaTeX 膨胀率

## 依赖

- `tokenizers>=0.19`（已在 pyproject.toml）
- 需要本地 tokenizer 模型（如 `/mnt/public/model/huggingface/Llama-2-7b-hf`）

## 运行示例

```bash
# Mock 数据快速验证
PYTHONPATH=. python stages/tokenization/run.py tokenize \
  --input data/mock.jsonl --dataset mock --input-format jsonl \
  --tokenizer /mnt/public/model/huggingface/Llama-2-7b-hf --max-docs 50

# 真实数据（带限量）
PYTHONPATH=. python stages/tokenization/run.py tokenize \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_en_l3/multi_style/ \
  --dataset ufw_en_l3 --config configs/stage10.yaml --max-docs 500
```
