# Stage 8: 专项能力

## 评估目标

评估预训练数据在专项能力维度的特性：

- **代码可解析率**：用 tree-sitter 解析代码文档，统计 AST ERROR 节点，评估代码质量
- **STEM 学科分布**：通过关键词密度分类，分析数据的学科覆盖和难度分层

## 子命令

| 子命令 | 功能 | 工具 |
|--------|------|------|
| `parsability` | 代码 AST 解析错误检测 | tree-sitter |
| `stem` | STEM 学科/难度关键词分类 | 内置关键词分类表 |

## parsability

**输出 scores**：`error_node_count, total_node_count, error_ratio`
**输出 flags**：`has_error`（有 ERROR/MISSING 节点）

当前支持语言：Python。扩展方法：安装对应 `tree-sitter-<lang>` 包，在 `utils.py::_get_parser()` 添加注册。

## stem

**输出 scores**：`subject_scores` (各学科关键词密度), `primary_subject`, `difficulty_level`, `word_count`, `stem_keyword_hits`
**输出 flags**：`is_stem`（关键词密度超阈值）

学科类别：cs / math / physics / chemistry / biology / engineering / medicine
难度级别：basic / intermediate / advanced / unknown

## 依赖

- `tree-sitter>=0.21`（已在 pyproject.toml）
- `tree-sitter-python>=0.21`（已在 pyproject.toml）

## 运行示例

```bash
# parsability
PYTHONPATH=. python stages/domain/run.py parsability \
  --input data/mock.jsonl --dataset mock --input-format jsonl

# stem
PYTHONPATH=. python stages/domain/run.py stem \
  --input data/mock.jsonl --dataset mock --input-format jsonl
```
