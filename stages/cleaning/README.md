# Stage 3：抽取质量 + 语言识别 + 多语言覆盖 + 文本质量

## 评估目标

对每条文档做抽取残留检测、语言识别（粗+细+交叉核对）与质量信号检测，输出问题文档列表供人工决策。

## 子命令

| 子命令 | 对应行 | 工具 | 说明 |
|--------|--------|------|------|
| `extraction` | 行 1 | 自实现（regex + 词表） | 抽取质量审计：检测已清洗文本里的 HTML/markup/boilerplate/mojibake 残留 |
| `langid` | 行 2 | fastText lid.176 | 176 种语言识别（粗）；语种标签 + 置信度 + 声明语种一致性 |
| `glotlid` | 行 3 | GlotLID v3（fastText） | 2000+ 语种-脚本组合（细）；低资源覆盖；`ISO3_Script` 标签 |
| `langcross` | 行 2↔3 | 自实现（langcodes） | 交叉核对粗细识别，暴露被 lid.176 吸收的低资源语种/方言 |
| `quality` | 行 4 | Gopher Quality + C4（中英双语：nltk + jieba 分词） | 只读：复刻 DataTrove 规则判每条文档是否通过及原因。**重复检测不在此，归 stage 4** |

> **行 1 重定位**：pipeline_overview 原写 Trafilatura HTML/PDF 正文抽取，针对 raw HTML。
> `extraction` 改为「抽取质量审计」——检测残留在成品里的杂质。不依赖 Trafilatura。
>
> **langid vs glotlid 不是二选一**：lid.176（粗、快）出主语种分布 + 声明一致性核查；
> GlotLID（细、2000+ 类）抓 lid.176 覆盖不到的低资源语种 + 脚本。两者机制同源
> （都是 fastText），但**不能用「langid 低置信再升级 glotlid」门控**——粗模型会
> 自信地把方言/低资源语种吸收进大语言、**不报歧义**，门控会系统性漏掉。正确做法是
> 两个都跑、用 `langcross` 把**分歧**当信号（尤其 glotlid 识出 lid.176 标签集里
> 根本没有的语种 = 必然被吸收）。

## 输入

标准 JSONL / Parquet，通过 `src/reader.py` 读取。

## 输出格式

### extraction
```json
{
  "scores": {"char_count": 1840, "html_tag_count": 0, "html_entity_count": 0,
             "markdown_artifact_count": 0, "url_count": 1, "boilerplate_count": 1, "mojibake_count": 0},
  "flags": {"has_html_residue": false, "has_boilerplate": true, "has_mojibake": false,
            "too_short_stub": false, "low_extraction_quality": true}
}
```

- `low_extraction_quality`：汇总红灯 —— HTML 残留 / mojibake / boilerplate / 残桩任一命中
- `summary` 含各残留率 + `boilerplate_phrase_distribution`（命中的样板短语 top20）
- 残桩判定按**字符数**（`short_stub_chars`，默认 50），对 CJK 比词数公平

### langid
```json
{
  "scores": {"lang": "en", "confidence": 0.997, "top3": [["en", 0.997], ["de", 0.001], ["fr", 0.001]]},
  "flags": {"lang_mismatch": false, "low_confidence": false}
}
```

- `lang_mismatch`：识别结果与文档 `language` 字段不一致
- `low_confidence`：置信度低于阈值（默认 0.7）

### glotlid
```json
{
  "scores": {"lang_script": "cmn_Hani", "confidence": 0.99, "top3": [["cmn_Hani", 0.99], ["jpn_Jpan", 0.002]]},
  "flags": {"low_confidence": false}
}
```

- 标签格式 `ISO3_Script`；`summary.lang_script_distribution` 给语种-脚本分布 top50

### langcross
```json
{
  "scores": {"langid_lang": "ar", "langid_conf": 0.91,
             "glotlid_lang_script": "arz_Arab", "glotlid_conf": 0.95},
  "flags": {"agree": false, "disagreement": true, "possibly_absorbed": true, "low_confidence": false}
}
```

- `agree`：两模型宏语言层一致（如 langid `zh` ↔ glotlid `cmn_Hani` → 都归 `zho`，判一致）
- `disagreement`：两者均高置信但宏语言不一致
- `possibly_absorbed`：分歧且 glotlid 宏语言**不在 lid.176 可表达集** → 粗模型结构性
  无法输出该语种 → langid 的标签必然是吸收（强信号，正是低资源审计要找的）
- `summary` 含 `possibly_absorbed_count`、`absorbed_lang_distribution`、`top_disagreement_pairs`

### quality
```json
{
  "scores": {"gopher_quality": true, "c4_quality": false, "c4_quality_reason": "too_few_sentences"},
  "flags": {"any_filter_failed": true}
}
```

- 复刻 DataTrove `GopherQualityFilter` + `C4QualityFilter` 全套规则，用 nltk(punkt) 真分词
- 中英双语适配：CJK 语料用 jieba 分词（解决逐字切分导致的平均词长/行级过滤误判），阈值按语言自动调整（平均词长下限 1.3 / 每行最少 2 词 / 中文停用词集）
- Gopher Quality 检查：词数上下限、平均词长、符号-词比(`#`/省略号)、bullet/省略号行比例、非字母词比(≥80% 词含字母)、停用词数(≥2)
- C4 逐行过滤：超长词行 / 无终止标点行 / 词数不足行 / javascript 行 / policy 行(cookies/terms/使用条款/隐私政策)剔除；引文剥除；含 lorem ipsum 或花括号直接判废；留存行句子数 <5 判废
- `summary.filter_fail_reasons` 给每个 filter 的失败原因分布

## 依赖

```
fasttext>=0.9.2        # langid (lid.176) + glotlid (GlotLID v3)
langcodes>=3.3         # langcross：ISO2/ISO3 + 宏语言归一（prefer_macrolanguage）
nltk>=3.8              # quality：Gopher/C4 真分词器（punkt）
jieba>=0.42            # quality：CJK 中文分词（Gopher/C4 中文适配）
```

> nltk 首次使用需 punkt 数据：`python -c "import nltk; nltk.download('punkt')"`（本环境已就绪）。

模型（共享区，无需下载）：
- `langid`：`/mnt/public/model/lid.176.bin`
- `glotlid`：`/mnt/public/model/glotlid/model.bin`（GlotLID v3，含 v1/v2/v3 三版）

路径在 `configs/stage3.yaml` 配置（`langid.model_path` / `glotlid.model_path`）。

## 运行示例

```bash
# mock 数据（任意子命令均带 --config）
PYTHONPATH=. python stages/cleaning/run.py extraction \
  --input data/mock.jsonl --dataset mock --input-format jsonl --config configs/stage3.yaml

PYTHONPATH=. python stages/cleaning/run.py glotlid \
  --input data/mock.jsonl --dataset mock --input-format jsonl --config configs/stage3.yaml

PYTHONPATH=. python stages/cleaning/run.py langcross \
  --input data/mock.jsonl --dataset mock --input-format jsonl --config configs/stage3.yaml

# UFW-L3 中文（细粒度语种 + 交叉核对）
PYTHONPATH=. python stages/cleaning/run.py langcross \
  --input /mnt/public/data/Ultra-FineWeb-L3/data/ultrafineweb_zh_l3/multi_style/part-00000-*.snappy.parquet \
  --dataset ufw_zh_l3 --config configs/stage3.yaml --max-docs 500
```
