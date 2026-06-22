"""
生成用于开发调试的 mock 数据集，输出到 data/mock.jsonl。

覆盖场景：
  - 多语种（en/zh/fr/de）
  - 多来源（web/code/arxiv/wikipedia/books）
  - 带 PII（邮箱、电话、身份证）
  - 含 Secret（API Key、密码硬编码）
  - 精确重复（10 对）
  - 近重复（改写变体，10 组）
  - 缺失可选字段（无 timestamp / 无 url）
  - 低质量文本（乱码、SEO 垃圾、boilerplate）
  - 代码文档（Python / JavaScript）
  - 学术文本（STEM，含 arXiv 分类号）
  - 疑似 AI 生成文本（模板化、重复句式）

用法：
  python scripts/gen_mock_data.py [--output data/mock.jsonl] [--n 300]
"""

import argparse
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

random.seed(42)

# ── 基础素材 ──────────────────────────────────────────────────────────────────

EN_PARAGRAPHS = [
    "The transformer architecture has fundamentally changed natural language processing. "
    "Attention mechanisms allow models to capture long-range dependencies efficiently.",

    "Pretraining on large corpora followed by task-specific fine-tuning has become "
    "the dominant paradigm in modern NLP research.",

    "Data quality is often more important than data quantity in training large language models. "
    "Careful curation can lead to significant improvements in downstream performance.",

    "The scaling laws proposed by Kaplan et al. suggest that model performance improves "
    "predictably with increases in compute, data, and parameters.",

    "Deduplication of training data is a critical preprocessing step. Duplicate documents "
    "can cause models to memorize specific texts rather than generalizing.",

    "Benchmark contamination occurs when evaluation data appears in the training corpus, "
    "leading to inflated performance estimates that do not reflect true generalization.",

    "Multilingual models trained on diverse language data exhibit cross-lingual transfer "
    "capabilities, performing well on low-resource languages without direct supervision.",

    "The educational value of a document is strongly correlated with its utility as "
    "training data for general-purpose language models.",
]

ZH_PARAGRAPHS = [
    "大规模语言模型的预训练数据质量对最终模型性能有决定性影响。"
    "数据清洗和去重是提升数据质量的关键步骤。",

    "在自然语言处理领域，基于Transformer架构的模型已经成为主流。"
    "自注意力机制使模型能够有效捕捉长距离依赖关系。",

    "多语言预训练模型能够在不同语言之间进行知识迁移，"
    "对低资源语言的理解能力尤为重要。",

    "数据集的去污染是评测中不可忽视的环节，"
    "评测集与训练集的重叠会导致模型性能被高估。",
]

FR_PARAGRAPHS = [
    "Les modèles de langage de grande taille ont révolutionné le traitement automatique "
    "du langage naturel ces dernières années.",

    "La qualité des données d'entraînement est un facteur déterminant pour les performances "
    "des modèles de langage pré-entraînés.",
]

DE_PARAGRAPHS = [
    "Große Sprachmodelle haben die natürliche Sprachverarbeitung grundlegend verändert. "
    "Die Qualität der Trainingsdaten spielt dabei eine entscheidende Rolle.",

    "Die Deduplizierung von Trainingsdaten ist ein wichtiger Vorverarbeitungsschritt, "
    "um Memorisierung zu vermeiden und die Generalisierung zu verbessern.",
]

PYTHON_SNIPPETS = [
    '''\
def compute_perplexity(log_probs):
    """Compute perplexity from log probabilities."""
    import math
    avg_log_prob = sum(log_probs) / len(log_probs)
    return math.exp(-avg_log_prob)
''',
    '''\
class DataLoader:
    def __init__(self, path, batch_size=32):
        self.path = path
        self.batch_size = batch_size

    def __iter__(self):
        with open(self.path) as f:
            batch = []
            for line in f:
                batch.append(json.loads(line))
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
''',
    '''\
import hashlib

def doc_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()
''',
]

JS_SNIPPETS = [
    '''\
async function fetchDocuments(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP error: ${response.status}`);
  return response.json();
}
''',
    '''\
const dedup = (arr) => [...new Set(arr.map(JSON.stringify))].map(JSON.parse);
module.exports = { dedup };
''',
]

ARXIV_ABSTRACTS = [
    "We present a novel approach to pretraining data curation that combines rule-based "
    "filtering with model-based quality scoring. Our method achieves state-of-the-art "
    "results on standard benchmarks while reducing training compute by 30%. "
    "ArXiv:cs.CL/2024.12345",

    "This paper investigates the relationship between data diversity and model generalization. "
    "We find that domain coverage is a stronger predictor of downstream performance than "
    "raw data volume. ArXiv:cs.LG/2024.67890",

    "We propose StemClassifier, a lightweight model for scientific document classification "
    "across 42 STEM disciplines. The model achieves F1=0.91 on our benchmark dataset. "
    "ArXiv:cs.IR/2024.11111",
]

SEO_SPAM = [
    "Buy cheap products online! Best deals guaranteed! Click here now! "
    "Top quality items at lowest prices. Free shipping worldwide. "
    "Order today and get 50% discount. Limited time offer. Best best best.",

    "| Home | About | Contact | Products | Services | Blog | FAQ | Privacy | Terms |"
    "Copyright 2024. All rights reserved. Powered by CMS. Back to top. "
    "| Home | About | Contact | Products | Services | Blog | FAQ | Privacy | Terms |",

    "aaa bbb ccc 123 456 ??? ### @@@ !!! aaa bbb ccc 123 456 ??? ### @@@ !!! "
    "乱码乱码乱码 ▓▓▓▓▓▓ ████ ░░░░ 乱码乱码乱码",
]

AI_GENERATED = [
    "Certainly! Here is a comprehensive overview of the topic you requested. "
    "Firstly, it is important to note that this subject has many facets. "
    "Secondly, we must consider the various perspectives involved. "
    "Thirdly, the implications are far-reaching. In conclusion, as I have outlined above, "
    "this is indeed a multifaceted issue that requires careful consideration.",

    "Great question! Large language models are trained on vast amounts of text data. "
    "This allows them to learn patterns and generate coherent text. "
    "As an AI language model, I can assist you with a wide range of tasks. "
    "Feel free to ask me anything! I am here to help you with your queries.",
]

PYTHON_SYNTAX_ERRORS = [
    '''\
def broken_function(x, y
    return x + y
''',
    '''\
class Foo:
    def bar(self):
        if True
            print("missing colon")
''',
    '''\
for i in range(10)
    x = i ** 2
    print(x
''',
    '''\
import os
def valid_function():
    return 42

def broken():
    x = [1, 2, 3
    return x
''',
]

LATEX_DOCS = [
    r"The quadratic formula is $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$. "
    r"This can be derived from $ax^2 + bx + c = 0$ by completing the square. "
    r"$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$",

    r"Consider the Schrödinger equation: "
    r"$$i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi$$ "
    r"where $\hat{H}$ is the Hamiltonian operator and $\Psi$ the wavefunction.",

    r"\begin{equation} E = mc^2 \end{equation} "
    r"Einstein's mass-energy equivalence relates energy $E$ to mass $m$ "
    r"via the speed of light $c \approx 3 \times 10^8$ m/s. "
    r"\begin{align} F &= ma \\ p &= mv \end{align}",

    r"The Taylor series expansion: $f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$. "
    r"For example, ```python\nimport math\ndef taylor_exp(x, n=10):\n    return sum(x**k / math.factorial(k) for k in range(n))\n``` "
    r"gives an approximation to $e^x$.",
]

SOURCES = ["common_crawl", "github", "arxiv", "wikipedia", "books", "stackexchange"]
DOMAINS = ["en.wikipedia.org", "github.com", "arxiv.org", "stackoverflow.com",
           "reddit.com", "news.ycombinator.com", "medium.com"]

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def make_id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:05d}"

def random_timestamp() -> Optional[str]:
    if random.random() < 0.15:
        return None
    base = datetime(2020, 1, 1)
    delta = timedelta(days=random.randint(0, 1600))
    return (base + delta).strftime("%Y-%m-%dT%H:%M:%SZ")

def random_url(domain: str) -> Optional[str]:
    if random.random() < 0.1:
        return None
    slug = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
    return f"https://{domain}/{slug}"

def make_doc(doc_id, text, source, language, domain=None, extra_meta=None):
    return {
        "doc_id": doc_id,
        "text": text,
        "source": source,
        "url": random_url(domain or random.choice(DOMAINS)),
        "timestamp": random_timestamp(),
        "language": language,
        "meta": extra_meta or {},
    }

# ── 各场景生成器 ──────────────────────────────────────────────────────────────

def gen_normal(counter):
    docs = []
    for para in EN_PARAGRAPHS * 4:
        text = para + " " + random.choice(EN_PARAGRAPHS)
        docs.append(make_doc(make_id("en", counter[0]), text, "common_crawl", "en",
                              domain="en.wikipedia.org"))
        counter[0] += 1
    for para in ZH_PARAGRAPHS * 3:
        text = para + random.choice(ZH_PARAGRAPHS)
        docs.append(make_doc(make_id("zh", counter[0]), text, "common_crawl", "zh",
                              domain="zh.wikipedia.org"))
        counter[0] += 1
    for para in FR_PARAGRAPHS * 2:
        docs.append(make_doc(make_id("fr", counter[0]), para * 2, "common_crawl", "fr"))
        counter[0] += 1
    for para in DE_PARAGRAPHS * 2:
        docs.append(make_doc(make_id("de", counter[0]), para * 2, "common_crawl", "de"))
        counter[0] += 1
    return docs

def gen_pii(counter):
    templates = [
        "Please contact {name} at {email} or call {phone} for more information.",
        "User account: {email}. Chinese ID: {id_cn}. Address: {addr}.",
        "From: {email}\nTo: support@example.com\nMy phone is {phone}.",
    ]
    names = ["Alice Zhang", "Bob Smith", "Carol Wang", "David Lee"]
    emails = ["alice@example.com", "bob.smith@corp.org", "carol_w@mail.net"]
    phones = ["138-0000-1234", "+86-21-1234-5678", "010-8765-4321"]
    id_cns = ["110101199001011234", "310115198506152345"]
    addrs = ["北京市海淀区中关村大街1号", "上海市浦东新区张江高科技园区"]

    docs = []
    for i in range(20):
        tmpl = random.choice(templates)
        text = tmpl.format(
            name=random.choice(names),
            email=random.choice(emails),
            phone=random.choice(phones),
            id_cn=random.choice(id_cns),
            addr=random.choice(addrs),
        )
        text = text + " " + random.choice(EN_PARAGRAPHS)
        docs.append(make_doc(make_id("pii", counter[0]), text, "common_crawl", "en",
                              extra_meta={"has_pii_mock": True}))
        counter[0] += 1
    return docs

def gen_secrets(counter):
    secret_templates = [
        'API_KEY = "sk-abcdef1234567890abcdef1234567890abcdef12"\n'
        'client = OpenAI(api_key=API_KEY)\n',

        'password = "P@ssw0rd!SuperSecret123"\n'
        'db.connect(host="db.internal", user="admin", password=password)\n',

        'private_key = "-----BEGIN RSA PRIVATE KEY-----\n'
        'MIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29D8rVyOGVEU9B\n'
        '-----END RSA PRIVATE KEY-----"\n',

        'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'
        'AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n',
    ]
    docs = []
    for i, tmpl in enumerate(secret_templates):
        text = tmpl + "\n" + random.choice(PYTHON_SNIPPETS)
        docs.append(make_doc(make_id("secret", counter[0]), text, "github", "en",
                              domain="github.com", extra_meta={"has_secret_mock": True}))
        counter[0] += 1
    return docs

def gen_code(counter):
    docs = []
    for snippet in PYTHON_SNIPPETS * 3:
        docs.append(make_doc(make_id("py", counter[0]), snippet, "github", "en",
                              domain="github.com", extra_meta={"lang": "python"}))
        counter[0] += 1
    for snippet in JS_SNIPPETS * 3:
        docs.append(make_doc(make_id("js", counter[0]), snippet, "github", "en",
                              domain="github.com", extra_meta={"lang": "javascript"}))
        counter[0] += 1
    return docs

def gen_exact_duplicates(base_docs, counter, n_pairs=10):
    """从已有文档中复制 n_pairs 对精确重复。"""
    docs = []
    chosen = random.sample(base_docs, min(n_pairs, len(base_docs)))
    for orig in chosen:
        dup = dict(orig)
        dup["doc_id"] = make_id("dup", counter[0])
        dup["url"] = random_url("duplicate.example.com")
        dup["timestamp"] = random_timestamp()
        docs.append(dup)
        counter[0] += 1
    return docs

def gen_near_duplicates(counter, n=10):
    """对同一段落做轻微改写，形成近重复对。"""
    base = (
        "Machine learning models require large amounts of training data to achieve "
        "good performance on downstream tasks."
    )
    variants = [
        base,
        base.replace("Machine learning", "Deep learning"),
        base.replace("large amounts of", "vast quantities of"),
        base.replace("training data", "labeled examples"),
        base.replace("good performance", "strong results"),
        base.replace("downstream tasks", "evaluation benchmarks"),
        base.replace("Machine learning models require", "Neural networks need"),
        base.replace("to achieve", "in order to achieve"),
        "Models for machine learning require large amounts of training data "
        "to achieve good performance on downstream tasks.",
        "To achieve good performance on downstream tasks, machine learning models "
        "require large amounts of training data.",
    ]
    docs = []
    for i, v in enumerate(variants[:n]):
        text = v + " " + random.choice(EN_PARAGRAPHS)
        docs.append(make_doc(make_id("near", counter[0]), text, "common_crawl", "en",
                              extra_meta={"near_dup_group": "ndg_0"}))
        counter[0] += 1
    return docs

def gen_low_quality(counter):
    docs = []
    for spam in SEO_SPAM * 3:
        docs.append(make_doc(make_id("lq", counter[0]), spam, "common_crawl", "en",
                              extra_meta={"quality_mock": "low"}))
        counter[0] += 1
    return docs

def gen_ai_generated(counter):
    docs = []
    for text in AI_GENERATED * 5:
        docs.append(make_doc(make_id("ai", counter[0]), text, "common_crawl", "en",
                              extra_meta={"ai_generated_mock": True}))
        counter[0] += 1
    return docs

def gen_arxiv(counter):
    docs = []
    for abstract in ARXIV_ABSTRACTS * 3:
        docs.append(make_doc(make_id("arxiv", counter[0]), abstract, "arxiv", "en",
                              domain="arxiv.org",
                              extra_meta={"arxiv_cat": random.choice(
                                  ["cs.CL", "cs.LG", "cs.IR", "math.ST", "physics.comp-ph"]
                              )}))
        counter[0] += 1
    return docs

def gen_missing_fields(counter, n=15):
    """部分字段缺失的文档（测试 Stage 1 字段完整性检查）。"""
    docs = []
    for i in range(n):
        text = random.choice(EN_PARAGRAPHS) + " " + random.choice(EN_PARAGRAPHS)
        doc = make_doc(make_id("miss", counter[0]), text, "common_crawl", "en")
        if i % 3 == 0:
            doc["timestamp"] = None
        if i % 3 == 1:
            doc["url"] = None
        if i % 3 == 2:
            doc["language"] = None
        docs.append(doc)
        counter[0] += 1
    return docs


def gen_code_with_errors(counter):
    """有语法错误的 Python 代码（Stage 8 parsability 测试）。"""
    docs = []
    for snippet in PYTHON_SYNTAX_ERRORS * 2:
        docs.append(make_doc(make_id("pyerr", counter[0]), snippet, "github", "en",
                              domain="github.com",
                              extra_meta={"lang": "python", "has_syntax_error": True}))
        counter[0] += 1
    return docs


def gen_latex_docs(counter):
    """含 LaTeX 公式的学术文档（Stage 10 tokenization 测试）。"""
    docs = []
    for text in LATEX_DOCS * 2:
        docs.append(make_doc(make_id("latex", counter[0]), text, "arxiv", "en",
                              domain="arxiv.org",
                              extra_meta={"has_latex": True}))
        counter[0] += 1
    return docs


def gen_megatron_config(output_dir: Path):
    """生成 mock Megatron-LM 训练配置文件（Stage 9 config-audit 测试）。"""
    config = {
        "model": {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_layers": 32,
            "seq_length": 4096,
        },
        "training": {
            "micro_batch_size": 4,
            "global_batch_size": 256,
            "lr": 3e-4,
            "min_lr": 3e-5,
            "weight_decay": 0.1,
        },
        "data": {
            "data_path": "/data/tokenized/train",
            "split": "98,2,0",
        },
        "reset_position_ids": True,
        "reset_attention_mask": True,
        "eod_mask_loss": True,
    }
    config_path = output_dir / "mock_megatron_config.yaml"
    import yaml
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"生成 Megatron 配置 -> {config_path}")

    # 生成一个缺失参数的配置（FAIL case）
    config_fail = dict(config)
    del config_fail["eod_mask_loss"]
    config_fail["reset_attention_mask"] = False
    fail_path = output_dir / "mock_megatron_config_fail.yaml"
    with fail_path.open("w", encoding="utf-8") as f:
        yaml.dump(config_fail, f, default_flow_style=False)
    print(f"生成 Megatron 配置 (FAIL) -> {fail_path}")

# ── 主入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/mock.jsonl")
    parser.add_argument("--n", type=int, default=300,
                        help="目标文档数（实际生成数可能略有差异）")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counter = [0]

    normal_docs = gen_normal(counter)
    all_docs = (
        normal_docs
        + gen_pii(counter)
        + gen_secrets(counter)
        + gen_code(counter)
        + gen_code_with_errors(counter)
        + gen_exact_duplicates(normal_docs, counter)
        + gen_near_duplicates(counter)
        + gen_low_quality(counter)
        + gen_ai_generated(counter)
        + gen_arxiv(counter)
        + gen_latex_docs(counter)
        + gen_missing_fields(counter)
    )

    random.shuffle(all_docs)

    with out_path.open("w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    gen_megatron_config(out_path.parent)

    print(f"生成 {len(all_docs)} 条文档 -> {out_path}")
    print(f"字段覆盖：doc_id / text / source / url / timestamp / language / meta")

if __name__ == "__main__":
    main()
