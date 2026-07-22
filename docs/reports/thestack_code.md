# The Stack 代码能力验证

本验证使用本地 The Stack dedup 代码样本检查代码相关子命令能否在真实代码结构上运行并
产生可解释候选。它不是对 The Stack 的完整质量审计，也没有建立总体估计。

| 维度 | 样本 | 观察 |
|---|---:|---|
| code-PII | Python 500 | 89.8% 文档有候选，多数来自注释/文档中的 URL；敏感候选含 email 39、IP 162 |
| Gitleaks | Python 500 | 0.4% 命中，规则以 generic/gcp-api-key 为主，未验证凭据真伪 |
| code-near | Python 2,000 对 EvalPlus 542 条 | MinHash 近重复命中 0% |
| code-AST | Python 2,000 对 EvalPlus | AST 候选 0.2%，精确指纹 0；短函数存在假阳 |
| parsability | 多语言各 500 | Go 0%、Ruby 0.4%、Python 1.2%、Rust 2.2%、JS 3.0%、C# 3.4%、C 62% AST error 文档 |

`code-PII` 的高命中主要反映 URL recognizer 对代码注释和文档的覆盖，不能解释为高隐私
泄漏率。C 语言 parsability 的异常高 error 率主要受预处理器、宏和代码片段完整性影响，
说明 tree-sitter error 率不能跨 grammar 直接比较。

零污染命中只覆盖当前小样本、EvalPlus 注册集和固定阈值。该验证支持“子命令可在真实代码
上工作并暴露典型混淆因素”，不支持“The Stack 无污染”或“代码质量合格”。
