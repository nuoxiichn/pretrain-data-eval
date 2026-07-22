# Stage 12: DataDecide 小模型 benchmark 复现

本 Stage 验证一个窄而明确的命题：候选预训练 recipe 在同配置小模型上的能力排序，能否预测
官方 1B/100B-token 模型的排序。它与 Stage 11 的跨 corpus loss conditioning 不同，直接运行
benchmark，但只把结果解释为对应能力的早期代理，不输出“数据总体质量分”。

方法来源：[DataDecide](https://arxiv.org/abs/2504.11393)、
[官方代码与模型](https://github.com/allenai/DataDecide)、
[公开评测矩阵](https://huggingface.co/datasets/allenai/DataDecide-eval-results)和
[OLMES](https://arxiv.org/abs/2406.08446)。预注册配置见
[`configs/stage12_datadecide.yaml`](../../configs/stage12_datadecide.yaml)。

## 选择的实验组

共 8 个 recipe、3 个家族：

- Falcon+CC：raw、复现质量分类器 QC20、原始质量分类器 Orig10；
- DCLM：raw、QC7+FineWeb-Edu 2 分、QC7+FineWeb-Edu 3 分；
- Dolma：raw、no-Flan source ablation。

五条目标边都要求官方 1B 三 seed 同向。其中 `DCLM raw > FW3` 是强制 crossover 反例：官方
20M/60M/150M 常预测反方向，防止只挑容易成功的 pair。主决策任务是 ARC-Easy、
ARC-Challenge、MMLU 的字符归一化连续答案似然；HellaSwag 单列，BoolQ 不进入决策分数。

## 1. 复算官方矩阵

```bash
PYTHONPATH=. python stages/datadecide/run.py official-matrix \
  --config configs/stage12_datadecide.yaml \
  --output-dir outputs/datadecide_reproduction/official_matrix
```

这一步验证筛选和统计代码，只是官方证据重算，不算本地训练复现。

## 2. OLMES 隔离环境

公开 OLMES 必须与仓库当前 Transformers 5 环境隔离，同时复用机器的 MetaX PyTorch：

```bash
python -m venv --system-site-packages /root/.venvs/datadecide-olmes
git clone https://github.com/allenai/olmes.git /root/src/olmes-datadecide
git -C /root/src/olmes-datadecide checkout 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e
/root/.venvs/datadecide-olmes/bin/pip install --no-deps -e /root/src/olmes-datadecide
/root/.venvs/datadecide-olmes/bin/pip install -r envs/olmes.txt
```

`pytrec_eval` 的 sdist 会在安装时访问 GitHub；若网络超时，需先把 trec_eval v9.0.8 源码放进
其 `trec_eval/` 目录后本地构建。不得让 pip 替换 `torch==2.8.0+metax...`。

发布权重的 evaluator golden test：

```bash
PATH=/root/.venvs/datadecide-olmes/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=1 \
olmes --model allenai/DataDecide-falcon-and-cc-20M \
  --task arc_easy:rc::olmes --limit 5 --batch-size 4 \
  --output-dir outputs/datadecide_reproduction/smoke
```

正式运行把配置中的十个任务全部传给 `--task`。每个 recipe/scale/seed 使用独立目录，随后聚合：

```bash
PYTHONPATH=. python stages/datadecide/run.py aggregate-olmes \
  --config configs/stage12_datadecide.yaml \
  --run falcon_raw:20M:default=/path/to/olmes-output \
  --run falcon_qc20:20M:default=/path/to/olmes-output \
  --output-dir outputs/datadecide_reproduction/local_olmes
```

聚合器从 predictions 重建 `correct_prob_per_char` 和 `total_prob_per_char`，MMLU 先在学科间
宏平均，再和其他任务组合。本地 OLMES-10 accuracy 用来预测官方 1B OLMES-10 accuracy；
ARC-Easy、ARC-Challenge、MMLU 的连续似然代理按 DataDecide 定义预测 1B 上相同三项的
accuracy，而不是预测 1B 连续似然。原始 OLMES 目录必须保留，不能只保存宏平均。
多个训练 seed 同时传入时，汇总还会输出逐 seed 排序、seed 均值排序和
`unanimous_seed_decisions`。后者仅在所有 seed 给出相同非零方向时才作决定，其余 pair 弃权；
它是保守的运行规则，不是三次训练即可成立的统计显著性检验。

## 3. 本地训练最小矩阵

本地训练只晋级 Falcon 三 recipe。它提供三条 pair：QC20 > raw、Orig10 > raw 是预期成功案例，
Orig10 > QC20 是官方 20M 均值失败但 1B 成立的边界案例。模型为官方 20M 架构：16 层、
`d_model=192`、SwiGLU、affine RMSNorm、RoPE、untied output，共 28,760,256 参数；训练 14,584
step、global batch 64、context 2048，即 1,911,554,048 token。

先取得固定 OLMo DataDecide 分支的 `olmo/data/named_data_mixes.py`，再跨完整 path list 物化代表性
样本：

```bash
PYTHONPATH=. python stages/datadecide/run.py prepare-recipe \
  --config configs/stage12_datadecide.yaml \
  --recipe falcon_raw \
  --named-mixes-source /path/to/OLMo/olmo/data/named_data_mixes.py \
  --output data/trainability/datadecide_local/falcon_raw.npy
```

每个输出约 3.56 GiB。文件虽沿用官方 `.npy` 后缀，实际是可直接 memmap 的无 header
little-endian `uint16` token 流，不能用 `np.load` 读取。manifest 记录格式、官方源文件 SHA、
所有 HTTP byte range 和最终 SHA。该抽样保持官方多来源构成，但不是作者训练时的精确 shuffle，
因此结论属于本地 recipe 复现，不是逐 bit 训练复现。

```bash
PYTHONPATH=. python stages/datadecide/run.py train-20m \
  --config configs/stage12_datadecide.yaml \
  --recipe falcon_raw \
  --tokens data/trainability/datadecide_local/falcon_raw.npy \
  --seed 6198 --device cuda:1 --micro-batch-size 32 \
  --output-dir outputs/datadecide_reproduction/local_train/falcon_raw_seed6198
```

`micro-batch-size 32` 是 MXC500-64G 的实测上限，约占 61GB；64 会 OOM，其他硬件需要重新探测。
梯度累积保持 global batch 64 不变。训练每 2,500 step 保存可恢复的 trainer state，最终
`hf-final/` 可直接传给 OLMES。先完成三 recipe 的 seed 6198；只有训练曲线正常且 benchmark
方向有信号时再补官方 small-aux seed 14/15。不要把单 seed 文档或题目 bootstrap 当作训练 seed
稳定性。

若要验证更短周期，在下一次 trainer state 被覆盖前冻结中间点：

```bash
PYTHONPATH=. python stages/datadecide/run.py export-state \
  --config configs/stage12_datadecide.yaml \
  --trainer-state outputs/datadecide_reproduction/local_train/falcon_raw_seed6198/trainer-state.pt \
  --output-dir outputs/datadecide_reproduction/local_train/falcon_raw_seed6198/hf-step2500
```

中间点必须作为单独 run 传给 OLMES；不能把同一训练 run 的多个 checkpoint 当成独立 seed。

## 判定

- 主要指标：五条预注册边的方向准确率，以及所有入选 recipe 的 pairwise accuracy；
- 次要指标：按任务的排序、跨 seed 一致性、连续似然 margin；
- 成功不要求 100%，但必须显著优于随机且在历史 recipe 回测中稳定；
- 代码、长程和通用知识数据必须分别校准 benchmark，本 Stage 的 OLMES 结果只覆盖通用知识与
  常识推理，不能给代码或长上下文数据排总体质量。

## 本次复现实测

Falcon raw/QC20/Orig10 已完成 seed 6198/14/15 的本地训练和 OLMES 评测。step 2,500 与完整
20M 的三任务 continuous 都在每个 seed 上命中本地 1B 的三个 pair，保守全 seed 同向规则覆盖
100%、条件正确率 100%；完整 20M 的离散 OLMES-10 seed 均值只命中 1/3，三个 pair 全部因
seed 分歧而弃权。聚合输出分别是：

- `outputs/datadecide_reproduction/local_step2500_three_seed_crossscale/summary.json`；
- `outputs/datadecide_reproduction/local_final_three_seed_crossscale/summary.json`。

DCLM raw/FW3 的本地 evaluator 复现了 OLMES-10 crossover：20M 是 FW3 高 0.476pp，1B 是 raw
高 0.702pp。结果在
`outputs/datadecide_reproduction/published_dclm_crossscale/summary.json`。这说明 continuous 可作为
已校准目标能力代理，但 single-scale 排序不是跨任务、跨规模的普遍定律。
