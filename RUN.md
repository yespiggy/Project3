# 在 Vanda 上跑实验

按教授 10-29 的反馈修订：CPU 用 cn 节点（每节点 72 核），从 1 核测到 10 节点 720 核；强扩展用两张图（LiveJournal 和 com-Friendster）；GPU 加了 MPI+CUDA 多卡（同节点 2 卡、跨 2 节点各 1 卡）。

## 0. 放到集群上

解压到 `~/project3`（脚本默认 `P3_ROOT=$HOME/project3`；放在别的目录的话先 `export P3_ROOT=...`）。

```bash
cd ~/project3
bash scripts/build.sh          # 在登录节点编译，看到 built: ... pagerank_mpi_cuda 才算成功
```

## 1. 数据（只需要一个人做）

```bash
bash scripts/download_data.sh  # 登录节点下载；Friendster 压缩包约 8.7 GB，不用解压
bash scripts/submit.sh prep    # 预处理 + 生成 R-MAT（1~720 核），约 1~2 小时，内存申请 160 GB
```

数据放在 `/scratch/$USER/p3data`，Friendster 处理完约 15 GB，R-MAT 全部约 15 GB。

**其他队友共用这份数据**：数据所在账号的人执行一次 `chmod -R a+rX /scratch/$USER/p3data` 并确认 `/scratch/$USER` 也对其他人可读可进入；其他人提交前先执行：

```bash
export P3_DATA=/scratch/<数据所在账号>/p3data
```

## 2. 提交实验

prep 跑完以后再提交（或者用 `after=<prep 作业号>` 让它们自动等 prep 成功）：

```bash
bash scripts/submit.sh cpu     # 7 个 CPU 作业
bash scripts/submit.sh gpu     # 2 个 GPU 作业
```

一个人一次提交全部也可以：`bash scripts/submit.sh all`（prep 成功后其他作业自动开始）。

| 作业 | 资源 | walltime | 内容 |
|---|---|---|---|
| `cpu_n1_lj.pbs` | 1 × 72 核 | 1.5 h | 正确性验证、E1 LiveJournal 1~72 核、E2 1~72 核、串行基准 |
| `cpu_n1_big.pbs` | 1 × 72 核 | 3 h | E1 Friendster 1~72 核（单核一轮几十秒，最慢的一个作业） |
| `cpu_nodes.pbs` × 5 | 2/4/6/8/10 × 72 核 | 1 h | E1 两张图、E2，各自在 144/288/432/576/720 核；4 节点的那个加跑 E3 |
| `gpu_1node.pbs` | 1 节点 × 2 A40 | 1 h | GPU 正确性验证、E4 单卡 kernel 对比、E6 1 卡和同节点 2 卡 |
| `gpu_2node.pbs` | 2 节点 × 1 A40 | 45 min | E6 1 卡和跨节点 2 卡 |

两个人分着提交也可以，比如一个人提交 `cpu`、另一个人提交 `gpu`（GPU 队列每人最多同时跑 2 个作业，正好 2 个）。每个作业写自己的结果文件，文件名不会冲突，最后把各自 `results/` 下的 CSV 拷到一起就行。

## 3. 检查

- 作业输出在 `jobs/*.o<作业号>`。CPU 作业开头会打印 `lscpu` 拓扑和 `layout: 72 cores/node, ...`；如果分到的不是 72 核节点会直接报 `ERROR` 退出。
- GPU 作业开头打印每个节点分到的 GPU（`GPU map ...`）和每个 rank 用的卡（`rank i/p host=... bus=...`）。跨节点作业要确认两个 rank 在两台机器上，而且 bus id 是 PBS 分给我们的那张卡。
- 全部跑完后：`qsub jobs/post.pbs`（与 NetworkX 比对 + 汇总），或者在登录节点直接 `python3 tools/aggregate.py results`，生成 `results/summary.json`。

## 结果文件

| 文件 | 内容 |
|---|---|
| `results/strong_n{1,2,4,6,8,10}_{lj,big}.csv` | E1 强扩展（`exp` 列是 `E1_mpi` / `E1_hybrid`） |
| `results/weak_n{...}.csv` | E2 弱扩展 |
| `results/design_n4.csv` | E3 设计对比（288 核） |
| `results/serial.csv` | LiveJournal 串行基准 |
| `results/e4.csv` | E4 单卡 kernel 对比 |
| `results/multigpu_{1node,2node}.csv` | E6 多卡；`t_stage` 是主机与 GPU 之间拷贝 ghost 的时间 |
| `results/val/` | web-Google 正确性验证 |

`t_comp`、`t_comm`、`t_stage` 都是整次运行的总和（取所有 rank 中的最大值），除以 `iters` 就是每轮的时间，`aggregate.py` 已经换算好。
