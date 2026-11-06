# PageRank 报告结果包（2026-11-06）

本目录汇总两位组员提供的结果包，以及聊天中补充的 720 核诊断和 256 核弱扩展结果。**这是实验材料整理，不是最终报告，也不是所有实现均已完成验证的声明。** 本次没有新提交 HPC 作业；作业状态来自包内日志或用户提供的历史快照。

推荐从本文、`tables/` 和 `figures/` 开始写报告。`raw/` 保留收到的证据，其中历史仪表板、旧 summary 和重复归档不参与主表汇总。与当前代码一起提交不代表每条旧实验都由当前 commit 运行。

## 1. 可以写进报告的材料

| 报告内容 | 已有证据 | 整理后的入口 | 使用限制 |
| --- | --- | --- | --- |
| CPU 强扩展 | R-MAT，1–72、256、720 核；Hybrid 与 pure MPI | [主表](tables/cpu_rmat_strong_and_layout.csv)、[图](figures/cpu_strong.png) | 720-rank pure MPI 失败；两个 720 核 Hybrid 批次必须分开 |
| 同核数下进程/线程选择 | 720 核，20×36、40×18、80×9，各 5 次 | [布局图](figures/cpu_720_layout.png)、主表 batch=1515216 | 未重跑 720 ranks；固定测试顺序，不能排除环境和顺序影响 |
| CPU 弱扩展 | R-MAT，1、2、4、8、16、36、72、256 核 | [弱扩展表](tables/cpu_rmat_weak.csv)、[图](figures/cpu_weak.png) | 近似固定每核边数；256 核节点和绑定待核对 |
| 计算与通信分量 | CPU、GPU CSV 的 t_comp、t_comm，GPU 的 t_stage | 各表的 `*_median_ms` | 分量不能直接相加或当成互斥墙钟时间占比 |
| 多 GPU 扩展 | LiveJournal、Friendster，1/2 GPU，同节点/跨节点，thread/hybrid 核函数，各 5 次 | [GPU 表](tables/gpu_scaling.csv)、[图](figures/gpu_scaling.png) | 待补 MPI+CUDA 全向量验证和运行代码版本/计时定义 |
| 同图 CPU/GPU 对比 | LiveJournal 串行、MPI+OpenMP、CUDA、MPI+CUDA | [同图比较表](tables/livejournal_cpu_gpu_comparison.csv) | 跨作业、代码版本待确认；不能与 R-MAT CPU 时间直接比较 |
| 并行设计取舍 | 历史 LiveJournal vertex/edge × allgather/ghost | [历史设计表](tables/historical_design_choices.csv) | 保留为历史批次，勿混入新计时主曲线 |
| 正确性 | CPU smoke；256 核与串行同 20 轮向量比较；历史 web-Google 验证 | 见第 5 节 | 固定迭代一致性、最终收敛、多 GPU 验证应分别陈述 |
| 真实图结论 | 历史 web-Google 分析 | [google_analysis.json](raw/teammate/results/val/google_analysis.json) | 缺原始向量，属于已有分析产物；不能推断真实网页身份 |

## 2. CPU 强扩展和 720 核诊断

固定图 `rmat_p144`：N=18,874,368，预处理后 M=297,337,356；20 iterations、tol=0。下表均为每轮时间的重复测量中位数。

强扩展基线是 **1 MPI rank × 1 OpenMP thread 的同一并行程序**，1370.543 ms/轮；不是另一个串行可执行文件的时间。报告应明确这是同实现的单核扩展基线。

| 配置 | 作业 | ms/轮 | 相对单核加速 | 效率 |
| --- | --- | ---: | ---: | ---: |
| 1 核 Hybrid | 1509305 | 1370.543 | 1.00× | 100% |
| 72 核 Hybrid | 1509305 | 27.981 | 48.98× | 68.03% |
| 256 核 Hybrid，8×32 | 1518822 | 19.558 | 70.08× | 27.37% |
| 256 ranks pure MPI | 1518822 | 541.854 | 2.53× | 0.99% |
| 720 核 Hybrid，20×36，旧批次 | 1512555 | 32.176 | 42.60× | 5.92% |
| 720 核 Hybrid，20×36，新批次 | 1515216 | 15.140 | 90.53× | 12.57% |
| 720 核 Hybrid，40×18 | 1515216 | 199.470 | 6.87× | 0.95% |
| 720 核 Hybrid，80×9 | 1515216 | 239.234 | 5.73× | 0.80% |

图中强扩展误差线表示重复测量最小值至最大值，不是置信区间。1–72、256、旧 720 各 3 次，新 720 各 5 次。完整布局、范围和分量见 CSV。

**可以支持的结论：** Hybrid 在这些实验中明显优于大量 pure MPI ranks；扩展效率随规模增大而下降。1515216 内 20×36 最快，40/80 ranks 的通信计时显著增加。但 20×36 两个批次的中位数相差很大（32.176 vs 15.140 ms），不能挑选快批次覆盖旧数据，也不能将全部性能差异归因于进程数量。

1512555 成功完成 3 条 Hybrid 测量，随后 720-rank pure MPI 在 `MPI_Alltoallv` 失败，日志含 `bind ... :0 failed: Address already in use (98)`，整个作业 Exit_status=16。保留 Hybrid 测量，pure MPI 记为失败、无有效性能数值。临时端口耗尽是待确认的解释；本包没有失败当时的端口占用快照。

1515216 Exit_status=0，15 条 Hybrid 测量完整。它只测试了 20、40、80 个 MPI ranks，**不能据此声称先前 720-rank pure MPI 故障已修复**。所收到附件不包含 `network_*.txt`，也不能声称已经核实网络限制。

## 3. CPU 弱扩展

1–72 核来自 1509305；256 核来自用户粘贴的 1521041 CSV，明确放在 `raw/user_transcribed_1521041/`。生成任务 1521040 和运行任务 1521041 用户提供的退出码均为 0。

256 核使用 `rmat_p256`：N=33,554,432，M=528,692,597，4 节点，8 ranks×32 threads，20 iterations、tol=0；3 次为 34.80268、33.67265、33.71405 ms，中位数 **33.71405 ms**。相对 1 核弱扩展基线 2.847124 ms，名义弱效率为 **8.44%**。

“Realized edges per core”定义为 **预处理/去重后实际边数 M ÷ 运行所用 CPU 核数**。1 核为 1,942,984，256 核约 2,065,205，增加 6.29%，因此应称为近似弱扩展。

256 核运行在 GN-A40-048/050/052/063，每节点分配 36+28=64 核。程序是 `mpi_omp`，节点名称含 A40 不代表使用了 GPU。CPU 型号、拓扑和实际绑定没有在所收材料中完整确认，因此跨节点类型的弱效率带有可比性限制。

## 4. GPU 结果与“GPU 为什么只快约两倍”

组员确实运行了 **MPI+CUDA 多 GPU 实验**：1508084 同节点两张 A40，1508085 跨节点各一张 A40。两份日志和 CSV 均包含完成的测试，但包内没有这两个作业的 PBS 最终 Exit_status。每份 CSV 40 条记录。日志可辨认不同 GPU PCI 地址，不能仅凭未设置 `CUDA_VISIBLE_DEVICES` 判为两进程共用一张卡；PBS 设备分配仍需另行确认。

下面是 `mpi_cuda_hybrid` 核函数的代表结果（其他核函数见完整表）：

| 图/布局 | 1 GPU ms/轮 | 2 GPU ms/轮 | 加速比 | 2 GPU 效率 |
| --- | ---: | ---: | ---: | ---: |
| LiveJournal，同节点 | 4.741 | 7.096 | 0.668× | 33.41% |
| LiveJournal，跨节点 | 4.740 | 8.737 | 0.542× | 27.12% |
| Friendster，同节点 | 583.861 | 360.448 | 1.620× | 80.99% |
| Friendster，跨节点 | 583.853 | 375.093 | 1.557× | 77.83% |

LiveJournal 使用 50 轮，Friendster 使用 20 轮。每个速度比使用同图、同核函数、同批次的 1 GPU 基线。结果与“大图有更多计算量可摊薄通信/数据传输开销”的解释一致，但缺少有效硬件计数器数据，不能把它写成已证实的硬件瓶颈。

LiveJournal 新批次串行中位数 155.9264 ms，36 核 Hybrid 为 6.0577 ms，72 核为 8.0549 ms；与同节点 1 GPU 的 MPI+CUDA hybrid 4.740727 ms 相比，分别约 **32.89×、1.28×、1.70×**。所以“GPU 仅两倍”强烈依赖 CPU 基线。此处为迭代计时比较，不能代表含加载/初始化的端到端性能；旧代码计时定义仍须确认。Friendster 没有对应的新 CPU 基线，不能计算其 CPU/GPU 加速比。

`cuda_hybrid`（独立单 GPU 程序）与 `mpi_cuda_hybrid`（MPI+CUDA）不同，不得合并为同一实现。`e4.csv` 有两段各 20 行、重复 rep 编号的数据；整理脚本将其分为两个来源未完全确认的批次，而不是当作每配置 10 次重复。`ncu_*.csv/.err` 显示 ERR_NVGPUCTRPERM，没有可用的硬件计数器测量。

## 5. 正确性与真实数据分析

- [CPU smoke 日志](raw/ours/jobs/p3smoke.o1509299)：52 次 NetworkX 比较，涵盖无边、稀疏、悬挂节点等小图；对应作业 1509299 Exit_status=0。
- [256 核验证 JSON](raw/user_transcribed_1521041/validation.json)：在 **rmat_p144** 上，串行与 Hybrid 同样执行 20 轮，L1=3.50708e-11、L∞=5.20070e-15，passed=true。它验证固定迭代实现一致性，不证明最终收敛，也不是 rmat_p256 的全向量验证。原始二进制向量未随本包提供。
- [历史 web-Google 验证](raw/teammate/results/val/validation.json)：包含 4 种 CPU 分区/通信组合和 4 种独立 CUDA 核函数，相对串行的 L1 约 3e-15；NetworkX 对照 L1 约 1.37e-7。报告中注明为已有验证产物，本包不含向量供独立重算。JSON **不含 MPI+CUDA 多 GPU 全向量比较**。
- residual、sumr 接近 1 是诊断值；概率和接近 1 不能替代全向量正确性检查。固定 20/50 轮性能测量不能直接声明达到最终收敛阈值。

真实数据小节可使用已有 web-Google 分析：875,713 个节点、5,105,039 条边；PageRank 与入度的 Spearman 相关系数约 0.847，二者 top100 重合 88 个；排名前 1% 节点占约 27.09% 概率质量。可以讨论排名集中以及 PageRank 与入度相关但不同；这些统计仅适用于该预处理后的图，不能推出网页身份或因果关系。原始数据分析代码/向量的重现仍待补齐。

## 6. 统计与术语

- 表格单位：`median_ms/min_ms/max_ms` 均为每轮毫秒；原始 CSV 的时间单位为秒。
- `t_iter` 是迭代阶段的每轮平均时间，再跨 repetitions 取中位数；不是端到端作业时间。PBS walltime、排队时间和图加载时间分别记录，不能混用。
- `t_comp_median_ms/t_comm_median_ms/t_stage_median_ms`：原始每次重复的累计分量 ÷ 该次迭代数，再取中位数。当前修订代码的总时间和各分量分别跨 ranks 取最大值，最大值可能来自不同 ranks，故不能将分量相加解释为总墙钟时间或直接计算互斥占比。旧批次的计时语义需核对。
- CPU 强扩展 `speedup=T_1/T_P`，`efficiency=speedup/P`，P=ranks×threads。
- 近似弱扩展 `nominal_weak_efficiency=T_1(small graph)/T_P(scaled graph)`；另列每核边数相对基线的比例。
- GPU `speedup=T_1GPU/T_G`，`gpu_efficiency=speedup/G`；G 为 GPU 数，在这些试验中等于 ranks 数。GPU CSV 的 `cores` 不应被当作 GPU 数的通用定义。
- CSV 效率字段为 0–1 比例；图和本文显示为百分比。
- `comm_bytes_iter` 是程序估计的每轮通信量，不是网卡实测吞吐。`gbps` 也不可直接当成有效硬件带宽计数器。

## 7. 目录、来源和复现

- `raw/ours/`：用户提供 `p3-results-20261105.tar.gz` 的材料。
- `raw/teammate/`：用户提供 `vanda_results_2026-11-04.zip` 的材料。旧 `summary.json`、`dash/`、e1/e2 等均保留为历史档案，不作为本次主表来源。特别是旧 summary 的串行基线与新 `serial.csv` 不同。
- `raw/diagnostic_1515216/`：用户粘贴的完整日志、提取 CSV 和已有小结。
- `raw/user_transcribed_1521041/`：从用户聊天消息转录的 CSV 和 JSON；未从 Vanda 重新下载。详见该目录 PROVENANCE。
- `normalized/`：拆分后的 E4 两批数据。
- `tables/`：7 张重新计算的汇总表，每行带来源路径和批次。
- `figures/`：4 张可用于报告草稿的 PNG。
- `SOURCE_MANIFEST.json`：输入文件哈希与来源说明；`SHA256SUMS`：本目录文件校验和（不含自身）。

在本目录运行（汇总脚本只需 Python 3 标准库；绘图另需 matplotlib）：

```bash
python3 summarize.py
python3 plot.py
```

脚本按明确文件白名单读取，不递归合并全部 CSV，避免 `archive_1512555` 双计、旧新批次混合。固定批次内检查 rep 编号唯一，生成 19 组 CPU 强扩展/布局、8 组弱扩展、16 组 GPU 扩展，以及其他补充表。CPU 强扩展图显示测量范围，其余图展示中位数或中位数之比，原始重复值全部保留。重新生成文件后原校验清单可能因运行环境/图像元数据变化需要更新。

## 8. 写最终报告前仍需补齐

1. MPI+CUDA 多 GPU 的同迭代全向量对照；保存代码 commit、编译器/MPI/CUDA 版本、GPU 绑定和最终作业退出状态。
2. 确认组员 GPU/历史 CPU 实验使用的准确代码版本与计时边界，尤其 rank0 时间与跨 rank 最大时间的差别。
3. 下载 1521041 的原始结果与脚本、参考向量，以及 1515216 的网络诊断文件；核对 256 核 GN 节点 CPU/绑定信息。目前转录数据仍可引用，但要标明来源。
4. 报告中分别讨论正确性、收敛、效率和失败案例；保留所有 720 核批次。1509302 GPU smoke 最后提供的状态仍是 Q，不能写成已通过。
5. 补上算法介绍、实现设计说明、所有组员姓名/学号和 AI usage；本结果包不能替代 PDF 报告。

AI usage 可如实写明：使用 Codex 协助代码审查/修改、作业脚本、日志整理、统计表和图生成；Vanda 的运行时间来自实际作业输出。1521041 数据由用户粘贴后转录；未生成或补造缺失测量。最终报告还应按组员实际使用情况补充，并人工审核结论。
