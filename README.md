# QT-Joint 


本方法核心流程是：一次性 HOSVD 结构初始化，然后在固定秩下交替更新浮点组件和位宽。量化类型及裁剪比例固定；缩放因子和零点随当前组件及位宽重新标定，不是独立优化变量。

## 1 安装与快速运行

建议 Python 3.10 或更新版本。在解压后的项目目录打开终端，逐行执行：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --ranks 8,8,8 --max-iter 20 --delta 0.0001 --output-dir result/ping
```

这些单行命令可用于 Windows PowerShell 和 Linux 终端。算法运行仅依赖 NumPy 和 SciPy；重新生成附带流程图才需要可选的 Matplotlib。

省略 `--ranks` 时，按式（5.2）的完整 Tucker 存储代价联合选择三个秩：

```bash
python main.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --rank-max 32 --max-iter 20 --delta 0.0001 --output-dir result/ping_auto_rank
```

`rank-max` 限制实际检查的候选集合；不能把有限搜索域上的最优称为所有可能秩上的最优。秩目标是 HOSVD 误差上界代理，不是原始 Tucker 联合目标的精确全局求解。

## 2 读取已有数据

读取 `.mat` 三阶张量：

```bash
python main.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --delta 0.0001 --max-iter 50 --output-dir result/ping
```

读取编号文本矩阵序列：

```bash
python main.py --matrix-sequence "Data/PlanetLab" PlanetLabData_ 18 --ranks 8,32,32 --delta 0.0001 --max-iter 50 --output-dir result/planetlab
```

序列读取器沿 axis0 堆叠时间，依次读取 `PlanetLabData_1` 至 `PlanetLabData_18`，文件名不自动添加扩展名。矩阵尺寸必须一致。输入必须为非空、有限实数的三阶数组。项目保留 PingData、PlanetLab、TaxiBJ_in、TaxiBJ_out、FT-AED 和 UCSD-Ped2 的数据与实验。

## 3 目标与参数迁移

```text
J = SSE + delta * Cost
SSE = sum((X - reconstructed_X)**2)
Cost = n_G*b_G + n_A*b_A + n_B*b_B + n_C*b_C  # bit
```

`Cost` 不含字节对齐、元数据、NPZ 容器和无损编码开销。秩代理目标共用同一个 `delta`，初始化时各组件按 32 bit 计价。

默认 `delta=1e-4` 只是可运行起点，不是已验证的最佳超参数。数值尺度改变后须调整权重：若仅 `X_scaled=X/a`，使用 `delta_scaled=delta_original/a**2` 才保留原误差与存储的权衡。

旧 `rate_weight` 不能原值代入 `delta`。兼容入口显式警告并按 `delta=rate_weight/32` 换算旧归一化目标的系数，但这不会恢复旧算法；旧 padding、秩目标、裁剪搜索等差异仍存在。新实验请使用 `--delta`。非零 L2、位宽切换惩罚、独立秩目标权重和强制周期 QR 不属于当前论文目标，旧设置会收到清晰的迁移错误。

## 4 固定量化设置

默认候选集合现为 `H={2,3,4,...,32}`（31 个整数位宽），固定非对称量化和 `c=0`。这是 2026 年 11 月 11 日按实验要求扩展的配置；旧实验使用 `H=(2,3,4,5,6,7,8,10,12,16)`。两版实测结果分别保存，32 bit 整数量化也不等同于无量化 float32。例如固定双侧总裁剪比例为 0.01：

```bash
python main.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --clip-ratio 0.01 --quantization-type asymmetric --delta 0.0001 --output-dir result/ping_clipped
```

`c` 表示双侧合计的名义裁剪比例，使用 `c/2`、`1-c/2` 分位点，并把标定区间扩展至包含零。它不是旧 `clip_quantile` 的保留比例。旧保留比例 0.99 在比例意义上对应新 c=0.01，但对称标定规则也已变化，不能直接复用旧结果。

分位数采用线性插值，舍入采用 NumPy 的 ties-to-even。先 `round(z/s)` 再加整数零点；仅真正全零取 `s=1,zero_point=0`，非零尾部造成退化时回退未裁剪区间。Python API 也允许按组件预设：

```python
from qulow_joint import joint_lowrank_quantize
result = joint_lowrank_quantize(
    X, ranks=(8, 8, 8), delta=1e-4,
    clip_ratio={"G": 0.0, "A": 0.01, "B": 0.01, "C": 0.01},
    symmetric={"G": False, "A": True, "B": True, "C": True},
    max_iter=50,
)
```

上述 c 和编码类型只在运行前设置，不在交替优化中搜索。

## 5 更新与停止

连续步骤使用半 SSE 梯度，乘以实际可表示区间的 STE 掩码，端点包含。四组件从同一旧状态同步生成候选。每个候选重新标定并完整重构，真实损失不增才接受，否则回退步长或保留旧状态；不对标定路径反向传播。量化平台上允许等损失浮点更新。

位宽按 G → A → B → C 顺序扫描。候选通过残差投影和 Gram 二次型计算精确全局 SSE 增量，再加 `delta*n_Z*(h-b_Z)`。当前位宽始终保留，严格改善才切换；提交后刷新残差。

可选偏度筛选：

```bash
python main.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --skew-filter --skew-thresholds 0.5,1.0,2.0 --delta 0.0001 --output-dir result/ping_filtered
```

阈值是公开运行设置，不是已验证的最佳值。参考集合与 H 取交集并保留当前位宽；峰度只用于诊断。默认结束前做完整扫描，研究纯筛选消融时可加 `--no-final-full-scan`。达到次数上限不等于坐标最优，应检查 `search_diagnostics` 的停止状态。

停止阈值只影响连续小改善的计数，不会丢弃小幅改善的最佳状态。理论保证的是接受目标值不增加，不声称浮点参数、真实驻点或联合全局最优收敛。

## 6 归一化和指标

默认 `--normalization none`。可选 `scale`（只缩放）、`global`（全局 min-max）和 `axis0`（沿 axis0 min-max）。包含平移时，归一化域相对误差不等于原始域相对误差。主程序重载模型后，分别报告两域的 SSE、MSE、RMSE、MAE、最大绝对误差和相对 Frobenius 误差。

| 指标 | 口径 |
| --- | --- |
| `model_bits` | 论文 `Cost=sum(n_Z*b_Z)` |
| `model_compression_ratio` | `32*N/model_bits`，理想压缩率 |
| `compressed_bits` / `compression_ratio` | 打包字节对齐及数值元数据估算，兼容旧接口，不是文件大小 |
| `actual_model_bytes` | 模型 NPZ 的实际大小 |
| `actual_decoder_bytes` | 模型加归一化文件的实际大小；扩展实验还加稀疏文件 |
| `actual_compression_ratio` | 明确以 float32 原数组的 `4*N` 字节为分子，除以上述实际大小 |

默认分子不是原 `.mat` 或文本文件大小。训练历史与配置不计入部署载荷。小张量实际压缩率可能低于 1，应如实报告。`--error-budget 0.1` 只在输出时检查原始域相对误差是否不超过 10%，不会自动搜索 delta，也不表示加权目标本身保证误差预算。

## 7 输出和独立解码

主入口输出：`joint_qulow_model.npz`、`normalization.npz`、`joint_qulow_history.csv`、`rank_selection.csv`、`run_config.json`、`summary.json`。归一化文件始终保存并计入实际大小，包括恒等变换，避免旧目录的残留元数据被误用。

模型只保存打包整数编码和解码元数据，不含浮点 master 参数、训练历史。v3 使用 float64 缩放因子、int64 零点；加载器兼容旧 v1/v2。显式允许 32 bit 整数编码，但它不等于无量化 float32。

```bash
python experiments/decode_model.py result/ping/joint_qulow_model.npz --normalization result/ping/normalization.npz --output result/ping/reconstructed.npy
```

Python 中仍可使用 `result.save_npz(path)`、`load_quantized_model(path)` 和 `tucker_reconstruct(core,factors)`。核心加载器只还原压缩域；归一化必须用同次保存的参数逆变换，不可凭位宽重新推导缩放因子。

## 8 保留的实验

当前项目保留六个真实数据集及其既有实测记录。`result/` 中的六数据集 Word、HTML 和图表已按保留记录重建；维护过程没有重新运行完整压缩实验。`result/maintenance_verification.json` 记录文件哈希、回归验证和待处理文件状态。

六数据集统一入口默认运行 PingData、PlanetLab、TaxiBJ_in、TaxiBJ_out、FT-AED 和 UCSD-Ped2：

```bash
python experiments/run_dataset_evaluation.py --source auto --max-iter 5 --rank-max 64 --output-dir result/dataset_evaluation --quiet
```

`--datasets` 可指定其中一个或多个名称。`--source auto` 优先读取已保存的实验输入，缺少时读取 `Data/` 中的原文件；`--source original` 强制读取原文件。默认使用 scale 归一化和 `H={2,3,...,32}`。每个数据集单独保存模型与指标，输出目录还包含汇总 CSV 和 JSON。

已有记录的汇总、报告及归档可分别通过 `experiments/summarize_retained_results.py`、`experiments/build_retained_reports.py` 和 `experiments/package_retained_results.py` 重新生成。这三个脚本只处理已保存结果。

真实输入上的异常保持性和时窗诊断可分别运行：

```bash
python experiments/anomaly_detection_test.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --ranks 4,4,4 --max-iter 5 --delta 0.0001 --output-dir result/ping_anomaly
python experiments/spatiotemporal_distribution_experiment.py --mat "Data/PingData/Pingsdata.mat" --mat-key data --time-axis 2 --time-length 8 --time-stride 4 --ranks 4,4,4 --max-iter 5 --delta 0.0001 --output-dir result/ping_windows
```

异常实验的低秩加稀疏分离是外接扩展，不是 QT-Joint 核心模块。稀疏值按实际保存精度重载，并计入文件大小。使用真实异常数量和 clean baseline 的 top-k 指标明确标为 oracle 保持性评价，不等于无标签部署检测性能。

时空实验在完整输入上拟合一次预处理参数，保持跨窗尺度一致；文本矩阵序列默认时间轴为 axis0，MAT 数据默认 axis2。固定秩必须适配每个窗口，末窗不适配时可调整窗口或使用 `--drop-last`。

## 9 验证范围

`tests/` 覆盖数学公式、边界情况、求解流程、命令行和导出。`REVISION_ANALYSIS.md` 列明修改依据，`VALIDATION_REPORT.md` 记录实际验证结果。`assets/` 提供更新后的流程图和可选生成脚本。

另外可以运行可复现的随机压力验证（90 组位宽穷举对照和 60 组交替优化检查）：

```bash
python tests/stress_validation.py --output result/stress_validation.json
```

请先运行单元测试，再使用真实数据进行实验。测试用于证明已检查路径的实现一致性，不替代论文中的真实数据集对比或消融实验；旧版实验数字不能直接更名为本版 QT-Joint 的结果。
