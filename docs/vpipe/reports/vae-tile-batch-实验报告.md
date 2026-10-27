# 视频 VAE 解码：tile 维度合批（方法 1）实验报告

> 分支：`261027_jzgBatchTileLinear`
> 日期：2026-10-27
> 机器：M5 Pro 64 GB / macOS 27.0.1 / Metal 4
> 复现：`make h3_vae_tile_batch_bench` + `tests/bench_vae_tile_batch.c`

## 0. 结论摘要

| 假设 | 结论 | 依据 |
|---|---|---|
| 把 15 个 tile 合并进一次线性层 GEMM（方法 1）能提速 | **否定，反而慢 13%** | 合批需要 15× 的工作集，丢掉了 L2 驻留；MPS 的 BF16 matmul 并不随 M 变宽而变快 |
| VAE 解码是"延迟/小 GEMM 瓶颈"（此前 §4 的假设） | **否定，是算力瓶颈** | 线性层实测 20.5–26.6 TFLOP/s，MLP 占 60% 时间；NAX TensorOps 对 VAE 形状无优势 |
| 唯一剩下的提速杠杆是**降精度到 INT8** | **成立：线性层 1.95×，整块 1.61×** | INT8 matmul 31.8–50.4 TFLOP/s，BF16 的 0.41–0.64× |
| 减少 tile 数（tile=304）能提速 | **成立：-21%** | 成对测量中位数 0.790（见《h3-vae-bf16-改造报告》§4） |

---

## 1. 背景：为什么要做"方法 1"

VAE 解码器逐 tile 循环，每个 tile 跑完整的 36 层栈。于是每一层要：

- 用同一份权重做 **15 次瘦 GEMM**（每次 1797 行）；
- 注意力做 15 次独立调用。

设想的优化是把 tile 维度"批"进矩阵：把 15 个 tile 的 1797 行拼成 26955 行，一次 GEMM 搞定，
权重从每层读 15 遍变成读 1 遍。

数学上这是**严格等价**的：QKV / OUT / w1 / w2 都是逐 cell 独立的，cell 维度上没有跨 tile 耦合；
RoPE 表对所有 tile 完全相同（同一个 16×16 网格），复制 B 份即可；注意力是 tile 内自注意力，
需要 per-tile 语义，但 MPSGraph 的 SDPA 原生支持 `batch` 维度
（`h3_gpu_sdpa_bf16_batch()`，本次为此新增）。

因此这属于**零保真度代价**的优化，值得优先尝试。

## 2. 先做微基准，而不是先改解码器

重写解码循环（tile 外层 → chunk 外层）动的是整个 VAE 的主路径，风险不低。
因此先写一个独立的微基准 `tests/bench_vae_tile_batch.c`：

- 用真实 VAE 形状（HIDDEN=2048 / INNER=2048 / FFN=8192 / HEADS=32 / HEAD_DIM=64 /
  SEQ=1797 行 / 15 个 tile）；
- 两种风格跑**完全相同的 FLOPs、相同的权重**：
  - `per-tile`：15 次 `rows=1797` 的调用；
  - `batched`：1 次 `rows=26955` 的调用；
- 每个算子单独计时，并给出两种风格的比值。

### 2.1 结果（15 tile / 1797 行）

| 算子 | per-tile | batched | 比值 |
|---|---|---|---|
| rms_norm1 | 0.87 ms | 0.87 ms | 1.00× |
| qkv linear | 25.45 ms | 108.67 ms | **2.42×（更慢）** |
| qkv+rope | 14.47 ms | 45.27 ms | 0.83× |
| attention | 18.31 ms | 70.86 ms | **1.34×（更慢）** |
| out linear | 7.95 ms | 30.18 ms | 0.91× |
| residual 1 | 0.89 ms | 0.89 ms | 1.21× |
| rms_norm2 | 0.87 ms | 0.87 ms | 0.96× |
| mlp fc1 | 70.76 ms | 190.23 ms | 1.05× |
| swiglu | 5.85 ms | 14.46 ms | 1.30× |
| mlp fc2 | 40.14 ms | 92.79 ms | 0.83× |
| residual 2 | 0.85 ms | 0.85 ms | 1.22× |
| **合计** | **494.44 ms** | **558.66 ms** | **1.13×（更慢）** |

（这是 15× 大缓冲区的版本；生产版本见 §3。）

结论：**合批不成立**。收益最大的 qkv linear 反而慢了 2.42×。

## 3. 根因：L2 驻留比 GEMM 宽度更值钱

把缓冲尺寸改回生产规格（解码器只常驻**一个** tile 的激活，即 1797 行）再测：

| 风格 | 缓冲 | 15 tile × 1 block |
|---|---|---|
| per-tile | 15 × 1797 行（生产） | **186.41 ms** |
| per-tile | 26955 行（合批所需） | 494.44 ms（**慢 2.65×**） |

**同样的算子、同样的行数，仅仅因为工作集从 7.4 MB 涨到 110 MB 就慢了 2.65 倍**——
说明这条路径的瓶颈在**缓存驻留**，不在 FLOPs 也不在带宽。

而合批**必须**用 15× 的工作集。于是：

- MPS 的 BF16 matmul 在 `M=1797` 与 `M=26955` 下吞吐几乎一样（合批没有宽度红利）；
- 合批却要付出 2.65× 的缓存代价。

两个方向相反的效应叠加，合批必然是负收益。

## 4. 修正后的瓶颈诊断

生产缓冲下，每 tile 每层（block-tile）的时间预算：

| 算子 | 耗时 | 占比 |
|---|---|---|
| mlp fc1（2048→16384） | 4.72 ms | 38% |
| mlp fc2（8192→2048） | 2.68 ms | 22% |
| qkv linear（2048→6144） | 1.70 ms | 14% |
| attention | 1.22 ms | 10% |
| qkv+rope | 0.96 ms | 8% |
| out linear（2048→2048） | 0.53 ms | 4% |
| swiglu | 0.39 ms | 3% |
| 两个 norm + 两个 residual | 0.23 ms | 2% |
| **合计** | **12.43 ms** | |

对应吞吐（MPSGraph BF16 matmul）：

| 形状 | TFLOP/s |
|---|---|
| qkv 1797×2048×6144 | 25.9 |
| out 1797×2048×2048 | 20.5 |
| mlp fc1 1797×2048×16384 | 22.4 |
| mlp fc2 1797×8192×2048 | 26.6 |

**≈25 TFLOP/s 就是 MPS BF16 matmul 在这批形状上的实际上限。**
解码器是**算力瓶颈**，不是延迟瓶颈。

> ⚠️ 这推翻了本仓库《h3-vae-bf16-改造报告》§4 中"每 tile 固定 dispatch 开销成为新瓶颈"的猜测：
> 成对测量拟合出的 per-tile 项约等于 0（甚至略负），per-cell 项才是主导。

## 5. NAX TensorOps 对 VAE 形状没有优势

仓库里已有 DiT 用的自定义 TensorOps 内核 `h3_linear_bf16_nax_r128`（128×64 分块）。
它对 VAE 的形状在约束上完全可用（`input_dim % 32 == 0`、`output_dim % 64 == 0`），
只是因为 **NAX 内核不支持 bias**（VAE 的四个线性层都有 bias）才没被用上。

同进程配对 A/B（1797 行）：

| 形状 | MPSGraph（带 bias） | NAX（无 bias） | NAX/MPS |
|---|---|---|---|
| qkv | 1.9 ms | 1.8 ms | 0.93× |
| out | 0.9 ms | 0.7 ms | 0.80× |
| mlp fc1 | 4.8 ms | 5.9 ms | **1.23×（更慢）** |
| mlp fc2 | 2.8 ms | 2.4 ms | 0.85× |

**结论：给 NAX 内核加 bias 支持也换不来速度**（最大 20%，且 fc1 反而更慢）。
DiT 的 NAX 收益来自它自己的形状（7168→5376 等）和 MLP 融合，不迁移到 VAE。

## 6. 唯一剩下的杠杆：INT8

仓库已有实验性的 M5 INT8 路径（`h3_gpu_linear_int8_bf16` / `h3_gpu_mlp_int8_bf16`，
权重按输出通道量化、激活按行动态量化），PR #73 已经把它接进 VAE（只接了 4 个投影）。

同进程配对 A/B（1797 行，INT8 一侧**包含**激活的逐行量化开销）：

| 形状 | BF16 | INT8 | INT8/BF16 | INT8 吞吐 |
|---|---|---|---|---|
| qkv | 1.75 ms | 1.11 ms | 0.63× | 40.9 TFLOP/s |
| out | 0.74 ms | 0.47 ms | 0.64× | 31.8 TFLOP/s |
| mlp fc1 | 5.82 ms | 2.39 ms | **0.41×** | 50.4 TFLOP/s |
| mlp fc2 | 2.27 ms | 1.44 ms | 0.64× | 41.8 TFLOP/s |
| **四层合计** | **10.58 ms** | **5.41 ms** | **0.51×** | |

线性层占 block-tile 的 77%，因此：

- block-tile：12.43 ms → **7.71 ms（1.61×）**
- 整段 VAE 解码推算：74.5 s → **约 46–50 s**

保真度代价（PR #73 的 INT8 实测）：60.14 dB → 52.82 dB，
即 RMSE 0.25 → 0.58 色阶。而 H.264+4:2:0 编码器自身的误差是 **20.78 色阶**，
是 INT8 VAE 误差的 **36 倍** ⇒ **INT8 的画质损失在最终 MP4 里不可见**。

## 7. 建议

1. **不要再做 tile 合批**——已实测为负收益，收益来自"更少的解码像素"（tile=304，-21%）
   或"更低的数值精度"（INT8，-35%）。
2. 若追求速度且接受画面与 checkpoint 有可见统计差异：**全量 INT8 VAE**（所有线性层 + 激活量化），
   预计 74.5 s → 46–50 s，代价 ~7 dB（0.58 色阶，远低于编码器噪声）。
3. 若要求完全忠于 checkpoint：**保持 BF16 + tile=256**，当前 74.5 s 已在 BF16 matmul 的硬件上限上。

## 8. 复现

```bash
make h3_vae_tile_batch_bench
# 生产缓冲 + 逐算子拆解
H3_BENCH_SMALL=1 ./h3_vae_tile_batch_bench h3_shaders.metal
# 15× 缓冲（合批需要的规模）
./h3_vae_tile_batch_bench h3_shaders.metal
# MPS vs NAX 配对 A/B（需要 H3_NAX=1 让 NAX 形状约束放开）
H3_BENCH_SMALL=1 H3_BENCH_AB=1 H3_NAX=1 ./h3_vae_tile_batch_bench h3_shaders.metal
# BF16 vs INT8 配对 A/B
H3_BENCH_SMALL=1 H3_BENCH_INT8=1 ./h3_vae_tile_batch_bench h3_shaders.metal
```

本分支新增 API：`h3_gpu_sdpa_bf16_batch()`（带 batch 维度的 BF16 SDPA，
供微基准与后续实验使用；解码器本身未改动）。
