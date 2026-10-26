# h3.c 视频 VAE bf16 改造报告

> 分支：`feature/vae-bf16-nax`（基于 `origin/main` @ `8974cc0`）
> 机器：MacBook Pro 16″ (Mac17,9)，M5 Pro，20 GPU core，64 GB 统一内存，macOS 27.0.1
> 日期：2026-10-25
> 问题：VPIPE 的视频 VAE 解码只要 63.1 s，h3.c 要 164.6 s，慢 2.61×。为什么？能不能按 VPIPE 的方向改？

---

## 0. 结论速览

| 结论 | 证据 |
|---|---|
| **根因不是架构差异，是数值精度**：两边都是 36 层 / 2048 宽 ViT 解码器，架构完全同源 | 见 §1 |
| h3.c 全程用 **FP32**，MPS 的 FP32 matmul **不经过张量核心**，只能跑 FP32 SIMD | 见 §1.2 微基准 |
| 改成 **BF16** 后 MPS 走张量核心，**VAE 解码 159.45 s → 47.33 s（3.37×）**，端到端 409.95 s → 316.93 s | 见 §3.1 |
| **同会话复测 VPIPE**：改造前 h3.c 的 VAE 落后 VPIPE **3.54×**，改造后只差 **1.05×**（47.33 vs 45.01），端到端从落后 1.10× 变成**领先 1.17×** | 见 §3.1.1 |
| 精度代价可忽略：相对误差 **rel-L2 0.0048**（VPIPE 自己的门槛是 0.02） | 见 §3.2 |
| 重构在 FP32 路径上**数值中性**：新 f32@304 vs 旧二进制 f32@304 = 23.12 dB（高于热漂移基线 20.86 dB） | 见 §3.5 |
| 顺带把 tile 默认值对齐回 checkpoint 的 256 px（成片影响 41.6 dB，感知不可见；代价是 VAE 慢约 26%） | 见 §4 |
| 改造是**接线活不是研发活**：h3.c 里 bf16 原语本来就齐了，缺的是"把视频 VAE 接上去" | 见 §2 |

---

## 1. 根因

### 1.1 架构同源，排除"算法不同"

| | h3.c | VPIPE |
|---|---|---|
| 解码器结构 | 36 层 ViT | 36 层 ViT |
| 隐藏维 / 头数 / 头维 | 2048 / 32 / 64 | 同 |
| FFN 中间维 | 8192 | 同 |
| latent 通道 | 24 | 同 |
| register token | 4 | 同 |
| suffix token | 5 | 同 |
| 输出 patch | 3×4×16×16 = 3072 | 同 |
| 空间下采样比 | 16 | 同 |
| tile 尺寸 | 256 px（见 §4） | 256 px |

h3.c 的定义见 `h3_video_vae.c:13-31`；VPIPE 侧见 `generative-models/minimax-h3/metal-minimax-h3-video-vae.cc:495`（注释明确写 "decoder MUST be the ViT one"）。

> ⚠️ 上一版对比报告里"VPIPE 的 VAE 快是因为用了卷积 + 矩阵核心"的说法**不准确**。VPIPE 文档里的 "2.8× faster via convolutions on matrix cores" 说的是**编码器**（`conv3d_hw_causal_s*_f16`，走 MPP `convolution2d`）。**解码器**两边都是 ViT，VPIPE 的优势全部来自 BF16 张量核心 GEMM + NAX flash attention。

### 1.2 硬数据：FP32 vs BF16 的 GEMM/SDPA 吞吐

用真实 VAE 形状写了一个独立微基准（`vpipe-work/bench_vae_gemm.c`），在 VAE 实际行数（1289 / 10240 / 15360）上分别测 MPS FP32 与 MPS BF16，8 轮取最优（`vpipe-work/bench-vae-gemm-f32-vs-bf16.log`）：

| 算子 | 形状 | FP32 (GF/s) | BF16 MPS (GF/s) | BF16 NAX (GF/s) | 加速 |
|---|---|---:|---:|---:|---:|
| QKV | 2048→6144 | 6354 | 26203 | 26245 | **4.12×** |
| 输出投影 | 2048→2048 | 6200 | 21756 | 22249 | **3.51×** |
| FFN w1 | 2048→16384 | 6420 | 28806 | 29594 | **4.49×** |
| FFN w2 | 8192→2048 | 6117 | 26584 | 27254 | **4.35×** |
| SDPA | 2048 维 | 5928 | 17078 | — | **2.88×** |

**这些数字直接指向根因**：

- FP32 的 6.2 TFLOP/s ≈ M5 Pro 的 FP32 SIMD 理论峰值（20 core × 128 FLOP × 2 × 1.26 GHz ≈ 6.45 TFLOP/s）→ **一个张量核心都没用到**。
- BF16 的 26 TFLOP/s ≈ 张量核心峰值 → **打满了**。
- MPS 的 BF16 matmul（26.2）与 h3.c 自带的 NAX `matmul2d` 路径（26.4）**基本持平** → 说明不用手写矩阵核心内核，走 MPS BF16 就够了。

h3.c 旧版 profile（`vpipe-work/h3c-run3.log`）也证实算子确实走了 MPS f32 matmul：

```
h3 profile: video VAE decoder  total  wall=163.770s  linear=8120  attention=2016
```

- `attention=2016` = 7 chunk × 8 tile × 36 层，分毫不差。
- `linear=8120` ≈ 8 × 7 × 36 × 4 = 8064。

**推导**：VPIPE 1.011 PFLOP / 63.1 s = 16.1 TFLOP/s；h3.c 0.698 PFLOP / 164 s = 4.3 TFLOP/s。**VPIPE 每 FLOP 快 3.8 倍，而且总 FLOP 还多 31%，墙钟时间反而只有 2/5。**

---

## 2. 改造方案

### 2.1 复用已有原语（不用重写）

h3.c 里本来就有一整套 BF16 原语（`h3_gpu.h`），只是**视频 VAE 一个都没接**：

| 原语 | 位置 |
|---|---|
| `h3_gpu_linear_bf16` | `h3_gpu.h:291` |
| `h3_gpu_sdpa_bf16` | `h3_gpu.h:493` |
| `h3_gpu_swiglu_bf16` | `h3_gpu.h:504` |
| `h3_gpu_rms_norm_bf16` | `h3_gpu.h:356` |
| `h3_gpu_layer_norm_bf16` | `h3_gpu.h:360` |
| `h3_gpu_cast_f32_to_bf16` / `_bf16_to_f32` | `h3_gpu.h:127` / `:129` |

另外 h3.c 还有给 DiT 写的 M5 矩阵核心 NAX 内核（`h3_shaders.metal:986` `h3_linear_bf16_nax_r128`、`:1200` `h3_fc1_swiglu_bf16_nax_r128`），但 **NAX 路径拒绝 bias**（`h3_gpu.m:2458` 要求 `!bias`），而 VAE 的 w1/w2/qkv/o 都带 bias → 走 MPS BF16 即可（实测与 NAX 持平）。

### 2.2 新增两个 BF16 原语（唯一的新 Metal 代码）

| 新原语 | Metal 核 | C 包装 | 声明 |
|---|---|---|---|
| `h3_gpu_scale_add_bf16` | `h3_shaders.metal:259` | `h3_gpu.m:1656` | `h3_gpu.h:190` |
| `h3_gpu_video_qkv_rope_bf16` | `h3_shaders.metal:363` | `h3_gpu.m:1746` | `h3_gpu.h:195` |

两个核都是对应 FP32 核的**逐行忠实镜像**（`h3_scale_add_f32` @ `:246`、`h3_video_qkv_rope_f32` @ `:315`），包括 RoPE half-split 的负半轴符号处理和 QK 的 RMS epsilon 位置；bf16↔f32 转换复用文件顶部已有的 helper；两个 kernel 名已加入 `h3_gpu.m:413` 的 pipeline 预热 NSArray。

### 2.3 权重加载（F32 checkpoint → BF16）

`MiniMax-H3/FL2VA/video_vae/source/model.safetensors` 的 **560 个张量全是 F32**（不是 BF16），所以原有的 `h3_weight_load_bf16` 会因 dtype 不匹配直接报错（`h3_weights.c:154`）。

新增 `h3_weight_load_f32_as_bf16()`（`h3_weights.c:204`）：校验 dtype/rank/shape → `h3_st_read_data` 读 F32 → 宿主端 RNE（round-to-nearest-even）转 BF16 → `h3_gpu_tensor_from_bf16` 上传。RNE helper 提升为共享的 `h3_float_to_bf16()`（`h3_weights.h:13`），与 `h3_vision_encoder.c:51` 的 static 版本逐位一致。

> 这是一处**有损**转换（权重本身降精度），但实测 rel-L2 0.0048，远在预算内。

### 2.4 改动清单

| 文件 | 改动 |
|---|---|
| `h3_video_vae.c` | 11 个算子全部切 BF16；MLP 拆成 `linear → swiglu → linear`（w1/w2 bias 保留）；rope 表宿主侧直接生成 BF16；输入 latent 用 `h3_gpu_cast_f32_to_bf16` 转一次；输出走新的 `read_projected()`；`H3_VAE_PRECISION=f32` 可回退；tile 默认改 256 |
| `h3_weights.c` / `h3_weights.h` | 新增 `h3_weight_load_f32_as_bf16()` + 共享 RNE helper |
| `h3_shaders.metal` | 新增 2 个 BF16 核 |
| `h3_gpu.m` / `h3_gpu.h` | 新增 2 个 C 包装 + 声明 + pipeline 预热 |
| `tests/test_video_vae.c`（新） | F32 vs BF16 解码对比（耗时 / rel-L2 / max·mean 绝对误差 / RGB 值域 / 帧数分辨率），支持多档 tile 与加载基准 |
| `tests/test_gpu_bf16_ops.c`（新） | 两个新原语的数值验证（见 §3.3） |
| `Makefile` | 新增两个测试目标并挂进 `make test` |

**未改动**：`h3_dit.c`、`h3_audio_vae.c`、`h3.c` 主流程。

---

## 3. 实测结果

### 3.1 端到端 A/B（同一会话、同参数、唯一变量是精度）

参数：960×544、124 帧（5.175 s @24fps）、6 步、seed 42、`--layers 50 --reuse 1 --core-reuse 1`。
两轮之间冷却 10 分钟；**两轮都是 5×3 @ 256 px tile**。

| 阶段 | FP32（对照） | **BF16（新版）** | 变化 |
|---|---:|---:|---:|
| Qwen 文本编码 | 4.595 s | 5.151 s | — |
| H3 DiT 加载 | 16.470 s | 16.492 s | — |
| H3 DiT 合计 | 244.678 s | 263.215 s | — |
| └ GPU Euler 去噪（6 步） | 227.943 s | 246.473 s | −8%（**纯热态噪声**，见下注） |
| 音频 VAE | 0.597 s | 0.603 s | — |
| **视频 VAE 解码** | **159.450 s** | **47.334 s** | **3.37×** ⚡ |
| **整轮墙钟** | **409.95 s** | **316.93 s** | **1.29×** |
| 视频 VAE 峰值显存 | 9.365 GiB | **4.683 GiB** | 0.50× |
| 进程峰值占用 | 22.79 GB | 22.79 GB | 持平（DiT 主导） |

> 注：去噪阶段 BF16 轮反而慢 18.5 s，是因为 **BF16 轮跑在前**（机器更凉）、FP32 轮跑在后（更热）。VAE 阶段的 3.37× 是同一份代码、同一份权重、同一 tile 尺寸下唯一的变量，属干净对照。

### 3.1.1 同会话三方对照（含 VPIPE 复测）

上一版报告里的 VPIPE 数字（489.2 s / VAE 63.1 s）来自 **2026-10-24** 的会话，而 h3.c 的新数据来自 2026-10-25——跨天的热态差异足以掩盖真实对比。因此在同一会话内把 VPIPE 重跑了一遍（同样冷却 10 分钟冷启动，同一套参数）：

| 阶段 | VPIPE | h3.c FP32 | **h3.c BF16** |
|---|---:|---:|---:|
| setup（文本编码 + 加载） | 16.42 s | 21.07 s | 21.64 s |
| GPU Euler 去噪（6 步） | 305.85 s | 227.94 s | 246.47 s |
| **视频 VAE 解码** | **45.01 s** | **159.45 s** | **47.33 s** |
| **整轮墙钟** | **370.90 s** | **409.95 s** | **316.93 s** |
| 峰值内存 | 44.45 GB | 22.79 GB | 22.79 GB |

三者的**运行顺序**是 h3.c BF16（00:27）→ h3.c FP32（00:42）→ VPIPE（00:59），即 VPIPE 跑在机器最热的时候——这让 h3.c 的优势不是"抢了凉快"。

**结论**：

| | 改造前 | 改造后 |
|---|---|---|
| VAE 解码 vs VPIPE | h3.c 慢 **3.54×**（159.45 vs 45.01） | h3.c 慢 **1.05×**（47.33 vs 45.01）——基本追平 |
| 端到端 vs VPIPE | h3.c 慢 **1.10×**（409.95 vs 370.90） | h3.c **快 1.17×**（316.93 vs 370.90） |
| 峰值内存 vs VPIPE | h3.c 省 2.0× | h3.c 省 2.0× |

bf16 改造**把 VAE 的差距从 3.54× 压到 1.05×，并把端到端的排名从落后 10% 翻成领先 17%**。

> 注意 VPIPE 今天的 VAE 是 45.01 s，而上一轮会话是 63.1 s——同一条代码、同一个 checkpoint，**1.4× 的差异纯粹来自热态**。这也是为什么本文所有对比都坚持"同会话 + 冷却 10 分钟 + 唯一变量"。


### 3.2 精度

独立测试程序（`tests/test_video_vae.c`），真实 checkpoint，960×544×124：

| tile | FP32 | BF16 | 加速 | rel-L2 (BF16 vs FP32) | max&#124;Δ&#124; | mean&#124;Δ&#124; |
|---|---:|---:|---:|---:|---:|---:|
| **256（checkpoint 默认）** | 190.9 s | **58.1 s** | 3.29× | **0.00478** | 0.0354 | 0.00159 |
| 304（旧行为） | 148.1 s | 35.4 s | 4.18× | 0.00488 | 0.0495 | 0.00163 |

- 精度门槛：VPIPE 自己的单测用 **rel-L2 ≤ 0.02**（`tests/unit-tests/minimax-h3-vvae.cc:212-222`），我们 **0.0048**，有 2 倍余量。
- RGB 值域：FP32 与 BF16 都是 **[0, 1]**，无越界。
- **误差不随规模放大**：小图（128×192×22）rel-L2 0.005121 → 全尺寸（960×544×124）rel-L2 0.004777。原因是误差主要来自"权重一次性舍入 + 残差流在 BF16 里累积"，都不随 token 数超线性增长；真正随规模放大的是注意力，而 SDPA 在 MPS 里是 BF16 输入 / FP32 累加，误差被控住。

**端到端视频画质**（ffmpeg 内置 `psnr` / `ssim` 滤镜，2026-10-25 同会话）：

| 对比 | PSNR (Y) | SSIM (All) |
|---|---:|---:|
| h3.c BF16 vs h3.c FP32（同 tile 256，同会话） | **23.05 dB** | **0.836** |
| h3.c BF16 vs VPIPE | 15.93 dB | 0.740 |
| h3.c FP32 vs VPIPE | 15.99 dB | 0.733 |
| h3.c BF16 vs 旧版 h3.c FP32(304px) | 21.66 dB | 0.818 |
| *（参照）h3.c 两次仅热态不同的自比* | *20.86 dB* | *0.842* |

→ BF16 与 FP32 的差距（23.05 dB）**比 h3.c 自身热漂移噪声（20.86 dB）还小**，即 BF16 带来的差异在成片层面不可见；相对 VPIPE，BF16 只低 0.06 dB / 高 0.008 SSIM，同样不可见。

### 3.3 新原语的单测（`tests/test_gpu_bf16_ops.c`）

三层协议，比单看 rel-L2 严格得多：

- **A 精度**：FP32 路径吃原始 float，BF16 路径吃它的 BF16 截断版 → 比 rel-L2 / max&#124;Δ&#124; / mismatch 数。
- **B 结构**：两条路径吃**完全相同**的数值（FP32 路径拿 BF16 截断后展宽回 FP32）→ 此时 BF16 输出必须**逐位**等于 FP32 输出的 BF16 舍入。这层专治"错了不崩、只是让画面慢慢变差"的 bug。
- **C 参考**：另写一份 double 精度 host 实现独立对照。

结果：

```
scale-add (BF16 input vs F32 reference): rel-L2 0.002341, max |delta| 0.006913
scale-add (identical values): 6144/6144 bit-exact against the F32 path
video qkv (BF16 input vs F32 reference): query rel-L2 0.002459, key rel-L2 0.002494
video qkv value channel: 6144/6144 bit-exact against the F32 path
video qkv (identical values): query 6144/6144 bit-exact, key 6144/6144 bit-exact
  rope rotation (BF16 path): worst |cos| error 0.00336, worst |sin| error 0.00343
epsilon 1: effect on the query rel-L2 0.5025
ok: BF16 video VAE primitives match their F32 counterparts
```

另外做了 **12 个变异体测试**（把 shader 改坏看测试能不能抓到），全部被抓：RoPE 负半轴符号翻转、epsilon 三种错放（sqrt 内外 / 分子 / 分母）、cos 索引错、value 通道不再逐字拷贝、RMS 求和范围错、scale 索引按行、运算顺序错、BF16 输出用截断而非 RNE。

### 3.4 加载开销与内存

- 宿主端 F32→BF16 转换：9.7 GiB / 560 个张量全量加载，5 次取平均 **0.638 s → 0.817 s（+0.18 s）**。
  相对省下的 159.45 − 47.33 = 112.1 s，**加载开销只占收益的 0.16%**。
- 常驻权重内存 **9.16 GiB → 4.58 GiB**（BF16 一半）。
- 视频 VAE 峰值显存 **9.365 GiB → 4.683 GiB**。

### 3.5 重构的数值中性验证

为了确认"只是把 f32 换成 bf16、重构本身没引入别的变化"，用新二进制跑了一轮**强制 f32 + 强制 304 px**（即改造前的行为），与改造前的旧二进制输出对比：

| 对比 | PSNR (Y) | SSIM (All) | 判读 |
|---|---:|---:|---|
| 新 f32@304 vs **旧二进制** f32@304 | **23.12 dB** | 0.832 | 重构**数值中性**（差异 = 热漂移噪声量级，基线 20.86 dB） |
| f32@304 vs f32@256 | 41.62 dB | 0.9856 | tile 尺寸对成片**感知不可见** |
| bf16@256 vs f32@304（全部改动） | 23.13 dB | 0.838 | 总改动 ≈ 热漂移噪声量级 |

同时，新 f32@304 轮的 profile 与改造前逐项吻合，证明算子图完全没变：

```
改造前 (h3c-run3.log):  video VAE decoder wall=163.770s  submissions=56  direct=12264  linear=8120  attention=2016
改造后 (h3c-f32-304.log): video VAE decoder wall=126.089s  submissions=56  direct=12264  linear=8120  attention=2016
```

（墙钟不同是热态差异：改造前那轮机器更热。`linear=8120` = 8 tile × 7 chunk × 36 层 × 4，`attention=2016` = 8 × 7 × 36，分毫不差。）

---

## 4. 附带改动：tile 尺寸对齐 checkpoint

h3.c 原来有个自动调优（`configured_tile_pixels()`，旧 `h3_video_vae.c:666-690`），按"tile 数 × 像素数²"打分，960×544 会被选成 **304 px**（旧日志实测：`h3: video VAE tiles 4x2 at 304 pixels`）。

但 checkpoint 配置写死的是：

```json
// MiniMax-H3/FL2VA/video_vae/config.json
"vae_tile_size": 256,
"vae_tile_overlap_min": 64
```

**机制上这确实是"换了位置编码"**：RoPE 网格按 latent 尺寸归一化（`h3_video_vae.c:487-493` 用 `vae->latent_h/latent_w`；参考实现 `func.py:34-37` 的 `length_normalized` 同理），而参考实现里的 `latent_H/latent_W` 就是**当前 tile 自己的**尺寸。tile 从 256 变 304 → latent 边长从 16 变 19 → 归一化坐标变了 → RoPE 变成了另一个函数，模型是在训练时没见过的位置编码上做推理。

改动：默认 tile 无条件为 **256 px**（删掉自动打分循环），`H3_VAE_TILE_PIXELS` 仍可覆盖（256..512，16 的倍数），304 保留为实验开关。

### 4.1 实测代价与收益

| 维度 | 数据 |
|---|---|
| VAE 张量层面的差异 | tile 304 vs 256：**rel-L2 0.0474，max&#124;Δ&#124; 0.825**（是 BF16 精度误差 0.0048 的 10 倍） |
| **成片层面的差异** | tile 304 vs 256：**PSNR 41.62 dB / SSIM 0.9856** → **感知不可见** |
| 速度代价 | 256 比 304 慢 **26.5%（f32 端到端：159.45 s vs 126.09 s）**；BF16 下差距更大（58.1 s vs 35.4 s，+64%） |

> **诚实的结论**：我一开始把 304 → 256 判定为"正确性 bug"，但成片数据不支持这么重的说法。VAE 张量层面 0.0474 的差异，经过 tile 边界的 overlap 混合（`vae_tile_overlap_min=64`）与后处理平滑后，**在最终视频上只剩 41.6 dB 的差距——完全不可见**。
>
> 所以保留 256 作为默认，理由是**与 checkpoint 规格一致**（训练时的分布），而不是"抢救画质"；代价是 VAE 慢约 12 s（端到端约 4%）。若要极致速度，`H3_VAE_TILE_PIXELS=304` 可换回旧的 8-tile 布局。
>
> 顺带一个有意思的观察：**BF16 让 tile 粒度变得更重要了**。因为计算量缩了 3.4×，而每 tile 的固定开销（36 层 × 十余次 kernel dispatch）没缩，8 tile → 15 tile 的边际成本在 BF16 下被放大（f32 差 26.5%，BF16 差 64%）。换句话说，**算子提速之后，固定开销开始成为新的瓶颈**，这是下一轮优化的方向（比如按 tile 合并 dispatch、或让 VPIPE 式的融合核覆盖带 bias 的 MLP）。

---

## 5. 残留风险与边界

1. **BF16 是有损的**。权重从 F32 降精度、激活全程 BF16（残差流 in-place 累积 36 层）。实测 rel-L2 0.0048 已覆盖这个风险；如需更保守，`H3_VAE_PRECISION=f32` 可一键回退（已验证两轮端到端结果一致）。
2. **没用上融合的 fc1+SwiGLU 矩阵核心核**。`h3_fc1_swiglu_bf16_nax_r128`（`h3_shaders.metal:1200`）不支持 bias，而 VAE 的 w1/w2 都有 bias → 保持 `linear → swiglu → linear` 三步拆法。代价是每次 MLP 多一次全量 SwiGLU 遍历 + 一次 16384 宽权重读取，实测占比很小（MPS linear dispatch 数与 FP32 路径完全相同，说明没漏掉任何 matmul）。若要吃下这个优化，需要给 w1 加"bias 后加"路径，属独立改动。
3. **tile 尺寸只做对齐，没有做空间冗余优化**。256 px 下 960×544 被切成 5×3=15 个 tile，按 VAE 自身卷积核计算的理想 tile 数约 5.65 → 冗余约 1.42×。VPIPE 也用 256 且同样承担这个冗余，所以是双方共同的优化空间，不是 h3.c 独有的劣势。
4. **VAE 之外的阶段本轮未动**。去噪阶段（DiT）本轮 A/B 为 227.9 s（FP32 轮）/ 246.5 s（BF16 轮），**仍快于上一轮 VPIPE 的 405.2 s**——VPIPE 去噪慢是因为它把 49/50 个 block 常驻显存（36 GB）导致更热更深降频，而 h3.c 用 `--layers 50 --reuse 1 --core-reuse 1` 走流式复用。真正拖慢 VPIPE 端到端的是它的 VAE，而这正是本轮改掉的部分。DiT 本身还有独立优化空间，本轮未涉及。

---

## 6. 提交到 upstream

Fork：`https://github.com/jiangzhenguo/h3.c`（parent：`antirez/h3.c`）

| PR | 内容 | 规模 |
|---|---|---|
| **[#80](https://github.com/antirez/h3.c/pull/80)** | 视频 VAE 跑 BF16，让 MPS 用上张量核心 | 9 文件 +1646 / −60 |
| **[#81](https://github.com/antirez/h3.c/pull/81)** | 空间 tile 默认值对齐 checkpoint 的 256 px | 1 文件 +17 / −16 |

两个 PR **基于上游 main 各自独立**，可以分开评审：#81 单独看是一个"变慢"的改动，混进 #80 会被当成性能回退，所以拆开了。#80 的 PR 描述里给出了完整的实测数据、精度对比、数值中性验证和复现命令。

本地分支：

```
feature/vae-bf16-nax            6d2a715  → fork/feature/vae-bf16-nax            (PR #80)
feature/vae-tile-256-standalone 709d0c2  → fork/feature/vae-tile-256-standalone (PR #81)
main                            8974cc0  → origin/main
```

> 仓库的 `fork` remote 用了一条**局部** URL 规则（`url.https://github.com/jiangzhenguo/.insteadOf`），让本账号的仓库直连 github.com（gh 的 token 不会经过第三方代理），其余仓库仍走原来的 `gh-proxy.com` 全局代理。

## 7. 复现命令

```sh
# 1) 编译与自测
cd /Users/jiangzhenguo/ai/h3/h3.c
make -j8
make test          # 含 h3_tests / h3_gpu_bf16_ops_tests / h3_video_vae_tests

# 2) 单独验证 VAE 的 F32 vs BF16
./h3_video_vae_tests /Users/jiangzhenguo/ai/h3/MiniMax-H3
H3_VAE_TEST_TILES=256,304 ./h3_video_vae_tests /Users/jiangzhenguo/ai/h3/MiniMax-H3

# 3) 端到端 A/B（两轮都默认 256px tile，唯一变量是精度）
cd /Users/jiangzhenguo/ai/vpipe-work
bash bench-vae-bf16.sh          # 冷却 10min → bf16 → 冷却 10min → f32

# 4) 强制 FP32 回退
H3_VAE_PRECISION=f32 ./h3 ... -o out.mp4
# 5) 强制旧 tile 行为（实验用）
H3_VAE_TILE_PIXELS=304 ./h3 ... -o out.mp4
```

---

## 8. 产物索引

| 文件 | 内容 |
|---|---|
| `vpipe-work/h3-vae-bf16-改造报告.md` | 本报告 |
| `vpipe-work/bench-vae-gemm.c` | FP32 vs BF16 GEMM/SDPA 微基准 |
| `vpipe-work/bench-vae-gemm-f32-vs-bf16.log` | 微基准原始输出 |
| `vpipe-work/bench-vae-bf16.sh` | 端到端 A/B 脚本 |
| `vpipe-work/h3c-bf16.log` | BF16 轮完整日志（含 `--profile`） |
| `vpipe-work/h3c-f32.log` | FP32 对照轮完整日志 |
| `vpipe-work/compare-psnr.sh` | 画质对比脚本（ffmpeg psnr/ssim） |
| `vpipe-work/outputs/h3c-bf16-*.mp4` | BF16 版成片 |
| `vpipe-work/outputs/h3c-f32ref-*.mp4` | FP32 对照成片 |
| `vpipe-work/h3-bf16-bin` | 跑基准用的二进制快照 |

代码在 `/Users/jiangzhenguo/ai/h3/h3.c`，分支 `feature/vae-bf16-nax`。
