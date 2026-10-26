# VPIPE vs h3.c — MiniMax-H3 视频生成速度对比（本机实测）

> 实测日期：2026-10-24 · 全部数据在本机实跑得出，非引用官方数字。

---

## 1. 测试机器

| 项 | 值 |
|---|---|
| 型号 | MacBook Pro 16″（Mac17,9） |
| 芯片 | Apple M5 Pro，18 核（6P + 12E） |
| 内存 | 64 GB 统一内存 |
| GPU | Apple M5 Pro（applegpu_g17s），Metal 4，**矩阵核心可用** |
| 系统 | macOS 27.0.1 |
| 存储 | 1.8 TB NVMe，405 GB 空闲；权重 196 GB 在内部 SSD |
| 散热 | 有风扇；**持续负载下 GPU 由 1618 MHz 降至 ~1230 MHz（-24%）**，壳温达 97 °C |

> ⚠️ 官方 README 里那条 M5 Pro 记录（VPIPE 5分29秒 / h3.c 7分11秒）来自一台 **24 GB** 的 M5 Pro MBP，且其 GPU **全程锁在 1620 MHz**。本机是 64 GB 且会热降频，两项前提都不同 —— 这是下文差异解读的关键。

## 2. 工作负载（与官方对比口径一致）

| 项 | 值 |
|---|---|
| 分辨率 / 时长 | 960 × 544，124 帧，24 fps → **5.175 s** |
| 去噪步数 | **6 个 DiT 步** |
| 随机种子 | 42 |
| 权重 | **bf16 发布快照**（MiniMaxAI 版，与 h3.c 读的是同一份文件），**无 LoRA、无量化** |
| 提示词 | 同一条 Context-IR 五段式提示词（Scene/Action/Camera/Look/Audio） |
| h3.c 模式 | `--layers 50 --reuse 1 --core-reuse 1`（**exact 模式**：全 50 块、零步复用），不加 `--ssd-streaming`（64 GB 装得下） |
| VPIPE 模式 | `i8_gemm=true`（复现官方 M5 行），`motion_cache` / `sol_attn` / `sage_attn` / `sage_pv8` **全部关闭** |

产物三方规格完全一致：960×544 / 124 帧 / 24 fps / 5.175 s / 32 kHz 立体声 AAC。

## 3. 主结果 —— 受控对照（两轮各冷却 10 分钟后冷启动）

| 阶段 | VPIPE | h3.c | 优势方 |
|---|---:|---:|---|
| 文本编码器 + DiT 准备 | 17.3 s | 22.0 s | VPIPE 1.27× |
| **去噪，6 步** | **405.2 s**（6分45秒） | **306.6 s**（5分07秒） | **h3.c 1.32×** |
| **VAE 解码（视频 + 音频）** | **63.1 s** | **164.6 s** | **VPIPE 2.61×** |
| **合计（启动→写盘）** | **489.2 s（8分09秒）** | **494.0 s（8分14秒）** | **打平（差 1.0%）** |
| 峰值内存 | **45.9 GB** | **22.8 GB** | h3.c 少用 2.0× |

## 3.5 追测：h3.c 视频 VAE 改 BF16 之后（2026-10-25）

> 上一轮发现的结论是"VPIPE 的 VAE 解码快 2.61×"。**追查后确认这不是架构差异**——两边的解码器都是 36 层 / 2048 宽 ViT，VPIPE 文档里"卷积 + 矩阵核心快 2.8×"说的是**编码器**，不是解码器。真正的根因是 **h3.c 全程用 FP32，MPS 的 FP32 matmul 不经过张量核心**（实测 6.2 TFLOP/s ≈ FP32 SIMD 理论峰值，即张量核心完全闲置）。
> 细节见独立报告：`vpipe-work/h3-vae-bf16-改造报告.md`。

改造后（同一套参数，两轮各冷却 10 分钟，**唯一变量是精度**）。为消除跨天热态差异，**VPIPE 也在同一会话内重跑了一遍**（同日、同样冷却 10 分钟冷启动；运行顺序 h3.c BF16 → h3.c FP32 → VPIPE，即 VPIPE 跑在最热的时候）：

| 阶段 | VPIPE（同会话复测） | h3.c FP32（对照） | **h3.c BF16** | 变化 |
|---|---:|---:|---:|---:|
| setup（文本编码 + 加载） | 16.42 s | 21.07 s | 21.64 s | — |
| GPU Euler 去噪（6 步） | 305.85 s | 227.943 s | 246.473 s | — |
| **视频 VAE 解码** | **45.01 s** | **159.450 s** | **47.334 s** | **3.37×**；相对 VPIPE 从慢 3.54× 变成慢 1.05× |
| **整轮墙钟** | **370.90 s** | **409.95 s** | **316.93 s** | **1.29×**；相对 VPIPE 从慢 1.10× 变成**快 1.17×** |
| 视频 VAE 峰值显存 | — | 9.365 GiB | **4.683 GiB** | 0.50× |
| 峰值内存 | 44.45 GB | 22.79 GB | 22.79 GB | h3.c 少用 2.0× |

> 注：VPIPE 今天的 VAE 是 45.01 s，而 §3 表里 2026-10-24 那轮是 63.1 s——同一条代码、同一个 checkpoint，**1.4× 的差异纯粹来自热态**。这也是为什么这里坚持同会话复测。

画质：BF16 vs FP32 相对误差 **rel-L2 0.0048**（VPIPE 自己的门槛是 0.02）；成片层面 PSNR 23.05 dB / SSIM 0.836，**比 h3.c 自身仅热态不同的两次自比（20.86 dB / 0.842）还接近**，即 BF16 的差异不可见。相对 VPIPE：BF16 15.93 dB / 0.740，FP32 15.99 dB / 0.733，同样不可区分。

另修一处规格偏差：h3.c 原来把 tile 自动挑成 304 px，而 checkpoint 的 `vae_tile_size` 是 **256 px**；RoPE 网格按 tile 的 latent 尺寸归一化，304 等于换了位置编码函数。已改回 256。**但实测成片差异只有 41.62 dB / SSIM 0.986（感知不可见）**——所以这是"与规格对齐"，不是画质抢救；代价是 VAE 慢约 26%。

重构的数值中性也做了验证：新二进制强制 `H3_VAE_PRECISION=f32 H3_VAE_TILE_PIXELS=304`（改造前的行为）与旧二进制的输出差异 **23.12 dB / 0.832**，高于 h3.c 自身热漂移噪声基线（20.86 dB）；profile 逐项吻合（`submissions=56 direct=12264 linear=8120 attention=2016`）。

## 4. 三轮全量（含非受控的连续运行）

| 轮次 | VPIPE 总耗时 | h3.c 总耗时 |
|---|---:|---:|
| run1（紧邻上一次运行） | 460.3 s（7分40秒） | 470.4 s（7分50秒） |
| run2（紧邻上一次运行） | 506.5 s（8分26秒） | 501.6 s（8分22秒） |
| run3（冷却 10 分钟后冷启动） | 489.2 s（8分09秒） | 494.0 s（8分14秒） |
| run4（冷却 10 分钟后冷启动，`i8_gemm` 关） | 567.0 s（9分27秒） | — |
| **均值（run1–3）** | **485.3 s** | **488.7 s** |
| **最好成绩** | **460.3 s** | **470.4 s** |

**结论：在本机（64 GB M5 Pro MBP）上两者端到端基本打平（VPIPE 快 0.7%，在噪声范围内）。**
差异不在总量，而在结构：**h3.c 的去噪快 1.32×，VPIPE 的 VAE 解码快 2.61×** —— 后者单项就把前者的劣势补回来了。

## 5. 与官方 README 数字的差异及其原因

| 来源 | VPIPE 去噪 | h3.c 去噪 | VPIPE VAE | h3.c VAE | VPIPE 总 | h3.c 总 |
|---|---:|---:|---:|---:|---:|---:|
| 官方 M5 Pro 16″ **24 GB** | 4分25秒 | 5分00秒 | 43 s | 1分59秒 | 5分29秒 | 7分11秒 |
| **本机 M5 Pro 16″ 64 GB** | 6分45秒 | 5分07秒 | 63 s | 2分45秒 | 8分09秒 | 8分14秒 |

- **VAE 解码：VPIPE 稳定领先 2.6–2.8×**，与官方一致（43 s vs 119 s）。这是 VPIPE 用矩阵核心做卷积解码的实打实收益。
- **去噪：VPIPE 比官方慢 53%，而 h3.c 与官方基本持平（+2%）。**
  主因是**热降频**：本机 GPU 在持续负载下从 1618 MHz 掉到 ~1230 MHz（有效约 1290 MHz ≈ 峰值的 80%）。VPIPE 的去噪时间几乎全部落在降频区。
- 为什么 VPIPE 比 h3.c 更容易被降频拖垮？VPIPE 把 **49/50 个 block 常驻内存**（36 GB，峰值 45.9 GB），h3.c 只常驻约 21 GB（峰值 22.8 GB）。**常驻权重越多 → 内存子系统越热 → GPU 降频越早、越深**。官方那台 24 GB 机器上 VPIPE 只能常驻 14 个 block，负载更轻，反而能锁在 1620 MHz。
- 另有两项无法在本机排除的次要差异：① 本机 VPIPE 走 **Metal runtime-compile 模式**（Metal Toolchain 组件下载在本机卡住未完成，与 h3.c 的 `newLibraryWithSource:` 同口径，但官方编译用的是 AOT metallib）；② VPIPE 读的是**发布方 per-head QKV 分组**权重，官方对比用的是 Comfy-Org flat 分组重打包版，GEMM 形状不同。

## 6. `i8_gemm` 开关实验（VPIPE 侧，第 4 轮）

同样冷却 10 分钟冷启动，只把 `generate-video::i8_gemm` 从 `true` 改成 `false`：

| | 去噪 6 步 | 后半程 GPU 持续频率 | 折算每 MHz 耗时 | VAE 解码 | 总计 |
|---|---:|---:|---:|---:|---:|
| `i8_gemm=true`（run3） | 405.2 s | 1254 MHz | 0.323 s/MHz | 63.1 s | 489.2 s |
| `i8_gemm=false`（run4） | 477.6 s | 1157 MHz | 0.413 s/MHz | 70.0 s | 567.0 s |

**`i8_gemm` 确实有效**：按 GPU 频率归一化后，关掉它每 MHz 要多花 28% 的时间（且这一轮整体散热更差，所以 0.413 vs 0.323 里有部分属于热状态差异，真实收益介于 10%–28% 之间）。它是有损量化，**评判画面质量时应关掉，比速度时应开着**。

### VPIPE DiT 逐算子耗时（run4，`VPIPE_H3_DIT_PROFILE=1`）

```
19386 rows (102 text + 0 cond + 414 audio + 18870 video), 50 blocks, 每步约 80.5 s/2 步
  input+refiner    7 ms     adaln     2.5 ms  (20914 GF/s)
  qkv         11902 ms (18830 GF/s)    o-proj   4052 ms (18438 GF/s)
  ff          25549 ms (17544 GF/s)    attn    37139 ms (14507 GF/s, steel flash)
```

- **注意力是最大头**：每步约 18.6 s（占 47%），且走的是 "steel flash" 路径，吞吐 14.5 TF/s 最低。
- FFN 每步约 12.8 s（32%），QKV 约 6.0 s（15%），O-proj 约 2.0 s（5%）。
- 首步额外一次性开销：`adaln 8093 ms` —— 把 24.87 GB 的 AdaLN 权重烘焙成 101 MB 查表。
- 归一到 1620 MHz 上限后，VPIPE 的 GEMM 达到 ~24.7 TF/s，约峰值的 75–80% —— 说明它**没有跑在理论上限上，瓶颈是热降频而非算子效率**。
- 同样的 GF/s 在后续步骤逐步下滑（qkv 20334 → 19346 → 18637 → 18660 → 18830），正是降频曲线在算子耗时上的投影。

## 7. 画质对照

| 对比 | PSNR (Y) | SSIM (All) |
|---|---:|---:|
| VPIPE vs h3.c（同 seed 同参数） | 14.08 dB | 0.728 |
| h3.c run1 vs run3（同运行时同参数，仅热状态不同） | 20.86 dB | 0.842 |

两者**不是逐像素对齐**：步数计数口径（VPIPE 用 ComfyUI/diffusers 口径的"前向次数"，h3.c 网格点数口径差 1）、调度细节与 `i8_gemm` 的有损量化都不同。同一个 h3.c 自己两次跑也只有 20.9 dB，说明 Metal 浮点归约顺序在热状态变化下也不是逐位确定的。

**若需要逐像素公平比较**：VPIPE 关掉 `i8_gemm`（`--stage-cfg 'generate-video::i8_gemm=false'`），且不要加 `--ssd-streaming`（两边都保持 bf16 常驻）。

## 8. 结论与建议

> ⚠️ **2026-10-25 更新**：下面第 1 条关于 VAE 的归因**已被修正并推翻**。VPIPE 的 VAE 优势不是"用了卷积"（那是编码器），而是 **BF16 张量核心 GEMM**；h3.c 当时用 FP32，MPS 的 FP32 matmul 完全不碰张量核心（实测 6.2 TFLOP/s = FP32 SIMD 理论峰值）。把 h3.c 的视频 VAE 改成 BF16 后，**同会话复测：VAE 解码 159.45 s → 47.33 s（3.37×），相对 VPIPE 从慢 3.54× 变成慢 1.05×；端到端 409.95 s → 316.93 s，从慢 1.10× 变成快 1.17×**。详见 §3.5 与 `h3-vae-bf16-改造报告.md`。以下第 1 条按当时数据保留以存档。

1. ~~**想更快出片：VPIPE 的卖点是 VAE 解码（快 2.6×）和端到端编排（一条 pipeline 出带声 mp4），而不是去噪。**~~ 在 64 GB 机器上它与 h3.c 总耗时打平。
2. **内存敏感场景选 h3.c**：峰值 22.8 GB vs VPIPE 45.9 GB，差 2 倍。官方 README 给 h3.c 加 `--ssd-streaming` 是为了 16/24 GB 小内存机器；**64 GB 机器不要加**，加上会把去噪拖慢。
3. **想复刻官方那条"VPIPE 快 1.31×"**：需要 24 GB 级内存的机器（VPIPE 常驻 block 少 → 发热低 → 能锁高频）。**64 GB 反而让 VPIPE 更容易降频**。
4. **想减少热降频的影响**：给机器留散热空间、避免连着跑、跑前冷却 10 分钟；连续跑第 2 轮普遍慢 5–10%。
5. **省时间的实用档**：VPIPE 侧用 Turbo LoRA（4–8 步）可把 6 步再压掉一半以上，h3.c 侧用 `--reuse 2 --layers 45`（官方标注人物场景下会有面部时序混叠）。

## 9. 复现命令

```sh
# —— VPIPE（本机）
cd /Users/jiangzhenguo/ai/vpipe-work
unset VPIPE_I8_GEMM VPIPE_I8_TIER
/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe \
  --launch h3-t2v-compare.vpipeline \
  --stage-cfg 'generate-video::i8_gemm=false'   # 逐像素对照时加这条

# —— h3.c（exact 模式）
cd /Users/jiangzhenguo/ai/h3/h3.c
./h3 -d /Users/jiangzhenguo/ai/h3/MiniMax-H3 \
  -p "$(cat /Users/jiangzhenguo/ai/vpipe-work/prompt.txt)" \
  --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
  --layers 50 --reuse 1 --core-reuse 1 --profile \
  -o /Users/jiangzhenguo/ai/vpipe-work/outputs/h3c.mp4
```

## 10. 产物与日志索引

| 文件 | 内容 |
|---|---|
| `vpipe-960x544-124f-s6-seed42.mp4` | VPIPE 产物（960×544 / 5.175 s / 带音轨） |
| `outputs/h3c-960x544-124f-s6-seed42.mp4` | h3.c 产物（规格完全相同） |
| `vpipe-run1.log` / `vpipe-run2.log` / `vpipe-run3.log` | VPIPE 三轮全量日志（逐阶段带时间戳） |
| `vpipe-run4.log` | 第 4 轮：`i8_gemm` 关闭 + 逐算子 DiT profile |
| `h3c-run1.log` / `h3c-run2.log` / `h3c-run3.log` | h3.c 三轮全量日志（含 `--profile` 分阶段计时） |
| `thermal-vpipe3.log` / `thermal-h3c3.log` / `thermal-vpipe4.log` | 受控轮次的 GPU 时钟/温度采样 |
| `controlled-bench.sh` / `controlled-bench-i8off.sh` | 受控基准脚本（冷却 → VPIPE → 冷却 → h3.c） |
| `h3-t2v-compare.vpipeline` | 对比用 pipeline（`hf_dir` 直指本地 bf16 权重） |

### 10.1 2026-10-25 bf16 改造轮新增产物

| 文件 | 内容 |
|---|---|
| `h3-vae-bf16-改造报告.md` | **bf16 改造的完整报告**（根因、方案、全部实测数据） |
| `bench-vae-gemm.c` / `bench-vae-gemm-f32-vs-bf16.log` | FP32 vs BF16 GEMM/SDPA 微基准（证明张量核心闲置） |
| `bench-vae-bf16.sh` / `h3c-bf16.log` / `h3c-f32.log` | 同会话 A/B：bf16 轮 vs f32 对照轮 |
| `bench-vpipe-recheck.sh` / `vpipe-recheck.log` | 同会话 VPIPE 复测（消除跨天热态差异） |
| `bench-f32-304.sh` / `h3c-f32-304.log` | 数值中性验证轮（强制 f32 + 旧 304 px tile） |
| `compare-psnr.sh` | 画质对比脚本（ffmpeg 内置 psnr / ssim 滤镜） |
| `outputs/h3c-bf16-*.mp4` / `outputs/h3c-f32ref-*.mp4` / `outputs/h3c-f32-304-*.mp4` | 三轮成片 |
| `h3-bf16-bin` | 跑基准用的二进制快照 |
| `h3.c` 代码 | `/Users/jiangzhenguo/ai/h3/h3.c`，分支 `feature/vae-bf16-nax`，提交 `d70e9d7` |

## 11. 安装位置

| 组件 | 路径 |
|---|---|
| VPIPE 源码 | `/Users/jiangzhenguo/ai/vpipe` |
| VPIPE 可执行文件 | `/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe` |
| VPIPE Web UI | `/Users/jiangzhenguo/ai/vpipe/build/apps/web-ui/vpipe-web-ui` |
| 工作目录（LMDB 注册表 / 模型 / 产物） | `/Users/jiangzhenguo/ai/vpipe-work` |
| 复用的权重（未下载、未量化） | `/Users/jiangzhenguo/ai/h3/MiniMax-H3/FL2VA`（bf16 发布快照） |
| h3.c 二进制 | `/Users/jiangzhenguo/ai/h3/h3.c/h3` |

Web UI 启动方式（从工作目录启动，才能找到模型注册表）：

```sh
cd /Users/jiangzhenguo/ai/vpipe-work
/Users/jiangzhenguo/ai/vpipe/build/apps/web-ui/vpipe-web-ui --bind 127.0.0.1
```

> **省下的 115 GB 下载**：`model-select` 的 `hf_dir` 直接写绝对路径即可复用本机已有 bf16 权重，既不用 `model-fetch` 下载 115 GB，也不用 `model-quantize` 量化（bf16 权重可直接运行，loader 会自己读出每个张量的位宽）。
