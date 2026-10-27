# VPIPE 对比与 VAE BF16 改造归档

本目录是 2026-10-24 ~ 2026-10-25 期间，在本机（MacBook Pro 16″ / Apple M5 Pro / 64 GB）
完成的 **VPIPE ↔ h3.c 速度与画质对比**，以及由此产生的 **h3.c 视频 VAE BF16 改造**
的全套材料归档：报告、复现脚本、原始日志、成片、补丁。

所有数据均为本机实跑得出，未引用任何官方数字。

## 结论速览

| 结论 | 数据 |
|---|---|
| 改造前 h3.c 的视频 VAE 解码比 VPIPE 慢 3.54×（159.45 s vs 45.01 s） | 见 [对比报告 §3.5](reports/VPIPE-vs-h3c-对比报告.md) |
| 根因不是架构差异，是**数值精度**：两边都是 36 层 / 2048 宽 ViT 解码器；h3.c 全程 FP32，MPS 的 FP32 matmul 不经过张量核心（实测 6.2 TFLOP/s ≈ FP32 SIMD 理论峰值） | 见 [改造报告 §1](reports/h3-vae-bf16-改造报告.md) |
| 改成 BF16 后：VAE 解码 **159.45 s → 47.33 s（3.37×）**，显存峰值 9.37 → 4.68 GiB，相对 VPIPE 从慢 3.54× 变成慢 1.05× | 见 [改造报告 §3.1](reports/h3-vae-bf16-改造报告.md) |
| 端到端从慢 1.10× 变成**领先 1.17×**（409.95 s → 316.93 s） | 同上 |
| 精度代价可忽略：rel-L2 **0.0048**（VPIPE 自身门槛 0.02），成片 23.05 dB / 0.836，比 h3.c 仅热态不同的自比（20.86 dB / 0.842）还接近 | 见 [对比报告 §3.5](reports/VPIPE-vs-h3c-对比报告.md) |
| 顺带把 VAE tile 默认值对齐回 checkpoint 的 `vae_tile_size` = **256 px**（原先自动挑成 304 px，等于换了 RoPE 位置编码函数） | 见 [改造报告 §4](reports/h3-vae-bf16-改造报告.md) |
| 内存敏感场景仍应选 h3.c：峰值内存 22.8 GB vs VPIPE 45.9 GB | 见 [对比报告 §3.5](reports/VPIPE-vs-h3c-对比报告.md) |
| 之后的第二轮实验：**tile 维度合批是净亏**（慢 13%），VAE 解码是算力瓶颈，唯一剩下的杠杆是降精度 | 见 [tile 合批实验报告](reports/vae-tile-batch-实验报告.md) |
| 第三轮：**全量 INT8 视频 VAE**，解码再快 **1.48×**（成对中位数），误差比 H.264 编码器自身低 30 dB，最终成片里不可见 | 见 [INT8 实验报告](reports/vae-int8-实验报告.md) |
| 一个反直觉的细节：**最后的输出投影不能量化**——它是进像素前最后一步，误差没人平均，单独值 11.4 dB | 见 [INT8 实验报告 §2.4](reports/vae-int8-实验报告.md) |

## 对应的代码改动

| 改动 | 分支 | 提交 |
|---|---|---|
| 视频 VAE 的 ViT decoder 走 BF16，用上 MPS 张量核心 | `feature/vae-bf16-nax` | `6d2a715` |
| VAE 空间 tile 默认固定为 checkpoint 的 256 px | `feature/vae-tile-256-standalone` | `709d0c2` |
| 合并到 `main` | — | `a417bf6` |
| tile 合批实验（否定结论，保留微基准与报告） | `261027_jzgBatchTileLinear` | `5c10b4d` |
| 视频 VAE 全量 INT8 路径（`H3_VAE_PRECISION=int8`） | `261027_jzgBatchTileLinear` | 见 [INT8 实验报告](reports/vae-int8-实验报告.md) |

完整补丁见 [`patch/vae-bf16-full.patch`](patch/vae-bf16-full.patch)（该补丁对应的两个提交已合入
`main`，此处仅作为存档与 review 依据）。

## 目录结构

```
docs/vpipe/
├── README.md                              本文件：索引与结论
├── reports/
│   ├── VPIPE-vs-h3c-对比报告.md            速度/画质/热降频/画质对照全量报告
│   ├── h3-vae-bf16-改造报告.md             BF16 改造的根因、方案与全部实测数据
│   ├── pr-body-bf16.md                    给上游的 BF16 改造 PR 正文
│   ├── pr-body-tile.md                    给上游的 tile-256 PR 正文
│   ├── vae-tile-batch-实验报告.md         tile 合批实验（否定结论 + 算力瓶颈论证）
│   └── vae-int8-实验报告.md               全量 INT8 实验（1.48× + 保真度评估）
├── scripts/                               复现用脚本与微基准源码
│   ├── controlled-bench.sh                受控基准（冷却 → VPIPE → 冷却 → h3.c）
│   ├── controlled-bench-i8off.sh          受控基准 + VPIPE 关 i8_gemm
│   ├── bench-vae-bf16.sh                  同会话 A/B：bf16 轮 vs f32 对照轮
│   ├── bench-vpipe-recheck.sh             同会话 VPIPE 复测（消除跨天热态差异）
│   ├── bench-f32-304.sh                   数值中性验证轮（强制 f32 + 旧 304 px tile）
│   ├── compare-psnr.sh                    画质对比（ffmpeg psnr / ssim 滤镜）
│   ├── sample-thermal.sh                  GPU 时钟/温度采样
│   └── bench_vae_gemm.c                   FP32 vs BF16 GEMM/SDPA 微基准
├── config/
│   ├── h3-t2v-compare.vpipeline           对比用 VPIPE pipeline（直指本地 bf16 权重）
│   └── prompt.txt                         Context-IR 五段式提示词
├── patch/
│   └── vae-bf16-full.patch                BF16 改造完整补丁（存档）
├── run-logs/                              全部原始日志（VPIPE / h3.c / 热态采样 / 微基准）
└── media/
    ├── vpipe-960x544-124f-s6-seed42.mp4   VPIPE 产物
    ├── h3c-*.mp4                          h3.c 各轮产物（bf16 / f32 / f32-304 / run2 / run3）
    └── frames/                            第 60 帧逐像素对照图（h3c_f60.png / vpipe_f60.png）
```

## 复现命令

```sh
# —— VPIPE
cd /Users/jiangzhenguo/ai/vpipe-work
unset VPIPE_I8_GEMM VPIPE_I8_TIER
/Users/jiangzhenguo/ai/vpipe/build/apps/vpipe/vpipe \
  --launch h3-t2v-compare.vpipeline \
  --stage-cfg 'generate-video::i8_gemm=false'   # 逐像素对照时加这条

# —— h3.c（exact 模式：不加 --ssd-streaming，64 GB 装得下）
cd /Users/jiangzhenguo/ai/h3/h3.c
./h3 -d /Users/jiangzhenguo/ai/h3/MiniMax-H3 \
  -p "$(cat /Users/jiangzhenguo/ai/vpipe-work/prompt.txt)" \
  --width 960 --height 544 --frames 124 --steps 6 --seed 42 \
  --layers 50 --reuse 1 --core-reuse 1 --profile \
  -o /Users/jiangzhenguo/ai/vpipe-work/outputs/h3c.mp4
```

## 测试口径

| 项 | 值 |
|---|---|
| 分辨率 / 时长 | 960 × 544，124 帧，24 fps → 5.175 s |
| 去噪步数 | 6 个 DiT 步 |
| 随机种子 | 42 |
| 权重 | bf16 发布快照（与 h3.c 读同一份文件），无 LoRA、无量化 |
| h3.c 模式 | `--layers 50 --reuse 1 --core-reuse 1`（exact 模式，全 50 块、零步复用） |
| VPIPE 模式 | `i8_gemm=true`，`motion_cache` / `sol_attn` / `sage_attn` / `sage_pv8` 全关 |

> ⚠️ 本机 64 GB M5 Pro 在持续负载下 GPU 由 1618 MHz 降至 ~1230 MHz（-24%），壳温 97 °C。
> 官方 README 那条对比来自 24 GB 的 M5 Pro 且 GPU 全程锁在 1620 MHz，两项前提都不同，
> 因此本目录中的所有绝对耗时都只在本机口径内横向比较。
