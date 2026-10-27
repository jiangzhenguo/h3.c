# INT8 视频 VAE 复现脚本

配套报告：[`../../reports/vae-int8-实验报告.md`](../../reports/vae-int8-实验报告.md)。

## 工具

| 文件 | 作用 |
|---|---|
| `vae_bench.c` | 解码一个 dump 出来的 latent，输出墙钟时间和相对参考解码的误差。已并入仓库 `tools/vae_bench.c`，用 `make h3_vae_bench` 构建 |
| `sweep.py` | 一键跑一串 (精度 × tile) 组合，把结果写成 `results.jsonl` |
| `paired.py` | 交替成对跑两个配置，输出**相邻成对**的耗时比值（消除热漂移） |
| `psnr.py` / `psnr_pairs.py` | `.rgbf32` dump 之间的 PSNR / RMSE / max_abs |
| `ssim.py` | `.rgbf32` dump 之间的逐帧 SSIM（11×11 高斯窗，K1=0.01 K2=0.03） |
| `codec_floor.py` | 把最终 MP4（H.264）解码回来，和 F32 解码比——量化"编码器自身的误差"这条噪声底线 |
| `groupsim.py` | 纯 numpy 模拟：逐行 vs 分组激活量化的误差（用于判断要不要写 grouped kernel） |
| `e2e_int8.sh` | 端到端 A/B：同一 prompt / 几何 / 种子，INT8 与 BF16 各跑一轮成片 |

## 依赖的外部路径

脚本按本机布局写死，搬家时改这几个常量即可：

| 变量 | 本机值 |
|---|---|
| `BENCH` | `/tmp/h3bench`（RGB dump、latent、results.jsonl 都在这） |
| `MODEL` | `/Users/jiangzhenguo/ai/h3/MiniMax-H3` |
| `FORK` | `/Users/jiangzhenguo/ai/h3/h3.c` |
| `PR73` | `/Users/jiangzhenguo/ai/h3/h3-pr73`（上游 PR #73 的 worktree，用于对照） |

## 典型流程

```sh
# 1) 构建基准工具
cd /Users/jiangzhenguo/ai/h3/h3.c && make h3_vae_bench

# 2) 单次解码，测误差
H3_VAE_PRECISION=int8 H3_VAE_TILE_PIXELS=256 ./h3_vae_bench \
    $MODEL/FL2VA/video_vae/source /tmp/h3bench/latent.bin \
    /tmp/h3bench/rgb/out.rgbf32 /tmp/h3bench/rgb/f32-256.rgbf32

# 3) 成对 A/B（只信相邻成对的比值）
python3 paired.py

# 4) 与编码器噪声底线对比
ffmpeg -i out-fork-bf16.mp4 -f rawvideo -pix_fmt rgb24 rgb/h264.rgbf24 -y
python3 codec_floor.py
```

> `sweep.py` / `paired.py` 里 "build" 一栏的 `fork` 指向本仓库，`73` 指向 PR #73 的
> worktree。两者用**同一个** `vae_bench` 源码编译，差别只有 `--precision` 环境变量
> 和 checkout 里的 `h3_video_vae.c`。
