# h3-metal

> 英文版：[README.md](README.md) · 本 fork 的本地改动：
> [本 fork 的优化](#本-fork-的优化)

面向 Apple Silicon 的原生 MiniMax-H3 推理实现。这个项目以一连串可运行的纵向切片
方式推进：先是确定性的 host/model 元数据，然后是可移植的 Metal block 对齐、
prompt 编码、prompt 到视频/音频、首尾帧条件控制，最后是有序参考输入。

prompt 到视频/音频、首尾帧条件控制，以及有序的 Ref2VA 图像/视频/音频参考输入
已经可以端到端跑通。当前的工作是在 M3 Max 和 M5 Max 上针对 H3 做增量的
Metal 性能与内存优化。

## 教程

### 1. 构建并检查模型

示例假定 Hugging Face 快照位于 `./MiniMax-H3`，并且 `PATH` 中可用 FFmpeg
和 FFprobe。

```sh
make -j8
mkdir -p outputs
./h3 --info -d ./MiniMax-H3
```

`--info` 会检查模型布局并打印选中的 Metal 设备，不会映射全部权重，也不会生成
媒体。完整的 CLI 参考请运行 `./h3 --help`。

不加 `-p` 时，同一个二进制会启动一个 Iris 风格的交互式会话：

```sh
./h3 -d ./MiniMax-H3 --width 512 --height 512 --steps 6
```

输入 prompt 即可生成带编号的视频。该会话会把精确的 BF16 prompt 条件、已准备的
DiT 和视频解码器常驻内存，因此用另一个 seed 重复同一 prompt 时无需重新加载和
编码。常用命令有 `!status`、`!seed random`、`!seconds 2`、`!show`、
`!save output.mp4` 和 `!cache`。用 `!help` 查看完整（简短的）命令列表。

首尾帧条件控制在会话中会持续生效：

```text
h3> !first opening.png
h3> !last ending.png
h3> The camera moves slowly around the subject.
```

用 `!first clear` 或 `!last clear` 移除某个锚点。生成的视频会写入启动时打印的
会话目录。

对于通用的 Ref2VA 条件图像，请改用 `!ref-image PATH`。图像会按顺序追加，并以
`<Picture 1>`、`<Picture 2>` 等形式暴露给模型；文件名对模型没有任何含义。

```text
h3> !ref-image person.png
h3> Make the person shown in Picture 1 wave to the camera.
```

`!refs` 列出当前顺序，`!ref-remove N` 移除其中一项，`!refs clear` 全部清除。
Ref2VA 参考输入不能与
`!first`/`!last` 锚点混用。

### 2. 生成第一个快速视频

先用经过验证的均衡预设。它会生成 22 帧 24 fps（约 0.92 秒）的视频，在支持的
图形终端里于每次去噪转移后显示中间视频帧的演化过程，并打印各阶段耗时：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --show \
  -o outputs/fox-fast.mp4
```

这并不是最激进的配置：

- `--steps 20` 执行默认的 20 轮去噪。
- `--reuse 2` 只计算 11 次新的去噪器速度，其余 20 次中被跳过的转移由外推得到。
- `--layers 45` 运行 50 个 transformer block 中的 45 个，同时减少耗时和
  统一内存占用。
- `--show` 是可选的。它支持 Kitty/Ghostty 和 iTerm2/WezTerm/Konsole 的图形协议。
  它会加载一个常驻的预览 VAE，在每次 Euler 转移后显示一帧有代表性的中间视频帧，
  然后显示全部最终帧。显示尺寸默认为 2 倍，以便在 macOS Retina 屏幕上呈现预期的
  逻辑尺寸；在非 HiDPI 显示器上请用 `--zoom 1`。它会增加预览解码时间和大约
  10 GiB 的临时模型常驻；不带 `--show` 的运行不受影响。
- `--profile` 是可选的，不会选择不同的生成路径。

第一次进程调用还要付出模型加载和文件系统缓存的开销。请用重复运行来比较性能，
并在机器升温时交替测试各个变体，因为该负载对热降频很敏感。

如果只想快速迭代一轮，直接请求四轮去噪：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur." \
  --width 512 --height 512 --frames 22 \
  --steps 4 --layers 50 --reuse 1 \
  --show \
  -o outputs/fox-four-step.mp4
```

`--steps N` 始终表示恰好 N 轮去噪。四到七轮使用的是赢得低预算对比的那套调度；
从 4 增加到 7 会逐步改善细节和运动幅度。在这么小的预算下请保持 `--reuse 1`，
以便每一轮请求都真正跑一遍模型。`--show` 会在每轮之后显示一次预览。

之所以评估了若干种尾部偏重的调度，是因为长时间运行中大部分可见的清理工作都
发生在后期。它们保留的早期构图更新太少，导致纹理交织、运动微弱或颜色过曝。
最终保留的模式使用发布版本的线性基础网格加一个终点。在 512 见方、22 帧的狐狸
测试中，选定的四轮结果相对 29 轮参考的全视频 SSIM 为 0.556；另一个独立的冲浪
者测试测得 0.547。四轮去噪在 M5 Max 上耗时约 3.5 秒，而参考耗时 26.4 秒。

要跑低内存版本，加上 `--ssd-streaming`：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 --ssd-streaming \
  -o outputs/fox-ssd.mp4
```

它使用原始 BF16 checkpoint，不做任何转换或量化。它在内存中保留两个 DiT block，
GPU 运行当前 block 时从 SSD 读取下一个。在 M5 Max 上，追踪到的 DiT 存储从约
36.5 GiB 降到 512 见方时 2.0 GiB、864x480 时 2.1 GiB。热身的 50-block 前向在
512 见方下实测 1.35 秒 vs 2.49 秒（慢 84%），在 864x480 下 2.14 秒 vs 2.68 秒
（慢 26%）。这些都是在与同一条全常驻 BF16 路径对比，且两次检查的结果字节一致。

2.0--2.1 GiB 这个数字指 DiT 被追踪的张量存储，而不是系统总内存。Prompt 编码和
两个 VAE 在各自独立的阶段运行，它们的峰值不会叠加到上面；操作系统、媒体缓冲
和输出分辨率仍然需要留出余量。`--show` 会让一个预览 VAE 常驻并额外占用大约
10 GiB，所以追求最低内存时请去掉它。

SSD 流式是显式的内存/速度权衡，并不是默认行为。它不能与 `--use-int8-row-fc2`
组合使用。在交互式会话中，请
使用 `!ssd-streaming on`。

### 3. 向参考画质靠拢

评估画质时每次只改一个控制项。先恢复全部层数，再恢复全部去噪器计算，最后把
默认的 20 轮调度提高到更慢的 50 轮参考：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest. Medium tracking shot, natural winter light, realistic fur, soft footsteps and wind." \
  --width 512 --height 512 \
  --frames 22 --steps 50 \
  --layers 50 --reuse 1 \
  -o outputs/fox-close.mp4
```

默认值是 `--steps 20 --layers 50 --reuse 1`；在这条精细路径上请显式写
`--steps 50`。它会执行 50 次完整的 50-block 去噪器前向，代价远高于默认值，但当
快速模式改变了主体、解剖结构、运动或构图时，它才是正确的对照基准。由于随机数
与执行引擎不同，不应期待与 MLX 逐像素一致；画面内容和运动应当一致。

### 4. 选择速度/画质档位

除非另有说明，以下控制项彼此独立：

| 控制项 | 慢速参考 | 默认 | 激进 | 主要影响 |
|---|---:|---:|---:|---|
| 去噪轮数 | `--steps 50` | `--steps 20` | `--steps 4..7` | 这个数字始终表示实际执行的去噪轮数。 |
| 整体去噪器复用 | `--reuse 1` | `--reuse 2` | `--reuse 3` | 20 步时：20、11 或 8 次新的 DiT 计算。 |
| 启用的 DiT block | `--layers 50` | `--layers 45` | `--layers 40` | 更少的 block 同时减少计算量和常驻的 transformer 权重。 |
| 核心残差复用 | `--core-reuse 1` | `--core-reuse 4` | `--core-reuse 6` | 每步都刷新 patch/head 的工作，但更少运行昂贵的核心部分。 |
| Token 削减 | 关闭 | 可选 | `--token-reduction` | 在中间 block 内成对合并水平方向的视频 token；更快，但可能改变构图。 |
| 内部画布 | 输出尺寸 | 512 见方输出时为 `384x384` | `320x320` | 以更小的尺寸运行 DiT/VAE，再用 vImage 放大。 |

在 M5 上，`--use-int8-row-fc2` 对每个 FC2 行使用一个激活 scale，并做一次全宽
TensorOps 乘积。它是可选的，因为相比分组 int8 它的数值上更不保守。在互易测试
中它把完整去噪器前向减少了大约 2.6%。四步的狐狸和冲浪者对照视频保持了相同的主体、
场景和运动（全视频 SSIM 分别为 0.919 和 0.828）。在交互式会话中，使用
`!int8-row-fc2 on`。

`--reuse` 和 `--core-reuse` 互斥。层数削减可以和其中任意一个组合。

想在保持输出分辨率的前提下让第一条命令更快，请加上 token 削减：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding inside a sharp blue ocean wave, one rider and one white board, realistic spray." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --token-reduction \
  -o outputs/surfer-fast.mp4
```

在经过验证的 512 见方尺寸下，token 削减把 IT M5 Max 上 `45 layers + reuse 2`
的去噪耗时从 16.69 秒降到 12.60 秒。独立的狐狸和冲浪者渲染结果仍然连贯，但构图
可能比精细路径偏离更多。

如果需要激进的预览，可以在内部以 320 见方渲染，再放大到请求的 512 见方输出：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A red fox walking through snow, realistic, tracking shot." \
  --width 512 --height 512 \
  --render-width 320 --render-height 320 \
  --frames 22 --steps 20 --layers 40 --reuse 3 \
  -o outputs/fox-aggressive.mp4
```

在验证中，这个组合产出了一只干净、可辨认的 22 帧狐狸，但会丢失细节并可能改变
取景。**不要**同时在 `--layers 40` 和 `--reuse 3` 上再加 `--token-reduction`：
测试中的该组合会产生颜色振铃、轮廓和残影化的肢体。

作为整体速度复用的替代方案，下面这种做法在每次转移时都保持依赖 timestep 的
patch head 和输出 head 为最新状态：

```sh
./h3 --profile \
  -d ./MiniMax-H3 \
  -p "A surfer riding a blue ocean wave." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --core-reuse 4 \
  -o outputs/surfer-core-reuse.mp4
```

只应把 `--core-reuse 6` 当作激进预览使用。大于 6 的取值
不建议暴露，因为验证中丢失了主体的保真度。

### 5. 选择分辨率与时长

宽和高都必须是 32 的倍数、不小于 32，且两者乘积不得超过 `768 * 1344` 像素。
这些是机械限制，并不意味着每个小画布都有好的模型画质。H3-Base 是一个 768p 模型。

| 画布 | 当前建议 |
|---|---|
| `512x512` | 最安全的开发尺寸；已用多个 prompt 反复验证。 |
| `768x768` | 验证过的精细画质正方形输出；成本高得多。 |
| `1344x768`、`768x1344` | 发布版本 768p 级别的横/竖屏上限。 |
| `1024x768`、`768x1024` | 合法的 4:3 与 3:4 768p 级别画布。 |
| `512x512` 内部用 `384x384` | 验证过的快速画质缩放点。 |
| `512x512` 内部用 `320x320` | 验证过的激进缩放点。 |
| `256x256` | 原生快速预览画布，带自动的低分辨率 RoPE 适配。 |

要生成快速的原生 256 见方预览：

```sh
./h3 -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow in a pine forest." \
  --width 256 --height 256 \
  --frames 22 --steps 20 \
  --layers 50 --reuse 1 \
  -o outputs/fox-256.mp4
```

在 256 见方下，H3 只有 `8x8` 的有效空间 token 网格，因此留给精细细节和复杂构图的
空间更少。H3 会在恰好 256 见方时自动把空间 RoPE 坐标减半。这消除了长狐狸渲染中
重复的格子状伪影，在一个独立的竖版测试中也保持了连贯，而且没有增加 token 或运行
时间。用 `--use-reference-rope` 可以恢复发布版本/MLX 的坐标以做一致性校验。在这个
尺寸下请保持 token 削减关闭。原生 128 见方仍然不支持：它的 `4x4` token 网格即使
调整了 RoPE 也无法恢复出可辨认的主体。

`--render-width` 和 `--render-height` 必须成对设置，必须与输出具有相同的宽高比，
并且不能超过输出尺寸。模型和 VAE 使用内部尺寸；终端帧和编码后的视频保持请求的
输出尺寸。

H3 输出 24 fps，并把帧数请求向上对齐到 `5 + 17*n`：

用 `--seconds N` 表示按时长请求，或用 `--frames N` 直接控制帧数；两者互斥。
允许小数秒。秒数会按 24 fps 换算，然后向上取整到下一个合法的 H3 时间形状，
因此 `--seconds 10` 会产生 243 帧（10.125 秒）。

| 帧数 | 大致视频时长 |
|---:|---:|
| 22 | 0.917 秒 |
| 39 | 1.625 秒 |
| 56 | 2.333 秒 |
| 107 | 4.458 秒 |
| 243 | 10.125 秒 |
| 362 | 15.083 秒 |

短片对开发很有用。发布版本的工作流面向大约 4–15 秒的视频。像 `--frames 23` 这样的
请求会被向上取整到 39 帧，而不是产生任意的时间形状。

### 6. 改进 prompt

简短的 prompt 也能工作，但发布版本的系统期望的是类似 Context-IR 的描述。
请说明主体、动作、场景、镜头、光照/风格以及想要的声音。例如：

```text
Scene: a single red fox in a snow-covered pine forest at dawn.
Action: the fox walks steadily left to right and looks toward the camera once.
Camera: medium-height lateral tracking shot, 50 mm lens, stable framing.
Look: photorealistic fur, cold blue ambient light, warm sunrise rim light.
Audio: soft footsteps in snow, light wind through pine branches, no music.
```

当身份和物体数量很重要时，请显式写明。`--seed N` 控制原生随机流，默认值为 42。
比较各选项时请使用相同的 prompt、seed、分辨率、帧数和步数。

### 7. 预览帧并诊断性能

- `--show` 在每次去噪转移后显示一帧有代表性的画面，随后显示完成视频的全部帧。
  与 Iris 一样，它默认对 Retina 终端声明 2 倍显示尺寸；`--zoom N` 可以改变该倍数，
  而不会重新缩放生成的视频或编码后的终端图像。
- `--frames-dir DIR` 会把最终回调帧写成 PPM 文件。中间的 `--show` 预览不会写到那里。
- `-o ''` 禁用 MP4 编码；当没有 FFmpeg 时可与 `--frames-dir` 组合使用。
- `--profile` 报告各阶段的墙钟时间、Metal 编码/等待时间、活跃张量存储峰值、
  累计分配量和 dispatch 次数。

例如：

```sh
./h3 --profile -d ./MiniMax-H3 -p "A hummingbird hovering over red flowers." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 --frames-dir outputs/hummingbird-frames \
  -o ''
```

### 8. 添加图像、视频和音频参考

首尾帧锚点会选择 FL2VA 路径：

```sh
./h3 -d ./MiniMax-H3 -p "The fox keeps walking through the snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 45 --reuse 2 \
  --first-frame fox.png --last-frame fox-later.png \
  -o outputs/fox-anchored.mp4
```

有序参考输入会选用另一个 Ref2VA checkpoint。请使用与媒体语义对应的参数：

```sh
# One image reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and setting in the reference." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-image fox.png -o outputs/fox-reference.mp4

# Continue a clip but ignore its soundtrack.
./h3 -d ./MiniMax-H3 -p "Continue the motion in this clip." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --ref-silent-video fox.mp4 -o outputs/fox-video-reference.mp4

# Preserve the clip's embedded audio.
./h3 -d ./MiniMax-H3 -p "Continue this audiovisual scene." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video fox-with-audio.mp4 -o outputs/fox-video-audio.mp4

# Replace a video's soundtrack explicitly.
./h3 -d ./MiniMax-H3 -p "Continue the scene with the supplied music." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-video-audio silent-fox.mp4 replacement.wav \
  -o outputs/fox-replaced-audio.mp4

# An ordered image plus standalone audio reference.
./h3 -d ./MiniMax-H3 -p "Use the animal and music from the references." \
  --width 512 --height 512 --frames 56 --steps 20 \
  --ref-image fox.png --ref-audio music.wav \
  -o outputs/fox-image-audio.mp4
```

参考参数可以重复使用，并且会保留它们在命令行中的顺序。独立音频必须与图像或
视频参考一起使用。音频参考必须是 2–15 秒；最多接受三个音频输入，它们的总解码
时长上限为 15 秒。

## 测试与运行时要求

```sh
make test
make parity
```

`make test` 运行确定性的 host 测试套件；当被忽略的 MLX fixture 安装在
`misc/fixtures/` 下时，它还会在运行时编译 Metal 源码，并把一个完整的玩具级
H3 block 与具名的 MLX 输出做比对。运行时编译是刻意为之：它沿用了 Iris 的做法，
不需要 Xcode 的可选离线 Metal 工具链。测试同时覆盖 F32 诊断路径和生产的 BF16
存储路径；宽矩阵 BF16 乘积和 SDPA 使用缓存的 MPSGraph graph，并带有直接的
Metal 正确性回退路径。`make parity` 只运行这些 Metal/MLX 检查。

媒体输入和 MP4 输出要求 `PATH` 中可用 FFmpeg 和 FFprobe（`H3_FFMPEG` 和
`H3_FFPROBE` 可指定具体可执行文件）。生成的 RGB24 和 32 kHz 立体声 F32 PCM
通过并发管道传输；没有中间
不会创建未压缩的媒体文件。

## 实现与性能说明

以下部分记录教程中各档位背后的实现，以及为精确 A/B 诊断而保留的环境变量。

### 采样器与 DiT 控制

默认采样器使用发布版本的 shifted 视频/音频调度。`--steps` 始终表示去噪轮数，
最后一轮之后再补一个终点零。整体去噪器复用会计算第一轮、最后一轮以及每一个
请求的间隔，然后在各自独立的调度上外推被跳过的视频和音频速度。步数非常少时，
请保持 `--reuse 1`。

在低预算路径上，发布版本的线性基础网格战胜了实际视频 sigma 线性间隔、二次和
三次 warp、精确的 30 点尾部子集、轻微的幂 warp、零阶保持的全网格速度、线性
速度外推以及 RES。尾部更重的候选方案常常让主体更锐利，但损害了运动，或留下
重复交织的背景；稀疏的 RES 和长外推间隔则失败得明显得多。

层数削减会按 checkpoint 中实际的 AdaLN 门控排序，同时保护结构上重要的首尾
block。未使用的权重和调度张量不会被保留，因此 `--layers 45` 和 `--layers 40`
同时降低 transformer 时间和统一内存占用。核心复用会保持上一次的完整 transformer
残差，同时刷新 patch 投影和感知 timestep 的 head；它与整体速度复用仍然互斥。

### 精确的 DiT 融合

每个启用的 DiT block 都把注意力的残差门控与后面的 MLP AdaLN 融合在一起。
舍入后的 BF16 残差仍会精确写入，但同一行会保留在线程组内存中用于归一化，从而
消除一次 dispatch 和一次全局重读。在 token 削减边界之外，MLP 残差门控还会产生
下一个 block 的注意力 AdaLN，并把该归一化状态跨循环传递。`H3_DISABLE_FUSED_GATE_ADALN=1`
和 `H3_DISABLE_FUSED_CROSS_BLOCK_ADALN=1` 可以恢复双 kernel 的对照实现。
最后的音频/视频 AdaLN kernel 直接绑定到残差流中的偏移量，避免了两次 slice blit
以及 512x512 下 18.8 MiB 的临时空间（在 864 级别基准尺寸下为 29.4 MiB）。
`H3_DISABLE_FUSED_FINAL_SLICE=1` 可在加载时恢复“拷贝 + AdaLN”的对照实现。
随后 BF16 的最终 head 在加载 16x16 投影 tile 的同时应用 AdaLN，保持独立的舍入
和累加顺序，同时又去掉了一个同等大小的归一化激活。这两项优化合计节省
37.5/58.9 MiB。`H3_DISABLE_FUSED_FINAL_HEAD=1` 可在加载时恢复
“偏移 AdaLN + linear”的对照实现。

### Token 削减内部机制

`--token-reduction` 是一种独立的激进 DiT 模式。在 block 3 之后，它把相邻的
水平目标视频 token 两两配对，而文本、音频、条件和参考 token 保持精确。完整的
全分辨率状态作为旁路保留。在最初十次噪声较大的计算中，它在 block 40 之前恢复；
之后成形的细节计算则在 block 30 之前恢复。每个 token 返回时等于它原本的值加上
其配对学到的更新，因此配对内部的细节不会被丢弃。
Pooling kernel 只把真正的成对基线写入已分配好的注意力 scratch 缓冲区的稠密尾部；
奇数宽度的孤立 token 不需要基线。完整旁路在放得下时使用扩大的 QKV 尾部，只有
在参考密集的布局下才启用带保护的专用回退路径。因此常见的纯文本画布在任何
token 网格宽度下都不会增加激活 arena。Pooling 还会在两个源 token 的 BF16 值
仍在寄存器中时对其做快照，从而避免单独的全 hidden blit 和重复的源读取。
同一个入口 kernel 让每个 pooling 后的行保留在线程组内存中，并输出第一个削减后
block 的注意力 AdaLN，又消除了一次全局残差重读。在恢复边界处，第一个全分辨率
注意力 AdaLN 被融合进扩展：一行 10.5 KiB 的线程组数据避免了全局残差重读，同时
仍然写出后续残差分支所需的精确旁路。
在热平衡的 512x512x22、IT M5 Max、19 次前向的 A/B 测试中，它把去噪耗时从
39.13 秒降到 28.06 秒（28.3%）。最终视频/音频 latent 的相对 L2 为
5.56%/15.14%。狐狸的首/中/尾帧保留了一个干净的口鼻、连贯的四肢和锐利的毛发；
一个独立的冲浪者在浪花中始终保持一名骑手和一块冲浪板的一致性。它会改变构图，
因此是可选开启，而不是精细参考的默认设置。
`H3_TOKEN_REDUCTION_BLOCKS` 可以覆盖后面的 `4:30` 区间；
`H3_TOKEN_REDUCTION_EARLY=STEPS:END` 覆盖早期调度，`0` 表示禁用它。
`H3_DISABLE_TOKEN_REDUCTION=1` 提供上下文内的精确对照实现。
`H3_DISABLE_FUSED_TOKEN_POOL_ADALN=1` 和 `H3_DISABLE_FUSED_TOKEN_ADALN=1`
可分别恢复双 kernel 的入口和出口边界以便诊断。
Token 削减与经过验证的 `--layers 45 --reuse 2` 设置配合良好：在同一个 512
基准上，它把该配置从 16.69 秒降到 12.60 秒（边际收益 24.5%），独立的狐狸和
冲浪者渲染结果依然连贯。不要把它与 `--layers 40` 和 `--reuse 3` 同时组合；
那次 6.47 秒的实验产生了色彩振铃和残影化的四肢，尽管 latent 范数尚可接受。

### 内部画布与视频 VAE

`--render-width` 和 `--render-height` 让模型和 VAE 在更低的、同宽高比的内部画布
上运行，然后在回调、终端显示和编码之前，用高质量的 vImage 缩放把 RGB 帧放大到
请求的输出尺寸。这是一个显式的画质/速度权衡：实测从 384 缩放到 512 的 prompt
渲染把 M5 的 DiT 时间减少了 33%、视频 VAE 时间减少了 18%，同时仍保持干净、
可辨认的写实结果。两个值都必须是 32 的倍数；精确的输出画布仍是默认行为。
对于 512 见方输出，384 是快速画质点，320 是验证过的激进点。后者产出了一只连贯
行走中的狐狸，DiT 耗时 8.02 秒，而原生方案约 15.82 秒。原生 256 使用上文描述的
同等代价空间 RoPE 适配；它仍然是快速的构图预览，而不是 512 或 768 级别最终渲染
的替代品。
视频 VAE 会根据请求的画布几何自动在 256-320 像素之间选择空间 tile，在尽量减少
重复重叠工作的同时把峰值存储控制在有界范围内。`H3_VAE_TILE_PIXELS=256` 可恢复
原本保守的 tile 方案，用于精细参考诊断。

### 权重常驻与流式 prompt 编码

在 M5 级别的 GPU 上，常驻的 transformer 权重会直接从它们的 safetensor 分片映射，
而不是拷贝进匿名共享缓冲区。这让 37 GiB 的模型保持 file-backed/可回收，并略微
改善 transformer 总时间；M3 使用更快的拷贝缓冲路径。`H3_ZERO_COPY_WEIGHTS=0`
可禁用 M5 的这一选择以便诊断。
流式 Qwen 文本编码器会预分配一个小的未来层缓冲环，并在 Metal 执行当前层的同时
由 8 个 I/O worker 填充。默认环深度在 M3/较老硬件上为 2 层，在 M5 上为 3 层
（目标机器有 128 GiB 内存）。`H3_QWEN_PREFETCH=0` 恢复单层同步参考路径；
取值 1-8 选择 worker 数量，`H3_QWEN_PREFETCH_DEPTH=1` 到 `6` 覆盖环深度。

`--ssd-streaming` 是 DiT 另一种更激进的常驻模式。只有每个 block 较小的归一化
权重保持常驻。两个完整的 BF16 矩阵槽位交替使用，后台读取器按 checkpoint 偏移
顺序填充下一个槽位，同时当前的 Metal command buffer 并发执行。Darwin 的
uncached 读取避免了在文件系统缓存中保留第二份副本。第一个启用的 block 会在最后
一个 block 期间被再次预取，这样缓存过的交互式 DiT 就能准备好进行下一次去噪器
计算。实测从内部 SSD 达到约 13--14.6 GiB/s。`H3_PROFILE=1` 会报告总字节数、
读取吞吐，以及读取等待中未被 GPU 工作掩盖的那部分。

### Metal 4 与 TensorOps 路径

M5 GPU 会自动在序列长度不超过 2,048 时，对 DiT 的 QKV 和注意力输出投影使用原生
BF16 Metal 4/TensorOps。紧凑的 Morton 调度把 Q/K/V 直接送入 head-major 的注意力
输入，避免了三次 MPSGraph 输入转置，并且与可移植路径字节一致。在多次 IT/US
M5 Max 的重复运行中，它把完整的 512x512、50-block 前向提升了约 2%。对于
2,049-3,072 行（包括 864x480），两次行偏移 Morton dispatch 保持了高效的 tile
几何，在平衡运行的测试中把完整前向提升约 2%。更大的序列仍走 MPSGraph。
`H3_NAX=0` 可禁用 TensorOps 以做精确 A/B 诊断。该选择在运行时受保护，编译不可用
时会回退到未改动的可移植库。

`H3_NAX=1` 强制使用更广泛的原生 BF16 linear 路径。它能通过完整的 50-block MLX
fixture，但仍然是可选的：精确形状的微基准偏好它的 128 行 tile，而完整的 DiT
运行目前更偏好 MPSGraph 调度。这样既保留了一个可用的 NAX 集成以备后续量化/融合
kernel 使用，又不会把基准性能回退变成默认行为。
`H3_NAX=mlp` 选择一条更专用的 Metal 4 路径：成对的 FC1 gate/up TensorOps tile 在
线程组内存中应用 SwiGLU，并且只写出 14,336 宽的激活中间结果，随后 FC2 也保持在
TensorOps 上。`H3_DISABLE_NAX_MLP=1` 会在以此方式创建的上下文中保留 MPSGraph
MLP，以便同进程 A/B 测试。这条路径刻意保持可选，因为调度效果依赖操作系统的
GPU 栈：macOS 26.5.2 的 M5 Max 在隔离的真实权重 MLP 运行中提升了 1.3-2.0%，
但在完整的 50-block 前向中损失约 1-3%；而另一台配置相同的 macOS 26.5 M5 Max
在同上下文前向 A/B 中提升了 1.4%。最终得到的 50-block 速度很接近（视频 1.9%、
音频 2.4% 的相对 L2），但并非字节一致。

### 专用投影 kernel

窄的 DiT 音频/视频输出 head 会一次性把其发布版本的小型 F32 权重转换为 BF16，
然后在 BF16 激活上直接使用源自 Iris 的 16x16 分块 linear。在生产环境的 320
渲染几何下，成对 head 的隔离测量在 M3 Max 上快 2.30 倍、在 M5 Max 上快 1.83 倍，
相对 L2 为 `8.64e-4`；M5 上绝对节省约每个求值步 0.6 ms。完整的狐狸和冲浪者序列
依然干净，相对 F32 head 渲染测得 29.9/38.4 dB。`H3_DIT_F32_FINAL=1` 可恢复
精细参考的 head 及其额外的激活缓冲区。
F32 的 `96->5376` 视频和 `32->5376` 音频 patch 投影使用专用的 16x16 协作 tile，
保留 F32 的权重、输入和累加，同时把 tile 结果直接舍入为 BF16。
生产形状的成对测量在 M3 上快 1.77 倍、在 M5 上快 1.62-1.78 倍；完整生成的 RGB
流与标量路径字节一致。融合最终 cast 把 2835 行的 tile 本身从 M3 上的 2.499 ms
优化到 1.734 ms、M5 上的 1.555 ms 优化到 1.186 ms，并在 512/864 级别几何下
去掉了 38.27/59.66 MiB 的 F32 临时空间。`H3_DISABLE_FUSED_PATCH_CAST=1` 可恢复
分块的 F32 输出加独立 cast；`H3_SCALAR_PATCH=1` 选择标量诊断路径。
同一 tile 会把输出直接绑定进打包后的 hidden 流，去掉了 BF16 媒体暂存缓冲区
及其 blit。这又节省了 19.13/29.83 MiB，并把 2835 行的边界从 M3 上的 1.847 ms
优化到 1.730 ms、M5 上的 1.282 ms 优化到 1.184 ms。连续的 T2VA 使用字节偏移；
FL2VA/Ref2VA 使用紧凑的目标行映射，使每个模态仍是一次大 dispatch。完整的
六段 Ref2VA M5 ABBA 测试保持字节一致，并把每次测量的前向对从 5.067 秒改善到
5.033 秒。`H3_DISABLE_FUSED_PATCH_PACK=1` 可恢复暂存缓冲区和打包 blit。

### 调度与激活内存

DiT 核心被拆成两个有序的 Metal command buffer，使第一部分的 GPU 执行可以与
第二部分的 CPU 编码重叠。热平衡的 ABBA 测量在 M5 上选择了 60% 深度的切分
（30/50、27/45 和 24/40），约有 0.5-1.8% 的收益；M3 只自动切分经过验证的
30/50 情形（实测快 1.2%），因为 24/40 在那里反而回退。操作顺序和生成的字节
保持不变。`H3_DIT_COMMAND_BLOCKS=0` 可恢复单个 command buffer；取值 1-50
可以覆盖切分点以便进一步调优。
DiT 的激活缓冲区也遵循其实际的 block 内生命周期：QKV 投影 arena 先被复用于
注意力 head，然后复用于归一化后的 MLP 输入；而当前的注意力输出 arena 在其分支
被消费之后就成为 MLP 输出。这在 512 级别几何下去掉 61.25 MiB、在 864 级别几何下
去掉 99.63 MiB，且不改变 dispatch 和算术。`H3_DISABLE_DIT_ACTIVATION_ALIAS=1`
可恢复独立的诊断缓冲区。
不可变 DiT 权重和偏置的 MPSGraph tensor-data 包装会连同其常驻缓冲区一起保留。
这样无需为每个 block 和每次去噪器计算重建相同的绑定元数据，也不会拷贝张量存储；
实测 ABBA 收益在 M3 Max 上为 1.6%，在 M5 Max 上为 0.4-1.1%。激活包装仍保持
临时，因为保留它们会让 M5 回退。输出保持字节一致，`H3_DISABLE_GRAPH_DATA_CACHE=1`
可为所有张量恢复临时包装。
在 M3/较老硬件上，每个 DiT block 中的四个 MPSGraph 段还会为它们共享的底层
Metal command buffer 复用同一个 `MPSCommandBuffer` 包装。重复的热平衡运行在
M3 Max 上测得快 1.0-1.6%；M5 上测量为中性，因此保留新的包装。
`H3_REUSE_MPS_COMMAND=0` 或 `1` 可以覆盖自动选择。结果字节一致。
在 M5 上，运行时 Euler 采样器把 patch 打包的 F32 latent 和缓存的 BF16 速度
保留在 Metal 缓冲区中。每次选定的去噪器刷新都在下一次编码之前完成，避免了
MPSGraph 的背压，同时去掉了所有中间 latent/速度的回读与重新打包。两次预热的
八轮 A/B 序列分别测得 0.1% 和 0.3% 的小幅收益，最终 latent 字节一致；
该路径还为每个视频 latent 元素省下约 16 字节的临时主机状态（在 768p 形状下
约 136 MB）。M3 和更老的 GPU 默认保留 CPU 采样器。`H3_CPU_SAMPLER=1` 可在 M5
上恢复它；`H3_GPU_SAMPLER=1` 显式选择 GPU 状态路径，`H3_GPU_SAMPLER_WINDOW=0`
启用更慢的、无限制的提前编码诊断模式。

### Checkpoint 布局与媒体管线

发布版本的 checkpoint 按注意力 head 交错存储 DiT 的 QKV 行。原生 Metal 在融合的
QK 归一化/RoPE kernel 中直接消费这一布局，避免了 checkpoint 转置和额外 RAM。
早期的“按恒等解释”正是那些嘈杂诊断输出的原因。

公开的生成路径用流式的原生 BigVGAN/AudioVAE 解码联合音频 latent，并写出同步的
H.264 加 32 kHz 立体声 AAC。原生波形与修正后的 MLX 对照实现的相对 L2 为
`6.94e-5`。`--first-frame`、`--last-frame` 及其组合使用发布版本的视觉 VAE 编码器、
Qwen3-VL 视觉塔和三层 deepstack 多模态呈现、0.999 的条件增强，以及原生 DiT 中的
固定条件行。第一张图会被拉伸到目标画布；最后一张图按 aspect-cover 缩放并居中
裁剪，与参考实现一致。`--ref-image` 选择另一个 Ref2VA transformer，保留有序的
`<Picture N>` 呈现，并使用发布版本的仅降采样、保持宽高比的参考画布。
`--ref-silent-video` 还会执行有界的 24 fps 解码、视觉 VAE 的因果 `ceil(T/4)`
压缩、两帧 Qwen 采样，以及带时间戳的 `<Video N>` 呈现。`--ref-video` 保留内嵌
音轨，`--ref-video-audio VIDEO AUDIO` 提供显式替换，`--ref-audio` 追加一段有序的
独立音频。参考音频被解码为 32 kHz 立体声 F32，由原生 AudioVAE 的后验均值路径
编码，按 0.999 干净 latent 加 0.001 种子噪声混合，固定在音频条件 timestep 1.0，
并在与视觉参考相同的旋转时间线上打包为宽度 32 的行。音频输入为 2-15 秒，最多
接受三个，其总解码时长上限为 15 秒，且独立音频参考必须与图像或视频参考组合使用。

原生音频编码器在真实的 2 秒立体声 fixture 上与修正后的 MLX 对照实现的相对 L2
为 `3.59e-6`。这一修正很重要：原始的 MLX reshape 把左右声道交错在一起，而官方
的 PyTorch/SGLang 路径把完整的立体声声道折叠进 batch 维度。在 128 GB 的
M5 Max 上，干净的端到端 图像+音频 和 内嵌视频+音频 渲染分别耗时 74.58 秒和
76.99 秒，每次的物理内存峰值约 40.1 GB，且没有发生 swap。

### Profiling 与诊断路径

`--profile` 会分别报告每个由 Metal 支撑的阶段：墙钟时间、CPU 侧的 command 编码、
从 commit 到 fence 的完整等待、根 command 的 GPU 时间戳、活跃张量存储峰值、
累计分配量和 dispatch 次数。等待测量的是完整的 command 周转时间；仅根 GPU
时间戳可能遗漏 MPSGraph 在内部调度的子 buffer，因此它有相应的标注。

DiT 快速路径把每个 BF16 的 `fc1 -> SwiGLU -> fc2` block 作为一张缓存的 graph
求值，避免了独立的 graph 边界和常驻的中间张量。设置 `H3_DISABLE_FUSED_MLP=1`
可以保留精细参考的操作边界以便做数值诊断。

在受支持的 M5 Metal 4 TensorOps 硬件上，原生 int8 MLP 引擎是默认路径。它动态
量化激活，使用按输出通道的权重 scale，并对敏感的 FC2 输入按每 1,024 个通道给
一个 scale。所选的 FC2 kernel 把带 scale 的部分积保留在私有的协作 fragment 中，
而不是反复溢出 32 KiB 的线程组 tile。一次固定的 50 层、19 次转移的 512x512 渲染
在 M5 Max 上用 BF16 MPS 测得 36.30 秒，用 int8 测得 25.80 秒。开头、中间和最终
的解码帧保留了相同的主体、构图和运动；细小的边缘和毛发细节可能不同。当前的
诊断实现只有在 A/B 诊断请求时才会同时保留 BF16 和 int8 两套 MLP 权重。常规的
int8 加载会在每个 block 提交的量化完成后释放它的 BF16 FC1/FC2 缓冲，把实测的
张量存储峰值从 BF16 路径的 36.4 GiB 降到 25.9 GiB。运行时的权重量化仍会增加
启动时间。

最快的 M5 路径还会量化每个 DiT QKV 投影，并在既有的 Q/K 归一化和 RoPE kernel
之前，直接以 head-major 注意力布局写出它的 Q/K/V tile。在一次固定的 50 层、
19 次转移的 512x512 渲染中，它再次把去噪时间从 25.80 秒降到 19.32 秒。采样的
开头、中间和最终帧仍是一只在雪中行走的连贯、细节丰富的狐狸；量化注意力可能
改变取景和精细细节。用 `--use-slower-bf16-qkv` 可使用精细参考的 BF16 投影。
常规的 int8 加载会在量化后释放多余的 BF16 QKV 权重。

在默认的 M5 路径上，后续的注意力输出投影同样是 int8。同模型交叉测试在 512 和
864 上把完整前向又提升了 4.5-5.5%。一次解码的狐狸渲染依然干净，且与仅用
int8-QKV 的构图高度接近；其热态去噪测得 19.18 秒。用
`--use-slower-bf16-attention-output` 可让该投影保持 BF16。

在该 int8 路径上，SDPA 现在把结果保留在原生的 `[head,row,dimension]` 顺序。
一个专用的 256 线程 kernel 直接把每个 H3 行收集并量化进投影的行主序 int8 缓冲，
消除了中间的全宽 BF16 转置，且不改变任何输出字节。热控的交叉运行把完整的 512
和 864 前向提升了约 0.2-1.2%。用 `--use-slower-row-major-attention-output`
可恢复显式的 BF16 行主序 SDPA 输出和普通量化器。

M5 路径还把 QKV 和 MLP 的激活量化折叠进前面的 gated AdaLN kernel。这在每次
50 层前向中去掉了 99 次独立的量化器 dispatch，同时保留此前的输出字节，把交叉
测量的 512/864 结果提升约 0.3-0.6%。用 `--use-slower-unfused-int8-inputs`
可恢复独立的量化器。

融合后的 gated-AdaLN 路径以 BF16x4 向量加载其完整的 5,376 宽 H3 行，并写出
int8x4。它先在本地暂存舍入后的值，再计算原有的逐线程 RMS 序列，因此归约树和
每一个输出字节都保持不变。交叉测量又节省了大约 0.1-0.5%。已有的
`--use-slower-unfused-int8-inputs` 选项保留了可移植的标量和独立量化器回退。

Q/K 的 RMS 归一化和 RoPE 同样在 int8 QKV 投影 tile 内部完成。融合后的 epilogue
字节一致，并在 M5 的交叉测量中把 512 的完整前向提升 2.1-3.2%、864 提升
1.0-1.8%。用 `--use-slower-unfused-qkv-rope` 可恢复独立的 Q/K kernel。

该 epilogue 用 BF16x4 的加载和存储，每个工作项处理四个相邻的 Q/K 维度。逐元素
算术和 BF16 舍入顺序保持不变，而冷态交叉测量在 512 和 864 上都把完整前向提升
约 0.4-1.0%。同一个 `--use-slower-unfused-qkv-rope` 选项可恢复标量独立路径。

在最多 2,048 行时，精确的 RMS 循环使用 BF16x4 加载后接四次显式的有序 FMA。
这保留了每一个输出比特，并把 512 级别的前向又提升约 0.5-0.6%；更大的形状仍使用
标量加载，因为两种形式在那里打平。用 `--use-slower-scalar-qkv-rms` 可强制
使用标量加载。

int8 注意力输出投影把它的 128 个行 scale 和列 scale 缓存在 1 KiB 的线程组内存中，
而不是为每个协作 fragment 元素重新读取。超过 2,048 行时，融合的 QKV kernel
采用同样的做法，随后把该存储复用于逆 RMS 值；更小的 QKV 形状保留直接加载，
因为两种形式在那里打平。两者字节一致，并在被选中的地方把完整前向提升约
0.2-0.7%。用 `--use-slower-uncached-int8-scales` 可恢复直接的设备 scale 加载。

对于最多 2,048 行的序列，H3 注意力输出投影还会把它的 7,168 x 5,376 形状编译进
TensorOps kernel。结果保持字节一致，同时在交叉的完整 512 前向测量中节省约
0.2-0.8%。更大的序列保留动态形状 kernel，因为该特化在那里反而回退。
`--use-slower-uncached-int8-scales` 可恢复通用的动态、直接加载 scale 的实现。

FC1 也使用 H3 专化的、编译期 5,376 宽的 TensorOps 循环。它与通用循环字节一致，
并在交叉的完整前向中节省约 0.1-0.4%。用 `--use-slower-dynamic-fc1-k` 可恢复
运行时绑定的循环。

```sh
./h3 --profile -d ./MiniMax-H3 \
  -p "A red fox walks through fresh snow." \
  --width 512 --height 512 --frames 22 --steps 20 \
  --layers 50 --reuse 1 -o outputs/fox-int8.mp4
```

用 `--use-slower-bf16-mlp` 可强制使用可移植的精细参考 MPS/BF16 MLP 路径做数值
比较。较老的 Metal 硬件在缺少所需原生 TensorOps kernel 时会自动选择该路径。
对于 FC2 的激活量化，最多 2,048 行的序列使用精确的 128 线程归约。每个线程在
计算组最大值的同时保留自己的八个 BF16 输入值，从而在写出 int8 值时避免第二次
设备内存读取；M5 的交叉测量把完整的 512 前向提升了约 0.2-0.8%，且不改变任何
输出字节。更大的序列保留实测的 256 线程 kernel。`--use-slower-grouped-quantizer`
可在任何尺寸下强制后者以便做 A/B 比较。

原生基线面向原始的 `FL2VA/` 和 `Ref2VA/` checkpoint 目录树。模型的各个阶段被
分别加载和释放，因此 33B 的 transformer、Qwen 编码器和各解码器从不需要同时
存在于统一内存中。

## 本 fork 的优化

本 fork 为视频 VAE 解码器加了三项本地优化，以及支撑它们的实测工作。完整归档
——报告、复现脚本、原始日志、补丁与成片——都在 [`docs/vpipe/`](docs/vpipe/)。

### 1. 视频 VAE 走 BF16：解码快 3.37 倍

MiniMax-H3 的视频 VAE 解码器是 36 层 / 2048 宽的 ViT，此前全程跑 FP32。MPS 会把
FP32 matmul 编译成普通的 FP32 SIMD 路径，于是 Apple 张量核心一直闲置：在解码器的
真实形状上做微基准，测得 6.2 TFLOP/s，正好是 M5 Pro 的 FP32 SIMD 峰值
（20 核 x 128 FLOP x 2 x 1.26 GHz）。把同样的 GEMM 换成 BF16 后就跑在张量核心上，
达到 26 TFLOP/s，大约 4 倍。

同一会话内实测，960x544、124 帧、6 步去噪、seed 42，M5 Pro 64 GB，唯一变量是精度：

| 指标 | FP32（改造前） | BF16（改造后） | 变化 |
|---|---:|---:|---:|
| 视频 VAE 解码 | 159.450 s | 47.334 s | 3.37 倍 |
| 端到端整轮 | 409.95 s | 316.93 s | 1.29 倍 |
| 视频 VAE 张量存储峰值 | 9.365 GiB | 4.683 GiB | 0.50 倍 |
| 解码器形状上的 GEMM 吞吐 | 6.2 TFLOP/s | 26 TFLOP/s | 约 4 倍 |
| 相对 FP32 解码器的相对 L2 | - | 0.0048 | VPIPE 自身的门槛是 0.02 |

同会话重跑 VPIPE（消除跨天热态差异）后，视频 VAE 从落后 3.54 倍变成落后 1.05 倍，
端到端从落后 1.10 倍变成领先 1.17 倍。

实现要点：

- 解码器的 11 个算子全部改走 BF16。BF16 原语在 `h3_gpu.h` 里本来就有
  （`h3_gpu_linear_bf16`、`h3_gpu_sdpa_bf16`、`h3_gpu_swiglu_bf16`、
  `h3_gpu_rms_norm_bf16`、`h3_gpu_layer_norm_bf16`），只是视频 VAE 一个都没接。
- 新增两个 Metal 原语补上最后的缺口：`h3_gpu_scale_add_bf16` 和
  `h3_gpu_video_qkv_rope_bf16`。它们是对应 FP32 核的逐行忠实镜像，包括 RoPE
  半轴切分的负半轴符号处理和 QK 的 RMS epsilon 位置。两个 kernel 名都已加入
  pipeline 预热列表。
- checkpoint 里 560 个解码器张量全是 FP32，因此新增 `h3_weight_load_f32_as_bf16()`，
  在宿主端用 round-to-nearest-even 转成 BF16 再上传。全量 9.7 GiB 转换耗时
  0.18 s，相对省下的 112 s 可以忽略；常驻权重从 9.16 GiB 降到 4.58 GiB。
- MLP 保留 bias，所以仍拆成 linear -> swiglu -> linear，而没有用融合的
  fc1+SwiGLU NAX kernel——后者不接受 bias。
- `H3_VAE_PRECISION=f32` 可以恢复原来的 FP32 路径，用于 A/B 对照和回归测试。

### 2. 空间 tile 固定回 checkpoint 的 256 像素

发布版本的 checkpoint 是在 256 像素的空间 tile 上训练解码器的
（`FL2VA/video_vae/config.json` 里的 `vae_tile_size`），而 RoPE 表由 tile 自身的
latent 尺寸推导，所以换一个 tile 尺寸解码出来的就是另一个函数。原来的自动调优
默认值在 960x544 上会挑 304 像素，也就是 19x19 的 latent 格，而 checkpoint 是
16x16。现在默认值固定为 256，`H3_VAE_TILE_PIXELS` 仍可覆盖。

| | 304 px（原默认值） | 256 px（checkpoint） |
|---|---:|---:|
| 960x544 的 tile 数 | 8 | 15 |
| 解码耗时，FP32 | 126.09 s | 159.45 s |
| 解码耗时，BF16 | 35.4 s | 58.1 s |
| 成片对比（304 对 256） | 41.62 dB PSNR，0.9856 SSIM | |

这是"用速度换规格一致"而不是抢救画质：两种尺寸的成片感知上无法区分，而 304 像素
在 FP32 下少解约 26%，在 BF16 下少解约 64%——此时固定的每 tile dispatch 开销相对
更小的矩阵乘已经不再可以忽略。

### 3. 视频 VAE 全量 INT8：再快 1.48 倍，误差远低于编码器噪声

tile 数固定之后，VAE 解码就是算力瓶颈：线性层跑在 20.5–26.6 TFLOP/s，MLP 占每个
block 的 60% 时间。剩下唯一能拿的常数因子只有"每次乘法用多少位"，而 M5 的整数
张量核在同样形状上能到 31.8–50.4 TFLOP/s。DiT 早就有 int8 路径，视频 VAE 现在
也接上了。

`H3_VAE_PRECISION=int8` 让每个 block 的四个投影（qkv、out、ff.w1、ff.w2）走
INT8 matmul。归一化、RoPE、注意力、SwiGLU 和残差流仍然是 BF16——它们不是矩阵乘，
量化没有收益。权重在加载时量化一次（每个输出通道一个 F32 scale），激活则每次
matmul 前按行动态量化。

**输出投影刻意留在 BF16**：它是进像素前的最后一个矩阵乘，误差不会被任何东西平均
掉——把它也量化要多付 11.4 dB（40.55 dB / 0.8965 SSIM，而不是 51.94 dB /
0.9962 SSIM），却几乎不省时间。`H3_VAE_INT8_PROJECT=1` 可以打开它做对照实验。

960x544、124 帧，对照 FP32 解码器实测：

| 指标 | FP32 | BF16 | INT8 |
|---|---:|---:|---:|
| 视频 VAE 解码 | 249.888 s | 84.803 s | 58.716 s |
| 相对 FP32 的加速 | 1.00 倍 | 2.95 倍 | 4.26 倍 |
| 相对 BF16 的加速 | - | 1.00 倍 | 1.44 倍 |
| 张量存储峰值 | 9.36 GiB | 4.68 GiB | 6.77 GiB（加载期） |
| 相对 FP32 解码器的相对 L2 | - | 0.0048 | 0.0170 |

墙钟时间随热态漂移，所以 1.48 倍取自三组相邻 BF16/INT8 成对测量的中位数
（0.764 / 0.611 / 0.677）；上表的单次独立测量给出 1.44 倍。

保真度代价比编码器自己引入的误差小得多：

| 误差来源 | PSNR | RMSE（换算成 8 位色阶） | SSIM |
|---|---:|---:|---:|
| 最终 H.264 编码，相对同一份 FP32 解码 | 21.78 dB | 20.764 | 0.7772 |
| **本轮 INT8 VAE，相对同一份 FP32 解码** | **51.94 dB** | **0.646** | **0.9962** |
| （参考）本 fork 的 BF16 VAE | 60.14 dB | 0.251 | 0.9994 |

INT8 解码的误差比视频编码器自身的误差低 30 dB，SSIM 也远高于编码器，
也就是说到最终 MP4 里根本看不出来。


实现要点：

- INT8 matmul 没有 bias epilogue，因此新增 `h3_bias_bf16_to_bf16` Metal kernel，
  沿行广播把 bias 原地加回去。
- BF16 权重副本只在量化 command buffer **提交之后**才释放。提前释放会让 GPU
  读到已释放的内存，量化结果全 0，解码会塌成一张常数灰图。
- 设备缺少整数张量核时会打印提示并自动回退 BF16。

端到端（同一 prompt / 几何 / 种子，相邻两轮）：视频 VAE 解码从 74.538 s 降到
49.771 s（1.50 倍），整轮渲染从 490.99 s 降到 480.60 s。

一个方法论提醒：**对比两次渲染的成片，衡量不到解码器的精度**。同一个二进制、同一个
种子，相隔 50 分钟跑两次，成片就差 18.41 dB——生成器跨运行不可复现，所以
"INT8 轮 vs BF16 轮"的 17.53 dB 其实落在整条流水线的噪声底上。本 fork 的精度数字
一律来自"把同一条 latent 在不同精度下反复重放"，从不用两次渲染做对比。

完整实测、失败分析和复现命令见
[`docs/vpipe/reports/vae-int8-实验报告.md`](docs/vpipe/reports/vae-int8-实验报告.md)。

### 4. 合入本 fork 的上游修复

那些既修真 bug、又不改变合法输入行为的上游 PR，本 fork 直接整条合入（保留上游
提交本身，而不是逐行挑拣）。每一条都先在独立分支上合并、编译、跑完测试套件，
再落到 main。

| PR | 上游修复 | 在本机为何安全 |
|---|---|---|
| #79 | Metal parity 检查失败时 `make test` 真的会失败 | 只改一个 shell 连接符；本地反正会跳过 fixture 分支 |
| #63 | 复用 GQA 归约 scratch 前补 `threadgroup_barrier` | 该 barrier 位于一致控制流中；它修掉的是真实存在的写后读竞争（上游 #52），正是它让因果 softmax 不确定 |
| #44 | 因果 GQA 的 threadgroup 内存长度向上对齐到 16 字节 | Metal 校验层遇到未对齐长度会直接 abort；向上取整只增加内核根本不读的填充 |
| #9 | 视频 VAE 的 RGB 输出改用 `fminf`/`fmaxf` 钳位 | 对有限值完全等价；NaN 现在落成 0，而不是污染整帧 |
| #11 | 帧数对齐改用 64 位计算 | 对齐后的帧数放不进 `int` 时 `h3_align_frame_count()` 返回 0，由调用方拒绝 |
| #62 | stdout 改为行缓冲 | "Done" 行能立刻送到管道客户端；linenoise 的交互提示符走裸 `write()`，REPL 不受影响 |
| #48 | safetensors 数据区必须连续且完全覆盖 | 本机全部 116 个 `.safetensors`（官方 checkpoint 与 Turbo LoRA 合并产物）都通过新校验 |
| #74 | JSON 字符串分配按编码长度设上限 | 编码长度是解码后长度的上界，不会截断内容 |
| #31 | 容忍 FFprobe 紧凑输出末尾多出的 `x` | 带 display matrix side-data 的 iPhone `.mov` 文件 |
| #59 | 兼容 macOS 26 之前的 SDK 编译 | 补 `<limits.h>`，并给 `MTLGPUFamilyMetal4` 加编译期开关 |
| #14（部分） | 相对 shader 路径改为相对可执行文件目录解析 | 该 PR 其余部分是功能特性，因此只取这一个提交 |

这张表背后的验证：

- `make test` 通过，1912 项检查，退出码 0。依赖 fixture 的 Metal、tokenizer 与
  官方权重套件在本机跳过，因为本机没有安装那些 released fixture。
- 真实权重：`h3 --info -d MiniMax-H3` 让整份 144 GB checkpoint 走一遍新的
  safetensors 校验；真实权重的视频 VAE 测试用全 NaN latent 解码，并检查输出
  仍是有限值且落在 [0, 1]。
- 端到端 A/B：256x256、22 帧、4 步、seed 42，分别用合入前的二进制和合入后的
  二进制各渲染一次。两者相差 23.05 dB PSNR，而**同一个**合入前二进制跑两次
  相差 23.13 dB。也就是说这些修复落在流水线自身的 run-to-run 噪声地板之内，
  不改变生成结果。

#14 的其余部分被有意排除。它的"数据区未对齐就拒绝"检查本身是对的，但它同时
会拒绝仓库自己的测试夹具：那段 137 字节的 JSON header 让数据区偏离 8 字节边界
1 个字节，所以上游这份 PR 按提交状态跑不过它自己的 `make test`。该 PR 里的
Turbo LoRA 折叠工具值得单独取用：它把步数蒸馏 LoRA 烘进 checkpoint，作者实测
124 帧 960x544 的渲染从 8.8 分钟降到 6.2 分钟。

### 5. 把 Turbo LoRA 烘进 checkpoint

h3.c 没有 LoRA 运行时，也不需要：LoRA 就是 `W' = W + strength*scale*(B@A)`，
可以离线一次性烘进权重。`tools/merge_lora_h3.py` 做这件事，随本 fork 提供。

```sh
python3 tools/merge_lora_h3.py \
  --lora minimax_h3_turbo_v4_step600_ema.safetensors \
  --src  ./MiniMax-H3/FL2VA/transformer \
  --dst  ./MiniMax-H3-turbo/FL2VA/transformer
```

- 支持三种键名约定：diffusers/sd-scripts（`lora_unet_blocks_24_attn_qkv_proj
  .lora_down.weight` 配 `.lora_up.weight` 与 `.alpha`）、ComfyUI
  （`diffusion_model.blocks.24...lora_A.weight`），以及
  `larryvrh/MiniMax-H3-Turbo-Lora` 用的 checkpoint 键名拼法。
  `--strength` 缩放 delta，sd-scripts 的 `alpha`/rank 因子会自动带上。
- 分片用 `cp -c`（APFS 写时复制）克隆，只重写每个被打补丁张量的字节区间。
  safetensors 的 header、张量顺序与数据区对齐逐字节不变，h3.c 直接就能加载
  折叠后的目录。
- 每个张量在写入任何字节之前都要过一遍随机探针检查：`folded@x` 必须在 1e-4
  相对误差内等于 `W@x + factor*(B@(A@x))`。A/B 装反、适配器转置、rank 不匹配
  都会在这里被拦下。
- base checkpoint 里找不到的 LoRA 目标默认直接报错（要放过得显式加
  `--allow-missing`），所以不会不小心做出"折了一半"的 checkpoint。

本机实测验证：把 `minimax_h3_turbo_v4_step600_ema` 的 259 对 adapter 折进
FL2VA transformer，结果与此前已经烘好的 checkpoint **逐字节相同**（535/535 个
张量一致），这交叉验证了折叠结果与之前实测过的模型是同一个。

LoRA 只影响 DiT transformer：text encoder、video VAE、audio VAE 都不受影响，
所以模型目录里这三块可以用软链指回原始快照，而不必复制。每对模型这样能省下
145 GB，这正是磁盘占用的大头。transformer 本身必须是真正打过补丁的文件：`cp -c`
会从原分片克隆一份，未被改动的字节是共享的，补丁只重写受影响的那些区间。

用折好的 Turbo LoRA 采样时：`--steps 5` 或 `6`，**不要**再叠加 `--reuse` 或
`--core-reuse`——蒸馏调度本身已经把这两者利用的冗余去掉了。4 步会拖影。

### 测试

新增两个测试二进制，并挂进 `make test`：

- `tests/test_video_vae.c`（`h3_video_vae_tests`）对比 F32、BF16 与 INT8 解码：
  墙钟时间、相对 L2、最大与平均绝对误差、RGB 值域、帧数与分辨率，覆盖多种 tile
  尺寸，并附带加载基准。
- `tests/test_gpu_bf16_ops.c`（`h3_gpu_bf16_ops_tests`）验证两个新原语与对应
  FP32 路径的一致性，相同输入下要求逐位一致。

数值中性验证还把新二进制强制回 `H3_VAE_PRECISION=f32` + 304 px tile 与旧二进制
对比：23.12 dB PSNR / 0.832 SSIM，即热漂移噪声量级，算子 profile 完全一致
（`submissions=56 direct=12264 linear=8120 attention=2016`）。
