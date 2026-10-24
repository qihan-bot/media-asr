# media-asr

[English](README.md) | [简体中文](README.zh-CN.md)

**版本 0.0.2** — 本地优先的视频 / 播客语音转文字工具。

`media-asr` 支持视频链接、播客 RSS、本地音视频文件，输出带时间戳的
Markdown、SRT、TXT、JSON，并把每个片段写入 SQLite FTS5，支持全文检索。
默认在本地 CPU 上用 `faster-whisper` 转写，也可以把下载、ffmpeg、转写、
渲染全部卸载到远程 GPU 主机。

## 功能

- YouTube、Bilibili、Vimeo、直接音频 URL、播客 RSS、本地文件。
- 本地 `faster-whisper` CPU 转写，默认 `small`。
- 远程 GPU 后端：OpenAI Whisper `large-v3-turbo`，通过 SSH 调用。
- 远程两种模式：
  - `asr`：本地下载 / ffmpeg / 渲染，只把转写放到远端。
  - `full`：下载、ffmpeg、转写、渲染全部在远端，本地只建索引。
- 说话人分离：`sherpa-onnx`（全离线）或 `pyannote.audio`（需要 HF 授权）。
- 中文字幕默认转简体，`--traditional` 保留原脚本。
- 可选摘要：Antigravity (`agy`) 或 DeepSeek API。
- 播客库：每天更新多档节目，补标点、识别说话人姓名，并对听错的人名和术语做可审计的纠错。
- Linux / Windows 都可运行；本地转写不需要任何云端账号。

## 环境要求

- Python 3.11+
- `ffmpeg`、`ffprobe`
- 远程模式：一台可通过 SSH 访问的 GPU 主机，安装 Python、ffmpeg 和
  [docs/remote-gpu.md](docs/remote-gpu.md) 里的依赖。

## 安装

```bash
git clone https://github.com/qihan-bot/media-asr.git
cd media-asr
python3 -m pip install -e .
```

也可以直接用仓库里的 `./media-asr`，它会自动定位项目目录。

部分功能需要的额外依赖：

```bash
python3 -m pip install feedparser zhconv paramiko
python3 -m pip install -e '.[library]'   # 播客库术语纠错
```

## 快速开始

```bash
# 视频
./media-asr add 'https://www.youtube.com/watch?v=...' --run

# B 站中文视频
./media-asr add 'https://www.bilibili.com/video/BV...' --language zh --run

# 本地文件
./media-asr add ~/Downloads/interview.mp4 --title '访谈' --language zh --run

# 播客 RSS：只加入最新 3 期
./media-asr add 'https://example.com/feed.xml' --latest 3
./media-asr run

# 查看与检索
./media-asr list
./media-asr search '信息过载'
./media-asr show <id>
```

不加 `--run` 时，条目会进入 `pending` 队列，之后用 `./media-asr run` 处理。

## 命令

| 命令 | 用途 |
| --- | --- |
| `add <url\|file>` | 解析 URL / RSS / 文件并加入队列（`--latest N`、`--title`、`--run`） |
| `run` | 处理 pending 条目（`--model`、`--language`、`--prompt`、`--keep-audio`） |
| `retranscribe <id>` | 用另一个模型重跑，结果写到 `runs/<label>/`，不覆盖主稿 |
| `refresh` | 检查订阅的 RSS 是否有新节目（`--latest N`、`--run`） |
| `list` | 列出条目（`--status pending`、`--limit`） |
| `show <id>` | 查看元数据和文稿开头（`--full` 查看全文） |
| `search <query>` | 检索所有片段，显示时间戳和说话人 |
| `summarize <id>` | 生成 `summary.md`（`--engine auto\|agy\|deepseek\|none`） |
| `remote-check` | 检查 SSH、GPU、Python 和远端依赖 |
| `feeds` | 列出订阅的 RSS |
| `stats` | 统计条目、小时数、片段数、订阅数 |
| `remove <id>` | 删除条目及其文件（`--yes`） |
| `library-update` | 播客库：刷新、转写、重新分离、补标点与姓名、术语纠错 |
| `rediarize` | 播客库：最大说话人标签混入多人的节目重新分离 |
| `enrich` | 播客库：补标点、识别说话人姓名 |
| `correct-terms` | 播客库：可审计的人名 / 术语纠错 |

常用参数：

```text
--model small|medium|large-v3-turbo|<path>   # 默认 small
--language zh|en|...                        # 留空自动识别
--prompt '人名, 术语, 缩写'                  # 热词 / 标点提示
--beam 5                                    # 精度与速度
--threads 8                                 # CPU 线程
--keep-audio                                # 保留 16kHz WAV
--offline                                   # 不下载模型
--traditional                               # 保留繁体
--backend local|remote                      # 本地或远程 GPU
--remote-mode asr|full                      # 远程范围
--diarize                                   # 说话人分离
--num-speakers 4                            # -1 自动判断
--diarization-backend sherpa|pyannote       # 默认 sherpa
--remote-shell cmd|sh                       # 远端 shell（默认 cmd）
```

## 远程 GPU 后端

本地 CLI 可以通过 SSH 把任务交给远程 GPU。音频会先上传到远端临时目录，
转写完成后打包下载，本地解包并写入 SQLite。除非加 `--remote-keep`，远端
文件会自动删除。

先配置连接：

```bash
export MEDIA_ASR_REMOTE_HOST=gpu-host
export MEDIA_ASR_REMOTE_USER="$USER"
export MEDIA_ASR_REMOTE_KEY="$HOME/.ssh/id_ed25519"
export MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
```

然后：

```bash
# 只远程转写：本地下载 / ffmpeg / 渲染
./media-asr run --backend remote --remote-mode asr

# 全远程：下载、ffmpeg、转写、渲染都在远端
./media-asr run --backend remote --remote-mode full

# 健康检查
./media-asr remote-check
```

`media-asr-remote` 包装脚本默认使用 full 模式，并可以读取本地配置
`~/.config/media-asr/remote.env`（不要提交到仓库）：

```bash
# ~/.config/media-asr/remote.env
MEDIA_ASR_REMOTE_HOST=gpu-host
MEDIA_ASR_REMOTE_USER=me
MEDIA_ASR_REMOTE_KEY=/home/me/.ssh/id_ed25519
MEDIA_ASR_REMOTE_WORKDIR=media-asr-work
# 可选：从 Avibe Vault 注入 SSH 密码
# MEDIA_ASR_REMOTE_SECRET=<vault-ref>
```

```bash
./media-asr-remote run
./media-asr-remote run --remote-mode asr
./media-asr-remote retranscribe <id> --label turbo-remote
```

SSH 主机密钥按 `~/.ssh/known_hosts` 校验。`--remote-auto-add-host-key` 只应
在首次可信连接时使用。密码通过 `MEDIA_ASR_REMOTE_PASSWORD` 提供，不要写进
仓库文件或 shell 历史。

多个任务可以共用一台主机：每个任务运行自己的一份 worker 代码，并通过远程工作目录下的
`gpu.lock` 轮流使用 GPU。

远端 CUDA PyTorch、OpenAI Whisper、ffmpeg、yt-dlp、说话人分离模型的安装
见 [docs/remote-gpu.md](docs/remote-gpu.md)。

## 说话人分离

说话人分离只在远程 **full** 模式下可用。

### sherpa-onnx（默认，全离线）

```bash
./media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 4
```

模型放在远端的 `models/diarization/`，安装步骤见
[docs/remote-gpu.md](docs/remote-gpu.md)。

### pyannote（精度更高，需要 HF 授权）

用持有 `HF_TOKEN` 的账号接受以下仓库条款：

- `pyannote/segmentation-3.0`
- `pyannote/speaker-diarization-3.1`
- `pyannote/speaker-diarization-community-1`

然后：

```bash
HF_TOKEN=... ./media-asr run --backend remote --remote-mode full \
  --diarize --num-speakers 5 \
  --diarization-backend pyannote \
  --pyannote-pipeline pyannote/speaker-diarization-3.1
```

两种后端都会使用 Whisper 的 word timestamps，把词对齐到说话人 turn，并把
`SPEAKER_00` 标签写进 Markdown、SRT、TXT、JSON 和 SQLite 检索结果。标签是
近似的，需要人工核对前几段再映射姓名。94 分钟、5 人的测试节目里，
pyannote 3.1 分离约 100 秒，能干净地分出 5 人；sherpa 约 250 秒，会把相近
男声拆分或合并。

## 播客库

播客库是一个数据目录，里面跟踪多档节目（可以是不同语言），持续完成转写、
说话人分离、补标点、说话人姓名识别，以及听错的人名和术语纠错。它需要远程
GPU 后端（pyannote）、DeepSeek API（`DEEPSEEK_API_KEY`）和 `library` 额外依赖：

```bash
python3 -m pip install -e '.[library]'          # wordfreq、jieba、pypinyin
export MEDIA_ASR_DATA=~/podcasts
./media-asr add 'https://feeds.fireside.fm/sv101/rss' --latest 20     # 每档节目订阅一次
cp docs/examples/podcast-library.toml "$MEDIA_ASR_DATA/library.toml"  # 再按需修改
HF_TOKEN=... DEEPSEEK_API_KEY=... ./media-asr library-update --jobs 3 \
  --gpu-env ~/.config/media-asr/gpu-a.env --fallback-gpu-env ~/.config/media-asr/remote.env
```

`$MEDIA_ASR_DATA/library.toml` 列出节目，每档一个 `[[show]]` 表（完整示例见
[docs/examples/podcast-library.toml](docs/examples/podcast-library.toml)）：

```toml
[[show]]
name = "Latent Space"                               # 可选，用于日志
feed = "https://api.substack.com/feed/podcast/1084089.rss"
language = "en"                                     # zh 或 en
hosts = "swyx (Shawn Wang) and Alessio Fanelli"     # 可选
```

- `feed`：用 `add` 订阅时的 RSS 地址；节目条目的 feed 地址包含这段文字即归入该节目。
- `language`：Whisper 转写语言，同时决定补标点用的提示词。
- `hosts`：很少在节目里报自己名字的常驻主持人，作为说话人识别的背景信息。

| 命令 | 用途 |
| --- | --- |
| `library-update` | 每日流程：`refresh --latest 5`（失败最多重试 3 次）→ 各档节目待处理和失败的单期（从旧到新，每档最多 100 期），每期单独通过 `media-asr-remote` 执行 `run --ids <id> --language <节目语言> --diarize --diarization-backend pyannote`，同时最多 `--jobs` 期 → `media-asr-remote rediarize` → `enrich` → `correct-terms` → `stats`。任何 feed、单期或步骤失败时，做完能做的部分后以 1 退出；每一步都幂等，失败后可原样重跑（`--latest`、`--workers`、`--jobs`） |
| `rediarize` | 某个说话人标签占说话时长超过 75%（`--share`）时，由 DeepSeek 判断是否混入了多人；严重合并（超过 90%）或 3 人以上的节目，在 GPU 上按“人数 + 1”重新转写分离。同一期的检查、重跑和完成标记都在这个进程里、持有同一把锁完成，所以要用 `media-asr-remote rediarize` 运行以连上 GPU 主机（`--dry-run`、`--shard i/n`；`--workers N` 让模型判断 N 期并行，重跑仍逐期进行） |
| `enrich` | 补标点、识别说话人姓名（`--ids`、`--force`、`--limit`、`--workers`、`--chunk-workers`） |
| `correct-terms` | 可审计的人名 / 术语纠错（`--ids`、`--force`、`--workers`） |

`library-update` 按 `media-asr-remote` 的 env 文件选择 GPU 主机：`--gpu-env`
（`MEDIA_ASR_GPU_ENV`）在 90 秒内通过 `remote-check` 就用它，同时最多 `--jobs`
期；否则在 `--fallback-gpu-env`（`MEDIA_ASR_FALLBACK_GPU_ENV`）通过检查时用它，
一次一期（Windows 主机没有 GPU 锁）。备用主机可以不配：给出的主机都没通过检查时，
跳过转写和 `rediarize`，本次以 1 退出（`enrich` 和 `correct-terms` 照常运行）。
两个选项都不给时，不做检查，直接用 `media-asr-remote` 的默认值
（`MEDIA_ASR_REMOTE_ENV` 或 `~/.config/media-asr/remote.env`）。它调用源码目录里的
`media-asr-remote` 包装脚本，所以要在克隆的仓库里运行。所有 DeepSeek 步骤使用
`MEDIA_ASR_LLM_MODEL`（默认 `deepseek-flash`）。

`--jobs N` 时，GPU 主机上同时最多处理 N 期：一期在下载和 ffmpeg 时，另一期可以用
GPU；各期通过远程 worker 的锁轮流使用 GPU（远程工作目录下的 `gpu.lock`，从加载
Whisper 一直持有到 pyannote 结束，见
[docs/remote-gpu.md](docs/remote-gpu.md#several-jobs-on-one-host)）。每期开始时日志
记一行 `start`，结束时把该期的输出作为一整段写入日志。

每期节目的文件：

```text
items/<id>/
  transcript.*         # 转写 + 说话人分离（SPEAKER_00 标签）；只有 run 会重写
  rediarize.json       # 合并检查：标签、前后占比、判断结果
  transcript_named.*   # 补标点、标姓名、术语纠错后的文稿；检索索引指向它
  named_base.json      # 术语纠错前的补标点、标姓名片段
  enrich.json          # 补标点的片段数、每个说话人姓名及依据
  terms.json           # 采用的替换对（层级、次数）和被拒绝的替换对（原因）
  .lock                # 某一步写这期文件时持有的锁
```

`rediarize.json`、`enrich.json`、`terms.json` 是完成标记：每一步跳过已有自己
标记的条目。`run` 写新转写之前会删除这三个标记，`enrich` 会删除 `terms.json`，
后续步骤因此会重新执行。每一步写某期的文件和索引时都持有 `items/<id>/.lock`，
先确认输入自读取后没有变化，并在覆盖任何文件之前删除自己的标记，所以写到一半
中断的条目会在下次运行时重做。`library-update` 持有数据目录下的
`.library-update.lock`，同时启动的第二个更新会以 1 退出。

审计与回滚：

- 补标点逐段校验，只有字母和数字完全不变时才采用，所以不会改动说过的话。
- 说话人姓名只采用高 / 中置信度的结果；其余标签写成 `主持人N`、`嘉宾N`、
  `说话人N`（英文节目为 `Host N`、`Guest N`、`Speaker N`）。
- 术语纠错时模型不改写文稿，只提出 `wrong -> right` 替换对。一个替换对要同时
  听起来像误听（中文两边字数相同，且每对汉字的无声调拼音相似度不低于 0.4，
  三个及以上音节时允许一对例外；英文拼写相似度不低于 0.55，短词 0.65；错误
  写法不是常用词），并且有依据：`reference` 层级（正确写法出现在节目简介、
  其他期标题或说话人姓名里），或 `model` 层级（模型高置信度认定的知名名词），
  才会被采用。采用的替换对在未纠错的文稿上一次性按字面替换：最长的匹配优先，
  已经是正确写法的文字不动，替换出来的文字不会再被匹配。`terms.json` 记录每个
  采用的替换对的层级和实际替换次数，以及每个被拒绝的替换对和原因（包括采用后
  已无可替换之处的替换对）。
- 每一步都保留输入：`enrich` 读取原始的 `transcript.json`，`correct-terms`
  总是从 `named_base.json` 开始，纠错不会叠加。`correct-terms --ids <id> --force`
  重建纠错后的文稿（例如规则修改之后）；`enrich --ids <id> --force` 从原始转写
  重建全部结果并删除 `terms.json`，下次运行时会重新纠错。

## 输出结构

```text
data/
  media.db
  items/<id>/
    meta.json
    audio.wav              # 仅 --keep-audio
    transcript.md
    transcript.txt
    transcript.srt
    transcript.json
    summary.md             # 可选
    run.json               # 远程运行指标
    runs/<label>/          # retranscribe / 不同分离版本
```

数据目录可用 `MEDIA_ASR_DATA` 覆盖；模型缓存默认在 `~/.cache/whisper`，
可用 `MEDIA_ASR_MODEL_DIR` 覆盖。

## 配置项

| 变量 | 含义 |
| --- | --- |
| `MEDIA_ASR_HOME` | 项目根目录默认值（默认源码目录） |
| `MEDIA_ASR_DATA` | 数据目录 |
| `MEDIA_ASR_MODEL` | 本地 faster-whisper 模型（默认 `small`） |
| `MEDIA_ASR_LANGUAGE` | 默认语言 |
| `MEDIA_ASR_THREADS` | CPU 线程数 |
| `MEDIA_ASR_COMPUTE` | CTranslate2 计算类型（默认 `int8`） |
| `MEDIA_ASR_BACKEND` | `local` 或 `remote` |
| `MEDIA_ASR_REMOTE_HOST` | SSH 主机 |
| `MEDIA_ASR_REMOTE_USER` | SSH 用户（默认当前用户） |
| `MEDIA_ASR_REMOTE_PORT` | SSH 端口 |
| `MEDIA_ASR_REMOTE_KEY` | SSH 私钥 |
| `MEDIA_ASR_REMOTE_WORKDIR` | 远端工作目录（默认 `media-asr-work`） |
| `MEDIA_ASR_REMOTE_PYTHON` | 远端 Python（默认 `.venv\Scripts\python.exe`） |
| `MEDIA_ASR_REMOTE_SHELL` | 远端 `cd`/Python 命令使用的 shell（`cmd` 或 `sh`） |
| `MEDIA_ASR_REMOTE_MODEL` | 远端 Whisper 模型（默认 `large-v3-turbo`） |
| `MEDIA_ASR_REMOTE_MODE` | `asr` 或 `full` |
| `MEDIA_ASR_REMOTE_PASSWORD` | SSH 密码（优先用 key 或密钥管理） |
| `MEDIA_ASR_REMOTE_SECRET` | 包装脚本使用的可选 Vault 引用 |
| `MEDIA_ASR_REMOTE_ENV` | `media-asr-remote` 读取的 env 文件（默认 `~/.config/media-asr/remote.env`） |
| `HF_TOKEN` | pyannote 模型所需的 HuggingFace token |
| `DEEPSEEK_API_KEY` | DeepSeek API key（摘要、播客库） |
| `MEDIA_ASR_LLM_MODEL` | 播客库各步骤使用的 DeepSeek 模型（默认 `deepseek-flash`） |
| `MEDIA_ASR_GPU_ENV` | `library-update`：首选 GPU 主机的 env 文件 |
| `MEDIA_ASR_FALLBACK_GPU_ENV` | `library-update`：备用 GPU 主机的 env 文件 |

## 性能参考

以下是参考数字，不是保证值。播客为 85 分钟中文音频，视频为 94 分钟英文音频。

| 主机 / 模式 | 模型 | 耗时 | 实时倍率 |
| --- | --- | ---: | ---: |
| 6 核 CPU，本地 | faster-whisper small/int8 | 15.0 分钟 | 5.66× |
| 6 核 CPU，本地 | faster-whisper medium/int8 | 39.2 分钟 | 2.17× |
| RTX 5090 Laptop，GPU | large-v3-turbo/fp16 | 8.8 分钟 | 9.61× |
| RTX 5090 Laptop，全远程 | large-v3-turbo/fp16 | 7.9 分钟 | 10.70× |
| RTX 5090 Laptop，GPU | medium/fp16 | 23.0 分钟 | 3.70× |

说话人分离在 94 分钟节目上大约增加 100 秒（pyannote 3.1）或 250 秒
（sherpa）。开启 word timestamps 会让 ASR 稍慢。

## 限制

- `small` 快，但人名、数字、中英混合术语容易错；可用 `--prompt`、`medium`
  或远程 GPU 提升。
- 不做翻译、章节切分。说话人姓名只在播客库的 `enrich` 步骤里识别（DeepSeek
  阅读节目简介和文稿）。
- pyannote 需要 gated HuggingFace 模型和 token。
- OpenAI Whisper 可能跳过短段落，或在结尾幻觉出“字幕志愿者 / 点赞订阅”等
  文本；远端 worker 会过滤常见结尾，但不能保证完美。
- 说话人聚类可能拆分或合并相近声音，标签只应视为近似。
- FTS5 使用 trigram 分词，少于 3 个字符的查询会退化为 `LIKE`。
- 远程模式默认按 Windows OpenSSH 的 `cmd.exe` + `.venv\Scripts\python.exe`
  布局设计；Linux 远端需要调整 `MEDIA_ASR_REMOTE_PYTHON` 和工作目录。

## 开发

```bash
python3 -m pytest -q
python3 -m ruff check .
```

目录结构：

```text
media_asr/          CLI、抓取、ASR、数据库、渲染、远程客户端，
                    播客库（library、llm、enrich、terms、rediarize、voiceprint）
scripts/            远端 worker、OpenAI Whisper 辅助脚本、对比工具、
                    每日任务包装脚本、All-In 迁移脚本（migrate_allin.py）
tests/              离线测试
docs/remote-gpu.md  远程 GPU 安装与基准
docs/examples/      播客库配置示例
```

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)