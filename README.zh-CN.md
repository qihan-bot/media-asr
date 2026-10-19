# media-asr

[English](README.md) | [简体中文](README.zh-CN.md)

**版本 0.01** — 本地优先的视频 / 播客语音转文字工具。

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
| `MEDIA_ASR_REMOTE_MODEL` | 远端 Whisper 模型（默认 `large-v3-turbo`） |
| `MEDIA_ASR_REMOTE_MODE` | `asr` 或 `full` |
| `MEDIA_ASR_REMOTE_PASSWORD` | SSH 密码（优先用 key 或密钥管理） |
| `MEDIA_ASR_REMOTE_SECRET` | 包装脚本使用的可选 Vault 引用 |
| `HF_TOKEN` | pyannote 模型所需的 HuggingFace token |

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
- 不做翻译、章节切分、说话人姓名识别。
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
media_asr/          CLI、抓取、ASR、数据库、渲染、远程客户端
scripts/            远端 worker、OpenAI Whisper 辅助脚本、对比工具
tests/              离线单元测试
docs/remote-gpu.md  远程 GPU 安装与基准
```

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 致谢

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)