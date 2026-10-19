from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .util import sanitize_proxy_env

PROMPT = """你是播客/视频精读助手。请阅读文稿文件：{path}
标题：{title}

只依据文稿内容，用简体中文输出 Markdown，结构如下：
## 一句话结论
## 核心要点
- 5-10 条，每条尽量带 [HH:MM:SS] 时间戳
## 值得回听
- 2-4 个片段，带时间戳和理由
## 提到的人物 / 公司 / 书籍 / 概念
## 可执行行动 / 对我的启发

不要编造文稿中没有的信息。不要创建或修改任何文件，直接把结果输出在回复里。"""


def available_engines() -> list[str]:
    engines = ["none"]
    if os.environ.get("DEEPSEEK_API_KEY"):
        engines.append("deepseek")
    if shutil.which("agy"):
        engines.append("agy")
    return engines


def summarize(
    transcript_path: Path,
    *,
    title: str,
    out_path: Path,
    engine: str = "auto",
    timeout: int = 900,
) -> tuple[str, str]:
    if engine == "auto":
        if os.environ.get("DEEPSEEK_API_KEY"):
            engine = "deepseek"
        elif shutil.which("agy"):
            engine = "agy"
        else:
            engine = "none"

    if engine == "none":
        raise RuntimeError(
            "no summarizer configured; set DEEPSEEK_API_KEY or install/authenticate agy"
        )

    if engine == "agy":
        text = _summarize_agy(transcript_path, title=title, timeout=timeout)
    elif engine == "deepseek":
        text = _summarize_deepseek(transcript_path, title=title, timeout=timeout)
    else:
        raise ValueError(f"unknown summarizer engine: {engine}")

    text = _dedupe(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return engine, text


def _dedupe(text: str) -> str:
    """Some summarizers repeat the first half; keep the last complete block."""
    heading = "## 一句话结论"
    positions = [index for index in range(len(text)) if text.startswith(heading, index)]
    if len(positions) > 1:
        candidate = text[positions[-1]:]
        if "## 可执行行动" in candidate:
            return candidate.strip() + "\n"
    return text


def _summarize_agy(transcript_path: Path, *, title: str, timeout: int) -> str:
    prompt = PROMPT.format(path=transcript_path.resolve(), title=title)
    proc = subprocess.run(
        ["agy", "-p", prompt],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"agy failed ({proc.returncode}): {proc.stderr[-1000:]}")
    text = proc.stdout.strip()
    if not text:
        raise RuntimeError(f"agy returned no summary: {proc.stderr[-1000:]}")
    return text + "\n"


def _summarize_deepseek(transcript_path: Path, *, title: str, timeout: int) -> str:
    sanitize_proxy_env()
    import httpx

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    transcript = transcript_path.read_text(encoding="utf-8")
    prompt = PROMPT.format(path=str(transcript_path.resolve()), title=title)
    response = httpx.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": os.environ.get("MEDIA_ASR_SUMMARY_MODEL", "deepseek-flash"),
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": transcript},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"].strip() + "\n"