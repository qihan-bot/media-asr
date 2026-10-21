#!/usr/bin/env python3
"""Diarize already-transcribed feed items into runs/<label>/.

This is a batch companion for long subscriptions. It does not overwrite the
primary transcript; each diarized variant is written under
``data/items/<id>/runs/<label>/``. The remote full pipeline is used, so the
source is downloaded again and ASR is run with word timestamps for clean
speaker alignment.
"""

from __future__ import annotations

import argparse
import os
import sys

from media_asr.config import ITEMS_DIR
from media_asr.db import Store
from media_asr.remote import RemoteGPU


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-contains", default="allinchamathjason")
    parser.add_argument("--label", default="diarized")
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--num-speakers", type=int, default=-1)
    parser.add_argument(
        "--pyannote-pipeline",
        default="pyannote/speaker-diarization-3.1",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    host = os.environ.get("MEDIA_ASR_REMOTE_HOST", "")
    if not host:
        print("set MEDIA_ASR_REMOTE_HOST or source the remote env file", file=sys.stderr)
        return 1
    if not os.environ.get("HF_TOKEN"):
        print("set HF_TOKEN for pyannote; run through `vibe vault run`", file=sys.stderr)
        return 1

    store = Store()
    rows = store.conn.execute(
        """
        SELECT * FROM items
         WHERE status = 'done'
           AND feed_url LIKE ?
         ORDER BY published_at ASC
        """,
        (f"%{args.feed_contains}%",),
    ).fetchall()
    store.close()
    if args.limit:
        rows = rows[: args.limit]

    remote = RemoteGPU(
        host,
        user=os.environ.get("MEDIA_ASR_REMOTE_USER", ""),
        workdir=os.environ.get("MEDIA_ASR_REMOTE_WORKDIR", "media-asr-work"),
        python=os.environ.get(
            "MEDIA_ASR_REMOTE_PYTHON", r".venv\Scripts\python.exe"
        ),
        shell=os.environ.get("MEDIA_ASR_REMOTE_SHELL", "cmd"),
        model=args.model,
        language=args.language,
        mode="full",
        diarize=True,
        num_speakers=args.num_speakers,
        diarization_backend="pyannote",
        pyannote_pipeline=args.pyannote_pipeline,
        key_filename=os.environ.get("MEDIA_ASR_REMOTE_KEY"),
    )
    remote.connect()
    failures = 0
    try:
        for index, row in enumerate(rows, start=1):
            item = dict(row)
            out_dir = ITEMS_DIR / item["id"] / "runs" / args.label
            transcript = out_dir / "transcript.md"
            if transcript.exists() and not args.force:
                print(f"[{index}/{len(rows)}] skip {item['title']}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{index}/{len(rows)}] diarize {item['title']} "
                f"({(item['duration'] or 0) / 60:.0f} min)",
                flush=True,
            )
            try:
                remote.run_full(
                    item,
                    out_dir,
                    language=args.language,
                    model=args.model,
                    diarize=True,
                    num_speakers=args.num_speakers,
                    beam=5,
                    progress=lambda message: print(message, flush=True),
                )
            except Exception as exc:
                failures += 1
                print(f"FAILED {item['id']}: {exc}", file=sys.stderr, flush=True)
    finally:
        remote.close()
    print(f"done: {len(rows) - failures}/{len(rows)} succeeded", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())