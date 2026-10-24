#!/usr/bin/env python3
"""Move the All-In archive into the podcast library, keeping episode ids.

    python3 scripts/migrate_allin.py --archive data --library data-podcasts          # dry run
    python3 scripts/migrate_allin.py --archive data --library data-podcasts --apply

The archive is the All-In data directory of media-asr 0.0.3 (``media.db``,
``items/<id>/runs/diarized/``, ``speaker_embeddings/<id>.npz``); it is only
read. For every done All-In episode with a diarized transcript (feed
allinchamathjason, and the YouTube copy whose digest is published under its
id), the library gets:

- ``items/<id>/``: ``transcript.{json,md,srt,txt}`` and ``run.json`` of the
  diarized run, ``meta.json`` of the episode;
- ``media.db``: the episode (done, same id, title, source and dates) and the
  segments of the diarized transcript, which ``enrich`` then replaces;
- ``voiceprint.json`` as ``media-asr voiceprint`` writes it, from the archived
  embeddings when they fit the diarized segmentation, else from the voice
  fields of ``speaker_map.json`` (none if neither: the voiceprint step embeds
  the episode on the GPU host);

plus the All-In feed and the reference voices ``voiceprints/all-in.npz``, built
as ``voiceprint_relabel.py`` built its templates, from the labels the LLM named
with high confidence (``speaker_map.llm.json``); an existing file is used as is.

An episode already in the library with the same transcript is left alone (a
missing voiceprint.json is added); one with another transcript, or whose id or
source the library has, is reported as a conflict and not touched, also when
it appears after the survey (checked again under the item lock and a database
write lock before anything is written). So the command can be re-run.
``--apply`` holds the library's ``.library-update.lock``; an archive and a
library that are the same directory or inside one another are refused. Afterwards run ``media-asr enrich`` and
``media-asr correct-terms`` (or ``library-update``) with the All-In show in
``library.toml`` (docs/examples/podcast-library.toml).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from media_asr import voiceprint as V  # noqa: E402
from media_asr.db import Store  # noqa: E402
from media_asr.library import LibraryError, load_shows  # noqa: E402
from media_asr.util import file_lock, item_lock, now_iso  # noqa: E402

FEED = "allinchamathjason"
# All-In episodes that did not come from the feed: the YouTube copy of 2026-10-17.
EXTRA_IDS = {"https-www-youtube-com-watch-v-f9423524"}
REFERENCES = Path("voiceprints") / "all-in.npz"
RUN_FILES = ("transcript.json", "transcript.md", "transcript.srt", "transcript.txt", "run.json")


def labels_fingerprint(segments: list[dict]) -> str:
    """Hash of boundaries and labels; a speaker map fits only this labelling
    (``speaker_common.labels_fingerprint`` of the archive)."""
    rows = [
        [round(float(s["start"]), 3), round(float(s["end"]), 3), s.get("speaker")] for s in segments
    ]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def read_only(path: Path, *, quiet: bool = False) -> sqlite3.Connection:
    """Open a database for reading. ``quiet`` (the archive): when its WAL holds
    nothing, open it immutable, so SQLite creates no -wal/-shm files next to it."""
    wal = path.with_name(path.name + "-wal")
    immutable = quiet and (not wal.exists() or wal.stat().st_size == 0)
    db = sqlite3.connect(f"file:{path}?mode=ro{'&immutable=1' if immutable else ''}", uri=True)
    db.row_factory = sqlite3.Row
    return db


class Episode:
    """One archived All-In episode and what the library makes of it."""

    def __init__(self, archive: Path, row: sqlite3.Row) -> None:
        self.row = row
        self.id = row["id"]
        self.item_dir = archive / "items" / self.id
        self.run = self.item_dir / "runs" / "diarized"
        raw = (self.run / "transcript.json").read_bytes()
        self.transcript_bytes = raw
        self.digest = hashlib.sha256(raw).hexdigest()
        self.data = json.loads(raw)
        self.segments = self.data.get("segments", [])
        self.embeddings, self.why = self._embeddings(
            archive / "speaker_embeddings" / f"{self.id}.npz"
        )

    def _embeddings(self, path: Path):
        import numpy as np

        if not path.exists():
            return None, "no embeddings"
        with np.load(path) as z:
            if "fp" not in z or str(z["fp"]) != V.segments_fingerprint(self.segments):
                return None, "embeddings of another segmentation"
            return (z["emb"], z["seg"], z["samples"]), None

    def maps(self) -> tuple[dict | None, dict | None]:
        """(the LLM names the archive's voiceprint pass started from, the final
        map), each None if missing or made for another labelling."""
        fit = labels_fingerprint(self.segments)
        current = _json(self.run / "speaker_map.json")
        if current is not None and current.get("labels_fp") != fit:
            current = None
        llm = current if current is not None and "voiceprint" not in current else None
        if llm is None:
            llm = _json(self.run / "speaker_map.llm.json")
            if llm is not None and llm.get("labels_fp") != fit:
                llm = None
        return llm, current

    def labels(self) -> dict | None:
        if self.embeddings is None:
            return None
        return V.label_voices(self.segments, *self.embeddings)

    def marker(self, references: dict) -> tuple[dict | None, str]:
        """voiceprint.json for this episode and where it came from."""
        if self.embeddings is not None:
            return V.report(self.segments, self.digest, *self.embeddings, references), "embeddings"
        _, current = self.maps()
        if current is None or "voiceprint" not in current:
            return None, "none"
        speakers = {}
        for label, rec in sorted(current["speakers"].items()):
            base = {"share": rec.get("share"), "seconds": rec.get("voice_seconds")}
            if rec.get("voice") in (None, "insufficient") or "voice_best" not in rec:
                speakers[label] = {"voice": "insufficient", **base}
                continue
            base.update(
                best=rec["voice_best"],
                score=rec["voice_score"],
                margin=rec["voice_margin"],
                purity=rec.get("voice_purity"),
            )
            if rec["voice"] == "match":
                confidence = "high" if rec["voice_score"] >= V.THRESHOLD + 0.1 else "medium"
                speakers[label] = {
                    "voice": "match",
                    "name": rec["voice_best"],
                    "confidence": confidence,
                    **base,
                }
            else:
                speakers[label] = {"voice": "no-match", **base}
        marker = {
            "voiceprinted_at": now_iso(),
            "references": sorted(current["voiceprint"].get("templates", [])),
            "threshold": current["voiceprint"].get("threshold", V.THRESHOLD),
            "margin": current["voiceprint"].get("margin", V.MARGIN),
            "transcript_sha256": self.digest,
            "speakers": speakers,
            "from": "speaker_map.json",
        }
        return marker, "speaker_map"


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def archived(archive: Path) -> tuple[list[Episode], list[tuple[str, str]], sqlite3.Row | None]:
    db = read_only(archive / "media.db", quiet=True)
    episodes, skipped = [], []
    for row in db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at"):
        if FEED not in (row["feed_url"] or "") and row["id"] not in EXTRA_IDS:
            continue
        if not (archive / "items" / row["id"] / "runs/diarized/transcript.json").exists():
            skipped.append((row["id"], "no diarized transcript"))
            continue
        episodes.append(Episode(archive, row))
    feed = db.execute("SELECT * FROM feeds WHERE url LIKE ?", (f"%{FEED}%",)).fetchone()
    db.close()
    return episodes, skipped, feed


def references(episodes: list[Episode], path: Path) -> tuple[dict, dict | None]:
    """The library's All-In reference voices: the existing file, else built from
    the archive (seed statistics returned only then)."""
    if path.exists():
        return V.load_references(path), None
    seeds = []
    for episode in episodes:
        labels = episode.labels()
        llm, _ = episode.maps()
        if labels is None or llm is None:
            continue
        seeds.append(
            {
                "id": episode.id,
                "labels": {
                    label: {**info, "llm": llm["speakers"].get(label, {})}
                    for label, info in labels.items()
                },
            }
        )
    built, info = V.build_references(seeds)
    if len(built) < 2:
        raise SystemExit(f"only {len(built)} reference voices could be built; nothing migrated")
    return built, info


def plan(episodes: list[Episode], library: Path, *, quiet: bool) -> dict[str, list[Episode]]:
    """migrate / same (already in the library) / conflict."""
    out: dict[str, list[Episode]] = {"migrate": [], "same": [], "conflict": []}
    path = library / "media.db"
    db = read_only(path, quiet=quiet) if path.exists() else None
    for episode in episodes:
        row = (
            db.execute("SELECT status FROM items WHERE id = ?", (episode.id,)).fetchone()
            if db
            else None
        )
        if row is None:
            taken = (
                db
                and db.execute(
                    "SELECT id FROM items WHERE source = ?", (episode.row["source"],)
                ).fetchone()
            )
            out["conflict" if taken else "migrate"].append(episode)
            continue
        transcript = library / "items" / episode.id / "transcript.json"
        same = (
            row["status"] == "done"
            and transcript.exists()
            and (transcript.read_bytes() == episode.transcript_bytes)
        )
        out["same" if same else "conflict"].append(episode)
    if db:
        db.close()
    return out


def copy(source: Path, target: Path) -> None:
    tmp = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, tmp)
    tmp.replace(target)


class Conflict(RuntimeError):
    """The library got the episode (or its source, or another transcript there)."""


def migrate_one(episode: Episode, store: Store, library: Path, marker: dict | None) -> int:
    """Migrate one episode; returns the number of segments indexed.

    Under the item lock and an immediate transaction (no other database writer
    until it ends), first checks again that the library has neither the id nor
    the source and no other transcript in ``items/<id>/`` (raising Conflict
    before writing anything), then writes the files, voiceprint.json and the
    rows, and commits: an interrupted episode is not in the library and is
    migrated again."""
    target = library / "items" / episode.id
    target.mkdir(parents=True, exist_ok=True)
    row = episode.row
    segments = [s for s in episode.segments if str(s.get("text", "")).strip()]
    with item_lock(target):
        conn = store.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            taken = conn.execute(
                "SELECT id FROM items WHERE id = ? OR source = ?", (episode.id, row["source"])
            ).fetchone()
            if taken:
                raise Conflict(f"the library has {taken[0]} with this id or source")
            transcript = target / "transcript.json"
            if transcript.exists() and transcript.read_bytes() != episode.transcript_bytes:
                raise Conflict("items/<id>/transcript.json is another transcript")
            for name in RUN_FILES:
                if (episode.run / name).exists():
                    copy(episode.run / name, target / name)
            if (episode.item_dir / "meta.json").exists():
                copy(episode.item_dir / "meta.json", target / "meta.json")
            if marker is not None:
                V.write(target, marker)
            conn.execute(
                """
                INSERT INTO items
                    (id, source, source_type, title, duration, language, status, audio_path,
                     transcript_path, summary_path, model, error, feed_url, meta_json,
                     published_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'done', NULL, ?, NULL, ?, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source"],
                    row["source_type"],
                    row["title"],
                    episode.data.get("duration") or row["duration"],
                    episode.data.get("language") or row["language"],
                    str(target / "transcript.md"),
                    episode.data.get("model") or row["model"],
                    row["feed_url"],
                    row["meta_json"],
                    row["published_at"],
                    row["created_at"],
                    now_iso(),
                ),
            )
            conn.executemany(
                "INSERT INTO segments (item_id, idx, start, end, text, speaker) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (row["id"], i, float(s["start"]), float(s["end"]), s["text"], s.get("speaker"))
                    for i, s in enumerate(segments)
                ],
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    return len(segments)


def finish_one(episode: Episode, store: Store, library: Path, marker: dict) -> bool:
    """Add the missing voiceprint.json of an episode migrated before, under its
    lock, if the library still has it done with the archive's transcript.
    Raises Conflict otherwise; False if the marker is there by now."""
    target = library / "items" / episode.id
    with item_lock(target):
        row = store.conn.execute("SELECT status FROM items WHERE id = ?", (episode.id,)).fetchone()
        transcript = target / "transcript.json"
        if (
            row is None
            or row["status"] != "done"
            or not transcript.exists()
            or (transcript.read_bytes() != episode.transcript_bytes)
        ):
            raise Conflict("the library's episode changed since the survey")
        if (target / V.MARKER).exists():
            return False
        V.write(target, marker)
    return True


def durations(segments: list[dict]) -> dict[str, float]:
    """Seconds of speech per speaker label (None: unlabelled), as in label_voices."""
    out: dict = collections.defaultdict(float)
    for s in segments:
        out[s.get("speaker")] += float(s["end"]) - float(s["start"])
    return out


def coverage(judged: list[tuple[list[dict], dict]]) -> dict:
    """How much the voiceprints name, over (segments, voiceprint.json) pairs:
    labels by decision; the share of all speaking time in labels named by voice;
    and the mean of that share per episode."""
    labels = collections.Counter()
    named = total = 0.0
    shares = []
    for segments, marker in judged:
        seconds = durations(segments)
        whole = sum(seconds.values())
        by_voice = 0.0
        for label, rec in marker["speakers"].items():
            labels[rec["voice"]] += 1
            by_voice += seconds.get(label, 0.0) if rec["voice"] == "match" else 0.0
        named, total = named + by_voice, total + whole
        if whole:
            shares.append(by_voice / whole)
    return {
        "labels": dict(labels),
        "speaking_time_named_by_voice": round(named / total, 4) if total else None,
        "mean_per_episode": round(sum(shares) / len(shares), 4) if shares else None,
    }


def agreement(episodes: list[Episode], markers: dict[str, dict]) -> dict:
    """New voice matches against the archive's final speaker_map.json."""
    out = collections.Counter()
    differ = []
    for episode in episodes:
        marker = markers.get(episode.id)
        _, current = episode.maps()
        if marker is None or current is None or "voiceprint" not in current:
            continue
        for label, rec in marker["speakers"].items():
            old = current["speakers"].get(label, {})
            new_name = rec.get("name") if rec["voice"] == "match" else None
            old_name = old.get("voice_best") if old.get("voice") == "match" else None
            out["same" if new_name == old_name else "different"] += 1
            if new_name != old_name:
                differ.append((episode.id, label, old_name, new_name))
    return {"labels": dict(out), "different": differ[:20]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", type=Path, required=True, help="the All-In data directory")
    parser.add_argument("--library", type=Path, required=True, help="the podcast library directory")
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args(argv)
    archive, library = args.archive.expanduser().resolve(), args.library.expanduser().resolve()
    if archive == library or archive in library.parents or library in archive.parents:
        print(f"error: the archive and the library overlap: {archive}, {library}", file=sys.stderr)
        return 2
    if not args.apply:
        summary, _ = survey(archive, library, quiet=True)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 1 if summary["conflicts"] else 0
    lock = library / ".library-update.lock"
    with file_lock(lock, wait=False) as acquired:
        if not acquired:
            print(f"error: a library-update holds {lock}; try again later", file=sys.stderr)
            return 1
        summary, work = survey(archive, library, quiet=False)
        summary["mode"] = "apply"
        summary["applied"] = apply(library, archive, work, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 1 if summary["conflicts"] or summary.get("failed") else 0


def survey(archive: Path, library: Path, *, quiet: bool) -> tuple[dict, dict]:
    """What a migration would do, and the material to do it. ``quiet`` (a dry
    run) reads the library database without leaving -wal/-shm files."""
    episodes, skipped, feed = archived(archive)
    refs_path = library / REFERENCES
    refs, built = references(episodes, refs_path)
    todo = plan(episodes, library, quiet=quiet)
    markers, sources = {}, collections.Counter()
    for episode in todo["migrate"] + todo["same"]:
        marker, source = episode.marker(refs)
        sources[source] += 1
        if marker is not None:
            markers[episode.id] = marker
    summary = {
        "mode": "dry run",
        "archive_episodes": len(episodes),
        "skipped": skipped,
        "migrate": len(todo["migrate"]),
        "already_in_library": len(todo["same"]),
        "conflicts": [e.id for e in todo["conflict"]],
        "segments": sum(
            sum(1 for s in e.segments if str(s.get("text", "")).strip()) for e in todo["migrate"]
        ),
        "voiceprint_from": dict(sources),
        "references": sorted(refs),
        "references_file": f"{refs_path} ({'exists' if built is None else 'built from the archive'})",
        "reference_seeds": {n: list(v) for n, v in built["seed_counts"].items()} if built else None,
        "voiceprint_coverage": coverage(
            [(e.segments, markers[e.id]) for e in episodes if e.id in markers]
        ),
        "agreement_with_archive": agreement(episodes, markers),
        "feed": feed["url"] if feed else None,
    }
    try:
        show = next(
            (s for s in load_shows(library / "library.toml") if feed and s.matches(feed["url"])),
            None,
        )
    except LibraryError as exc:
        show, summary["library_toml"] = None, str(exc)
    if show is None or not show.voiceprints:
        summary["library_toml"] = "WARN: add the All-In show with voiceprints (see docs/examples)"
    elif V.references_path(show, library) != refs_path:
        summary["library_toml"] = (
            f"WARN: its voiceprints point at {show.voiceprints}, not {REFERENCES}"
        )
    work = {"todo": todo, "markers": markers, "refs": refs, "built": built, "feed": feed}
    return summary, work


def apply(library: Path, archive: Path, work: dict, summary: dict) -> dict:
    """Write the migration; per-episode failures go to summary["failed"]."""
    todo, markers, feed = work["todo"], work["markers"], work["feed"]
    if work["built"] is not None:
        V.save_references(
            library / REFERENCES,
            work["refs"],
            {
                "built_at": now_iso(),
                "from": str(archive),
                "seed_counts": work["built"]["seed_counts"],
            },
        )
    applied = collections.Counter()
    store = Store(library / "media.db")
    try:
        if feed is not None:
            with store.conn:
                store.conn.execute(
                    "INSERT OR IGNORE INTO feeds (url, title, latest, last_checked, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        feed["url"],
                        feed["title"],
                        feed["latest"],
                        feed["last_checked"],
                        feed["created_at"],
                    ),
                )
        for episode in todo["migrate"]:
            try:
                applied["segments"] += migrate_one(episode, store, library, markers.get(episode.id))
                applied["episodes"] += 1
            except Conflict:
                summary["conflicts"].append(episode.id)
            except Exception as exc:  # noqa: BLE001 - this episode only; re-run
                summary.setdefault("failed", []).append(f"{episode.id}: {exc}")
        for episode in todo["same"]:  # finish an interrupted earlier run
            target = library / "items" / episode.id
            if episode.id in markers and not (target / V.MARKER).exists():
                try:
                    if finish_one(episode, store, library, markers[episode.id]):
                        applied["voiceprints added"] += 1
                except Conflict:
                    summary["conflicts"].append(episode.id)
    finally:
        store.close()
    return dict(applied)


if __name__ == "__main__":
    raise SystemExit(main())
