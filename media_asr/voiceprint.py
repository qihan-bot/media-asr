"""Name speakers by voice (``media-asr voiceprint``) in shows with reference voices.

A show opts in with ``voiceprints = "<file>"`` in ``library.toml`` (relative to
the data directory): an ``.npz`` with ``names`` and unit ``vectors``, one
256-dimensional WeSpeaker voice per recurring speaker (``build_references``).
For every done episode of such a show without a current ``voiceprint.json``:

1. The GPU host downloads the audio and embeds each diarized segment with the
   speaker-embedding model of pyannote 3.1 (edges trimmed by 0.2 s, at most the
   middle 20 s, at least 1 s), under the host's GPU lock.
2. Each speaker label gets the mean of its segments' embeddings, weighted by the
   seconds embedded, and is scored (cosine) against every reference voice.
3. A label with at least 15 s of embedded speech whose best score is at least
   0.65 and beats the runner-up by 0.10 (0.75 and 0.20 under 30 s) is that
   person: ``match``. Otherwise ``no-match`` (best guess kept) or
   ``insufficient``.

``voiceprint.json``, the completion marker, records each label's decision,
best reference, score, margin, seconds, speaking-time share and mean embedding,
the reference names, and the SHA-256 of the transcript.json it judged; the step
redoes an episode whose transcript changed. ``enrich`` takes matched names as
given, asks the model only about the other labels, and rejects a model's name
that is a reference voice the label does not sound like. ``run`` removes the
marker before writing a new transcript; writing it removes ``enrich.json`` and
``terms.json``, so the episode is named and corrected again.

The rules and numbers are those of the All-In ``voiceprint_relabel.py`` this
replaces; the references are what that script built from the archive
(``scripts/migrate_allin.py``).
"""

from __future__ import annotations

import collections
import hashlib
import json
import sqlite3
from pathlib import Path

from .config import DATA_DIR, ITEMS_DIR
from .library import Show, file_digest, read_json
from .util import atomic_write, item_lock, now_iso

MARKER = "voiceprint.json"
THRESHOLD = 0.65
MARGIN = 0.10
MIN_LABEL_SECONDS = 15.0  # embedded speech needed to judge a label
SHORT_LABEL_SECONDS = 30.0  # below this, threshold and margin are 0.10 stricter
SEED_MIN_SECONDS = 120.0  # a reference voice is built from labels this long
SEED_MIN_EPISODES = 4  # in at least this many episodes
PRUNE = 0.45  # a seed must sound at least this much like its own name


def unit(vectors):
    import numpy as np

    return vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-9)


def segments_fingerprint(segments: list[dict]) -> str:
    """Hash of the segment boundaries; embeddings fit only this segmentation."""
    bounds = [[round(float(s["start"]), 3), round(float(s["end"]), 3)] for s in segments]
    return hashlib.sha256(json.dumps(bounds).encode()).hexdigest()[:16]


def references_path(show: Show, data_dir: Path = DATA_DIR) -> Path:
    path = Path(show.voiceprints).expanduser()
    return path if path.is_absolute() else data_dir / path


def load_references(path: Path) -> dict:
    """Name -> unit vector."""
    import numpy as np

    with np.load(path) as z:
        names = [str(name) for name in z["names"]]
        vectors = unit(z["vectors"])
    if len(names) < 2 or len(set(names)) != len(names) or len(vectors) != len(names):
        raise ValueError(f"{path}: needs two or more distinct names with one vector each")
    return dict(zip(names, vectors))


def save_references(path: Path, references: dict, info: dict) -> None:
    import numpy as np

    names = sorted(references)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp,
        names=np.array(names),
        vectors=np.stack([references[name] for name in names]).astype(np.float32),
        info=np.array(json.dumps(info, ensure_ascii=False)),
    )
    tmp.replace(path)


def label_voices(segments: list[dict], emb, seg, samples) -> dict[str, dict]:
    """Per speaker label: share of speaking time, seconds embedded, mean
    embedding (weighted by the seconds embedded per segment) and purity (mean
    similarity of its segments to that mean). ``seg[j]`` is the index in
    ``segments`` of embedding ``emb[j]``, which covers ``samples[j]`` samples."""
    import numpy as np

    emb, samples = unit(np.asarray(emb)), np.asarray(samples)
    total = sum(float(s["end"]) - float(s["start"]) for s in segments) or 1.0
    labels = {}
    for label in sorted({s.get("speaker") for s in segments if s.get("speaker")}):
        rows = [j for j, k in enumerate(seg) if segments[k].get("speaker") == label]
        airtime = sum(
            float(s["end"]) - float(s["start"]) for s in segments if s.get("speaker") == label
        )
        info = {"share": airtime / total, "seconds": 0.0, "centroid": None, "purity": None}
        if rows:
            weights = samples[rows] / 16000.0
            vectors = emb[rows]
            centroid = unit((vectors * weights[:, None]).sum(0))
            info.update(
                seconds=float(weights.sum()),
                centroid=centroid,
                purity=float(np.average(vectors @ centroid, weights=weights)),
            )
        labels[label] = info
    return labels


def judge(
    labels: dict[str, dict], references: dict, threshold: float = THRESHOLD, margin: float = MARGIN
) -> dict[str, dict]:
    """Each label's decision: ``match`` with a name and confidence, ``no-match``
    or ``insufficient``."""
    out = {}
    for label, info in sorted(labels.items()):
        record = {"share": round(info["share"], 4), "seconds": round(info["seconds"], 1)}
        if info["centroid"] is None or info["seconds"] < MIN_LABEL_SECONDS:
            out[label] = {"voice": "insufficient", **record}
            continue
        scores = sorted(
            ((float(info["centroid"] @ vector), name) for name, vector in references.items()),
            reverse=True,
        )
        (best, best_name), (second, _) = scores[0], scores[1]
        record.update(
            best=best_name,
            score=round(best, 3),
            margin=round(best - second, 3),
            purity=round(info["purity"], 3),
            centroid=[round(float(x), 5) for x in info["centroid"]],
        )
        short = info["seconds"] < SHORT_LABEL_SECONDS
        need_score, need_margin = (
            (threshold + 0.10, margin + 0.10) if short else (threshold, margin)
        )
        if best >= need_score and best - second >= need_margin:
            confidence = "high" if best >= threshold + 0.1 else "medium"
            out[label] = {"voice": "match", "name": best_name, "confidence": confidence, **record}
        else:
            out[label] = {"voice": "no-match", **record}
    return out


def build_references(episodes: list[dict], prune: float = PRUNE) -> tuple[dict, dict]:
    """Reference voices from labels a model named with high confidence.

    ``episodes``: ``[{"id", "labels": {label: {"llm": {"name", "confidence"},
    "seconds", "centroid"}}}]``. A name needs ``SEED_MIN_SECONDS`` of speech in
    labels of at least ``SEED_MIN_EPISODES`` episodes; in three rounds, a seed
    that sounds less like the rest of its name's seeds than ``prune``, or more
    like another name, is dropped. Returns name -> unit vector and
    ``{"seed_counts": name -> (seeds, kept), "seeds": name -> [(episode, label)]}``.
    """
    import numpy as np

    seeds = collections.defaultdict(list)
    for episode in episodes:
        for label, info in episode["labels"].items():
            llm = info["llm"]
            if (
                llm.get("confidence") == "high"
                and info["centroid"] is not None
                and info["seconds"] >= SEED_MIN_SECONDS
                and llm.get("name")
                and not llm["name"].startswith(("Guest", "SPEAKER", "Unidentified"))
            ):
                seeds[llm["name"]].append((episode["id"], label, info["centroid"]))
    seeds = {n: v for n, v in seeds.items() if len({e for e, _, _ in v}) >= SEED_MIN_EPISODES}
    stats = {}
    for _ in range(3):
        templates = {n: unit(np.mean([c for _, _, c in v], 0)) for n, v in seeds.items()}
        kept = {}
        for name, items in seeds.items():
            good = []
            for episode_id, label, centroid in items:
                others = [x for x in items if x[:2] != (episode_id, label)]
                own = centroid @ unit(np.mean([x[2] for x in others], 0)) if others else 1.0
                rival = max((centroid @ t for n, t in templates.items() if n != name), default=-1.0)
                if own >= prune and own > rival:
                    good.append((episode_id, label, centroid))
            kept[name] = good
            stats[name] = (len(items), len(good))
        seeds = {n: v for n, v in kept.items() if len({e for e, _, _ in v}) >= SEED_MIN_EPISODES}
    templates = {n: unit(np.mean([c for _, _, c in v], 0)) for n, v in seeds.items()}
    return templates, {
        "seed_counts": stats,
        "seeds": {n: [(e, label) for e, label, _ in v] for n, v in seeds.items()},
    }


def report(segments: list[dict], digest: str, emb, seg, samples, references: dict) -> dict:
    """The voiceprint.json of a transcript (digest) from its segment embeddings."""
    return {
        "voiceprinted_at": now_iso(),
        "references": sorted(references),
        "threshold": THRESHOLD,
        "margin": MARGIN,
        "transcript_sha256": digest,
        "speakers": judge(label_voices(segments, emb, seg, samples), references),
    }


def current(item_dir: Path, digest: str | None) -> dict | None:
    """The episode's voiceprint.json if it judged the transcript with this digest."""
    try:
        data = json.loads((item_dir / MARKER).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return data if digest and data.get("transcript_sha256") == digest else None


def voices(item_dir: Path, digest: str) -> tuple[dict | None, str | None]:
    """voiceprint.json for ``enrich`` and the SHA-256 of the bytes read, which
    ``enrich`` checks again before it commits; (None, None) without one. Raises
    if it judged another transcript."""
    try:
        raw = (item_dir / MARKER).read_bytes()
    except FileNotFoundError:
        return None, None
    try:
        data = json.loads(raw)
    except ValueError:
        data = {}
    if not isinstance(data, dict) or data.get("transcript_sha256") != digest:
        raise RuntimeError("voiceprint.json is for another transcript; run media-asr voiceprint")
    return data, hashlib.sha256(raw).hexdigest()


def write(item_dir: Path, data: dict) -> None:
    """Write voiceprint.json, after removing the markers of the steps that use
    it (the names may change)."""
    for marker in ("enrich.json", "terms.json"):
        (item_dir / marker).unlink(missing_ok=True)
    atomic_write(item_dir / MARKER, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def voiceprint_item(row, references: dict, embed, items_dir: Path) -> dict | None:
    """The episode's voiceprint.json contents, or None if it has no speaker labels."""
    data, digest = read_json(items_dir / row["id"] / "transcript.json")
    segments = data.get("segments", [])
    if not any(s.get("speaker") for s in segments):
        return None
    fingerprint = segments_fingerprint(segments)
    emb, seg, samples, got = embed(row, segments, fingerprint)
    if got != fingerprint:
        raise RuntimeError("the embeddings are for another segmentation")
    return report(segments, digest, emb, seg, samples, references)


def commit(row, data: dict, items_dir: Path, *, force: bool = False) -> bool:
    """Write the episode's voiceprint.json under its lock. False if another run
    voiceprinted this transcript meanwhile (unless ``force``); raises if the
    transcript changed since it was read."""
    item_dir = items_dir / row["id"]
    with item_lock(item_dir):
        digest = file_digest(item_dir / "transcript.json")
        if not force and current(item_dir, digest):
            return False
        if digest != data["transcript_sha256"]:
            raise RuntimeError(
                "transcript.json changed while voiceprinting; the next run redoes it"
            )
        write(item_dir, data)
    return True


def run(
    db: sqlite3.Connection,
    shows: list[Show],
    *,
    embed,
    ids: list[str] | None = None,
    force: bool = False,
    items_dir: Path = ITEMS_DIR,
    data_dir: Path = DATA_DIR,
) -> int:
    """Voiceprint the done episodes of the shows with ``voiceprints``; return 1
    if any failed. ``embed(row, segments, fingerprint)`` returns the arrays
    ``(emb, seg, samples, fingerprint)`` for the episode's segments
    (``media-asr voiceprint`` runs it on the GPU host). A failure affects only
    its episode, which is tried again on the next run."""
    voiced = [show for show in shows if show.voiceprints]
    todo = []
    for row in db.execute("SELECT * FROM items WHERE status = 'done' ORDER BY published_at"):
        show = next((s for s in voiced if s.matches(row["feed_url"])), None)
        transcript = items_dir / row["id"] / "transcript.json"
        if show is None or (ids is not None and row["id"] not in ids) or not transcript.exists():
            continue
        if force or not current(items_dir / row["id"], file_digest(transcript)):
            todo.append((row, show))
    print(f"items: {len(todo)}", flush=True)
    references: dict[str, dict] = {}
    failed = 0
    for index, (row, show) in enumerate(todo, start=1):
        tag = f"[{index}/{len(todo)}]"
        try:
            if show.voiceprints not in references:
                references[show.voiceprints] = load_references(references_path(show, data_dir))
            data = voiceprint_item(row, references[show.voiceprints], embed, items_dir)
            if data is None:
                print(f"{tag} skip {row['id']}: no speaker labels", flush=True)
                continue
            if not commit(row, data, items_dir, force=force):
                print(f"{tag} skip {row['id']}: voiceprinted by another run", flush=True)
                continue
        except Exception as exc:  # noqa: BLE001 - this episode only; retried next run
            failed += 1
            print(f"{tag} FAIL {row['id']} {row['title'][:40]}: {exc}", flush=True)
            continue
        matched = [
            f"{label}={rec['name']}"
            for label, rec in data["speakers"].items()
            if rec["voice"] == "match"
        ]
        print(
            f"{tag} ok {row['title'][:40]} | {len(matched)}/{len(data['speakers'])} by voice: "
            + (", ".join(matched) or "-"),
            flush=True,
        )
    print(f"done={len(todo) - failed} failed={failed}", flush=True)
    return 1 if failed else 0
