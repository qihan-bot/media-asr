#!/usr/bin/env python3
"""Re-check LLM speaker names against cross-episode voiceprints.

The LLM mapping (``map_speakers.py`` + ``repair_speakers.py``) infers names
from who-addresses-whom in the text, which breaks down when four or five
hosts talk over each other. This script builds a voice template for each
recurring speaker from labels the LLM identified with high confidence,
prunes seeds that disagree with the consensus voice, and then scores every
diarized label in every episode against those templates.

Decision per label (only labels with enough embedded speech are judged):
  * voice matches a recurring speaker (score >= T and margin >= M)
      -> that speaker's name, method "voice"
  * voice matches nobody, but the LLM named a recurring speaker
      -> the LLM name is rejected; label becomes "Unidentified N"
  * otherwise -> the LLM name is kept, method "llm"

``--apply`` re-renders ``transcript_named.*`` and writes ``speaker_map.json``
with a ``voiceprint`` block. The LLM result it started from is kept as
``speaker_map.llm.json``; a map without a ``voiceprint`` block (e.g. after
re-running the LLM mapping) is treated as a new baseline. Episodes whose
embeddings do not match the current segmentation are skipped.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repair_speakers import render_named  # noqa: E402
from speaker_common import is_allin, labels_fingerprint, segments_fingerprint  # noqa: E402

from media_asr.config import DATA_DIR, ITEMS_DIR  # noqa: E402
from media_asr.util import atomic_write, now_iso  # noqa: E402

EMB_DIR = DATA_DIR / "speaker_embeddings"
MIN_LABEL_SECONDS = 15.0  # embedded speech needed to judge a label
SHORT_LABEL_SECONDS = 30.0  # below this, use a stricter threshold/margin
SEED_MIN_SECONDS = 120.0
SEED_MIN_EPISODES = 4


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def load_episode(item_dir: Path) -> dict | None:
    run = item_dir / "runs" / "diarized"
    npz = EMB_DIR / f"{item_dir.name}.npz"
    current_path = run / "speaker_map.json"
    llm_path = run / "speaker_map.llm.json"
    if not is_allin(item_dir) or not npz.exists() or not current_path.exists():
        return None
    current = json.loads(current_path.read_text(encoding="utf-8"))
    # A map without a "voiceprint" block is a fresh LLM result (first run, or the
    # LLM mapping was redone) and becomes the new baseline; otherwise the saved
    # LLM baseline is used so re-runs are idempotent.
    fresh_llm = "voiceprint" not in current
    if not fresh_llm and not llm_path.exists():
        print(f"WARN {item_dir.name}: voiceprint map without LLM baseline; skipped", file=sys.stderr)
        return None
    mapping = current if fresh_llm else json.loads(llm_path.read_text(encoding="utf-8"))
    data = json.loads((run / "transcript.json").read_text(encoding="utf-8"))
    segs = data["segments"]
    if mapping.get("labels_fp") != labels_fingerprint(segs):
        print(f"WARN {item_dir.name}: LLM speaker map does not match current diarization; "
              "re-run map_speakers.py --force", file=sys.stderr)
        return None
    with np.load(npz) as z:
        if "fp" not in z or str(z["fp"]) != segments_fingerprint(segs):
            print(f"WARN {item_dir.name}: embeddings do not match current segmentation; re-run embed_speakers.py",
                  file=sys.stderr)
            return None
        emb, idx, samples = unit(z["emb"]), z["seg"], z["samples"]
    total = sum(s["end"] - s["start"] for s in segs) or 1.0
    labels = {}
    for label in sorted({s.get("speaker") for s in segs if s.get("speaker")}):
        rows = [j for j, k in enumerate(idx) if segs[k].get("speaker") == label]
        airtime = sum(s["end"] - s["start"] for s in segs if s.get("speaker") == label)
        info = {
            "llm": mapping["speakers"].get(label, {}),
            "share": airtime / total,
            "seconds": 0.0,
            "centroid": None,
            "purity": None,
        }
        if rows:
            weights = samples[rows] / 16000.0  # seconds actually embedded (after edge trim / cap)
            vecs = emb[rows]
            centroid = unit((vecs * weights[:, None]).sum(0))
            info.update(
                seconds=float(weights.sum()),
                centroid=centroid,
                purity=float(np.average(vecs @ centroid, weights=weights)),
            )
        labels[label] = info
    return {"id": item_dir.name, "dir": run, "title": data.get("title") or "", "data": data, "fresh_llm": fresh_llm,
            "mapping": mapping, "map_path": run / "speaker_map.json", "labels": labels}


def build_templates(episodes: list[dict], prune: float) -> tuple[dict, dict]:
    seeds = collections.defaultdict(list)
    for ep in episodes:
        for label, info in ep["labels"].items():
            llm = info["llm"]
            if (
                llm.get("confidence") == "high"
                and info["centroid"] is not None
                and info["seconds"] >= SEED_MIN_SECONDS
                and llm.get("name")
                and not llm["name"].startswith(("Guest", "SPEAKER", "Unidentified"))
            ):
                seeds[llm["name"]].append((ep["id"], label, info["centroid"]))
    seeds = {n: v for n, v in seeds.items() if len({e for e, _, _ in v}) >= SEED_MIN_EPISODES}
    stats = {}
    for _ in range(3):  # prune seeds that disagree with their own template
        templates = {n: unit(np.mean([c for _, _, c in v], 0)) for n, v in seeds.items()}
        kept = {}
        for name, items in seeds.items():
            good = []
            for ep_id, label, c in items:
                others = [x for x in items if x[:2] != (ep_id, label)]
                own = c @ unit(np.mean([x[2] for x in others], 0)) if others else 1.0
                rival = max((c @ t for n, t in templates.items() if n != name), default=-1.0)
                if own >= prune and own > rival:
                    good.append((ep_id, label, c))
            kept[name] = good
            stats[name] = (len(items), len(good))
        seeds = {n: v for n, v in kept.items() if len({e for e, _, _ in v}) >= SEED_MIN_EPISODES}
    templates = {n: unit(np.mean([c for _, _, c in v], 0)) for n, v in seeds.items()}
    return templates, {"seed_counts": stats, "seeds": {n: [(e, lab) for e, lab, _ in v] for n, v in seeds.items()}}


def decide(ep: dict, templates: dict, threshold: float, margin: float) -> dict:
    roster = set(templates)
    out = {}
    unidentified = 0
    for label, info in sorted(ep["labels"].items()):
        llm = info["llm"]
        llm_name = llm.get("name") or label
        rec = {"name": llm_name, "confidence": llm.get("confidence", "low"),
               "evidence": llm.get("evidence", ""), "llm_name": llm_name,
               "method": "llm", "share": round(info["share"], 4),
               "voice_seconds": round(info["seconds"], 1)}
        if info["centroid"] is None or info["seconds"] < MIN_LABEL_SECONDS:
            rec["voice"] = "insufficient"
            out[label] = rec
            continue
        scores = sorted(((float(info["centroid"] @ t), n) for n, t in templates.items()), reverse=True)
        (best, best_name), (second, _) = scores[0], scores[1]
        rec.update(voice_best=best_name, voice_score=round(best, 3), voice_margin=round(best - second, 3),
                   voice_purity=round(info["purity"], 3))
        short = info["seconds"] < SHORT_LABEL_SECONDS
        need_t, need_m = (threshold + 0.10, margin + 0.10) if short else (threshold, margin)
        if best >= need_t and best - second >= need_m:
            rec["voice"] = "match"
            if best_name != llm_name:
                rec.update(name=best_name, method="voice", confidence="high" if best >= threshold + 0.1 else "medium",
                           evidence=f"voiceprint {best:.2f} vs template (margin {best - second:.2f}); LLM said {llm_name}")
        elif llm_name in roster and best_name == llm_name:
            rec["voice"] = "weak"  # e.g. live-event acoustics; voice leans the same way
            rec["confidence"] = "low"
        elif llm_name in roster:
            unidentified += 1
            rec["voice"] = "reject"
            rec.update(name=f"Unidentified {unidentified}", method="voice", confidence="low",
                       evidence=f"voice does not match {llm_name} (best {best_name} {best:.2f}); LLM name rejected")
        else:
            rec["voice"] = "no-match"
        out[label] = rec
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--prune", type=float, default=0.45)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", default=None,
                        help="write the change list as JSON here (default with --apply: "
                             "data/allin-2026-voiceprint-report.json; dry runs write nothing)")
    args = parser.parse_args()

    episodes = [ep for d in sorted(ITEMS_DIR.iterdir()) if (ep := load_episode(d))]
    print(f"episodes with embeddings: {len(episodes)}")
    templates, info = build_templates(episodes, args.prune)
    print("templates (seeds before -> after pruning):")
    for name, (before, after) in sorted(info["seed_counts"].items(), key=lambda x: -x[1][0]):
        print(f"  {name:<24} {before:>3} -> {after:>3}{'' if name in templates else '  (dropped)'}")
    if len(templates) < 2:
        print("need at least two voice templates to score labels; leaving maps unchanged", file=sys.stderr)
        return 1

    # Calibration: leave-one-out own score vs best other-template score for seed labels.
    own, rival = [], []
    for name, seeds in info["seeds"].items():
        for ep_id, label in seeds:
            ep = next(e for e in episodes if e["id"] == ep_id)
            c = ep["labels"][label]["centroid"]
            others = [x for x in seeds if x != (ep_id, label)]
            if not others:
                continue
            loo = unit(np.mean([next(e for e in episodes if e["id"] == x[0])["labels"][x[1]]["centroid"] for x in others], 0))
            own.append(float(c @ loo))
            rival.append(max(float(c @ t) for n, t in templates.items() if n != name))
    if own:
        q = lambda a, p: float(np.percentile(a, p))  # noqa: E731
        print(f"calibration own  p1={q(own,1):.2f} p5={q(own,5):.2f} p50={q(own,50):.2f}")
        print(f"calibration rival p50={q(rival,50):.2f} p95={q(rival,95):.2f} p99={q(rival,99):.2f} max={max(rival):.2f}")

    report = []
    changes = collections.Counter()
    for ep in episodes:
        decided = decide(ep, templates, args.threshold, args.margin)
        for label, rec in decided.items():
            if rec["name"] != rec["llm_name"]:
                changes[rec["voice"]] += 1
                report.append({"id": ep["id"], "title": ep["title"][:70], "label": label, **rec})
        if args.apply:
            # Order matters for resumability: baseline first, named transcripts next,
            # and the voiceprint map (the completion marker) last.
            if ep["fresh_llm"]:
                atomic_write(ep["dir"] / "speaker_map.llm.json",
                             json.dumps(ep["mapping"], ensure_ascii=False, indent=2) + "\n")
            render_named(ep["dir"], ep["data"], decided)
            payload = dict(ep["mapping"])
            payload["speakers"] = decided
            payload["voiceprint"] = {"applied_at": now_iso(), "threshold": args.threshold,
                                     "margin": args.margin, "templates": sorted(templates)}
            atomic_write(ep["map_path"], json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    report_path = args.report or (str(DATA_DIR / "allin-2026-voiceprint-report.json") if args.apply else None)
    if report_path:
        atomic_write(Path(report_path), json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"label changes: {dict(changes)}" + (f" (report: {report_path})" if report_path else ""))
    for r in sorted(report, key=lambda r: -r["share"])[:60]:
        print(f"  {r['share']*100:5.1f}% {r['title'][:38]:<38} {r['label']} {r['llm_name'][:20]:<20} -> {r['name']:<22} "
              f"{r.get('voice_best','')[:14]} {r.get('voice_score','')} m={r.get('voice_margin','')}")
    print("applied" if args.apply else "dry run (use --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
