from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import DB_PATH
from .util import item_id, now_iso

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    title TEXT NOT NULL,
    duration REAL,
    language TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    audio_path TEXT,
    transcript_path TEXT,
    summary_path TEXT,
    model TEXT,
    error TEXT,
    feed_url TEXT,
    meta_json TEXT,
    published_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_status ON items(status);
CREATE INDEX IF NOT EXISTS idx_items_feed ON items(feed_url);

CREATE TABLE IF NOT EXISTS segments (
    item_id TEXT NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    speaker TEXT,
    PRIMARY KEY (item_id, idx)
);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text,
    item_id UNINDEXED,
    idx UNINDEXED,
    start UNINDEXED,
    content='segments',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text, item_id, idx, start)
    VALUES (new.rowid, new.text, new.item_id, new.idx, new.start);
END;

CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text, item_id, idx, start)
    VALUES ('delete', old.rowid, old.text, old.item_id, old.idx, old.start);
END;

CREATE TRIGGER IF NOT EXISTS segments_au AFTER UPDATE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text, item_id, idx, start)
    VALUES ('delete', old.rowid, old.text, old.item_id, old.idx, old.start);
    INSERT INTO segments_fts(rowid, text, item_id, idx, start)
    VALUES (new.rowid, new.text, new.item_id, new.idx, new.start);
END;

CREATE TABLE IF NOT EXISTS feeds (
    url TEXT PRIMARY KEY,
    title TEXT,
    latest INTEGER NOT NULL DEFAULT 3,
    last_checked TEXT,
    created_at TEXT NOT NULL
);
"""


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


class Store:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        rebuilt = self._migrate_fts()
        self.conn.executescript(SCHEMA)
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(segments)")}
        if "speaker" not in columns:
            self.conn.execute("ALTER TABLE segments ADD COLUMN speaker TEXT")
        if rebuilt:
            self.conn.execute("INSERT INTO segments_fts(segments_fts) VALUES('rebuild')")
        self.conn.commit()

    def _migrate_fts(self) -> bool:
        """Upgrade databases created before FTS5 was external-content."""
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='segments_fts'"
        ).fetchone()
        if row and "content='segments'" not in (row[0] or ""):
            self.conn.executescript(
                """
                DROP TRIGGER IF EXISTS segments_ai;
                DROP TRIGGER IF EXISTS segments_ad;
                DROP TRIGGER IF EXISTS segments_au;
                DROP TABLE IF EXISTS segments_fts;
                """
            )
            self.conn.commit()
            return True
        return False

    def close(self) -> None:
        self.conn.close()

    # ---- items ---------------------------------------------------------
    def get_item(self, item: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM items WHERE id = ?", (item,)).fetchone()

    def get_item_by_source(self, source: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM items WHERE source = ?", (source,)).fetchone()

    def add_item(
        self,
        *,
        source: str,
        source_type: str,
        title: str,
        duration: float | None = None,
        published_at: str | None = None,
        feed_url: str | None = None,
        meta_json: str | None = None,
    ) -> tuple[str, bool]:
        existing = self.get_item_by_source(source)
        if existing:
            return existing["id"], False
        if feed_url and title:
            # A feed that moves hosts (e.g. Substack -> beehiiv) changes every
            # enclosure URL; the same title in the same feed is the same episode.
            row = self.conn.execute(
                "SELECT id, status FROM items WHERE feed_url = ? AND title = ?", (feed_url, title)
            ).fetchone()
            if row:
                if row["status"] != "done":
                    # Not transcribed yet: download from the feed's current address,
                    # the old one may be gone.
                    with self.conn:
                        self.conn.execute(
                            "UPDATE items SET source = ?, meta_json = ?, updated_at = ? WHERE id = ?",
                            (source, meta_json, now_iso(), row["id"]),
                        )
                return row["id"], False

        new_id = item_id(source)
        if self.get_item(new_id):
            new_id = item_id(source + "#1")
        now = now_iso()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO items
                    (id, source, source_type, title, duration, status, feed_url,
                     meta_json, published_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                """,
                (
                    new_id,
                    source,
                    source_type,
                    title or "(untitled)",
                    duration,
                    feed_url,
                    meta_json,
                    published_at,
                    now,
                    now,
                ),
            )
        return new_id, True

    def set_status(self, item: str, status: str, error: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE items SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (status, error, now_iso(), item),
            )

    def update_fields(self, item: str, **fields: object) -> None:
        allowed = {
            "title",
            "duration",
            "language",
            "status",
            "audio_path",
            "transcript_path",
            "summary_path",
            "model",
            "error",
            "feed_url",
            "meta_json",
            "published_at",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown item fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        values = list(fields.values()) + [now_iso(), item]
        with self.conn:
            self.conn.execute(
                f"UPDATE items SET {assignments}, updated_at = ? WHERE id = ?", values
            )

    def list_items(
        self,
        *,
        status: str | None = None,
        limit: int = 100,
        retry_failed: bool = False,
    ) -> list[sqlite3.Row]:
        if status:
            return self.conn.execute(
                "SELECT * FROM items WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        if retry_failed:
            return self.conn.execute(
                "SELECT * FROM items WHERE status IN ('pending', 'failed') "
                "ORDER BY created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM items ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    def save_result(
        self,
        item: str,
        *,
        segments: Sequence[Segment],
        duration: float | None,
        language: str | None,
        model: str,
        transcript_path: str,
        summary_path: str | None = None,
    ) -> None:
        now = now_iso()
        with self.conn:
            self.conn.execute("DELETE FROM segments WHERE item_id = ?", (item,))
            self.conn.executemany(
                "INSERT INTO segments (item_id, idx, start, end, text, speaker) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (item, i, seg.start, seg.end, seg.text, seg.speaker)
                    for i, seg in enumerate(segments)
                ],
            )
            self.conn.execute(
                """
                UPDATE items
                   SET status = 'done', duration = ?, language = ?, model = ?,
                       transcript_path = ?, summary_path = ?, error = NULL,
                       updated_at = ?
                 WHERE id = ?
                """,
                (duration, language, model, transcript_path, summary_path, now, item),
            )

    def delete_item(self, item: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM segments WHERE item_id = ?", (item,))
            self.conn.execute("DELETE FROM items WHERE id = ?", (item,))

    def segments_for(self, item: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT idx, start, end, text, speaker FROM segments WHERE item_id = ? ORDER BY idx",
            (item,),
        ).fetchall()

    # ---- feeds ---------------------------------------------------------
    def add_feed(self, url: str, title: str | None, latest: int) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO feeds (url, title, latest, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = COALESCE(excluded.title, feeds.title),
                    latest = excluded.latest
                """,
                (url, title, latest, now_iso()),
            )

    def list_feeds(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM feeds ORDER BY created_at").fetchall()

    def touch_feed(self, url: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE feeds SET last_checked = ? WHERE url = ?", (now_iso(), url)
            )

    # ---- search / stats ------------------------------------------------
    def search(self, query: str, *, limit: int = 20) -> list[dict[str, object]]:
        q = query.strip()
        if not q:
            return []
        rows: list[sqlite3.Row] = []
        if len(q) >= 3:
            match = '"' + q.replace('"', '""') + '"'
            try:
                rows = self.conn.execute(
                    """
                    SELECT i.id, i.title, s.start, s.text, s.speaker
                      FROM segments_fts f
                      JOIN segments s ON s.rowid = f.rowid
                      JOIN items i ON i.id = f.item_id
                     WHERE segments_fts MATCH ?
                     ORDER BY bm25(segments_fts), s.start
                     LIMIT ?
                    """,
                    (match, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []

        if len(rows) < limit:
            seen = {(r["id"], float(r["start"])) for r in rows}
            like = f"%{q}%"
            extra = self.conn.execute(
                """
                SELECT i.id, i.title, s.start, s.text, s.speaker
                  FROM segments s
                  JOIN items i ON i.id = s.item_id
                 WHERE s.text LIKE ?
                 ORDER BY s.start
                 LIMIT ?
                """,
                (like, limit * 3),
            ).fetchall()
            for row in extra:
                key = (row["id"], float(row["start"]))
                if key in seen:
                    continue
                rows.append(row)
                seen.add(key)
                if len(rows) >= limit:
                    break

        if not rows:
            items = self.conn.execute(
                "SELECT id, title FROM items WHERE title LIKE ? LIMIT ?", (f"%{q}%", limit)
            ).fetchall()
            return [
                {"id": row["id"], "title": row["title"], "start": None, "snippet": ""}
                for row in items
            ]

        results = []
        for row in rows:
            text = row["text"]
            pos = text.lower().find(q.lower())
            if pos < 0:
                snippet = text[:160]
            else:
                start = max(0, pos - 60)
                end = min(len(text), pos + len(q) + 100)
                snippet = ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")
            results.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "start": float(row["start"]),
                    "speaker": row["speaker"],
                    "snippet": snippet,
                }
            )
        return results

    def stats(self) -> dict[str, object]:
        counts = {
            row["status"]: row["n"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) AS n FROM items GROUP BY status"
            )
        }
        total_duration = self.conn.execute(
            "SELECT COALESCE(SUM(duration), 0) FROM items WHERE status = 'done'"
        ).fetchone()[0]
        segments = self.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        return {
            "counts": counts,
            "total_duration": float(total_duration or 0),
            "segments": int(segments),
            "feeds": len(self.list_feeds()),
        }