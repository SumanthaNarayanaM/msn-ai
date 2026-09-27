"""Knowledge store: chunk persistence, lexical + vector retrieval.

Storage is deliberately boring and self-contained:

* ``knowledge/chunks.jsonl`` — the authoritative chunk text and metadata, one
  JSON object per line, appended on index and rewritten on forget.
* ``knowledge/index.sqlite3`` — an FTS5 table for keyword search plus a small
  vector table, both keyed on ``file_id`` so deletion is a single statement.

Retrieval blends FTS5 keyword hits with cosine similarity over the cached
vectors, then de-duplicates by file so one long document cannot crowd out the
others.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from app.core.logging import get_logger
from app.ingestion.chunking import Chunk
from app.ingestion.embeddings import EmbeddingProvider, cosine_similarity

logger = get_logger("msn_ai.ingestion.knowledge")

#: How many FTS candidates to consider before vector re-ranking.
FTS_CANDIDATE_LIMIT = 40
#: RRF constant for blending the two rankings.
RRF_K = 60


@dataclass
class SearchHit:
    """One retrieved chunk plus its score and citation."""

    chunk: Chunk
    score: float
    matched_by: str = "vector"

    def source_label(self) -> str:
        return self.chunk.source_label()

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.chunk.file_id,
            "filename": self.chunk.filename,
            "locator": self.chunk.locator,
            "kind": self.chunk.kind.value,
            "score": round(self.score, 4),
            "matched_by": self.matched_by,
            "text": self.chunk.text,
            "source": self.source_label(),
        }


class KnowledgeBase:
    """Chunk storage and retrieval, scoped by session/project/personal."""

    def __init__(self, settings, embedder: EmbeddingProvider) -> None:
        self._settings = settings
        self._embedder = embedder
        self._dir = settings.knowledge_path()
        self._embed_dir = settings.embeddings_path()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._embed_dir.mkdir(parents=True, exist_ok=True)
        self._chunks_path = self._dir / "chunks.jsonl"
        self._db_path = self._dir / "index.sqlite3"
        self._lock = threading.RLock()
        self._chunks: list[dict[str, Any]] = []
        # Vectors live in one contiguous matrix, not per-chunk Python lists:
        # a few thousand chunks as lists of floats costs hundreds of MB and
        # makes every search a Python loop over them.
        self._keys: list[str] = []
        self._matrix: Any = None  # numpy.ndarray (n, dim), L2-normalised
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._init_db()
        self._load()
        if self._fts_enabled and self._chunks:
            # The FTS table may have just been created (or replaced after a
            # schema change) while chunks already exist on disk; make sure the
            # keyword index matches the authoritative chunk list.
            with self._lock:
                self._rebuild_fts()
                self._conn.commit()

    # --- schema ----------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            # FTS5 may be missing in a stripped-down Python build; degrade to
            # vector-only retrieval rather than failing to start.
            self._fts_enabled = True
            try:
                cur.execute("SELECT chunk_key FROM chunks_fts LIMIT 1")
            except sqlite3.OperationalError:
                # Table absent, or left over from the older rowid-keyed schema.
                cur.execute("DROP TABLE IF EXISTS chunks_fts")
            try:
                cur.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
                    "chunk_key UNINDEXED, file_id UNINDEXED, locator, text,"
                    " tokenize='unicode61')"
                )
            except sqlite3.OperationalError as exc:
                self._fts_enabled = False
                logger.warning("FTS5 unavailable (%s); using vector retrieval only.", exc)
            cur.execute(
                "CREATE TABLE IF NOT EXISTS chunk_meta ("
                " chunk_key TEXT PRIMARY KEY,"
                " file_id TEXT NOT NULL,"
                " chunk_index INTEGER NOT NULL,"
                " scope TEXT NOT NULL DEFAULT 'session',"
                " session_id TEXT NOT NULL DEFAULT '',"
                " project TEXT NOT NULL DEFAULT '',"
                " filename TEXT NOT NULL DEFAULT '')"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunk_meta_file ON chunk_meta(file_id)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_meta_scope ON chunk_meta(scope)")
            self._conn.commit()

    def _chunk_key(self, file_id: str, index: int) -> str:
        return f"{file_id}:{index}"

    # --- loading / writing ----------------------------------------------

    def _load(self) -> None:
        with self._lock:
            self._chunks = []
            if self._chunks_path.exists():
                for line in self._chunks_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed knowledge row.")
            vectors: dict[str, list[float]] = {}
            for path in sorted(self._embed_dir.glob("*.npy")):
                try:
                    import numpy as np

                    matrix = np.load(path, allow_pickle=False)
                    if matrix.ndim == 1:
                        matrix = matrix.reshape(1, -1)
                    file_id = path.stem
                    for index, row in enumerate(matrix):
                        vectors[self._chunk_key(file_id, index)] = [float(x) for x in row]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Could not load vectors from %s: %s", path.name, exc)
            self._rebuild_matrix(vectors)
            logger.info("Loaded %d knowledge chunk(s).", len(self._chunks))

    def _rebuild_matrix(self, vectors: dict[str, list[float]] | None = None) -> None:
        """Recompute the in-memory vector matrix from per-file ``.npy`` files.

        Called on load and after any index/remove so the matrix always matches
        the chunk list. ``vectors`` supplies freshly computed vectors to overlay
        on top of the persisted ones (used right after indexing, before the new
        file has been written to disk).
        """
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy is a hard dependency
            self._keys = []
            self._matrix = None
            return

        by_key: dict[str, list[float]] = {}
        for path in sorted(self._embed_dir.glob("*.npy")):
            try:
                matrix = np.load(path, allow_pickle=False)
                if matrix.ndim == 1:
                    matrix = matrix.reshape(1, -1)
                file_id = path.stem
                for index, row in enumerate(matrix):
                    by_key[self._chunk_key(file_id, index)] = [float(x) for x in row]
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not load vectors from %s: %s", path.name, exc)
        if vectors:
            by_key.update(vectors)

        keys: list[str] = []
        rows: list[list[float]] = []
        for row in self._chunks:
            key = row.get("key")
            vector = by_key.get(key) if key else None
            if vector:
                keys.append(key)
                rows.append(vector)
        self._keys = keys
        if rows:
            self._matrix = np.asarray(rows, dtype="float32")
        else:
            self._matrix = None

    def _rebuild_fts(self) -> None:
        """Re-create the keyword index from the authoritative chunk list.

        A regular (non-contentless) FTS5 table cannot use the ``delete-all``
        command, so the rows are deleted and re-inserted instead.
        """
        if not self._fts_enabled:
            return
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks_fts")
        cur.executemany(
            "INSERT INTO chunks_fts (chunk_key, file_id, locator, text) VALUES (?,?,?,?)",
            [
                (row["key"], row["file_id"], row.get("locator", ""), row["text"])
                for row in self._chunks
            ],
        )

    def _rewrite_jsonl(self) -> None:
        lines = [json.dumps(row, ensure_ascii=False) for row in self._chunks]
        tmp = self._chunks_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(self._chunks_path)

    def _save_vectors(self, file_id: str) -> None:
        """Persist one file's vectors as a float32 ``.npy``."""
        try:
            import numpy as np

            if self._matrix is None:
                return
            positions = [
                i for i, key in enumerate(self._keys) if key.startswith(f"{file_id}:")
            ]
            if not positions:
                return
            block = self._matrix[positions]
            np.save(self._embed_dir / f"{file_id}.npy", block.astype("float32"))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Could not persist vectors for %s: %s", file_id, exc)

    # --- indexing --------------------------------------------------------

    def index_chunks(
        self,
        file_id: str,
        chunks: Sequence[Chunk],
        *,
        scope: str = "session",
        session_id: str = "",
        project: str = "",
    ) -> int:
        """Embed and store chunks for a file. Returns the number indexed."""
        with self._lock:
            if not chunks:
                # A re-index that yields nothing must clear the old content
                # rather than leave stale chunks behind.
                self.remove_file(file_id)
                return 0
            self.remove_file(file_id)  # re-index replaces, never duplicates

            rows: list[dict[str, Any]] = []
            fresh: dict[str, list[float]] = {}
            for chunk in chunks:
                key = self._chunk_key(file_id, chunk.index)
                fresh[key] = self._embedder.embed(chunk.text)
                rows.append(
                    {
                        "key": key,
                        "file_id": file_id,
                        "chunk_index": chunk.index,
                        "filename": chunk.filename,
                        "locator": chunk.locator,
                        "kind": chunk.kind.value,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "scope": scope,
                        "session_id": session_id,
                        "project": project,
                    }
                )
            self._chunks.extend(rows)
            self._rewrite_jsonl()
            # Pass the fresh vectors in: this file has not been written to the
            # vector store yet, so a disk-only rebuild would drop them.
            self._rebuild_matrix(fresh)
            self._save_vectors(file_id)

            cur = self._conn.cursor()
            for row in rows:
                cur.execute(
                    "INSERT OR REPLACE INTO chunk_meta"
                    " (chunk_key, file_id, chunk_index, scope, session_id, project, filename)"
                    " VALUES (?,?,?,?,?,?,?)",
                    (
                        row["key"],
                        file_id,
                        row["chunk_index"],
                        scope,
                        session_id,
                        project,
                        row["filename"],
                    ),
                )
                if self._fts_enabled:
                    cur.execute(
                        "INSERT INTO chunks_fts (chunk_key, file_id, locator, text)"
                        " VALUES (?,?,?,?)",
                        (row["key"], file_id, row["locator"], row["text"]),
                    )
            self._conn.commit()
            logger.info("Indexed %d chunk(s) for %s", len(rows), file_id)
            return len(rows)

    # --- removal ---------------------------------------------------------

    def remove_file(self, file_id: str) -> int:
        """Drop every chunk belonging to a file (used by forget and delete)."""
        with self._lock:
            before = len(self._chunks)
            self._chunks = [row for row in self._chunks if row["file_id"] != file_id]
            removed = before - len(self._chunks)
            if removed:
                self._rewrite_jsonl()
                self._rebuild_matrix()
            cur = self._conn.cursor()
            cur.execute("DELETE FROM chunk_meta WHERE file_id = ?", (file_id,))
            if self._fts_enabled and removed:
                # Rebuild from the chunk list, which is the authoritative copy.
                self._rebuild_fts()
            self._conn.commit()
            try:
                (self._embed_dir / f"{file_id}.npy").unlink(missing_ok=True)
            except OSError:  # pragma: no cover - defensive
                pass
            if removed:
                logger.info("Removed %d chunk(s) for %s", removed, file_id)
            return removed

    def chunk_count_for(self, file_id: str) -> int:
        """How many chunks a file currently contributes to the index."""
        with self._lock:
            return sum(1 for row in self._chunks if row["file_id"] == file_id)

    def set_scope(self, file_id: str, scope: str, *, project: str = "") -> int:
        """Change a file's memory scope (session -> project/personal)."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE chunk_meta SET scope = ?, project = ? WHERE file_id = ?",
                (scope, project, file_id),
            )
            self._conn.commit()
            updated = 0
            for row in self._chunks:
                if row["file_id"] == file_id:
                    row["scope"] = scope
                    row["project"] = project
                    updated += 1
            if updated:
                self._rewrite_jsonl()
            return updated

    # --- retrieval -------------------------------------------------------

    def _scope_clause(
        self, scope: str, session_id: str, project: str
    ) -> tuple[str, list[Any]]:
        """SQL predicate selecting the chunks visible to this request."""
        if scope == "personal":
            return "scope = 'personal'", []
        if scope == "project":
            return "scope = 'project' AND (project = ? OR project = '')", [project]
        if scope == "all":
            return "1=1", []
        # Session scope sees the current session *and* anything remembered
        # more broadly, which is the whole point of Remember.
        return "(scope = 'session' AND session_id = ?) OR scope IN ('project','personal')", [
            session_id
        ]

    def _fts_search(self, query: str, limit: int) -> dict[str, float]:
        """Return ``{chunk_key: score}`` for a keyword match, best first."""
        if not self._fts_enabled or not query.strip():
            return {}
        match = self._fts_query(query)
        if not match:
            return {}
        try:
            cur = self._conn.execute(
                "SELECT chunk_key, bm25(chunks_fts) FROM chunks_fts"
                " WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, limit),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError as exc:
            logger.debug("FTS query failed (%s); falling back to vectors.", exc)
            return {}
        if not rows:
            return {}
        # bm25() returns lower-is-better; normalise to 0.5..1.0.
        best = max(abs(score) for _key, score in rows) or 1.0
        return {key: 1.0 - (abs(score) / best) * 0.5 for key, score in rows}

    @staticmethod
    def _fts_query(query: str) -> str:
        """Build a safe FTS5 MATCH expression from free-form user text."""
        import re

        terms = [
            term
            for term in re.findall(r'[\w"]+', query or "")
            if len(term) > 1
        ][:24]
        if not terms:
            return ""
        quoted = [f'"{term}"' for term in terms]
        return " OR ".join(quoted)

    def search(
        self,
        query: str,
        *,
        scope: str = "session",
        session_id: str = "",
        session_ids: Optional[Iterable[str]] = None,
        project: str = "",
        top_k: int = 5,
        min_score: float = 0.12,
        file_ids: Optional[Iterable[str]] = None,
    ) -> list[SearchHit]:
        """Retrieve the most relevant chunks for ``query``.

        ``session_ids`` accepts several owning sessions at once. Chat needs
        that: the client pins its own session id when uploading files, and the
        server may hand out a different one for the conversation, so both must
        stay visible.
        """
        with self._lock:
            rows = list(self._chunks)
        if not rows:
            return []
        by_key = {row["key"]: row for row in rows}
        allowed = set(file_ids) if file_ids is not None else None
        owner_ids = (
            set(session_ids) if session_ids is not None else {session_id}
        )

        def visible(row: dict[str, Any]) -> bool:
            if allowed is not None and row["file_id"] not in allowed:
                return False
            row_scope = row.get("scope")
            if scope == "personal":
                return row_scope == "personal"
            if scope == "project":
                return row_scope == "project" and (
                    not row.get("project") or row.get("project") == project
                )
            if scope == "all":
                return True
            return (row_scope == "session" and row.get("session_id") in owner_ids) or row_scope in (
                "project",
                "personal",
            )

        candidates = [row for row in rows if visible(row)]
        if not candidates:
            return []
        candidate_keys = {row["key"] for row in candidates}

        # --- lexical ---
        fts_scores = self._fts_search(query, FTS_CANDIDATE_LIMIT)
        fts_keys = [key for key in fts_scores if key in candidate_keys]

        # --- vector (one matrix-vector product, not a Python loop) ---
        vector_scores: dict[str, float] = {}
        query_vector = self._embedder.embed(query or "")
        with self._lock:
            matrix = self._matrix
            keys = list(self._keys)
        if matrix is not None and keys and any(query_vector):
            try:
                import numpy as np

                q = np.asarray(query_vector, dtype="float32")
                if q.shape[0] == matrix.shape[1]:
                    scores = matrix @ q
                    ranked = sorted(
                        (i for i, key in enumerate(keys) if key in candidate_keys),
                        key=lambda i: -float(scores[i]),
                    )
                    # min_score is a *cosine similarity* floor, so it applies to
                    # the vector leg before fusion -- not to the rank-based RRF
                    # score, where 0.12 would be meaningless.
                    vector_scores = {
                        keys[i]: float(scores[i])
                        for i in ranked[:FTS_CANDIDATE_LIMIT]
                        if float(scores[i]) >= min_score
                    }
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Vector scoring failed, using keywords only: %s", exc)

        # --- fuse with reciprocal rank fusion ---
        # RRF uses rank only. Folding the raw score in here (e.g. 1-score) would
        # invert the ranking, because the strongest match carries the highest
        # score and would then contribute the *least*.
        fused: dict[str, float] = {}
        matched_by: dict[str, str] = {}
        for rank, key in enumerate(fts_keys):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            matched_by.setdefault(key, "keyword")
        for rank, (key, _score) in enumerate(
            sorted(vector_scores.items(), key=lambda kv: kv[1], reverse=True)
        ):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
            matched_by.setdefault(key, "vector")

        if not fused:
            return []

        # Keep the best chunk per file first, then fill with the runners-up so a
        # single long document cannot crowd out the rest of the evidence.
        per_file_best: dict[str, tuple[float, str]] = {}
        for key, score in fused.items():
            file_id = by_key[key]["file_id"]
            if file_id not in per_file_best or score > per_file_best[file_id][0]:
                per_file_best[file_id] = (score, key)

        limit = max(int(top_k), 1)
        ordered_keys = [k for _s, k in sorted(per_file_best.values(), key=lambda v: -v[0])]
        for key, _score in sorted(fused.items(), key=lambda kv: -kv[1]):
            if len(ordered_keys) >= limit * 2:
                break
            if key not in ordered_keys:
                ordered_keys.append(key)

        hits: list[SearchHit] = []
        for key in ordered_keys:
            row = by_key.get(key)
            if row is None:
                continue
            hits.append(
                SearchHit(
                    chunk=_row_to_chunk(row),
                    score=fused.get(key, 0.0),
                    matched_by=matched_by.get(key, "vector"),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    # --- diagnostics -----------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            per_file: dict[str, int] = {}
            for row in self._chunks:
                per_file[row["file_id"]] = per_file.get(row["file_id"], 0) + 1
            return {
                "chunks": len(self._chunks),
                "files": len(per_file),
                "vectors": len(self._keys),
                "embedder": self._embedder.status(),
                "fts": self._fts_enabled,
                "chunks_per_file": per_file,
            }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - defensive
                pass


def _row_to_chunk(row: dict[str, Any]) -> Chunk:
    from app.ingestion.models import SourceKind

    try:
        kind = SourceKind(row.get("kind", "document"))
    except ValueError:
        kind = SourceKind.DOCUMENT
    return Chunk(
        text=row["text"],
        file_id=row["file_id"],
        filename=row.get("filename", ""),
        index=int(row.get("chunk_index", 0)),
        kind=kind,
        locator=row.get("locator", ""),
        metadata=row.get("metadata") or {},
    )
