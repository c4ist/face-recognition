from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List

import numpy as np

from .types import KnownEmbedding

# credit to some random romanian guy for this database stuff i have no idea how to work these still lol 
def connect_db(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS identities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            identity_id INTEGER NOT NULL,
            source_path TEXT NOT NULL,
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL,
            det_score REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(identity_id) REFERENCES identities(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_embeddings_identity_id ON embeddings(identity_id);
        CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(name);
        """
    )
    conn.commit()


def clear_database(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM embeddings;")
    conn.execute("DELETE FROM identities;")
    conn.commit()


def upsert_identity(conn: sqlite3.Connection, name: str) -> int:
    conn.execute("INSERT OR IGNORE INTO identities(name) VALUES (?);", (name,))
    row = conn.execute("SELECT id FROM identities WHERE name = ?;", (name,)).fetchone()
    if row is None:
        raise RuntimeError(f"Unable to create or fetch identity: {name}")
    return int(row["id"])


def insert_embedding(
    conn: sqlite3.Connection,
    identity_id: int,
    source_path: str,
    embedding: np.ndarray,
    det_score: float,
) -> None:
    vector = np.asarray(embedding, dtype=np.float32).flatten()
    if vector.size == 0:
        raise ValueError("Embedding vector is empty.")

    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("Embedding vector norm is zero.")

    normalized = (vector / norm).astype(np.float32)

    conn.execute(
        """
        INSERT INTO embeddings(identity_id, source_path, embedding, embedding_dim, det_score)
        VALUES (?, ?, ?, ?, ?);
        """,
        (
            identity_id,
            source_path,
            sqlite3.Binary(normalized.tobytes()),
            int(normalized.size),
            float(det_score),
        ),
    )


def load_known_embeddings(conn: sqlite3.Connection) -> List[KnownEmbedding]:
    rows = conn.execute(
        """
        SELECT
            e.identity_id,
            i.name,
            e.source_path,
            e.embedding,
            e.embedding_dim
        FROM embeddings e
        INNER JOIN identities i ON i.id = e.identity_id
        ORDER BY i.name ASC, e.id ASC;
        """
    ).fetchall()

    known: List[KnownEmbedding] = []
    for row in rows:
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        expected_dim = int(row["embedding_dim"])
        if expected_dim > 0 and vector.size != expected_dim:
            continue
        norm = np.linalg.norm(vector)
        if norm <= 0:
            continue
        vector = (vector / norm).astype(np.float32)
        known.append(
            KnownEmbedding(
                identity_id=int(row["identity_id"]),
                name=str(row["name"]),
                embedding=vector,
                source_path=str(row["source_path"]),
            )
        )
    return known


def count_identities(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM identities;").fetchone()
    return int(row["c"] if row else 0)


def count_embeddings(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM embeddings;").fetchone()
    return int(row["c"] if row else 0)
