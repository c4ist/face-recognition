from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import ensure_directory

CSV_FIELDS = [
    "source_type",
    "source_file",
    "frame_index",
    "timestamp_sec",
    "face_index",
    "match_name",
    "confidence",
    "is_unknown",
]


def _safe_source_file(source_path: Any) -> str:
    if source_path is None:
        return ""
    try:
        return Path(str(source_path)).name
    except Exception:
        return str(source_path)


def _sanitize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_type": row.get("source_type"),
        "source_file": _safe_source_file(row.get("source_path")),
        "frame_index": row.get("frame_index"),
        "timestamp_sec": row.get("timestamp_sec"),
        "face_index": row.get("face_index"),
        "match_name": row.get("match_name"),
        "confidence": row.get("confidence"),
        "is_unknown": row.get("is_unknown"),
    }


def build_summary(rows: List[Dict[str, Any]], extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    counts = Counter(row.get("match_name", "unknown") for row in rows if not row.get("is_unknown", True))
    unknown_faces = sum(1 for row in rows if row.get("is_unknown", True))

    summary: Dict[str, Any] = {
        "total_faces": len(rows),
        "matched_faces": len(rows) - unknown_faces,
        "unknown_faces": unknown_faces,
        "matches_by_name": dict(counts),
    }

    if extra:
        summary.update(extra)

    return summary


def write_reports(output_dir: Path | str, rows: Iterable[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    output_path = ensure_directory(output_dir)
    rows_list = [_sanitize_row(row) for row in rows]

    csv_path = output_path / "results.csv"
    json_path = output_path / "results.json"
    summary_path = output_path / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows_list:
            prepared = {field: row.get(field) for field in CSV_FIELDS}
            writer.writerow(prepared)

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(rows_list, json_file, indent=2)

    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
