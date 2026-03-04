from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List

from facerec.config import EngineConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return parsed


def _confidence_threshold(value: str) -> float:
    parsed = float(value)
    if parsed < -1.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("Threshold must be between -1.0 and 1.0.")
    return parsed


def _face_score(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("Min face score must be between 0.0 and 1.0.")
    return parsed


def _raise_missing_dependency(exc: ModuleNotFoundError) -> None:
    package = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency '{package}'. Install requirements with: pip install -r requirements.txt"
    ) from exc


def _engine_config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model_name=args.model_name,
        det_size=args.det_size,
        provider=args.provider,
        min_face_score=args.min_face_score,
    )


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", default="buffalo_l", help="InsightFace model name.")
    parser.add_argument("--det-size", type=_positive_int, default=640, help="Detection input size.")
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNXRuntime provider. Use cuda only when properly configured.",
    )
    parser.add_argument(
        "--min-face-score",
        type=_face_score,
        default=0.5,
        help="Minimum detector confidence to keep a face (0.0-1.0).",
    )


def _format_label(name: str, confidence: float) -> str:
    return f"{name} ({confidence:.3f})"


def _result_row(
    source_type: str,
    source_path: str,
    face_index: int,
    bbox: tuple[float, float, float, float],
    det_score: float,
    match_name: str,
    confidence: float,
    is_unknown: bool,
    identity_id: int | None,
    frame_index: int | None = None,
    timestamp_sec: float | None = None,
) -> Dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "source_type": source_type,
        "source_path": source_path,
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "face_index": face_index,
        "bbox_x1": round(float(x1), 3),
        "bbox_y1": round(float(y1), 3),
        "bbox_x2": round(float(x2), 3),
        "bbox_y2": round(float(y2), 3),
        "det_score": round(float(det_score), 6),
        "match_name": match_name,
        "confidence": round(float(confidence), 6),
        "is_unknown": bool(is_unknown),
        "identity_id": identity_id,
    }


def enroll_command(args: argparse.Namespace) -> None:
    try:
        from facerec.db import (
            clear_database,
            connect_db,
            count_embeddings,
            count_identities,
            initialize_db,
            insert_embedding,
            upsert_identity,
        )
        from facerec.engine import FaceEngine
        from facerec.io import list_identity_dirs, list_image_files, load_bgr_image
    except ModuleNotFoundError as exc:
        _raise_missing_dependency(exc)

    known_dir = Path(args.known_dir)
    if not known_dir.exists() or not known_dir.is_dir():
        raise SystemExit(f"Known directory not found: {known_dir}")

    conn = connect_db(args.db_path)
    try:
        initialize_db(conn)
        if args.clear_db:
            clear_database(conn)

        engine = FaceEngine(_engine_config_from_args(args))

        identity_dirs = list_identity_dirs(known_dir)
        if not identity_dirs:
            raise SystemExit(f"No identity folders found in: {known_dir}")

        stats = {
            "identity_dirs": len(identity_dirs),
            "images_scanned": 0,
            "images_with_no_face": 0,
            "images_failed": 0,
            "embeddings_saved": 0,
        }

        for identity_dir in identity_dirs:
            identity_name = identity_dir.name.strip()
            if not identity_name:
                continue

            identity_id = upsert_identity(conn, identity_name)
            image_files = list_image_files(identity_dir)
            if args.max_images_per_identity is not None:
                image_files = image_files[: args.max_images_per_identity]

            for image_path in image_files:
                stats["images_scanned"] += 1
                try:
                    image = load_bgr_image(image_path)
                    faces = engine.detect_faces(image)
                except Exception:
                    stats["images_failed"] += 1
                    continue

                face = engine.largest_face(faces)
                if face is None:
                    stats["images_with_no_face"] += 1
                    continue

                insert_embedding(
                    conn=conn,
                    identity_id=identity_id,
                    source_path=str(image_path.resolve()),
                    embedding=face.embedding,
                    det_score=face.det_score,
                )
                stats["embeddings_saved"] += 1

        conn.commit()

        print("Enrollment complete")
        print(f"- identity folders: {stats['identity_dirs']}")
        print(f"- images scanned: {stats['images_scanned']}")
        print(f"- no-face images: {stats['images_with_no_face']}")
        print(f"- failed images: {stats['images_failed']}")
        print(f"- embeddings saved: {stats['embeddings_saved']}")
        print(f"- identities in db: {count_identities(conn)}")
        print(f"- total embeddings in db: {count_embeddings(conn)}")
    finally:
        conn.close()


def analyze_images_command(args: argparse.Namespace) -> None:
    try:
        from facerec.db import connect_db, initialize_db, load_known_embeddings
        from facerec.engine import FaceEngine
        from facerec.io import (
            draw_face_annotation,
            ensure_directory,
            list_image_files,
            load_bgr_image,
            save_bgr_image,
        )
        from facerec.matching import EmbeddingMatcher
        from facerec.reports import build_summary, write_reports
    except ModuleNotFoundError as exc:
        _raise_missing_dependency(exc)

    input_path = Path(args.input_path)
    image_files = list_image_files(input_path)
    if not image_files:
        raise SystemExit(f"No image files found at: {input_path}")

    conn = connect_db(args.db_path)
    try:
        initialize_db(conn)
        known_embeddings = load_known_embeddings(conn)
    finally:
        conn.close()

    if not known_embeddings:
        raise SystemExit("No embeddings found in database. Run enroll first.")

    matcher = EmbeddingMatcher(known_embeddings)
    engine = FaceEngine(_engine_config_from_args(args))

    output_dir = ensure_directory(args.output_dir)
    annotated_dir = ensure_directory(output_dir / "annotated_images") if args.annotate else None

    rows: List[Dict[str, Any]] = []

    for image_path in image_files:
        try:
            image = load_bgr_image(image_path)
        except Exception:
            continue

        faces = engine.detect_faces(image)
        annotated = image.copy() if args.annotate else None

        for face_index, face in enumerate(faces):
            match = matcher.match(face.embedding, threshold=args.threshold)

            row = _result_row(
                source_type="image",
                source_path=str(image_path.resolve()),
                frame_index=None,
                timestamp_sec=None,
                face_index=face_index,
                bbox=face.bbox,
                det_score=face.det_score,
                match_name=match.name,
                confidence=match.confidence,
                is_unknown=match.is_unknown,
                identity_id=match.identity_id,
            )
            rows.append(row)

            if annotated is not None:
                draw_face_annotation(
                    annotated,
                    face.bbox,
                    _format_label(match.name, match.confidence),
                    match.is_unknown,
                )

        if annotated_dir is not None:
            if input_path.is_file():
                output_file = annotated_dir / image_path.name
            else:
                output_file = annotated_dir / image_path.relative_to(input_path)
            save_bgr_image(output_file, annotated)

    summary = build_summary(
        rows,
        extra={
            "mode": "analyze_images",
            "input_target": input_path.name,
            "total_input_images": len(image_files),
            "threshold": args.threshold,
        },
    )
    write_reports(output_dir, rows, summary)

    print("Image analysis complete")
    print(f"- images processed: {len(image_files)}")
    print(f"- faces found: {summary['total_faces']}")
    print(f"- matched faces: {summary['matched_faces']}")
    print(f"- unknown faces: {summary['unknown_faces']}")
    print(f"- reports: {output_dir.resolve()}")


def analyze_video_command(args: argparse.Namespace) -> None:
    try:
        import cv2

        from facerec.db import connect_db, initialize_db, load_known_embeddings
        from facerec.engine import FaceEngine
        from facerec.io import draw_face_annotation, ensure_directory, save_bgr_image
        from facerec.matching import EmbeddingMatcher
        from facerec.reports import build_summary, write_reports
    except ModuleNotFoundError as exc:
        _raise_missing_dependency(exc)

    video_path = Path(args.video_path)
    if not video_path.exists() or not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")

    conn = connect_db(args.db_path)
    try:
        initialize_db(conn)
        known_embeddings = load_known_embeddings(conn)
    finally:
        conn.close()

    if not known_embeddings:
        raise SystemExit("No embeddings found in database. Run enroll first.")

    matcher = EmbeddingMatcher(known_embeddings)
    engine = FaceEngine(_engine_config_from_args(args))

    output_dir = ensure_directory(args.output_dir)
    annotated_dir = ensure_directory(output_dir / "annotated_frames") if args.annotate_frames else None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Unable to open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    frames_read = 0
    sampled_frames = 0
    rows: List[Dict[str, Any]] = []

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frames_read += 1

            should_process = frame_index % args.frame_step == 0
            if should_process:
                sampled_frames += 1
                faces = engine.detect_faces(frame)
                annotated = frame.copy() if args.annotate_frames else None

                timestamp_sec = round(frame_index / fps, 4) if fps > 0 else None

                for face_index, face in enumerate(faces):
                    match = matcher.match(face.embedding, threshold=args.threshold)

                    row = _result_row(
                        source_type="video",
                        source_path=str(video_path.resolve()),
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        face_index=face_index,
                        bbox=face.bbox,
                        det_score=face.det_score,
                        match_name=match.name,
                        confidence=match.confidence,
                        is_unknown=match.is_unknown,
                        identity_id=match.identity_id,
                    )
                    rows.append(row)

                    if annotated is not None:
                        draw_face_annotation(
                            annotated,
                            face.bbox,
                            _format_label(match.name, match.confidence),
                            match.is_unknown,
                        )

                if annotated_dir is not None:
                    output_file = annotated_dir / f"frame_{frame_index:06d}.jpg"
                    save_bgr_image(output_file, annotated)

                if args.max_sampled_frames is not None and sampled_frames >= args.max_sampled_frames:
                    break

            frame_index += 1
    finally:
        capture.release()

    summary = build_summary(
        rows,
        extra={
            "mode": "analyze_video",
            "video_file": video_path.name,
            "fps": fps,
            "frames_read": frames_read,
            "sampled_frames": sampled_frames,
            "frame_step": args.frame_step,
            "threshold": args.threshold,
        },
    )
    write_reports(output_dir, rows, summary)

    print("Video analysis complete")
    print(f"- frames read: {summary['frames_read']}")
    print(f"- sampled frames: {sampled_frames}")
    print(f"- faces found: {summary['total_faces']}")
    print(f"- matched faces: {summary['matched_faces']}")
    print(f"- unknown faces: {summary['unknown_faces']}")
    print(f"- reports: {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline facial recognition from image/video files (no webcam required)."
    )
    parser.add_argument("--debug", action="store_true", help="Show full traceback on errors.")
    subparsers = parser.add_subparsers(dest="command")

    enroll = subparsers.add_parser("enroll", help="Enroll labeled known faces into the embedding database.")
    enroll.add_argument("--known-dir", required=True, help="Directory with known/<person_name>/*.jpg structure.")
    enroll.add_argument("--db-path", default="data/faces.db", help="SQLite DB path.")
    enroll.add_argument("--clear-db", action="store_true", help="Delete existing identities/embeddings before enrolling.")
    enroll.add_argument(
        "--max-images-per-identity",
        type=_positive_int,
        default=None,
        help="Optional cap for how many images to ingest per identity.",
    )
    _add_common_engine_args(enroll)
    enroll.set_defaults(func=enroll_command)

    analyze_images = subparsers.add_parser(
        "analyze-images",
        help="Analyze one image file or all images in a directory.",
    )
    analyze_images.add_argument("--input-path", required=True, help="Image file or folder path.")
    analyze_images.add_argument("--db-path", default="data/faces.db", help="SQLite DB path.")
    analyze_images.add_argument("--output-dir", default="runs/images", help="Output folder for reports.")
    analyze_images.add_argument(
        "--threshold",
        type=_confidence_threshold,
        default=0.45,
        help="Cosine similarity threshold for known/unknown decision.",
    )
    analyze_images.add_argument("--annotate", action="store_true", help="Write annotated images to output.")
    _add_common_engine_args(analyze_images)
    analyze_images.set_defaults(func=analyze_images_command)

    analyze_video = subparsers.add_parser("analyze-video", help="Analyze faces from a video file.")
    analyze_video.add_argument("--video-path", required=True, help="Input video file.")
    analyze_video.add_argument("--db-path", default="data/faces.db", help="SQLite DB path.")
    analyze_video.add_argument("--output-dir", default="runs/video", help="Output folder for reports.")
    analyze_video.add_argument(
        "--threshold",
        type=_confidence_threshold,
        default=0.45,
        help="Cosine similarity threshold for known/unknown decision.",
    )
    analyze_video.add_argument(
        "--frame-step",
        type=_positive_int,
        default=5,
        help="Process every Nth frame.",
    )
    analyze_video.add_argument(
        "--max-sampled-frames",
        type=_positive_int,
        default=None,
        help="Optional cap on sampled frames processed.",
    )
    analyze_video.add_argument(
        "--annotate-frames",
        action="store_true",
        help="Write sampled annotated frames to output.",
    )
    _add_common_engine_args(analyze_video)
    analyze_video.set_defaults(func=analyze_video_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(1)

    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}")
        if args.__dict__.get("debug"):
            traceback.print_exc()
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
