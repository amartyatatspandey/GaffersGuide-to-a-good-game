from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FixtureSpec:
    clip_id: str
    category: str
    source_filename: str
    start_sec: int
    duration_sec: int
    output_filename: str


SPECS: tuple[FixtureSpec, ...] = (
    FixtureSpec(
        clip_id="short_czech_england_120s",
        category="short",
        source_filename="08_czech_england.mp4",
        start_sec=0,
        duration_sec=120,
        output_filename="short_czech_england_120s.mp4",
    ),
    FixtureSpec(
        clip_id="medium_liverpool_mancity_900s",
        category="medium",
        source_filename="02_liverpool_mancity.mp4",
        start_sec=0,
        duration_sec=900,
        output_filename="medium_liverpool_mancity_900s.mp4",
    ),
    FixtureSpec(
        clip_id="long_barca_madrid_1200s",
        category="long",
        source_filename="01_barca_madrid.mp4",
        start_sec=0,
        duration_sec=1200,
        output_filename="long_barca_madrid_1200s.mp4",
    ),
    FixtureSpec(
        clip_id="stress_psg_intermilan_1200s",
        category="stress",
        source_filename="03_psg_intermilan.mp4",
        start_sec=0,
        duration_sec=1200,
        output_filename="stress_psg_intermilan_1200s.mp4",
    ),
)


def _probe(video_path: Path) -> dict[str, object]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    payload = json.loads(
        subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    )
    stream = payload.get("streams", [{}])[0]
    fmt = payload.get("format", {})
    fps_raw = str(stream.get("r_frame_rate", "0/1"))
    num, den = fps_raw.split("/")
    fps = (float(num) / float(den)) if float(den) else 0.0
    return {
        "duration_sec": round(float(fmt.get("duration", 0.0)), 2),
        "fps": round(fps, 3),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
    }


def _extract(source: Path, target: Path, start_sec: int, duration_sec: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_sec),
        "-i",
        str(source),
        "-t",
        str(duration_sec),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(target),
    ]
    subprocess.check_call(cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare experiment-local fixture dataset.")
    parser.add_argument(
        "--training-samples-dir",
        type=str,
        default="../../../backend/data/training_samples",
        help="Path to source training samples directory.",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=str,
        default="../data/fixtures",
        help="Target fixtures root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    source_dir = (root / args.training_samples_dir).resolve()
    fixtures_dir = (root / args.fixtures_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Training samples dir not found: {source_dir}")

    manifest: list[dict[str, object]] = []
    for spec in SPECS:
        source = source_dir / spec.source_filename
        if not source.is_file():
            raise FileNotFoundError(f"Missing source sample: {source}")
        target = fixtures_dir / spec.category / spec.output_filename
        _extract(source, target, spec.start_sec, spec.duration_sec)
        meta = _probe(target)
        manifest.append(
            {
                "clip_id": spec.clip_id,
                "category": spec.category,
                "source_relpath": f"backend/data/training_samples/{spec.source_filename}",
                "fixture_relpath": str(target.relative_to(fixtures_dir)),
                "requested_duration_sec": spec.duration_sec,
                "start_sec": spec.start_sec,
                **meta,
            }
        )

    manifest_path = fixtures_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"clips": manifest}, indent=2), encoding="utf-8")
    print(f"Wrote fixture manifest: {manifest_path}")


if __name__ == "__main__":
    main()
