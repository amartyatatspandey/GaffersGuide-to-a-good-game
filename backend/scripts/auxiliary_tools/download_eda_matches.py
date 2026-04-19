from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class Match(TypedDict):
    name: str
    url: str
    start: str
    end: str


MATCHES: list[Match] = [
    {
        "name": "01_barca_madrid",
        "url": "https://youtu.be/T2TAHYKo3UU",
        "start": "00:17:56",
        "end": "00:38:13",
    },
    {
        "name": "02_liverpool_mancity",
        "url": "https://youtu.be/XckOkRuE3-4",
        "start": "00:12:30",
        "end": "00:33:04",
    },
    {
        "name": "03_psg_intermilan",
        "url": "https://youtu.be/onpqXWqQqxQ",
        "start": "00:00:13",
        "end": "00:20:13",
    },
    {
        "name": "04_intermilan_atalanta",
        "url": "https://youtu.be/412v722hRDA",
        "start": "00:16:29",
        "end": "00:36:29",
    },
    {
        "name": "05_psg_arsenal",
        "url": "https://youtu.be/CbE8h0lvJ1s",
        "start": "00:08:05",
        "end": "00:28:05",
    },
    {
        "name": "06_realmadrid_liverpool",
        "url": "https://youtu.be/oXn0KPPHzuY",
        "start": "00:09:00",
        "end": "00:29:00",
    },
    {
        "name": "07_manu_spurs",
        "url": "https://youtu.be/emKNZKhefsc",
        "start": "01:02:19",
        "end": "01:22:19",
    },
    {
        "name": "08_czech_england",
        "url": "https://youtu.be/HOwLN4PhL10",
        "start": "00:11:13",
        "end": "00:21:13",
    },
    {
        "name": "09_belgium_italy",
        "url": "https://youtu.be/PQB22q_qJRc",
        "start": "00:09:03",
        "end": "00:29:03",
    },
    {
        "name": "10_barca_psg",
        "url": "https://youtu.be/5mTGRWX1Vcg",
        "start": "00:17:00",
        "end": "00:27:00",
    },
]


def clean_youtube_url_remove_playlist_params(url: str) -> str:
    """
    Remove `list` query parameter to avoid yt-dlp playlist expansion.
    """

    parts = urlsplit(url)
    if not parts.query:
        return url

    query_items = parse_qsl(parts.query, keep_blank_values=True)
    filtered = [(k, v) for (k, v) in query_items if k != "list"]

    new_query = urlencode(filtered, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def download_match_segment(
    match: Match,
    output_dir: Path,
) -> None:
    out_path = output_dir / f"{match['name']}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        logging.info("Skipping existing: %s", out_path.name)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_url = clean_youtube_url_remove_playlist_params(match["url"])

    # Download only the requested segment.
    # Note: yt-dlp expects the pattern without quotes, e.g. "*00:00:10-00:00:20".
    download_sections = f"*{match['start']}-{match['end']}"

    cmd: list[str] = [
        "yt-dlp",
        "--no-playlist",
        "--no-progress",
        "--download-sections",
        download_sections,
        "--format",
        "bestvideo+bestaudio/best",
        "--merge-output-format",
        "mp4",
        "--recode-video",
        "mp4",
        "-o",
        str(out_path),
        cleaned_url,
    ]

    logging.info("Downloading %s (%s - %s)", match["name"], match["start"], match["end"])
    subprocess.run(cmd, check=True)
    ensure_quicktime_compatible(out_path)


def ensure_quicktime_compatible(video_path: Path) -> None:
    """
    Re-encode to H.264 (video) + AAC (audio), the most reliable QuickTime combo.
    """

    temp_path = video_path.with_suffix(".qt.mp4")
    cmd: list[str] = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(temp_path),
    ]
    logging.info("Normalizing codecs for QuickTime: %s", video_path.name)
    subprocess.run(cmd, check=True)
    temp_path.replace(video_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    backend_dir = Path(__file__).resolve().parents[2]
    output_dir = backend_dir / "data" / "training_samples"

    failed_matches: list[str] = []
    for match in MATCHES:
        try:
            download_match_segment(match=match, output_dir=output_dir)
        except subprocess.CalledProcessError as exc:
            logging.exception("yt-dlp failed for %s", match["name"])
            failed_matches.append(match["name"])

    if failed_matches:
        logging.error("Failed matches: %s", ", ".join(failed_matches))
        return 1

    logging.info("All EDA match segments downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

