"""Download SoccerNet ReID dataset zips (train/valid/test/challenge) using SoccerNetDownloader.

This script is a thin, robust wrapper around the official instructions in
`backend/references/sn-reid/README.md` and the `SoccerNet` pip package:

    from SoccerNet.Downloader import SoccerNetDownloader
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="/path/to/project/datasets/soccernetv3")
    mySoccerNetDownloader.downloadDataTask(task="reid", split=["train", "valid", "test", "challenge"])

Here we:
- Resolve paths explicitly relative to `backend/` (script location).
- Download the ReID zips to `backend/data/soccernet_reid/soccernetv3/reid/`.
- Optionally unzip each split to match the expected layout:
    backend/data/soccernet_reid/soccernetv3/reid/{train, valid, test, challenge}

Note: The official sn-reid repo does not provide separate pre-trained baseline
weights via the `SoccerNetDownloader` API. This script focuses on downloading
the ReID dataset itself; baseline weights should be trained or fetched
following the sn-reid documentation if needed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

try:
    from SoccerNet.Downloader import SoccerNetDownloader
except ImportError as exc:  # pragma: no cover - runtime environment concern
    SoccerNetDownloader = None  # type: ignore[assignment]
    _SOCCERNET_IMPORT_ERROR: Exception | None = exc
else:
    _SOCCERNET_IMPORT_ERROR = None


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_ROOT = BACKEND_ROOT / "data" / "soccernet_reid"


def _ensure_soccernet_available() -> None:
    """Raise a clear error if the SoccerNet package is not installed."""

    if _SOCCERNET_IMPORT_ERROR is not None or SoccerNetDownloader is None:
        logger.error(
            "SoccerNet package is not installed. "
            "Install it with `pip install SoccerNet` in your active environment."
        )
        raise RuntimeError("Missing dependency: SoccerNet") from _SOCCERNET_IMPORT_ERROR


def _extract_zip_files(reid_root: Path, splits: Iterable[str]) -> None:
    """Extract downloaded ReID zip files into split-specific folders.

    This follows the official sn-reid layout:
    - Zips are located at:   <reid_root>/{split}.zip
    - Extracted folders at:  <reid_root>/{split}/

    Args:
        reid_root: Directory where `reid/*.zip` files are stored.
        splits: Iterable of split names (e.g. ["train", "valid", "test", "challenge"]).
    """

    for split in splits:
        zip_path = reid_root / f"{split}.zip"
        target_dir = reid_root / split

        if not zip_path.exists():
            logger.warning("Expected zip for split '%s' not found at %s", split, zip_path)
            continue

        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info("Split '%s' already extracted at %s; skipping unzip.", split, target_dir)
            continue

        logger.info("Extracting %s into %s ...", zip_path, target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
        except zipfile.BadZipFile as exc:
            logger.error("Failed to extract %s (corrupted zip?): %s", zip_path, exc)
            raise


def download_reid_data(
    save_root: Path,
    splits: Sequence[str] | None = None,
    unzip: bool = True,
) -> None:
    """Download SoccerNet ReID dataset zips (and optionally unzip them).

    Args:
        save_root: Base directory under which the Soccernet ReID data should live.
            The effective downloader directory will be:
                save_root / "soccernetv3"
            and ReID zips will be written under:
                save_root / "soccernetv3" / "reid" / {train, valid, test, challenge}.zip
        splits: Sequence of splits to download. Defaults to all:
            ["train", "valid", "test", "challenge"].
        unzip: If True, automatically extract each zip into a folder of the same name
            under `.../reid/` (recommended).
    """

    _ensure_soccernet_available()

    resolved_root = save_root.resolve()
    sn_local_dir = resolved_root / "soccernetv3"
    reid_root = sn_local_dir / "reid"
    reid_root.mkdir(parents=True, exist_ok=True)

    splits = list(splits or ["train", "valid", "test", "challenge"])

    logger.info("Using SoccerNetDownloader with LocalDirectory=%s", sn_local_dir)
    logger.info("Requested ReID splits: %s", ", ".join(splits))

    try:
        downloader = SoccerNetDownloader(LocalDirectory=str(sn_local_dir))
        downloader.downloadDataTask(task="reid", split=splits)
    except Exception as exc:  # pragma: no cover - defensive against network/API errors
        logger.exception("Error while downloading SoccerNet ReID data: %s", exc)
        raise

    if unzip:
        logger.info("Unzipping downloaded ReID archives into %s", reid_root)
        _extract_zip_files(reid_root=reid_root, splits=splits)

    logger.info(
        "SoccerNet ReID data is available under: %s",
        reid_root,
    )
    logger.info(
        "Expected structure: %s",
        "(train/, valid/, test/, challenge/ with player thumbnails)",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the downloader script."""

    parser = argparse.ArgumentParser(
        description=(
            "Download the SoccerNet ReID dataset using SoccerNetDownloader, "
            "storing data under backend/data/soccernet_reid/ by default."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "Base directory where Soccernet ReID data will be stored. "
            "Within this directory, the downloader will create 'soccernetv3/reid/'. "
            f"Default: {DEFAULT_DATA_ROOT}"
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test", "challenge"],
        choices=["train", "valid", "test", "challenge"],
        help="Which ReID splits to download. Default: train valid test challenge",
    )
    parser.add_argument(
        "--skip-unzip",
        action="store_true",
        help="If set, do not unzip the downloaded archives (leave only .zip files).",
    )

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for CLI execution."""

    args = _parse_args(argv)
    output_dir = Path(args.output_dir)

    try:
        download_reid_data(
            save_root=output_dir,
            splits=args.splits,
            unzip=not args.skip_unzip,
        )
    except Exception:
        logger.exception("Failed to download SoccerNet ReID dataset.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

