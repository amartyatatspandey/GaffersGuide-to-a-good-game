"""
Download SoccerNet calibration and tracking task data (validation split).

Uses the SoccerNet pip package. Data is written to backend/data/soccernet.
Zips in calibration-2023 and tracking-2023 are extracted and deleted.
Password is read from SOCCERNET_PASSWORD env var with a default so it can be
supplied without interactive prompts. Test split is private; we use valid.
"""
import logging
import os
import zipfile
from pathlib import Path

from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default password for SoccerNet NDA/public data; override via SOCCERNET_PASSWORD.
DEFAULT_PASSWORD = "s0cc3rn3t"

# Tasks to download (SoccerNet downloadDataTask task names).
CALIBRATION_TASK = "calibration-2023"
TRACKING_TASK = "tracking-2023"


def extract_and_cleanup(directory: Path) -> None:
    """Find any .zip under directory, extract into same dir, delete zip to save space."""
    for zip_path in directory.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(path=zip_path.parent)
            zip_path.unlink()
            logger.info("Extracted and removed: %s", zip_path.name)
        except (zipfile.BadZipFile, OSError) as e:
            logger.warning("Skipped %s: %s", zip_path, e)


def get_soccernet_data_dir() -> Path:
    """Return backend/data/soccernet as an absolute path (relative to this script)."""
    return Path(__file__).resolve().parent.parent / "data" / "soccernet"


def download_calibration_and_tracking(
    local_dir: Path | str | None = None,
    splits: list[str] | None = None,
    password: str | None = None,
) -> None:
    """Download calibration and tracking task data from SoccerNet.

    Why: Populates backend/data/soccernet for football CV without interactive
    password prompts. Validation split (test is private for leaderboard).

    Args:
        local_dir: Directory to save data; defaults to backend/data/soccernet.
        splits: Dataset splits to download; defaults to ["valid"].
        password: SoccerNet NDA password; defaults to os.getenv("SOCCERNET_PASSWORD", DEFAULT_PASSWORD).
    """
    if local_dir is None:
        local_dir = get_soccernet_data_dir()
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    if splits is None:
        splits = ["valid"]
    if password is None:
        password = os.getenv("SOCCERNET_PASSWORD", DEFAULT_PASSWORD)

    config_path = os.path.expanduser("~/.SoccerNet/config.ini")
    if os.path.isfile(config_path):
        try:
            os.remove(config_path)
            logger.info("Removed cached credentials: %s", config_path)
        except OSError as e:
            logger.warning("Could not remove %s: %s", config_path, e)

    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    downloader.password = password
    logger.info("🔐 Credentials Hard-Reset and Applied")

    # Connectivity check: download one small public file (Labels-v2.json for one valid game).
    try:
        valid_games = getListGames("valid")
        if valid_games:
            downloader.downloadGame(
                game=valid_games[0],
                files=["Labels-v2.json"],
                spl="valid",
                verbose=False,
            )
            logger.info("Connectivity check passed (downloaded Labels-v2.json).")
        else:
            logger.warning("Connectivity check skipped (no valid games listed).")
    except Exception as e:
        raise RuntimeError(
            "Connectivity check failed. Check internet and SoccerNet library."
        ) from e

    logger.info(
        "⚠️ Test split is private. Downloading Validation split for development."
    )

    for task in (CALIBRATION_TASK, TRACKING_TASK):
        logger.info("Downloading task=%s splits=%s", task, splits)
        downloader.downloadDataTask(task=task, split=splits, password=password)
    logger.info("Done. Data under %s", local_dir)

    has_zip = any((local_dir / CALIBRATION_TASK).glob("*.zip")) or any(
        (local_dir / TRACKING_TASK).glob("*.zip")
    )
    has_extracted = False
    for split in splits:
        if (local_dir / CALIBRATION_TASK / split).is_dir() and any(
            (local_dir / CALIBRATION_TASK / split).iterdir()
        ):
            has_extracted = True
            break
        if (local_dir / TRACKING_TASK / split).is_dir() and any(
            (local_dir / TRACKING_TASK / split).iterdir()
        ):
            has_extracted = True
            break
    if not (has_zip or has_extracted):
        raise RuntimeError(
            "Download failed (401 or empty). Set SOCCERNET_PASSWORD and complete NDA at "
            "soccer-net.org."
        )

    extract_and_cleanup(local_dir)
    logger.info("✅ Extraction complete. Ready for training.")


if __name__ == "__main__":
    download_calibration_and_tracking()
