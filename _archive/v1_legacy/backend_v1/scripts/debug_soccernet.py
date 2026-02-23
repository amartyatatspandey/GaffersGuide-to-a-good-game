"""
Debug SoccerNet 401: test train split and legacy calibration, gather evidence for support.
"""
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

# Reduce noise from third-party libs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("SoccerNet").setLevel(logging.DEBUG)

DEFAULT_PASSWORD = "s0cc3rn3t"
OWNCLOUD_SERVER = "https://exrcsdrive.kaust.edu.sa/public.php/webdav/"


def main() -> None:
    local_dir = Path(__file__).resolve().parent.parent / "data" / "soccernet"
    local_dir.mkdir(parents=True, exist_ok=True)
    password = os.getenv("SOCCERNET_PASSWORD", DEFAULT_PASSWORD)

    config_path = os.path.expanduser("~/.SoccerNet/config.ini")
    if os.path.isfile(config_path):
        try:
            os.remove(config_path)
            logger.info("Removed cached credentials: %s", config_path)
        except OSError as e:
            logger.warning("Could not remove config: %s", e)

    from SoccerNet.Downloader import SoccerNetDownloader

    downloader = SoccerNetDownloader(LocalDirectory=str(local_dir))
    downloader.password = password

    results = []
    errors = []

    # Test 1: calibration-2023 train
    logger.info("Test 1: calibration-2023 split=['train']")
    try:
        downloader.downloadDataTask(
            task="calibration-2023",
            split=["train"],
            password=password,
            verbose=True,
        )
        has_zip = (local_dir / "calibration-2023" / "train.zip").is_file()
        results.append(("calibration-2023", "train", has_zip))
        if has_zip:
            logger.info("calibration-2023 train: SUCCESS (train.zip present)")
        else:
            logger.warning("calibration-2023 train: no train.zip after download")
            errors.append("calibration-2023 train: requested but train.zip not found")
    except Exception as e:
        results.append(("calibration-2023", "train", False))
        errors.append(f"calibration-2023 train: {type(e).__name__}: {e}")
        logger.exception("calibration-2023 train failed")

    # Test 2: legacy calibration (2022) train
    logger.info("Test 2: calibration (legacy 2022) split=['train']")
    try:
        downloader.downloadDataTask(
            task="calibration",
            split=["train"],
            password=password,
            verbose=True,
        )
        has_zip = (local_dir / "calibration" / "train.zip").is_file()
        results.append(("calibration", "train", has_zip))
        if has_zip:
            logger.info("calibration (legacy) train: SUCCESS (train.zip present)")
        else:
            logger.warning("calibration (legacy) train: no train.zip after download")
            errors.append("calibration (legacy) train: requested but train.zip not found")
    except Exception as e:
        results.append(("calibration", "train", False))
        errors.append(f"calibration (legacy) train: {type(e).__name__}: {e}")
        logger.exception("calibration (legacy) train failed")

    all_failed = all(not r[2] for r in results)
    if all_failed:
        out_dir = Path(__file__).resolve().parent.parent / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        draft_path = out_dir / "support_email_draft.txt"
        failed_tasks = [f"{t} split={s}" for t, s, ok in results if not ok]
        body = f"""Subject: 401 Unauthorized on SoccerNet calibration downloads (pip tool, NDA password)

Hello,

I completed the SoccerNet NDA and received the password for API/download access (as per your confirmation email). I am using the official pip package (SoccerNet) to download data, but all calibration download attempts return HTTP 401 Unauthorized.

Forensic details:
- Tool: pip install SoccerNet (official package)
- Password: As provided in NDA confirmation email (used via SOCCERNET_PASSWORD / script default)
- Server (OwnCloud): {OWNCLOUD_SERVER}
- Tasks/splits attempted: {', '.join(failed_tasks)}
- Error: HTTP 401 Unauthorized (repeated for each requested file)

URLs that would be requested (library builds these from server + filename):
- {OWNCLOUD_SERVER}train.zip (for calibration-2023 and calibration task, share-specific)
- Same server used for valid.zip, test.zip; all return 401 with the NDA password.

Main game labels (e.g. Labels-v2.json for a valid-split game) download successfully, so connectivity and the library work. Only the calibration (and tracking) task shares reject the credentials.

Could you confirm that the NDA password is enabled for the calibration and calibration-2023 task shares, and whether any extra step is required after NDA approval?

Thank you.
"""
        draft_path.write_text(body, encoding="utf-8")
        logger.info("All attempts failed. Draft email written to %s", draft_path)
        for err in errors:
            logger.error("%s", err)
        sys.exit(1)

    logger.info("Summary: %s", results)
    if errors:
        for err in errors:
            logger.warning("%s", err)


if __name__ == "__main__":
    main()
