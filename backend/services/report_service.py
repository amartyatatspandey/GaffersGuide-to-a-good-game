"""
Report Service — persistent tactical report storage backed by GCS.

Public interface is identical to the original filesystem-based implementation.
All callers in main.py are unmodified.

Storage layout (GCS):
    gs://{GCS_BUCKET_NAME}/reports/{filename}   (e.g. report_{job_id}_{ts}.json)

When GCS_ENABLED=false (local dev) gcs_service falls back to the local filesystem
under /tmp/gaffer_gcs_fallback/reports/ automatically.
"""

import json
import logging
import time
from typing import Any, List, Optional

from services import gcs_service

LOGGER = logging.getLogger(__name__)

# In-process cache: blob_name → {"etag": str, "summary": dict}
# Avoids re-downloading unchanged blobs on list_reports() calls.
_reports_cache: dict[str, dict] = {}


class ReportService:
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # save_report
    # ------------------------------------------------------------------
    @staticmethod
    def save_report(report_data: dict, user_id: str) -> str:
        """
        Saves a tactical report to GCS with a timestamped filename and registers it in SQLite.
        Returns the filename (without path prefix).
        """
        timestamp = int(time.time())
        job_id = report_data.get("job_id", "manual")
        filename = f"report_{job_id}_{timestamp}.json"

        # Attach metadata if missing
        if "metadata" not in report_data:
            report_data["metadata"] = {}
        
        report_data["metadata"]["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report_data["metadata"]["timestamp"] = timestamp
        report_data["metadata"]["version"] = "elite-v1"
        report_data["metadata"]["user_id"] = user_id

        # Write to GCS
        blob_name = gcs_service.report_blob_name(filename)
        gcs_service.write_json(blob_name, report_data)
        LOGGER.info("Report saved to GCS: %s", blob_name)

        # Register in SQLite reports table for fast multi-tenant queries
        from services.db_service import DatabaseService
        conn = DatabaseService.get_connection()
        try:
            with conn:
                # Get flaw count
                flaws = len(report_data.get("advice_items") or []) - 1
                if flaws < 0:
                    flaws = 0

                conn.execute("""
                INSERT INTO reports (id, match_id, user_id, filename, video_title, flaw_count)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    video_title = excluded.video_title,
                    flaw_count = excluded.flaw_count;
                """, (
                    filename.replace(".json", ""),
                    job_id,
                    user_id,
                    filename,
                    report_data.get("video_title") or f"Analyzed Match ({job_id[:8]})",
                    flaws
                ))
                LOGGER.info("Report registered in SQLite: %s (User: %s)", filename, user_id)
        except Exception as db_err:
            LOGGER.error("Failed to register report in SQLite: %s", db_err)
        finally:
            conn.close()

        return filename

    # ------------------------------------------------------------------
    # list_reports
    # ------------------------------------------------------------------
    @staticmethod
    def list_reports(user_id: str) -> List[dict]:
        """
        Lists all saved reports belonging to a specific user using SQLite.
        """
        from services.db_service import DatabaseService
        conn = DatabaseService.get_connection()
        reports: List[dict] = []
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.*, m.quality_profile, 
                       mm.tactical_power_red, mm.tactical_power_blue,
                       mm.compactness_red, mm.compactness_blue,
                       mm.transition_speed_red, mm.transition_speed_blue
                FROM reports r
                JOIN matches m ON r.match_id = m.id
                LEFT JOIN match_metrics mm ON r.match_id = mm.match_id
                WHERE r.user_id = ?
                ORDER BY r.created_at DESC
            """, (user_id,))
            rows = cursor.fetchall()
            for row in rows:
                r_dict = dict(row)
                reports.append({
                    "id": r_dict["id"],
                    "filename": r_dict["filename"],
                    "job_id": r_dict["match_id"],
                    "saved_at": r_dict["created_at"],
                    "video_title": r_dict["video_title"],
                    "quality_profile": r_dict["quality_profile"] or "balanced",
                    "win_probability": {},
                    "tactical_power": {
                        "red": r_dict["tactical_power_red"] or 0.0,
                        "blue": r_dict["tactical_power_blue"] or 0.0,
                    },
                    "flaw_count": r_dict["flaw_count"] or 0,
                })
        except Exception as exc:
            LOGGER.error("Error listing reports from SQLite: %s", exc)
        finally:
            conn.close()

        return reports

    # ------------------------------------------------------------------
    # get_report
    # ------------------------------------------------------------------
    @staticmethod
    def get_report(report_id: str, user_id: str) -> Optional[dict]:
        """
        Retrieves a full report verifying ownership.
        """
        from services.db_service import DatabaseService
        conn = DatabaseService.get_connection()
        filename = None
        try:
            cursor = conn.cursor()
            # Verify owner in DB first
            cursor.execute("SELECT filename FROM reports WHERE id = ? AND user_id = ?", (report_id, user_id))
            row = cursor.fetchone()
            if row:
                filename = row["filename"]
            else:
                # Try finding by match_id
                cursor.execute("SELECT filename FROM reports WHERE match_id = ? AND user_id = ?", (report_id, user_id))
                row = cursor.fetchone()
                if row:
                    filename = row["filename"]
        except Exception as exc:
            LOGGER.error("DB check failed for get_report: %s", exc)
        finally:
            conn.close()

        if not filename:
            # Try directly loading from GCS and verify the metadata.user_id (fallback for unmigrated reports)
            blob_name = gcs_service.report_blob_name(f"{report_id}.json")
            data = gcs_service.read_json(blob_name)
            if data and (data.get("metadata") or {}).get("user_id") == user_id:
                return data
            
            # Pattern search
            all_blobs = gcs_service.list_blobs(gcs_service.GCS_REPORTS_PREFIX + "/")
            for name in all_blobs:
                fn = name.split("/")[-1]
                stem = fn[: -len(".json")] if fn.endswith(".json") else fn
                if stem == report_id or f"report_{report_id}_" in fn:
                    data = gcs_service.read_json(name)
                    if data and (data.get("metadata") or {}).get("user_id") == user_id:
                        return data
            return None

        # Fetch report from GCS securely
        blob_name = gcs_service.report_blob_name(filename)
        return gcs_service.read_json(blob_name)

    # ------------------------------------------------------------------
    # delete_report
    # ------------------------------------------------------------------
    @staticmethod
    def delete_report(report_id: str, user_id: str) -> bool:
        """
        Deletes a report from GCS and SQLite verifying ownership.
        """
        from services.db_service import DatabaseService
        conn = DatabaseService.get_connection()
        filename = None
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT filename FROM reports WHERE id = ? AND user_id = ?", (report_id, user_id))
            row = cursor.fetchone()
            if row:
                filename = row["filename"]
            
            with conn:
                conn.execute("DELETE FROM reports WHERE id = ? AND user_id = ?", (report_id, user_id))
        except Exception as exc:
            LOGGER.error("DB check failed for delete_report: %s", exc)
        finally:
            conn.close()

        if not filename:
            return False

        blob_name = gcs_service.report_blob_name(filename)
        deleted = gcs_service.delete_blob(blob_name)
        if deleted:
            _reports_cache.pop(blob_name, None)
            return True
        return False

