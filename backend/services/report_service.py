import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = BACKEND_ROOT / "data" / "reports"

_reports_cache = {}

class ReportService:
    @staticmethod
    def save_report(report_data: dict) -> str:
        """
        Saves a tactical report to disk with a timestamped filename.
        Returns the filename.
        """
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        # Use job_id if available, otherwise just timestamp
        job_id = report_data.get("job_id", "manual")
        filename = f"report_{job_id}_{timestamp}.json"
        
        file_path = REPORTS_DIR / filename
        
        # Add metadata if not present
        if "metadata" not in report_data:
            report_data["metadata"] = {
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "timestamp": timestamp,
                "version": "elite-v1"
            }
            
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4, ensure_ascii=False)
            
        return filename

    @staticmethod
    def list_reports() -> List[dict]:
        """
        Lists all saved reports with summary info.
        """
        if not REPORTS_DIR.exists():
            return []
            
        reports = []
        for p in REPORTS_DIR.iterdir():
            if p.is_file() and p.suffix == ".json":
                try:
                    stat = p.stat()
                    mtime = stat.st_mtime
                    size = stat.st_size
                    
                    cached = _reports_cache.get(p)
                    if cached and cached["mtime"] == mtime and cached["size"] == size:
                        reports.append(cached["summary"])
                        continue
                        
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                        # Extract summary info
                        summary_card = {}
                        for item in (data.get("advice_items") or []):
                            if item and item.get("flaw") == "Match Summary":
                                summary_card = item
                                break
                        
                        summary = {
                            "id": p.stem,
                            "filename": p.name,
                            "job_id": data.get("job_id", "unknown"),
                            "saved_at": (data.get("metadata") or {}).get("saved_at", "unknown"),
                            "video_title": data.get("video_title", "Unnamed Match"),
                            "quality_profile": data.get("quality_profile", "balanced"),
                            "win_probability": (summary_card.get("summary_data") or {}).get("win_probability", {}),
                            "tactical_power": {
                                "red": (summary_card.get("summary_data") or {}).get("team_0", {}).get("tactical_power", 0),
                                "blue": (summary_card.get("summary_data") or {}).get("team_1", {}).get("tactical_power", 0)
                            },
                            "flaw_count": len(data.get("advice_items") or []) - 1 # Exclude summary
                        }
                        
                        _reports_cache[p] = {
                            "mtime": mtime,
                            "size": size,
                            "summary": summary
                        }
                        reports.append(summary)
                except Exception as e:
                    print(f"Error reading report {p}: {e}")
                    
        # Sort by saved_at descending
        reports.sort(key=lambda x: x["saved_at"], reverse=True)
        return reports

    @staticmethod
    def get_report(report_id: str) -> Optional[dict]:
        """
        Retrieves a full report by its ID (filename stem) with fallback checks.
        """
        # 1. Try direct match
        file_path = REPORTS_DIR / f"{report_id}.json"
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        # 2. Try prefix/suffix pattern search in reports directory
        if REPORTS_DIR.exists():
            for p in REPORTS_DIR.iterdir():
                if p.is_file() and p.suffix == ".json":
                    if p.stem == report_id or f"report_{report_id}_" in p.name:
                        with open(p, "r", encoding="utf-8") as f:
                            return json.load(f)

        # 3. Try fallback to outputs directory
        output_path = BACKEND_ROOT / "output" / f"{report_id}_report.json"
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # 4. Try fallback to enriched output format
        output_enriched = BACKEND_ROOT / "output" / f"{report_id}_report_enriched.json"
        if output_enriched.exists():
            with open(output_enriched, "r", encoding="utf-8") as f:
                cards = json.load(f)
                return {
                    "job_id": report_id,
                    "video_title": "Analyzed Match",
                    "advice_items": cards
                }

        return None

    @staticmethod
    def delete_report(report_id: str) -> bool:
        """
        Deletes a report.
        """
        file_path = REPORTS_DIR / f"{report_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
