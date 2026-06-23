import sqlite3
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_ROOT / "data" / "gaffer.db"
REPORTS_DIR = BACKEND_ROOT / "data" / "reports"
UPLOADS_DIR = BACKEND_ROOT / "data" / "uploads"

class DatabaseService:
    @staticmethod
    def get_connection() -> sqlite3.Connection:
        """Returns a sqlite3 connection with Row factory enabled."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @classmethod
    def init_db(cls):
        """Initializes the database schema, inserts default records, and migrates old reports."""
        LOGGER.info("Initializing database at %s", DB_PATH)
        conn = cls.get_connection()
        try:
            with conn:
                # 1. Create tables
                conn.execute("""
                CREATE TABLE IF NOT EXISTS accounts (
                    id VARCHAR(36) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id VARCHAR(36) PRIMARY KEY,
                    account_id VARCHAR(36) NOT NULL,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    role VARCHAR(50) DEFAULT 'analyst',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );
                """)
                
                conn.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id VARCHAR(36) PRIMARY KEY,
                    account_id VARCHAR(36) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    video_filename VARCHAR(255) NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_date TIMESTAMP,
                    cv_engine VARCHAR(50) NOT NULL,
                    llm_engine VARCHAR(50) NOT NULL,
                    quality_profile VARCHAR(50) NOT NULL,
                    chunking_interval VARCHAR(100) NOT NULL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES accounts(id)
                );
                """)
                
                conn.execute("""
                CREATE TABLE IF NOT EXISTS match_metrics (
                    match_id VARCHAR(36) PRIMARY KEY,
                    tactical_power_red FLOAT DEFAULT 0.0,
                    tactical_power_blue FLOAT DEFAULT 0.0,
                    compactness_red FLOAT DEFAULT 0.0,
                    compactness_blue FLOAT DEFAULT 0.0,
                    transition_speed_red FLOAT DEFAULT 0.0,
                    transition_speed_blue FLOAT DEFAULT 0.0,
                    flaw_count INTEGER DEFAULT 0,
                    FOREIGN KEY (match_id) REFERENCES matches(id) ON DELETE CASCADE
                );
                """)

                # 2. Insert default tenant (Default Team account & user)
                conn.execute("""
                INSERT OR IGNORE INTO accounts (id, name)
                VALUES ('default-account-id', 'Default Team');
                """)
                
                conn.execute("""
                INSERT OR IGNORE INTO users (id, account_id, email, role)
                VALUES ('default-user-id', 'default-account-id', 'coach@gaffersguide.com', 'coach');
                """)
            
            # 3. Migrate existing reports to DB
            with conn:
                cls._migrate_legacy_reports(conn)
            
        except Exception as e:
            LOGGER.exception("Failed to initialize database: %s", e)
        finally:
            conn.close()

    @classmethod
    def _migrate_legacy_reports(cls, conn: sqlite3.Connection):
        """Scans both reports and outputs folder and imports reports into SQLite."""
        LOGGER.info("Starting legacy reports migration...")
        migrated_count = 0
        
        # Helper to validate hex UUIDs
        def is_valid_job_id(jid: str) -> bool:
            return len(jid) == 32 and all(c in "0123456789abcdef" for c in jid.lower())
            
        candidates: List[tuple[Path, str]] = []
        
        # 1. Scan reports directory
        if REPORTS_DIR.exists():
            for p in REPORTS_DIR.iterdir():
                if p.is_file() and p.suffix == ".json":
                    candidates.append((p, "reports"))
                    
        # 2. Scan outputs directory
        output_dir = BACKEND_ROOT / "output"
        if output_dir.exists():
            for p in output_dir.iterdir():
                if p.is_file() and p.name.endswith("_report.json"):
                    jid = p.name.replace("_report.json", "")
                    if is_valid_job_id(jid):
                        candidates.append((p, "output"))

        for p, source in candidates:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json_load_fail_safe(f)
                
                if not data or not isinstance(data, list):
                    continue
                
                # Scan list elements for job_id and video_title
                job_id = None
                video_title = None
                for item in data:
                    if isinstance(item, dict):
                        if not job_id and item.get("job_id"):
                            job_id = item["job_id"]
                        if not video_title and item.get("video_title"):
                            video_title = item["video_title"]
                
                # Fallback parsed job_id from filename if not found in list items
                if not job_id or job_id == "unknown" or job_id == "manual":
                    if source == "output":
                        job_id = p.name.replace("_report.json", "")
                    else:
                        parts = p.stem.split("_")
                        if len(parts) >= 2 and is_valid_job_id(parts[1]):
                            job_id = parts[1]
                        else:
                            job_id = p.stem
                
                if not job_id or not is_valid_job_id(job_id):
                    continue

                # Check if match already in DB
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM matches WHERE id = ?", (job_id,))
                if cursor.fetchone():
                    continue
                
                # Locate or infer video file
                if not video_title:
                    video_title = f"Analyzed Match ({job_id[:8]})"
                    
                video_filename = f"{job_id}.mp4"
                if UPLOADS_DIR.exists():
                    for upload in UPLOADS_DIR.iterdir():
                        if upload.stem == job_id:
                            video_filename = upload.name
                            break
                
                # Extract metadata
                saved_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                cv_engine = "cloud"
                llm_engine = "cloud"
                quality_profile = "balanced"
                chunking_interval = "15-minute intervals"
                
                for item in data:
                    if isinstance(item, dict) and item.get("metadata"):
                        meta = item["metadata"]
                        saved_at = meta.get("saved_at") or saved_at
                        cv_engine = meta.get("cv_engine") or cv_engine
                        llm_engine = meta.get("llm_engine") or llm_engine
                        quality_profile = meta.get("quality_profile") or quality_profile
                        chunking_interval = meta.get("chunking_interval") or chunking_interval
                        break

                # Extract performance metrics
                metrics = cls._extract_metrics(data)
                flaws = len(data) - 1
                if flaws < 0:
                    flaws = 0
                
                # Insert match record
                conn.execute("""
                INSERT INTO matches (
                    id, account_id, name, status, video_filename, 
                    upload_date, analysis_date, cv_engine, llm_engine, 
                    quality_profile, chunking_interval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    job_id,
                    'default-account-id',
                    video_title,
                    'completed',
                    video_filename,
                    saved_at,
                    saved_at,
                    cv_engine,
                    llm_engine,
                    quality_profile,
                    chunking_interval
                ))

                # Insert match metrics
                conn.execute("""
                INSERT INTO match_metrics (
                    match_id, tactical_power_red, tactical_power_blue, 
                    compactness_red, compactness_blue, 
                    transition_speed_red, transition_speed_blue, flaw_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    job_id,
                    metrics["tactical_power_red"],
                    metrics["tactical_power_blue"],
                    metrics["compactness_red"],
                    metrics["compactness_blue"],
                    metrics["transition_speed_red"],
                    metrics["transition_speed_blue"],
                    flaws
                ))
                
                migrated_count += 1
            except Exception as e:
                LOGGER.warning("Error migrating report file %s: %s", p.name, e)
        
        if migrated_count > 0:
            LOGGER.info("Migrated %d legacy reports to database.", migrated_count)

    @staticmethod
    def _extract_metrics(data: list) -> dict:
        """Parses performance metrics from the Match Summary card item."""
        summary_card = {}
        for item in data:
            if isinstance(item, dict) and item.get("flaw") == "Match Summary":
                summary_card = item
                break
                
        summary_data = summary_card.get("summary_data") or {}
        
        tp_red = summary_data.get("team_0", {}).get("tactical_power", 0.0)
        tp_blue = summary_data.get("team_1", {}).get("tactical_power", 0.0)
        comp_red = summary_data.get("team_0", {}).get("compactness", 0.0)
        comp_blue = summary_data.get("team_1", {}).get("compactness", 0.0)
        trans_red = summary_data.get("team_0", {}).get("transition_speed", 0.0)
        trans_blue = summary_data.get("team_1", {}).get("transition_speed", 0.0)
        
        # Fallback to parsing from evidence string
        evidence = summary_card.get("evidence", "")
        import re
        
        if tp_red == 0.0 and tp_blue == 0.0:
            tp_match = re.search(r"Tactical Power:\s*Red\s*(\d+(?:\.\d+)?)\.?\s*vs\s*Blue\s*(\d+(?:\.\d+)?)\.?", evidence)
            if tp_match:
                tp_red = float(tp_match.group(1).rstrip('.'))
                tp_blue = float(tp_match.group(2).rstrip('.'))
                
        if comp_red == 0.0 and comp_blue == 0.0:
            comp_match = re.search(r"Compactness:\s*Red\s*(\d+(?:\.\d+)?)\.?\s*/\s*Blue\s*(\d+(?:\.\d+)?)\.?", evidence)
            if comp_match:
                comp_red = float(comp_match.group(1).rstrip('.'))
                comp_blue = float(comp_match.group(2).rstrip('.'))
                
        if trans_red == 0.0 and trans_blue == 0.0:
            trans_match = re.search(r"Transition Speed:\s*Red\s*(\d+(?:\.\d+)?)\.?\s*/\s*Blue\s*(\d+(?:\.\d+)?)\.?", evidence)
            if trans_match:
                trans_red = float(trans_match.group(1).rstrip('.'))
                trans_blue = float(trans_match.group(2).rstrip('.'))
                
        return {
            "tactical_power_red": tp_red,
            "tactical_power_blue": tp_blue,
            "compactness_red": comp_red,
            "compactness_blue": comp_blue,
            "transition_speed_red": trans_red,
            "transition_speed_blue": trans_blue
        }

    @classmethod
    def list_matches(
        cls, 
        search: Optional[str] = None, 
        sort_by: str = "newest", 
        limit: int = 100, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lists matches with cached metrics from the DB."""
        conn = cls.get_connection()
        try:
            query = """
                SELECT m.*, 
                       mm.tactical_power_red, mm.tactical_power_blue,
                       mm.compactness_red, mm.compactness_blue,
                       mm.transition_speed_red, mm.transition_speed_blue,
                       mm.flaw_count
                FROM matches m
                LEFT JOIN match_metrics mm ON m.id = mm.match_id
            """
            params = []
            
            if search:
                query += " WHERE m.name LIKE ? OR m.upload_date LIKE ?"
                search_val = f"%{search}%"
                params.extend([search_val, search_val])
            
            if sort_by == "oldest":
                query += " ORDER BY m.upload_date ASC"
            else:
                query += " ORDER BY m.upload_date DESC"
                
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    @classmethod
    def create_match(
        cls, 
        match_id: str, 
        name: str, 
        video_filename: str, 
        cv_engine: str, 
        llm_engine: str, 
        quality_profile: str, 
        chunking_interval: str
    ):
        """Inserts a new match record into the database."""
        conn = cls.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT INTO matches (
                    id, account_id, name, status, video_filename,
                    cv_engine, llm_engine, quality_profile, chunking_interval
                ) VALUES (?, 'default-account-id', ?, 'pending', ?, ?, ?, ?, ?);
                """, (match_id, name, video_filename, cv_engine, llm_engine, quality_profile, chunking_interval))
        finally:
            conn.close()

    @classmethod
    def update_match_status(cls, match_id: str, status: str, error: Optional[str] = None):
        """Updates match status, setting analysis date if complete."""
        conn = cls.get_connection()
        try:
            with conn:
                if status == "done":
                    conn.execute("""
                    UPDATE matches 
                    SET status = ?, error_message = NULL, analysis_date = CURRENT_TIMESTAMP
                    WHERE id = ?;
                    """, (status, match_id))
                else:
                    conn.execute("""
                    UPDATE matches 
                    SET status = ?, error_message = ?
                    WHERE id = ?;
                    """, (status, error, match_id))
        finally:
            conn.close()

    @classmethod
    def update_match_metrics(cls, match_id: str, metrics: Dict[str, Any]):
        """Inserts or replaces metrics for a match."""
        conn = cls.get_connection()
        try:
            with conn:
                conn.execute("""
                INSERT OR REPLACE INTO match_metrics (
                    match_id, tactical_power_red, tactical_power_blue,
                    compactness_red, compactness_blue,
                    transition_speed_red, transition_speed_blue, flaw_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    match_id,
                    metrics.get("tactical_power_red", 0.0),
                    metrics.get("tactical_power_blue", 0.0),
                    metrics.get("compactness_red", 0.0),
                    metrics.get("compactness_blue", 0.0),
                    metrics.get("transition_speed_red", 0.0),
                    metrics.get("transition_speed_blue", 0.0),
                    metrics.get("flaw_count", 0)
                ))
        finally:
            conn.close()

    @classmethod
    def delete_match(cls, match_id: str) -> bool:
        """Deletes match from DB."""
        conn = cls.get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM matches WHERE id = ?;", (match_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()


def json_load_fail_safe(f) -> Optional[dict]:
    """Graceful JSON decoder helper."""
    import json
    try:
        return json.load(f)
    except Exception:
        return None
