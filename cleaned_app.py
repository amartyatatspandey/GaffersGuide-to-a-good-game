from flask import Flask, render_template, request, redirect, url_for, abort, send_file, send_from_directory
from pathlib import Path
import json
import os

# -------------------------------------------------
# PROJECT ROOT & MATCH DIRECTORY
# -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
MATCHES_DIR = REPO_ROOT / "opendata" / "data" / "matches"

# -------------------------------------------------
# MODEL IMPORTS
# -------------------------------------------------
from models.model0_load_data import load_match
from models.model1 import model_main as run_model1
from models.model2 import model_main as run_model2
from models.model3 import model_main as run_model3

app = Flask(__name__)

# -------------------------------------------------
# SAFE SERVING OF MODEL OUTPUTS
# -------------------------------------------------
@app.route("/match_output/<path:subpath>")
def match_output(subpath):
    safe_base = REPO_ROOT / "opendata" / "data" / "matches"
    full = safe_base / subpath

    try:
        full = full.resolve()
        if not str(full).startswith(str(safe_base.resolve())):
            abort(404, description="Path traversal detected")
    except Exception:
        abort(404)

    if not full.exists():
        abort(404, description=f"File not found: {subpath}")

    return send_file(full)


# -------------------------------------------------
# HOME PAGE
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


# -------------------------------------------------
# LIST MATCHES
# -------------------------------------------------
@app.route("/matches")
def matches():
    match_dirs = [d for d in MATCHES_DIR.iterdir() if d.is_dir()]

    match_list = []
    for m in match_dirs:
        meta_files = list(m.glob("*match.json"))
        if not meta_files:
            continue

        try:
            meta = json.loads(meta_files[0].read_text())
            match_list.append({
                "id": m.name,
                "home": meta.get("home_team", {}).get("name", "Home"),
                "away": meta.get("away_team", {}).get("name", "Away")
            })
        except:
            match_list.append({
                "id": m.name,
                "home": "Unknown",
                "away": "Unknown"
            })

    return render_template("matches.html", matches=match_list)


# -------------------------------------------------
# MATCH PAGE
# -------------------------------------------------
@app.route("/match/<match_id>")
def match_page(match_id):
    match_dir = MATCHES_DIR / match_id
    meta_file = list(match_dir.glob("*match.json"))[0]

    meta = json.loads(meta_file.read_text())

    return render_template(
        "match_detail.html",
        match_id=match_id,
        home=meta.get("home_team", {}).get("name", "Home"),
        away=meta.get("away_team", {}).get("name", "Away")
    )


# -------------------------------------------------
# MODEL 1 — WORKING VERSION
# -------------------------------------------------
@app.route("/run_model1/<match_id>")
def do_model_1(match_id):
    match_dir = MATCHES_DIR / match_id

    tracking = list(match_dir.glob("*tracking_extrapolated.jsonl"))
    if not tracking:
        abort(500, description=f"Tracking file not found in {match_dir}")

    result = run_model1(str(match_dir))

    cleaned = []
    for p in result["plots"]:
        p = Path(p)
        rel = p.relative_to(REPO_ROOT / "opendata" / "data" / "matches")
        cleaned.append(f"/match_output/{rel}")

    cleaned = sorted(cleaned)
    pairs = []

    def side_from_name(path):
        low = path.lower()
        if "_a" in low or "team_a" in low or "teama" in low:
            return "A"
        if "_b" in low or "team_b" in low or "teamb" in low:
            return "B"
        return None

    def base_title(path):
        name = path.split("/")[-1]
        name = name.replace(".png", "").replace(".jpg", "")
        for token in ["_A", "_B", "teamA", "teamB", "TeamA", "TeamB"]:
            name = name.replace(token, "")
        return name.strip()

    bucket = {}
    for img in cleaned:
        side = side_from_name(img)
        title = base_title(img)

        if title not in bucket:
            bucket[title] = {"A": None, "B": None}

        if side == "A":
            bucket[title]["A"] = img
        elif side == "B":
            bucket[title]["B"] = img

    for title, sides in bucket.items():
        pairs.append({
            "title": title.replace("_", " ").title(),
            "A": sides["A"],
            "B": sides["B"]
        })

    return render_template("results_model1.html", match_id=match_id, pairs=pairs)


# -------------------------------------------------
# MODEL 2 — WORKING ORIGINAL VERSION (YOU WANT THIS)
# -------------------------------------------------
@app.route("/run_model2/<match_id>")
def do_model_2(match_id):
    try:
        match_folder = os.path.join("opendata", "data", "matches", match_id)

        if not os.path.exists(match_folder):
            return f"Match folder not found: {match_folder}", 404

        output = run_model2(match_id, match_folder)

        return render_template(
            "results_model2.html",
            match_id=match_id,
            total_distance_A=output["total_distance_A"],
            total_distance_B=output["total_distance_B"],
            sprint_A=output["sprint_A"],
            sprint_B=output["sprint_B"],
            fatigue_A=output["fatigue_A"],
            fatigue_B=output["fatigue_B"],
            plots=output.get("served_plots", [])
        )

    except Exception as e:
        print("MODEL 2 FAILED:", e)
        return f"Error running Model 2: {str(e)}", 500

# -------------------------------------------------
# MODEL 3 — WORKING VERSION
# -------------------------------------------------
@app.route("/run_model3/<match_id>")
def do_model_3(match_id):
    match_dir = MATCHES_DIR / match_id

    result = run_model3(str(match_dir))
    served = result.get("served_plots", [])

    cleaned = []
    for p in served:
        if str(p).startswith("/static/"):
            cleaned.append(p)
            continue

        p_path = Path(p)
        try:
            rel = p_path.relative_to(REPO_ROOT / "opendata" / "data" / "matches")
            cleaned.append(f"/match_output/{rel.as_posix()}")
        except:
            try:
                rel = p_path.relative_to(match_dir)
                cleaned.append(f"/match_output/{match_id}/{rel.as_posix()}")
            except:
                cleaned.append(p)

    cleaned = sorted(cleaned)
    pairs = []

    def side_from_name(path):
        low = path.lower()
        if "_a" in low or "team_a" in low or "teama" in low:
            return "A"
        if "_b" in low or "team_b" in low or "teamb" in low:
            return "B"
        return None

    def base_title(path):
        name = Path(path).name
        name = name.replace(".png", "").replace(".jpg", "")
        for s in ["_A", "_B", "TeamA", "TeamB", "teamA", "teamB"]:
            name = name.replace(s, "")
        return name.strip()

    bucket = {}
    for img in cleaned:
        side = side_from_name(img)
        title = base_title(img)

        if title not in bucket:
            bucket[title] = {"A": None, "B": None}

        if side == "A":
            bucket[title]["A"] = img
        elif side == "B":
            bucket[title]["B"] = img

    for title, sides in bucket.items():
        pairs.append({
            "title": title.replace("_", " ").title(),
            "A": sides["A"],
            "B": sides["B"],
        })

    return render_template("results_model3.html", match_id=match_id, pairs=pairs)


if __name__ == "__main__":
    app.run()