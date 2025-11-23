# batch_classify.py
"""Batch classify matches using model3.classify_all.

Run:
    python batch_classify.py

Outputs:
 - batch_classification.csv
 - classification_results/<match_id>.json
"""

import json
import csv
import glob
import os
from pathlib import Path
from typing import List, Dict, Any

# Try to import your adapter(s)
candidate_modules = ["model0_load_data", "model0"]
model0 = None
for mname in candidate_modules:
    try:
        model0 = __import__(mname)
        print(f"Using feature provider module: {mname}")
        break
    except Exception:
        model0 = None

# Import model3 (classification engine)
import model3

# Helper: validation
REQUIRED_KEYS = {
    "attacking_focal", "attacking_focal_value", "attacking_presence",
    "midfield_focal", "midfield_presence",
    "defensive_focal", "defensive_presence",
}

def validate_feature_dict(d: Dict[str, Any]) -> bool:
    missing = REQUIRED_KEYS - set(d.keys())
    return len(missing) == 0

def load_matches_from_model0() -> List[Dict[str, Any]]:
    """Try multiple provider names inside the module to get list/dict of matches."""
    if model0 is None:
        return []
    # 1) try get_all_matches / get_matches
    for fn in ("get_all_matches", "get_matches", "load_all_matches"):
        if hasattr(model0, fn) and callable(getattr(model0, fn)):
            out = getattr(model0, fn)()
            if isinstance(out, list):
                return out
            if isinstance(out, dict):
                return [out]

    # 2) try FEATURES_LIST or FEATURES
    for var in ("FEATURES_LIST", "features_list", "FEATURES", "features"):
        if hasattr(model0, var):
            val = getattr(model0, var)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return [val]

    # 3) try single get_match_features() and wrap
    for fn in ("get_match_features", "get_features", "extract_features", "load_match", "load_features"):
        if hasattr(model0, fn) and callable(getattr(model0, fn)):
            single = getattr(model0, fn)()
            if isinstance(single, list):
                return single
            if isinstance(single, dict):
                return [single]

    return []

def load_matches_from_opedata_dir(opedir: Path = Path("opedata")) -> List[Dict[str, Any]]:
    """Load all JSON files in opedata/ or a CSV matches.csv"""
    matches = []
    if not opedir.exists():
        return matches

    # JSON files
    for p in sorted(opedir.glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if isinstance(d, list):
                matches.extend(d)
            else:
                matches.append(d)
        except Exception as e:
            print(f"Skipping {p} (json parse error): {e}")

    # CSV fallback
    csv_path = opedir / "matches.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # convert numeric fields to floats/ints as needed
                for k in row:
                    try:
                        if "." in row[k]:
                            row[k] = float(row[k])
                        else:
                            row[k] = int(row[k])
                    except Exception:
                        pass
                matches.append(row)

    return matches

def main():
    # 1) try model0 providers
    matches = load_matches_from_model0()
    if matches:
        print(f"Loaded {len(matches)} matches from model0 provider.")
    else:
        # 2) try opedata directory JSON/CSV files
        matches = load_matches_from_opedata_dir(Path("opedata"))
        if matches:
            print(f"Loaded {len(matches)} matches from ./opedata/*.json or opedata/matches.csv")
        else:
            print("No matches found from model0 or opedata. Exiting.")
            return

    # validate & classify
    out_folder = Path("classification_results")
    out_folder.mkdir(exist_ok=True)
    csv_out = Path("batch_classification.csv")
    rows = []
    for i, feats in enumerate(matches):
        # assign an id if none
        match_id = feats.get("match_id", f"match_{i+1}")
        if not validate_feature_dict(feats):
            print(f"Skipping {match_id}: missing required keys -> {REQUIRED_KEYS - set(feats.keys())}")
            continue
        try:
            result = model3.classify_all(feats)
        except Exception as e:
            print(f"Error classifying {match_id}: {e}")
            continue

        # save json per match
        outp = {"match_id": match_id, "features": feats, "result": result}
        p = out_folder / f"{match_id}.json"
        p.write_text(json.dumps(outp, indent=2))

        # flatten one-line row for CSV
        row = {
            "match_id": match_id,
            "attacking_labels": "|".join(result["attacking"]["labels"]),
            "midfield_labels": "|".join(result["midfield"]["labels"]),
            "defensive_labels": "|".join(result["defensive"]["labels"]),
            "match_style_labels": "|".join(result["match_style"]["labels"]),
        }
        rows.append(row)
        print(f"Processed {match_id}: {row['match_style_labels']}")

    # write CSV
    if rows:
        keys = ["match_id", "attacking_labels", "midfield_labels", "defensive_labels", "match_style_labels"]
        with csv_out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote {len(rows)} rows to {csv_out.resolve()}")
    else:
        print("No rows to write (all matches skipped or errored).")

if __name__ == "__main__":
    main()
