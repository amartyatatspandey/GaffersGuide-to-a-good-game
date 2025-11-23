# test_model3.py
"""
Robust runner for model3 over matches discovered by model0_load_data.

This version:
 - reloads model0_load_data and prints which file was loaded
 - tries multiple fallbacks to obtain matches (get_all_matches, FEATURES_LIST, FEATURES)
 - if the module doesn't expose those, it attempts to execute the file directly with runpy
 - prints clear diagnostics on failure
 - otherwise runs model3 on every match, prints short summaries, and saves JSON/CSV
"""

import json
import csv
import importlib
import inspect
import runpy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

OUT_JSON = Path("model3_results.json")
OUT_CSV = Path("model3_summary.csv")


def find_classifier(module) -> Tuple[Any, str]:
    """
    Try to find a classifier function in model3.
    Preference order:
      - classify_all(features)
      - classify(features)
      - run(features)
      - main() (callable without args)
    Returns (callable, call_mode) where call_mode is 'fn' or 'main'
    """
    for name in ("classify_all", "classify", "run"):
        if hasattr(module, name) and callable(getattr(module, name)):
            return getattr(module, name), "fn"
    if hasattr(module, "main") and callable(getattr(module, "main")):
        return getattr(module, "main"), "main"
    return None, None


def flatten_labels(labels):
    if isinstance(labels, (list, tuple)):
        return "|".join(str(x) for x in labels)
    return "" if labels is None else str(labels)


def load_matches_from_provider(m0_module) -> List[Dict]:
    """
    Attempts to get a list of match feature dicts from m0_module using several strategies.
    Returns list of matches (possibly empty).
    """
    # 1) direct function
    if hasattr(m0_module, "get_all_matches") and callable(getattr(m0_module, "get_all_matches")):
        try:
            lst = m0_module.get_all_matches()
            if isinstance(lst, list):
                return lst
        except Exception as e:
            print(f"Calling get_all_matches() raised: {e}")

    # 2) FEATURES_LIST or FEATURES variable exported
    for attr in ("FEATURES_LIST", "FEATURES"):
        if hasattr(m0_module, attr):
            val = getattr(m0_module, attr)
            if isinstance(val, list):
                return val

    # 3) get any list-like attribute (best-effort)
    for name in dir(m0_module):
        if name.startswith("_"):
            continue
        try:
            val = getattr(m0_module, name)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                print(f"Using list found on module attribute: {name}")
                return val
        except Exception:
            continue

    return []


def try_run_model3_on_matches(matches: List[Dict], model3_module):
    if not matches:
        raise SystemExit("No matches to run on (empty list).")

    classifier, mode = find_classifier(model3_module)
    if classifier is None:
        raise SystemExit("Could not find a classifier function in model3. Expected classify_all/classify/run/main")

    results = []
    csv_rows = []

    print(f"Running model3 on {len(matches)} matches...\n")

    for i, match in enumerate(matches, start=1):
        match_id = match.get("match_id", f"match_{i}")
        print(f"--- [{i}/{len(matches)}] Match: {match_id} ---")
        try:
            if mode == "fn":
                out = classifier(match)
            else:
                maybe = classifier()
                out = maybe if maybe is not None else {}
        except Exception as e:
            print(f"ERROR running model3 on {match_id}: {e}")
            out = {"error": str(e)}

        # pretty print compact summary
        if isinstance(out, dict):
            for third in ("attacking", "midfield", "defensive", "match_style"):
                sec = out.get(third, {})
                labels = sec.get("labels") if isinstance(sec, dict) else None
                reasons = sec.get("reasons") if isinstance(sec, dict) else None
                print(f"{third.upper():10}: {flatten_labels(labels)}")
                if reasons:
                    print(f"  reason: {reasons[0]}")
            print()
        else:
            print("Model3 returned non-dict result; saved raw output.\n")

        results.append({"match_id": match_id, "features": match, "result": out})
        csv_rows.append({
            "match_id": match_id,
            "attacking_labels": flatten_labels(out.get("attacking", {}).get("labels") if isinstance(out, dict) else ""),
            "midfield_labels": flatten_labels(out.get("midfield", {}).get("labels") if isinstance(out, dict) else ""),
            "defensive_labels": flatten_labels(out.get("defensive", {}).get("labels") if isinstance(out, dict) else ""),
            "match_style_labels": flatten_labels(out.get("match_style", {}).get("labels") if isinstance(out, dict) else ""),
        })

    # Save outputs
    OUT_JSON.write_text(json.dumps(results, indent=2))
    keys = ["match_id", "attacking_labels", "midfield_labels", "defensive_labels", "match_style_labels"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(f"Saved full results to {OUT_JSON.resolve()}")
    print(f"Saved CSV summary to {OUT_CSV.resolve()}")


def main():
    # Import providers
    try:
        import model0_load_data as m0
    except Exception as e:
        raise SystemExit(f"Could not import model0_load_data: {e}")

    try:
        import model3
    except Exception as e:
        raise SystemExit(f"Could not import model3: {e}")

    # show where model0_load_data was loaded from
    try:
        print("model0_load_data loaded from:", inspect.getfile(m0))
    except Exception:
        print("Could not determine file path for model0_load_data module.")

    # Reload module to pick up edits
    try:
        importlib.reload(m0)
    except Exception as e:
        print("Reloading model0_load_data raised:", e)

    matches = load_matches_from_provider(m0)
    if not matches:
        # fallback: try to run the module file directly with runpy to capture globals
        try:
            m0_path = inspect.getfile(m0)
            print(f"No matches found via import; attempting to execute file directly: {m0_path}")
            g = runpy.run_path(m0_path, run_name="__main__")
            # first try get_all_matches in returned globals
            if "get_all_matches" in g and callable(g["get_all_matches"]):
                matches = g["get_all_matches"]()
            elif "FEATURES_LIST" in g and isinstance(g["FEATURES_LIST"], list):
                matches = g["FEATURES_LIST"]
            elif "FEATURES" in g and isinstance(g["FEATURES"], list):
                matches = g["FEATURES"]
            else:
                print("Execution returned globals but no matches found in those globals.")
        except Exception as e:
            print("Running model0_load_data.py directly failed:", e)

    if not matches:
        # final diagnostic: print exported names for user to inspect
        names = [n for n in dir(m0) if not n.startswith("_")]
        print("Diagnostics: model0_load_data exported names:", names)
        raise SystemExit("Could not locate matches. Ensure model0_load_data exposes get_all_matches() or FEATURES_LIST/FEATURES.")

    # If we have matches, run model3
    try:
        try_run_model3_on_matches(matches, model3)
    except Exception as e:
        raise SystemExit(f"Error while running model3: {e}")


if __name__ == "__main__":
    main()
