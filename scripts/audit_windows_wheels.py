from pathlib import Path
from zipfile import ZipFile
import sys

def main():
    wheel_files = sorted(Path("dist_windows").glob("gaffers_guide-*.whl"))
    if not wheel_files:
        print("[ERROR] No wheels found in dist_windows/")
        sys.exit(1)
        
    wheel = wheel_files[-1]
    
    protected = {
        "gaffers_guide/cv/temporal_ball_prior",
        "gaffers_guide/cv/pitch_roi_provider",
        "gaffers_guide/cv/ball_candidate_fuser",
        "gaffers_guide/cv/slice_batch_inferencer",
        "gaffers_guide/cv/optimized_sahi_wrapper",
        "gaffers_guide/pipeline/advanced_pitch_calibration",
        "gaffers_guide/pipeline/dynamic_homography",
        "gaffers_guide/pipeline/track_teams",
        "gaffers_guide/pipeline/track_teams_reid_hybrid",
        "gaffers_guide/pipeline/reid_healer",
        "gaffers_guide/pipeline/global_refiner",
        "gaffers_guide/pipeline/e2e_shared_impl",
        "gaffers_guide/pipeline/tactical_radar",
        "gaffers_guide/pipeline/generate_analytics",
    }

    with ZipFile(wheel) as zf:
        names = zf.namelist()

    compiled = [
        name for name in names
        if any(name.startswith(module + ".") and name.endswith(".pyd") for module in protected)
    ]
    leaks = [
        name for name in names
        if name.rsplit(".", 1)[0] in protected and name.endswith((".py", ".pyx", ".c"))
    ]

    print(f"wheel={wheel}")
    print(f"compiled_extensions={len(compiled)}")
    print(f"source_leaks={len(leaks)}")
    for leak in leaks:
        print(f"LEAK: {leak}")

    if len(compiled) == len(protected) and not leaks:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
