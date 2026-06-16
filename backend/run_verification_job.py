import asyncio
from pathlib import Path
from services.parallel_pipeline import run_e2e_parallel

async def main():
    video_path = Path("/Users/nishoosingh/Documents/GaffersGuide-to-a-good-game/psg_inter.mp4")
    job_id = "1b226175902d41da8d45266b53338b90"
    print(f"Starting E2E parallel execution for video={video_path} job={job_id}")
    report_path = await run_e2e_parallel(
        video=video_path,
        output_prefix=job_id,
        progress_callback=print,
        llm_engine="cloud",
        device="mps"
    )
    print(f"Pipeline finished! Report generated at: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
