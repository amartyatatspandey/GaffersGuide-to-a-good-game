import asyncio
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        with open("match_test.mp4", "rb") as f:
            files = {"file": ("test.mp4", f, "video/mp4")}
            data = {
                "cv_engine": "local",
                "llm_engine": "local",
                "quality_profile": "balanced",
                "chunking_interval": "15-minute intervals"
            }
            res = await client.post("http://127.0.0.1:8000/api/v1/jobs", data=data, files=files)
            print("Status:", res.status_code)
            print("Response:", res.text)

asyncio.run(main())
