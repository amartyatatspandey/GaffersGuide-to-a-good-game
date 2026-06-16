import asyncio
import sys
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
backend_dir = tests_dir.parent

sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(tests_dir))

from test_parallel_pipeline import test_parallel_e2e_execution

if __name__ == "__main__":
    print("Running parallel pipeline E2E test...")
    asyncio.run(test_parallel_e2e_execution())
    print("Test passed successfully!")
