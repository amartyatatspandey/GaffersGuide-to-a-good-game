# Cursor Implementation Prompt — Gaffer's Guide Pipeline Stabilization
# Strategic Streaming Refactor (Medium Refactor Path)

---

## ROLE AND CONTEXT

You are a senior systems engineer helping refactor the `Gaffer's Guide` experiment pipeline —
a computer vision benchmarking backend that processes football match video using YOLOv8,
ByteTrack, and pitch homography. The pipeline runs inside a Linux container with a hard
cgroup memory cap of ~56.8 GiB.

The existing code is broken in a critical way: it loads entire video chunks (300 seconds ×
25fps = 7,500 frames) into memory as uncompressed RGB NumPy arrays (~46.6 GiB per chunk),
which causes the Linux OOM killer to send SIGKILL to worker processes — leaving no stack
trace, corrupting output files, and silently dropping Redis tasks.

We are implementing the **Medium Refactor (Strategic Streaming)** path. Do NOT suggest
a full microservices redesign. Do NOT suggest only surface-level semaphore patches.
Build the streaming architecture described below, file by file, with full working code.

Before touching any file, output a brief plan listing:
1. What the file currently does (wrong).
2. What the analogy is (to explain the bug simply).
3. What the fix does differently.
4. The exact code changes.

---

## GUIDING MENTAL MODEL (use this when writing docstrings and comments)

Think of the pipeline like a **conveyor belt sushi restaurant**:
- **Old code**: The kitchen cooks all 7,500 plates at once, stacks them everywhere, and the
  restaurant collapses under the weight.
- **New code**: The kitchen cooks one plate at a time (`yield` generator), passes it directly
  to the customer (inference loop), and clears the counter before the next plate arrives.

Each section below is one "station" in the restaurant. Fix them in order.

---

## IMPLEMENTATION PLAN

---

### STATION 1 — `services/cv_pipeline.py`
**Fix: Replace bulk VideoCapture with a streaming ffmpeg generator**

**The bug (analogy):** OpenCV's `VideoCapture` acts like a photocopier that must scan the
*entire* document before handing you a single page. We need a scanner that hands you one
page at a time as it scans.

**What to do:**

1. **Deprecate** `cv2.VideoCapture`. Remove all code that reads frames into a large list or
   NumPy array accumulator before processing.

2. **Implement** a Python generator function called `stream_yuv_frames` that:
   - Opens an `ffmpeg` subprocess with these flags:
     ```
     ffmpeg -i {input_path} -f rawvideo -pix_fmt yuv420p -vf scale={width}:{height} pipe:1
     ```
   - Reads exactly `width * height * 3 // 2` bytes per iteration (that is one YUV420p frame —
     1.5 bytes per pixel instead of 3 bytes for RGB, cutting raw frame size in half).
   - `yield`s a NumPy array of shape `(height * 3 // 2, width)` with dtype `uint8` for each
     frame, directly to the inference loop.
   - In the `finally` block: calls `.terminate()` then `.wait()` on the subprocess to prevent
     zombie processes. Never allow the subprocess to outlive the generator.

3. **Wrap** the `ThreadPoolExecutor.submit(...)` call with a `threading.BoundedSemaphore`
   initialized to `max_workers`. Acquire the semaphore before submitting; release it in the
   task's `done_callback`. This prevents the executor from staging more chunks in memory
   than there are active threads.

4. **Add** a module-level comment block explaining the YUV420p memory math:
   ```
   # RGB24: 1920 × 1080 × 3 bytes = 6.22 MB per frame
   # YUV420p: 1920 × 1080 × 1.5 bytes = 3.11 MB per frame
   # 300s chunk at 25fps = 7,500 frames
   # Old footprint: 7,500 × 6.22 MB ≈ 46.6 GiB  ← kills container
   # New footprint: 1 frame in memory at a time   ← bounded and safe
   ```

**Success condition:** Peak RAM during a 300s chunk must be measurable in megabytes,
not gigabytes. Verify with `tracemalloc` or `/proc/self/status` VmRSS logging.

---

### STATION 2 — `services/splitter.py`
**Fix: Replace static 300s chunk size with dynamic memory-aware chunk sizing**

**The bug (analogy):** The current splitter is like a pizza cutter that always cuts slices
of a fixed size — regardless of whether the pizza is 10 inches or 30 inches. We need a
cutter that measures first, then cuts to fit the plate.

**What to do:**

1. **Remove** the hardcoded `CHUNK_DURATION_SECONDS = 300` constant.

2. **Implement** a `compute_max_chunk_frames(cgroup_limit_bytes, width, height,
   safety_factor=0.60)` function that:
   - Reads the cgroup memory limit from `/sys/fs/cgroup/memory.max` (or falls back to
     `resource.getrlimit(resource.RLIMIT_AS)[0]`).
   - Computes: `max_frames = floor((cgroup_limit_bytes * safety_factor) / (width * height * 1.5))`
   - Returns `max_frames` as an integer.
   - Logs the computed limit at startup: `logger.info(f"Dynamic chunk limit: {max_frames} frames ({max_frames/fps:.1f}s at {fps}fps)")`

3. **Dispatch** multiple smaller chunk tasks to Redis instead of one massive payload.
   Each dispatched task JSON must include `{"chunk_start_frame": N, "chunk_end_frame": M,
   "video_path": "...", "retry_count": 0, "chunk_size_override": null}`.

4. **Never** allow a single dispatched payload to exceed the computed `max_frames`.

**Success condition:** On a 64 GiB container, `compute_max_chunk_frames` for 1080p video
should return approximately 8,192 frames (~327s), NOT 7,500 frames from a static constant.

---

### STATION 3 — `services/task_backend_redis.py`
**Fix: Replace RPUSH/LPOP with BLMOVE reliable queue pattern + Dead Letter Queue**

**The bug (analogy):** The current Redis queue is like a waiter who takes your order ticket,
immediately destroys it, then goes to the kitchen. If the waiter trips on the way, the order
is gone forever. We need a waiter who keeps a copy of the ticket in their apron until the
food is delivered — and has a manager who re-seats the order if the waiter never returns.

**What to do:**

1. **Replace** all `RPUSH` / `LPOP` commands with the reliable queue pattern:
   - **Enqueue:** `RPUSH queue:main {payload_json}`
   - **Dequeue (worker):** `BLMOVE queue:main queue:processing:{worker_id} LEFT RIGHT 30`
     (blocking pop with 30-second timeout, atomically moves to a worker-specific holding list)
   - **Acknowledge success:** `LREM queue:processing:{worker_id} 1 {payload_json}`
   - **Route failure to DLQ:** `RPUSH queue:dead_letter {payload_json_with_error_info}`

2. **Implement** a `SweepThread` class that:
   - Runs as a daemon thread every 60 seconds.
   - Scans all `queue:processing:*` keys.
   - For each payload, checks the `enqueued_at` timestamp embedded in the JSON.
   - If `(now - enqueued_at) > visibility_timeout_seconds` (default 1800 = 30 minutes):
     - Increments `retry_count` in the payload.
     - If `retry_count < 3`: uses `LREM` + `RPUSH queue:main` to re-queue with a halved
       chunk size hint (`chunk_size_override = original_size // 2`).
     - If `retry_count >= 3`: routes to `queue:dead_letter` and logs a `CRITICAL` alert.
   - Uses `SETNX queue:sweep_lock` as a distributed lock before sweeping, to prevent
     multiple sweeper threads from re-queuing the same job simultaneously. Expire the
     lock after 90 seconds.

3. **Implement** a failure taxonomy in the worker's exception handler:
   ```python
   except MemoryError:
       # Resource failure: re-queue with smaller chunk hint
   except (ValueError, RuntimeError) as e:
       if is_deterministic_failure(e):
           # Route straight to DLQ — don't retry poison payloads
   except (ConnectionError, TimeoutError):
       # Transient: exponential backoff, then re-queue
   ```

4. **Add** `enqueued_at: time.time()` and `worker_id: socket.gethostname()` to every
   enqueued payload JSON before dispatch.

**Success condition:** Kill a worker mid-task with `kill -9 {pid}`. After the sweep timeout,
the job must appear back in `queue:main` with `retry_count: 1` — confirmed via `redis-cli
LRANGE queue:main 0 -1`.

---

### STATION 4 — `services/merge.py`
**Fix: Eliminate in-memory list accumulation; use streaming JSONL serialization**

**The bug (analogy):** The current merge is like an accountant who insists on laying every
single receipt on the floor before adding them up. We need one who types each receipt
directly into the spreadsheet as it arrives, then presses total at the end.

**What to do:**

1. **Remove** all `tracking_data.extend(...)` and `sorted(tracking_data, ...)` patterns
   that accumulate the full result set in RAM.

2. **Replace** `json.dump(all_data, f)` output with a streaming **JSON Lines (JSONL)**
   writer: one JSON object per line, written immediately as each record is processed.
   Use `f.write(json.dumps(record) + '\n')` inside a loop, never a single bulk dump.

3. **For sort operations:** Use Python's `heapq.merge` on pre-sorted per-chunk JSONL
   files (disk-backed), rather than loading all chunks into RAM and calling `list.sort()`.

4. **Transition output format** from `output.json` (single large array) to `output.jsonl`
   (one tracking record per line). Update any downstream readers accordingly.
   Add a migration note in the changelog.

**Success condition:** `merge.py` processing 11 match files should not cause VmRSS to
exceed 500 MB regardless of the combined output size.

---

### STATION 5 — `services/dense_pass.py`
**Fix: Replace list comprehensions with generator expressions for analytical windows**

**The bug (analogy):** Current code is like a librarian who photocopies every book in the
library before checking one out. We need a librarian who hands you one book at a time.

**What to do:**

1. Replace every `[row for row in window]` list comprehension with a generator:
   `(row for row in window)`.

2. Never materialize a full analytical window array into memory if it can be yielded
   row-by-row to the disk-backed merge queue.

3. Feed yielded rows directly into the JSONL writer from Station 4.

---

### STATION 6 — `main.py` and `worker_main.py`
**Fix: Enforce hard memory limits at startup via `resource.setrlimit`**

**The bug (analogy):** Right now Python is like a driver who doesn't know the height
limit of the tunnel until the truck is already stuck inside it. We need a warning sign
*before* the tunnel that stops the truck early — with a recoverable error, not a crash.

**What to do:**

1. **At the top of the startup routine**, before any work begins:
   ```python
   import resource, os

   def enforce_memory_ceiling():
       cgroup_path = '/sys/fs/cgroup/memory.max'
       try:
           with open(cgroup_path) as f:
               cgroup_limit = int(f.read().strip())
       except (FileNotFoundError, ValueError):
           cgroup_limit = resource.getrlimit(resource.RLIMIT_AS)[1]

       safe_limit = int(cgroup_limit * 0.90)  # 90% of cgroup cap
       try:
           resource.setrlimit(resource.RLIMIT_AS, (safe_limit, safe_limit))
           logger.info(f"Memory ceiling enforced: {safe_limit / 1e9:.1f} GiB")
       except ValueError as e:
           logger.critical(f"setrlimit failed: {e}. Falling back to internal polling.")
   ```
   This converts an unrecoverable SIGKILL into a catchable Python `MemoryError`.

2. **Remove** all logic that allows fallback to local in-memory backends when Redis is
   unreachable during distributed matrix mode. Fail fast with a clear error message:
   `SystemExit("FATAL: Redis broker unreachable. Cannot guarantee task durability.")`

3. **Add** a `preflight_check()` function that validates before any work starts:
   - Redis is reachable and responsive (ping).
   - cgroup memory limit was successfully read.
   - Output directory is writable.
   - ffmpeg binary exists on PATH.
   Raise `SystemExit` with a descriptive message if any check fails.

---

### STATION 7 — Artifact Integrity (`services/merge.py`, `worker_main.py`)
**Fix: Implement write-then-rename atomic file pattern with SHA-256 verification**

**The bug (analogy):** Currently, writing output files is like painting the Mona Lisa
directly on the museum wall. If someone trips the fire alarm halfway through, you're
left with half a painting that looks like a full one from a distance — and ruins all
comparisons. We need to paint on a canvas, then hang it only when complete.

**What to do:**

1. **Never** write directly to the final output path. Always write to a `.tmp` sibling:
   ```python
   import os, hashlib

   def atomic_write_json(final_path: str, data_generator):
       tmp_path = final_path + '.tmp'
       sha256 = hashlib.sha256()

       with open(tmp_path, 'w') as f:
           for line in data_generator:
               encoded = (line + '\n').encode()
               f.write(line + '\n')
               sha256.update(encoded)
           f.flush()
           os.fsync(f.fileno())   # Force OS to flush write buffers to physical disk

       os.replace(tmp_path, final_path)  # Atomic rename — POSIX guaranteed

       manifest_path = final_path + '.sha256'
       with open(manifest_path, 'w') as mf:
           mf.write(sha256.hexdigest())
   ```

2. **In the benchmark comparison script**, before reading any artifact:
   ```python
   def verify_artifact(path: str) -> bool:
       sha_path = path + '.sha256'
       if not os.path.exists(sha_path):
           raise IntegrityError(f"Missing manifest for {path}")
       stored = open(sha_path).read().strip()
       actual = hashlib.sha256(open(path, 'rb').read()).hexdigest()
       if stored != actual:
           raise IntegrityError(f"SHA-256 mismatch on {path}. File may be corrupted.")
       return True
   ```

3. **Never** read an artifact file that lacks its `.sha256` sidecar. Raise and log
   `IntegrityError`, do not silently continue with potentially corrupted benchmark data.

---

### STATION 8 — `scripts/benchmark_decoders.py`
**Fix: Empirical hardware decoder provenance via pynvml**

**The bug (analogy):** OpenCV silently switches from GPU decoding to CPU decoding if
the GPU isn't cooperating — like a race car driver quietly swapping to a bicycle mid-race
without telling the timekeepers. Your benchmark records bicycle speed as car speed.

**What to do:**

1. **Install** `pynvml` (`pip install pynvml`).

2. **Before** each benchmark trial, record baseline decoder utilization:
   ```python
   import pynvml

   def get_decoder_utilization(handle) -> int:
       util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
       return util  # percentage 0–100
   ```

3. **During** and **after** each trial, poll `get_decoder_utilization` at 1-second intervals.
   Compute the mean utilization across the trial duration.

4. **Classify decoder provenance** in the output artifact:
   - Mean utilization > 5%: `"decoder_provenance": "hardware_nvdec"`
   - Mean utilization == 0%: `"decoder_provenance": "software_fallback_cpu"`
   - pynvml unavailable (no NVIDIA GPU): `"decoder_provenance": "non_nvidia_unknown"`

5. **Refuse** to label a trial as a hardware benchmark if provenance is `software_fallback_cpu`.
   Log a `WARNING` and tag the trial in the output JSON with `"hw_benchmark_valid": false`.

6. **Wrap** all `pynvml` calls in try/except `pynvml.NVMLError` so that the benchmarking
   script doesn't crash on non-NVIDIA machines — it just records `non_nvidia_unknown`.

---

### STATION 9 — `scripts/compare_benchmarks.py`
**Fix: Replace arithmetic mean comparisons with statistically rigorous methodology**

**The bug (analogy):** Using arithmetic mean to compare benchmark trials is like judging
a student's typical performance by including the one exam where their alarm didn't go off.
Median and confidence intervals tell you what the student actually does day-to-day.

**What to do:**

1. **Remove** all `statistics.mean()` comparisons from performance reporting.

2. **Implement** the following reporting pipeline for each metric (e.g., frame decode time):
   ```python
   import numpy as np
   from scipy import stats

   def analyze_metric(samples: list[float]) -> dict:
       arr = np.array(samples)
       median = np.median(arr)
       # Maritz-Jarrett bootstrap confidence interval for the median
       n_bootstrap = 10_000
       bootstrap_medians = [np.median(np.random.choice(arr, size=len(arr), replace=True))
                            for _ in range(n_bootstrap)]
       ci_low, ci_high = np.percentile(bootstrap_medians, [2.5, 97.5])
       return {
           "median": round(median, 4),
           "ci_95_low": round(ci_low, 4),
           "ci_95_high": round(ci_high, 4),
           "n_samples": len(arr),
       }
   ```

3. **Implement** `compare_two_conditions(baseline: list, experiment: list)`:
   - Run Welch's t-test: `stats.ttest_ind(baseline, experiment, equal_var=False)`
   - Report a win for `experiment` ONLY if `p < 0.05` AND the 95% CI of experiment
     does not overlap the 95% CI of baseline.
   - Output a verdict string: `"statistically_significant_improvement"`,
     `"no_significant_difference"`, or `"statistically_significant_regression"`.

4. **Mandate a warm-up phase:** Discard the first 3 iterations of any benchmark trial
   before recording samples. Log discarded iterations clearly: `logger.debug("Warm-up iter {i} discarded")`.

5. **Minimum sample size:** Enforce a minimum of 10 valid samples per condition before
   reporting. Raise `InsufficientSamplesError` if fewer samples are available.

---

### STATION 10 — `services/observability.py`
**Fix: Prevent indefinite in-memory timer accumulation**

**What to do:**

1. Implement a `flush_timers_to_disk()` function that serializes all in-memory timer
   objects to a local SQLite database (using the stdlib `sqlite3` module) or a JSONL log file.

2. Call `flush_timers_to_disk()` on a schedule — every 60 seconds via a background
   `threading.Timer` that reschedules itself.

3. After flushing, call `timer_store.clear()` to allow the garbage collector to reclaim memory.

4. On startup, load any persisted timers back from disk so long-running benchmark sessions
   survive worker restarts.

---

## TESTING CHECKLIST (implement these as `pytest` tests)

After implementing all stations, create `tests/test_pipeline_stability.py` with:

```python
# Test 1 — Memory ceiling: setrlimit converts OOM into catchable MemoryError
def test_oom_raises_memory_error_not_sigkill():
    ...

# Test 2 — Atomic write: kill -9 during write leaves no partial file
def test_atomic_write_no_partial_on_sigkill():
    ...

# Test 3 — SHA-256 integrity: tampered file raises IntegrityError before read
def test_sha256_mismatch_raises_integrity_error():
    ...

# Test 4 — Reliable queue: job re-appears after simulated worker crash
def test_job_requeued_after_worker_crash():
    ...

# Test 5 — Decoder provenance: 0% utilization tagged as software_fallback
def test_zero_nvdec_utilization_flagged_as_software_fallback():
    ...

# Test 6 — Benchmark stats: overlapping CI intervals returns no_significant_difference
def test_overlapping_ci_returns_no_significant_difference():
    ...
```

---

## CONSTRAINTS AND GUARDRAILS

- Do NOT introduce any new cloud infrastructure or external object stores (S3/MinIO).
  That belongs to the Robust Redesign path, which is out of scope here.
- Do NOT use `asyncio`. Stick to `threading` and subprocess-based concurrency.
  The existing architecture uses `concurrent.futures.ThreadPoolExecutor`.
- Do NOT change the public API of any function visible to `worker_main.py` without first
  flagging it as a breaking change and listing all call sites that must be updated.
- YUV420p is the mandatory color space. If a downstream model requires RGB24, implement
  a `yuv_to_rgb(frame_yuv)` conversion step at the point of model input, not at decode time.
  Color conversion for one frame costs ~1ms and keeps the memory footprint bounded.
- All new code must be compatible with Python 3.13.

---

## EXPECTED OUTCOME AFTER FULL IMPLEMENTATION

| Metric | Before refactor | After refactor |
|---|---|---|
| Peak RAM per chunk (300s, 1080p) | ~46.6 GiB | ~50–200 MB |
| OOM SIGKILL events | Frequent | Zero |
| Task loss on worker crash | Silent, unrecoverable | Zero (BLMOVE + sweeper) |
| Corrupt output files | Possible on crash | Impossible (atomic rename) |
| Hardware benchmark validity | Unverified | Empirically proven |
| Statistical test validity | None (arithmetic mean) | p < 0.05, non-overlapping CI |
