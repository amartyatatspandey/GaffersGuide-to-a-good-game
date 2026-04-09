# Beta Execution Isolation Design

## Problem

Running CV-heavy jobs directly in API flow can saturate API instances and degrade request latency.

## Beta Solution

- Introduce queue-backed execution (`backend/services/beta_queue.py`):
  - API enqueue path (`POST /api/v1beta/jobs`)
  - worker loop executes CV pipeline off request path
- Persist job metadata in `BetaJobStore` for deterministic status reads.

## Failure Handling

- Routing errors -> terminal `error` status with machine-readable message.
- Unexpected worker exceptions -> terminal `error` and counter increment.
- Queue item always marked `task_done` in `finally`.

## Future Evolution

- Replace in-process queue with external queue (Cloud Tasks/PubSub).
- Replace JSON store with durable managed datastore.
- Scale worker pool independently from API pod count.
