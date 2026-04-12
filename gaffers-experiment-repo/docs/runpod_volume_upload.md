# RunPod network volume upload (S3 API)

Upload **only** this experiment repository to your RunPod network volume using the S3-compatible API. Credentials come from the RunPod UI for the volume; do not commit them.

## Your endpoint and bucket

- **Endpoint:** `https://s3api-eu-ro-1.runpod.io`
- **Bucket:** `6x4jz4yym1`

Keys are written under prefix `gaffers-experiment-repo/` by default (configurable via `RUNPOD_S3_PREFIX`).

## Prerequisites

- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) (`aws --version`)
- RunPod **Access Key** and **Secret Key** for the volume’s S3 access

## One-time environment

```bash
export AWS_ACCESS_KEY_ID="…"
export AWS_SECRET_ACCESS_KEY="…"
export AWS_DEFAULT_REGION=us-east-1
```

Optional (defaults already match your bucket):

```bash
export RUNPOD_S3_BUCKET=6x4jz4yym1
export RUNPOD_S3_ENDPOINT=https://s3api-eu-ro-1.runpod.io
export RUNPOD_S3_PREFIX=gaffers-experiment-repo
```

Or copy [`.env.runpod.example`](../.env.runpod.example) to `.env.runpod`, fill keys, then:

```bash
set -a && source .env.runpod && set +a
```

## Sync from your laptop (experiment repo root)

From the **parent** of `scripts/` (this repo root):

```bash
chmod +x scripts/sync_to_runpod_s3.sh
./scripts/sync_to_runpod_s3.sh
```

Dry run (no uploads):

```bash
DRY_RUN=1 ./scripts/sync_to_runpod_s3.sh
```

Mirror mode (also **delete** remote objects missing locally — use with care):

```bash
RUNPOD_S3_SYNC_DELETE=1 ./scripts/sync_to_runpod_s3.sh
```

Omit large local job output directories:

```bash
EXCLUDE_LOCAL_OUTPUT=1 ./scripts/sync_to_runpod_s3.sh
```

## What is included / excluded

**Included:** all source, `docs/`, `experiment-backend/data/fixtures/` (if present), configs, infra YAMLs, etc.

**Excluded by default:** `.git/`, `**/.venv/`, `**/node_modules/`, `__pycache__`, `.pytest_cache`, `experiment-frontend/dist/`, `.pyc`, `.DS_Store`.

**Optional:** set `EXCLUDE_LOCAL_OUTPUT=1` to skip `experiment-backend/output/` (regenerate on pod if needed).

## On the RunPod pod

Mount the network volume (e.g. under `/runpod-volume`). Your files appear under the prefix you used, for example:

`/runpod-volume/gaffers-experiment-repo/experiment-backend/`

Then on the pod:

```bash
cd /runpod-volume/gaffers-experiment-repo/experiment-backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-gpu.txt   # or requirements.txt for CPU
```

## Troubleshooting

- **403 / signature errors:** confirm keys, endpoint URL, and bucket name; keep `AWS_DEFAULT_REGION=us-east-1` unless RunPod docs specify otherwise.
- **Path-style:** if the CLI fails listing objects, try `aws configure set default.s3.addressing_style path` (or set `AWS_S3_ADDRESSING_STYLE=path` in the environment for AWS CLI v2 where supported).

## Security

Never commit `AWS_SECRET_ACCESS_KEY` or `.env.runpod`. Add `.env.runpod` to local ignore if you use a personal exclude list.
