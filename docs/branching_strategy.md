# Branching Strategy and Guardrails

This repository uses a monorepo model with three durable branches:

- `main`: Integrated product (frontend + backend together, release-ready).
- `backend-dev`: Backend-focused work that may not yet be integrated.
- `frontend-dev`: Frontend-focused work that may not yet be integrated.

## Integration Flow

1. Build in `backend-dev` and `frontend-dev`.
2. Open PRs into `main` regularly (small batches).
3. Resolve conflicts in PRs.
4. `main` must pass all required checks before merge.

## Guardrails (Repository-Level)

- CI checks in `.github/workflows/branch_guardrails.yml`:
  - `Backend quality` (ruff + backend pytest)
  - `Frontend quality` (npm build)
  - `Integration smoke` on `main` PRs/pushes
- PR template in `.github/pull_request_template.md` enforces intent and validation.

## Required Remote Settings (GitHub)

Enable branch protection for `main` in GitHub settings:

- Require pull request before merging
- Require status checks to pass:
  - `Backend quality`
  - `Frontend quality`
  - `Integration smoke`
- Restrict direct pushes to `main`

Optional branch protection for `backend-dev` and `frontend-dev`:

- Require PR for merges
- Require checks relevant to that branch's scope

## One-Time Branch Setup Commands

```bash
git checkout main
git pull origin main
git branch backend-dev main
git branch frontend-dev main
git push -u origin backend-dev
git push -u origin frontend-dev
```

## Daily Sync Routine

```bash
git checkout backend-dev
git fetch origin
git merge origin/main

git checkout frontend-dev
git fetch origin
git merge origin/main
```
