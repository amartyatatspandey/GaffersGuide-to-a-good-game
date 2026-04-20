# Gaffer's Guide Agent Runtime Rules

## Graphify Usage Protocol (Token Efficiency + Speed)

Graphify is required for discovery and architecture-scale work, but should be skipped for trivial local edits.

### MUST use Graphify first when:

- The task touches 3+ files or multiple modules.
- The code area is unfamiliar and dependency flow is unclear.
- The request is architectural (refactor, decoupling, boundary enforcement).
- Root-cause analysis requires cross-file call-path reasoning.

### MUST skip Graphify when:

- The task is a trivial single-file edit.
- The fix is an obvious lint/type/test issue with a known file path.
- The change is a small docs/comment/rename update.
- The task is a direct follow-up tweak in already-open local context.

### OPTIONAL (agent discretion):

- Medium tasks (2-3 files) where uncertainty is moderate.

### Fast decision gate (run before implementation):

- If scope <= 2 files and the change is obvious -> skip Graphify.
- Else -> use Graphify (`query`, `path`, or `explain`) to narrow context first.

### Audit line required in each substantial agent response:

- `Graphify: used (reason: <cross-module/architecture/unknown-flow>)`
- `Graphify: skipped (reason: <trivial-known-local-change>)`
