# Gaffer's Guide – Quality Profile System

## Introduction
This project implements a quality/speed profile system to allow users to control tradeoffs between runtime performance and output accuracy in a football analytics pipeline.

## Features Implemented
- CLI flag: --quality-profile
- Backward compatibility with --precision
- Centralized configuration (profiles.py)
- Runtime integration into pipeline
- Structured logging of active profile

## Profiles
- fast: prioritizes speed
- balanced: default tradeoff
- high_res: higher accuracy
- sahi: maximum quality (slowest)

## Command to Run

PYTHONPATH=backend:src python -m gaffers_guide.cli run \
--video "<path_to_video>" \
--output out \
--quality-profile fast

## Example

PYTHONPATH=backend:src python -m gaffers_guide.cli run \
--video "/path/to/video.mp4" \
--output out \
--quality-profile fast

## Outputs
- backend/output/*_tracking_data.json
- backend/output/*_tactical_metrics.json
- backend/output/*.mp4

## Verification
- Profile is resolved and logged
- Pipeline executes end-to-end
- Output files generated successfully

## Conclusion
The system provides a clean, scalable way to control runtime vs quality tradeoffs with minimal impact on existing architecture.
