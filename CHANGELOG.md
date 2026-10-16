# Changelog

## 0.1.0 — unreleased

First version.

- HTTP job API per [docs/api.md](docs/api.md): health (vpipe-api / wan-api compatible), capabilities,
  workflow listing with JSON Schemas, typed submit route, job status with progress over all images, queue
  position, `waiting_for`, estimate and timings (incl. `coordination_seconds`), per-index PNG download,
  cancel. One error envelope for every failure.
- Workflow `flux2-klein`: FLUX.2 [klein] text (and, once enabled, reference images) to 1–4 images; ~1 MP
  generation sizes in multiples of 64, lanczos cover crop to the exact size, 8-bit RGB PNG verified per image.
- Backend `drawthings`: `draw-things-cli` headless under a pty, `--local --offline --no-download-missing
  --disable-preview`, 4 steps / CFG 1, one process per image (`seed + i`), stopped on cancel/timeout, orphans
  reaped after a restart. Output parsing isolated in `drawthings_output.py`.
- Coordination with other GPU servers (`coordinate_with`, `settle_seconds`): double idle check before a job
  starts; unreachable servers count as idle (warning); peer tokens never logged, redirects not followed.
- Job store/queue, security and process handling adapted from wan-api 0.1.0.
