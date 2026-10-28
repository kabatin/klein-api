# Changelog

## 0.2.0 — 2026-10-28

- `klein-api setup [--models klein-9b,klein-4b,z-image-turbo] [--check] [--yes] [--json] [--lan] [--service]
  [--port N] [--prefix DIR] [--label-suffix S] [--coordinate-with NAME=URL]`: the uniform setup command shared
  with vpipe-api, wan-api, ace-step-api and sfx-api. Installs draw-things-cli + gRPCServerCLI-macOS v26.1011.0
  from the GitHub release (SHA-256 pinned; a hand-configured CLI is never replaced), fetches models with
  `draw-things-cli models ensure` under a silence watchdog (resumes), checks disk first, merges the config
  without removing keys (0600, atomic, token never printed), and with `--service` writes and loads a
  LaunchAgent, waits for `/v1/health` and reports the macOS "access other apps' data" dialog as `needs_human`.
- Workflow `z-image-turbo`: Z Image Turbo (Apache-2.0) text to image, 1–4 PNGs at the exact size; text encoder
  and VAE from Draw Things' model definition, the model's recommended settings (8 steps, CFG 1, UniPC Trailing) passed explicitly so an offline Mac without
  Draw Things' cached catalog does not fall back to generic defaults. Timing measured (33 s per
  image); memory and non-square sizes not yet.
- `/v1/health` gains `workflows[id].{available, reason}`; a submit to a workflow (or `flux2-klein` tier) whose
  model is missing gets `503 workflow_unavailable`. `serve` starts while any model is installed; `doctor`
  warns (instead of failing) about the others and `doctor --smoke` skips them.

## 0.1.0 — 2026-10-17

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
- Coordination with other GPU servers (`coordinate_with`, `settle_seconds`): a job starts on an all-idle
  check, and not before `settle_seconds` after any server was last seen busy; unreachable servers count as idle (warning); peer tokens never logged, redirects not followed.
- Resident engine for `standard` jobs (`resident`, default on): a loopback-only `gRPCServerCLI-macOS` keeps the
  9B loaded between jobs (~19.5 s instead of ~26 s per image); stopped when idle, when a coordinated server is
  busy, before draft jobs, after failed jobs and on shutdown; leftovers reaped after a crash.
- Live findings applied: references are FLUX.2 references at strength 1.0 (default `max_reference_images` = 2,
  `--strength` never passed), measured per-model sizes and conservative estimates, text-encoder `-tensordata`
  checks, recorded CLI output fixtures, paths removed from engine error messages.
- Job store/queue, security and process handling adapted from wan-api 0.1.0.
