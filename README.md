# klein-api

**Run FLUX.2 [klein] text/reference-to-image on your Apple Silicon Mac as an HTTP job API.**

`klein-api` is a small, safe job server around FLUX.2 [klein], driven headless through
[`draw-things-cli`](https://drawthings.ai). Send a prompt and an exact output size, get back 1–4 PNGs:

```
POST /v1/workflows/flux2-klein/jobs   → 202 {id}
GET  /v1/jobs/{id}                     → queued → running (progress) → succeeded
GET  /v1/jobs/{id}/output?index=N      → image/png
```

It speaks the same job protocol as [wan-api](https://github.com/kabatin/wan-api) and
[vpipe-api](https://github.com/kabatin/vpipe-api) (job lifecycle, error envelope, `429` + `Retry-After`,
`Idempotency-Key`), so a client that drives them needs only a different workflow id, body and output type.
The full contract is [docs/api.md](docs/api.md).

- **Closed workflow.** Clients send a prompt, an output size, `draft`/`standard`, a seed, a count and
  (when enabled) reference images — never model files, steps, CFG, samplers, LoRAs, strength or paths.
- **Exact output.** Each image is generated at a ~1 MP size (multiples of 64) closest to the requested aspect
  ratio, then cover-cropped (lanczos) to the exact size: 8-bit RGB PNG, no alpha, decoded and checked before
  the job succeeds.
- **One GPU slot, honest backpressure.** One job at a time (its images run one after another), a small waiting
  queue, `429 busy` with `Retry-After` when full.
- **Plays nicely with other GPU servers.** With `coordinate_with`, a job starts only once vpipe-api / wan-api
  report idle twice, `settle_seconds` apart; meanwhile it stays `queued` with `waiting_for`.
- **Restart-safe.** Jobs live on disk; queued jobs run again, interrupted ones fail as retryable, and an engine
  process left behind by a crashed server is stopped.

## Requirements

- Apple Silicon Mac, macOS 14+. A `standard` image peaks at ~10 GB of memory.
- Python 3.12+ and [uv](https://docs.astral.sh/uv/).
- `draw-things-cli` (measured with v26.1011.0) and the FLUX.2 [klein] model files:

| tier | model file | also needs |
|---|---|---|
| `standard` | `flux_2_klein_9b_i8x.ckpt` | `qwen_3_8b_q8p.ckpt`, `flux_2_vae_f16.ckpt` |
| `draft` | `flux_2_klein_4b_i8x.ckpt` | `flux_2_vae_f16.ckpt` (+ its text encoder, not checked yet) |

Measured on the target Mac (1024×1024, one cold process per image): 9B ~26 s, 4B ~14 s.

## Setup on the Mac

```sh
uv tool install git+https://github.com/kabatin/klein-api     # or: uv sync && uv run klein-api …
```

The defaults expect:

- the CLI at `~/.local/opt/draw-things-cli/v26.1011.0/draw-things-cli`
- the models in `~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models` (the Draw Things app's
  folder; the app itself may stay closed)

Both can be changed in `~/.config/klein-api/config.toml`:

```toml
# ~/.config/klein-api/config.toml   (chmod 600 if it holds tokens)
port = 8767

[backends.drawthings]
cli = "~/.local/opt/draw-things-cli/v26.1011.0/draw-things-cli"
models_dir = "~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
max_reference_images = 0     # keep 0 until reference editing is verified (see below)

# wait for other GPU servers on this Mac before each job
[[coordinate_with]]
name = "vpipe-api"
health_url = "http://127.0.0.1:8765/v1/health"
token = "…vpipe-api's bearer token…"

[[coordinate_with]]
name = "wan-api"
health_url = "http://127.0.0.1:8766/v1/health"
```

klein-api always runs the CLI with `--local --offline --no-download-missing --disable-preview`: a job never moves
to the cloud and never starts a download. A missing file is a clear failure.

### `doctor`

```sh
klein-api doctor           # platform, macOS, Python, config permissions, CLI, model files, disk, power,
                           # memory, and every coordinate_with server
klein-api doctor --smoke   # + one real draft image through the workflow (stop the server first)
```

### `serve`

```sh
klein-api serve            # http://127.0.0.1:8767   (OpenAPI: /openapi.json, docs: /docs)
```

`serve` checks the CLI and the model files first and exits with a clear message if it cannot generate.

### Run at login (launchd)

[`examples/com.github.kabatin.klein-api.plist`](examples/com.github.kabatin.klein-api.plist):

```sh
cp examples/com.github.kabatin.klein-api.plist ~/Library/LaunchAgents/   # then edit the paths
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.github.kabatin.klein-api.plist
launchctl bootout gui/$(id -u)/com.github.kabatin.klein-api                # stop
```

Logs are JSON lines on stderr (job id, state transitions, queue/coordination/backend/postprocess/total seconds,
failure category) — never prompts, images or tokens.

## Configuration

`~/.config/klein-api/config.toml` (or `$KLEIN_API_CONFIG`); every scalar can also be set as `KLEIN_API_<NAME>`.

| key | default | |
|---|---|---|
| `host` / `port` | `127.0.0.1` / `8767` | a non-loopback host **requires** `token` |
| `token` | — | when set, every request needs `Authorization: Bearer <token>` (≥ 32 printable ASCII) |
| `max_waiting` | `2` | jobs allowed to wait behind the running one |
| `retention_days` | `7` | finished jobs and outputs are deleted after this |
| `data_dir` | `~/.local/share/klein-api` | `jobs/<id>/` (record, outputs, engine logs) |
| `job_timeout_factor` | `3.0` | a job is stopped after `estimate × factor + 5 min` |
| `max_body_mb` | `64` | request size limit (a 20 MB image is ~27 MB of base64) |
| `coordinate_with` | `[]` | `[[coordinate_with]]` tables: `name`, `health_url`, optional `token` (file only) |
| `settle_seconds` | `30` | the two idle checks of every coordinated server are this far apart |
| `coordination_poll_seconds` | `5` | how often a busy server is asked again |

`[backends.drawthings]`: `cli`, `models_dir`, `max_reference_images` (0–4, default 0),
`reference_strength` (default 1.0), `stop_grace_s`.

### Coordination with other GPU servers

The Mac has one GPU and one pool of unified memory. When `coordinate_with` is set, the job at the head of the
queue starts only after **every** listed server's `/v1/health` reports `running == 0 && waiting == 0` (and
`outside_runs == 0` when present) **twice, `settle_seconds` apart**. Until then it is `queued` with
`waiting_for: "<name>"` and does not count as running in klein-api's own `/v1/health`. A server that cannot be
reached (or answers anything but a valid health body) counts as idle and is logged as a warning. A running job
is never paused. Redirects are not followed, so a peer token is only ever sent to its own `health_url`.

Note that the double check costs `settle_seconds` before **every** job, even when the other servers are idle
(it shows up as `timings.coordination_seconds`).

### Reference images (off until verified)

`--image` is passed to the CLI once per reference (the first is the main one) with `--strength
<reference_strength>`. Whether FLUX.2 [klein] in draw-things-cli treats these as FLUX.2 reference editing or as
img2img has **not been verified yet**, so `max_reference_images` defaults to `0`: `/v1/capabilities` then says
`reference_images.max: 0` and any request with references gets `422 invalid_params`. Enable it only after
checking on the Mac that references are honoured. References are normalised to upright RGB PNGs of at most
2048 px on the longest side before they reach the engine.

## API examples

Full reference: [docs/api.md](docs/api.md); more in [examples/curl.md](examples/curl.md).

```sh
API=http://127.0.0.1:8767
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $KLEIN_API_TOKEN")

curl -s "${AUTH[@]}" "$API/v1/capabilities" | jq '.workflows[0] | {output, count, reference_images}'

JOB=$(curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/flux2-klein/jobs" \
        -H 'Content-Type: application/json' -H 'Idempotency-Key: poster-001' \
        -d '{"prompt": "A red tram in the rain at night, neon reflections, 35mm photo",
             "output": {"width": 1920, "height": 1080}, "quality": "draft", "seed": 7, "count": 2}' \
      | jq -r .id)
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings}'
curl -s "${AUTH[@]}" -o image-0.png "$API/v1/jobs/$JOB/output?index=0"
curl -s "${AUTH[@]}" -o image-1.png "$API/v1/jobs/$JOB/output?index=1"
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"          # cancel (200 canceled / 202 stopping)
```

## Security

- Binds `127.0.0.1` by default. A non-loopback `host` without a `token` refuses to start.
- Without a token, only loopback `Host` headers are served (blocks DNS rebinding) and proxy headers are refused.
  With a token, every endpoint (including `/docs`, `/openapi.json`, `/healthz`) needs it.
- **No TLS.** On a LAN the token and the images travel in clear text: use a trusted network or a TLS reverse proxy
  (keep the token set). `doctor` warns if a config file holding tokens is readable by others.
- Body limit (`max_body_mb`; any `Transfer-Encoding` is refused with 411), at most 64 concurrent connections,
  images ≤ 20 MB / 50 MP decoded by the PNG/JPEG/WebP decoders only (size checked before decoding), control
  characters refused in prompts, strict job ids, no client-chosen paths, models or settings. The engine runs
  without a shell and without klein-api's environment (the token); prompts are passed in files. Client-facing
  errors never contain paths or internals. Job files are private to the server's user (umask 077). Do not add
  CORS: the API relies on browsers not being able to send cross-site JSON.

## Not verified yet on the real Mac

The server was developed against a fake `draw-things-cli`. Check these with `doctor --smoke` and
`KLEIN_API_REAL_TESTS=1 uv run pytest -m real -p no:cacheprovider --no-cov`:

- the output line formats parsed in `src/klein_api/backends/drawthings_output.py` (progress
  `Sampling... i / n`, `Sampling step time (n step(s)): avg x s`, `Wrote: <path>`, `Error: …`);
- that `--output raw-<i>.png` is written exactly at that path, as PNG;
- the six unmeasured generation sizes in `src/klein_api/backends/sizes.py`;
- whether `--image` + `--strength 1` is reference editing for klein (references are off until then);
- that seeds above 2³¹−1 (`seed + i` with `seed` near the maximum) are accepted;
- the draft model's text encoder file (not checked by `doctor`).

## Development

```sh
uv sync
uv run pytest            # fake engine; 80%+ coverage gate; `real` tests are skipped
uv run ruff check && uv run ruff format --check && uv run pyright
KLEIN_API_REAL_TESTS=1 uv run pytest -m real --no-cov   # on the Mac with the engine installed
```

## License

Apache-2.0 (this repository; it ships no weights). draw-things-cli is GPL-3.0 and is run as a separate program,
not linked. Check the license of every model you download before using its output.
