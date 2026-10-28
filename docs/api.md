# klein-api HTTP API (v1)

Request and response bodies are JSON (UTF-8) unless noted. The machine-readable schema is served at
`/openapi.json`; interactive docs at `/docs`. The shapes deliberately match
[wan-api](https://github.com/kabatin/wan-api) and [vpipe-api](https://github.com/kabatin/vpipe-api)
(same job lifecycle, error envelope, `429` + `Retry-After`, `Idempotency-Key`), so a client that drives
them needs only a different workflow id, body and output type.

The engines are FLUX.2 [klein] (workflow `flux2-klein`) and Z Image Turbo (workflow `z-image-turbo`)
through `draw-things-cli` on Apple Silicon. **Nothing about the engine (model files, steps, CFG,
sampler, quantization) is part of this contract.**

## Authentication

Identical to wan-api:

- If the server has a token configured (`KLEIN_API_TOKEN`, ≥ 32 printable ASCII characters), **every**
  request — including `/healthz`, `/docs` and `/openapi.json` — must send `Authorization: Bearer <token>`;
  otherwise `401`.
- A server bound to a non-loopback address refuses to start without a token.
- Without a token the server only answers requests whose `Host` is `127.0.0.1`, `localhost` or `[::1]`
  and that carry no proxy headers (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`).
- There is no TLS. On a LAN the token and the images travel in clear text.

## Error envelope

```json
{ "error": { "code": "busy", "message": "human readable", "retryable": true, "details": null } }
```

| HTTP | `code` | `retryable` | When |
|---|---|---|---|
| 401 | `unauthorized` | false | missing / wrong bearer token |
| 403 | `forbidden_host` / `proxy_requires_token` | false | token-less server reached via a foreign Host or a proxy |
| 404 | `not_found` | false | unknown workflow, job, output index or path |
| 409 | `conflict` | false | output requested before the job succeeded, cancel of a finished job |
| 409 | `idempotency_conflict` | false | `Idempotency-Key` reused with different params |
| 409 | `idempotency_in_flight` | true | another request with the same key is being accepted right now |
| 411 | `length_required` | false | any `Transfer-Encoding` — send a plain `Content-Length` |
| 413 | `payload_too_large` | false | request body over `max_body_mb` (default 64 MB) |
| 422 | `invalid_params` | false | params failed validation; `details` lists `{loc, msg}` (inputs are never echoed) |
| 429 | `busy` | true | the GPU slot and the waiting queue are full. `Retry-After` header (seconds) is set |
| 500 | `internal` | true | unexpected server error (details only in the server log) |
| 503 | `insufficient_storage` | true | less than 2 GB free in `data_dir`; nothing was queued |
| 503 | `workflow_unavailable` | false | this workflow's (or this `quality` tier's) engine or model files are not installed; the message says which (`/v1/health` → `workflows`); other workflows keep working |

## Endpoints

### `GET /healthz` and `GET /v1/health`

```json
{ "status": "ok", "version": "0.1.1", "running": 1, "waiting": 0, "max_waiting": 2,
  "workflows": {
    "flux2-klein":   { "available": true,  "reason": "draft: model files missing: flux_2_klein_4b_i8x.ckpt",
                       "qualities": {
                         "draft":    { "available": false, "reason": "model files missing: flux_2_klein_4b_i8x.ckpt" },
                         "standard": { "available": true,  "reason": null } } },
    "z-image-turbo": { "available": false, "reason": "model files missing: z_image_turbo_1.0_i8x.ckpt",
                       "qualities": null }
  } }
```

`503` with `"status": "degraded"` if the worker has stopped.

`workflows` (added in 0.1.1; the other fields are unchanged) has one entry per workflow: `available` —
it can take jobs (for `flux2-klein`: at least one `quality` tier can); `reason` — why it, or one of its
tiers, cannot run (`null` when everything can; never a path); `qualities` — per `quality` tier
(`flux2-klein`: `draft` = the 4B model files, `standard` = the 9B), `null` for a workflow without tiers. A submit that cannot run gets
`503 workflow_unavailable`. One unavailable workflow does not make `status` `degraded`.

### `GET /v1/capabilities`

What a client may ask for, without engine details. **This is the source of truth for client
descriptors** — a client must not assume values that are not listed here.

```json
{
  "version": "0.1.1",
  "queue": { "max_running": 1, "max_waiting": 2 },
  "workflows": [{
    "id": "flux2-klein",
    "output": { "media_type": "image/png",
                "width":  { "min": 256, "max": 2048, "multiple_of": 2 },
                "height": { "min": 256, "max": 2048, "multiple_of": 2 },
                "aspect_ratio": { "min": "9:21", "max": "21:9" },
                "fit": "cover (generated at a supported size, then centre-cropped and resized)" },
    "count": { "min": 1, "max": 4 },
    "quality": ["draft", "standard"],
    "seed": true,
    "reference_images": { "max": 4, "media_types": ["image/jpeg", "image/png", "image/webp"],
                          "max_bytes": 20971520, "max_pixels": 50000000 },
    "typical_seconds": { "draft": 15, "standard": 26 }
  }, {
    "id": "z-image-turbo",
    "output": { "…": "same as flux2-klein" },
    "count": { "min": 1, "max": 4 },
    "seed": true,
    "reference_images": { "max": 0 },
    "typical_seconds": 34
  }],
  "coordination": { "waits_for": ["vpipe-api"] }
}
```

`reference_images.max` is `0` when the engine in use cannot take references; a client must then not
send any. `typical_seconds` is per image and only a hint.

### `GET /v1/workflows`

Lists workflows with the JSON Schema of their POST body (`params_schema`).

### `POST /v1/workflows/flux2-klein/jobs`

Body = the params below, `Content-Type: application/json`.
Optional header `Idempotency-Key: <1–128 of [A-Za-z0-9._:-]>`.

| field | type | default | notes |
|---|---|---|---|
| `prompt` | string, 1–4000 chars | — | No control characters (tab/newline are fine). |
| `output` | `{width, height}` | — | Exact output size: even integers 256–2048, aspect 9:21 … 21:9. |
| `quality` | `"draft"` \| `"standard"` | `"standard"` | `draft` is for fast iteration, `standard` for final candidates. |
| `seed` | int 0 … 2³¹−1 \| null | random | Image *i* (0-based) uses `seed + i`. Echoed back per image. |
| `count` | int 1–4 | `1` | Number of images. They run one after another inside the one job. |
| `reference_images` | image[] (0 … `max`) | `[]` | Subjects, wardrobe, places or style to keep. Order matters: the first is the main reference. |

`image` = `{ "data": "<base64>", "media_type": "image/png" | "image/jpeg" | "image/webp" }`, decoded
size ≤ 20 MB, ≤ 50 megapixels, ≥ 64 px per side. EXIF orientation is applied.

There is **no `negative_prompt`**: the engine runs without classifier-free guidance, so it would be
silently ignored. Anything else in the body — `negative_prompt`, model names, steps, CFG, samplers,
LoRAs, strength, file paths — is rejected with `422`.

Success is `202 Accepted`:

```json
{ "id": "job_01J9ZK3N2Q8V7W6X5Y4Z3A2B1C", "workflow": "flux2-klein", "status": "queued", "created_at": "2026-10-16T12:00:00Z" }
```

`429 busy`: nothing was queued; try again after `Retry-After` seconds. Answered before the body is read.

**Idempotency** works exactly as in wan-api: same key + same params → the existing job with `200 OK`;
same key + different params → `409 idempotency_conflict`; concurrent → `409 idempotency_in_flight`.

### `GET /v1/jobs/{job_id}`

```json
{
  "id": "job_…",
  "workflow": "flux2-klein",
  "status": "succeeded",
  "progress": 1.0,
  "queue_position": null,
  "waiting_for": null,
  "created_at": "…", "started_at": "…", "finished_at": "…",
  "estimate_seconds": 52.0,
  "cancel_requested": false,
  "timings": { "queue_seconds": 0.8, "coordination_seconds": 0.0, "backend_seconds": 50.2,
               "sampling_seconds": 32.1, "postprocess_seconds": 0.4, "total_seconds": 51.4 },
  "result": {
    "outputs": [
      { "index": 0, "media_type": "image/png", "width": 1920, "height": 1080, "seed_used": 12345 },
      { "index": 1, "media_type": "image/png", "width": 1920, "height": 1080, "seed_used": 12346 }
    ],
    "seed_used": 12345,
    "details": { "generation": { "width": 1344, "height": 768, "quality": "standard", "references": 1 } }
  },
  "error": null
}
```

- `status`: `queued` → `running` → `succeeded` | `failed` | `canceled` (one "l").
- `running` only while the job owns the GPU slot. A job that is first in line but is waiting for another
  GPU user on the machine (see *Coordination*) stays `queued` with `waiting_for: "vpipe-api"`.
- `progress`: `0.0..0.99` while running, over all images of the job; `1.0` when succeeded; `null` when
  unknown.
- `queue_position`: 1-based while `queued`, else `null`.
- `result.seed_used` is the seed of image 0 (wan-api/vpipe-api compatible); each output has its own.
- `error` (only when `failed`): `{code, message, retryable}`; e.g. `generation_failed`, `timeout`,
  `invalid_output`, `server_restarted`, `server_stopped`, `engine_unstoppable`.
- Restart behaviour is identical to wan-api (`server_stopped` / `server_restarted`, retryable; queued
  jobs run again).

### `GET /v1/jobs/{job_id}/output` and `GET /v1/jobs/{job_id}/output?index=N`

The PNG (`image/png`) of image `N` (default `0`). `409 conflict` unless the job has succeeded,
`404 not_found` for an index outside `0 … count-1`. Outputs are kept for `retention_days` (default 7).

Every output is checked before the job succeeds: a decodable PNG of exactly `output.width × height`.

### `DELETE /v1/jobs/{job_id}`

As in wan-api: `200` `canceled`, `202` while the engine is stopping, `409` if already finished.

## Coordination with other GPU servers

The Mac has one GPU and one pool of unified memory. wan-api does not know about klein-api. vpipe-api
(0.1.4 and later), ace-step-api and sfx-api can list it in their own `coordinate_with`; each of them
yields while it waits (`waiting_for_peer`, `waiting: 0`), and klein-api does not, so klein-api goes
first. When `coordinate_with` is configured (a list of `{name, health_url, token}`),
the job at the head of the queue starts only after a check in which **every** listed server reports
idle: `running == 0 && waiting == 0` (and `outside_runs == 0` when present). The server remembers the
time it last saw any listed server busy (`running`, `waiting` or `outside_runs` > 0). If no server has
been seen busy within the last `settle_seconds` (default 30) — including when none has ever been seen
busy — a single all-idle check is enough and the job starts at once. If one was seen busy within that
window, the job waits and starts only if a second all-idle check, made `settle_seconds` after that busy
observation, still finds every server idle; a busy answer at any check starts the window again from
that observation — clients often submit follow-up jobs right after one finishes. While waiting the
job stays `queued` with `waiting_for` set. An unreachable server counts as idle (logged as a warning).
Coordination is checked only before a job starts; a running job is never paused.

### `POST /v1/workflows/z-image-turbo/jobs`

Same as `flux2-klein` with fewer fields: `prompt`, `output`, `seed`, `count` (same rules). There is no
`quality` (one model) and no `reference_images` (text only); sending either is `422`.

## Workflow `flux2-klein`

- Output: PNG, 8-bit sRGB, no alpha, exact `output.width × output.height`.
- Size: each image is generated at the engine's supported size (multiples of 64, about one megapixel)
  closest to the requested aspect ratio, then **cover + centre crop** (lanczos) to the exact size.
  The aspect ratio is never distorted and there are no bars.
- `draft` and `standard` map to engine settings chosen by the server (e.g. a smaller model for `draft`).
- One job runs at a time.
- Consecutive `standard` jobs may run faster than the first one: the server can keep the engine warm between
  them and releases it when idle, before a `draft` job, or when another GPU server needs the memory. This
  changes only timings; `estimate_seconds` and `typical_seconds` stay conservative (cold) values.

## Workflow `z-image-turbo`

- Z Image Turbo (Tongyi-MAI, Apache-2.0): text to image, delivered exactly like `flux2-klein` (PNG,
  8-bit sRGB, no alpha, exact size, cover + centre crop from a ~1 MP generation size).
- Steps, guidance and sampler are the model's recommended settings (8 steps, CFG 1), set by the server.
- Measured: 33 s per image (56 s the very first run). Its generation sizes reuse the table verified with
  FLUX.2 [klein]; only the timing above is measured for this model, its peak memory is not.
- It never shares the resident engine with `standard` jobs: a `z-image-turbo` job first releases it.
