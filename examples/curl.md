# curl examples

```sh
API=http://127.0.0.1:8767
AUTH=()                                   # or: AUTH=(-H "Authorization: Bearer $KLEIN_API_TOKEN")
```

## Is it up? What can it do?

```sh
curl -s "${AUTH[@]}" "$API/healthz"
# {"status":"ok","version":"0.1.0","running":0,"waiting":0,"max_waiting":2}
curl -s "${AUTH[@]}" "$API/v1/capabilities" | jq '{queue, coordination, w: .workflows[0] | {count, quality, reference_images, typical_seconds}}'
```

## Text → two 16:9 images

```sh
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/flux2-klein/jobs" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: poster-001' \
  -d '{"prompt": "A lighthouse on a cliff at dusk, long exposure, crashing waves",
       "output": {"width": 1920, "height": 1080}, "quality": "standard", "seed": 42, "count": 2}'
# → 202 {"id":"job_…","workflow":"flux2-klein","status":"queued","created_at":"…"}
# Sending the same Idempotency-Key + body again returns the same job (200), never a duplicate.
```

## With a reference image (only when `reference_images.max` > 0)

```sh
jq -n --arg img "$(base64 -i character.png)" '{
  prompt: "The same woman in a yellow raincoat, standing on a tram platform at night",
  output: {width: 1080, height: 1350},
  quality: "draft",
  reference_images: [{data: $img, media_type: "image/png"}]
}' > body.json
curl -s "${AUTH[@]}" -X POST "$API/v1/workflows/flux2-klein/jobs" \
  -H 'Content-Type: application/json' --data-binary @body.json
# → 422 invalid_params while the server has reference images disabled (max 0)
```

## Poll, download, cancel

```sh
JOB=job_…
curl -s "${AUTH[@]}" "$API/v1/jobs/$JOB" | jq '{status, progress, queue_position, waiting_for, timings, error}'
curl -s "${AUTH[@]}" -o image-0.png "$API/v1/jobs/$JOB/output"           # index 0; 409 until succeeded
curl -s "${AUTH[@]}" -o image-1.png "$API/v1/jobs/$JOB/output?index=1"   # 404 outside 0 … count-1
curl -s "${AUTH[@]}" -X DELETE "$API/v1/jobs/$JOB"                        # 200 canceled / 202 stopping
```

## Waiting for another GPU server

With `[[coordinate_with]]` configured, the first job in line stays queued while e.g. vpipe-api is busy:

```json
{ "status": "queued", "queue_position": 1, "waiting_for": "vpipe-api", "progress": null }
```

## Handling `429 busy`

One job runs and `max_waiting` (default 2) wait. When full, the server answers before reading the body:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 30
{"error":{"code":"busy","message":"the GPU slot and the waiting queue are full","retryable":true,"details":null}}
```

Wait `Retry-After` seconds and submit again. Nothing was queued.
