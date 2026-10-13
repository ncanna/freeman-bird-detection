# System One detector experiments

Harness for asking **System One** models typed questions about Freeman camera-trap frames. Same role in this repo as `detr_detector/`: a self-contained package you import or run from the repo root. It is not an `hlwdetector` adapter, and its YAML files are not `ExperimentConfig` files.

System One models do not emit boxes or free text. You send a **state** and a map of **questions**. Each question is one of:

| type | asks | returns |
| --- | --- | --- |
| `noul` | yes/no | probability the answer is yes |
| `choice` | pick one option you defined | option, full distribution, confidence |
| `score` | rate on an ordered rubric (2–10 levels) | probability-weighted score, distribution, confidence |

Detection is mapped onto that interface. COCO boxes become gold labels for three questions:

- `bird_present` — noul, at least one bird
- `bird_count` — score over none / one / 2–3 / 4+
- `largest_bird_cell` — choice over a 3×3 grid, plus `none`

The text state is only the frame name and video stem. It does not include the labels.

## Which model actually sees the frame

Checked against each project's own docs on 2026-10-02.

| backend | model | sees pixels? | how this package calls it |
| --- | --- | --- | --- |
| `jev` | [TypeSafe Jev](https://docs.typesafe.ai/introduction) | no — text, JSON, or arrays of text only | `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`, model `jev-latest` |
| `laya` | [Laya](https://github.com/NandhaKishorM/laya) (`convaiinnovations/laya`) | no — text state | `laya.Router.predict(state, questions)` or `laya.load(...).predict` |
| `jev_omni` | [akhilaaa3/Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) | yes — image, audio, or video | `classifier.predict(state=, question=, options=, media=, modality=)`. One call per question. CUDA. Not affiliated with TypeSafe. |
| `clef` | [Cloudflare Clef](https://developers.cloudflare.com/workers-ai/models/clef/) / Clef-Flash | yes — up to 4 images | Workers AI `POST /client/v4/accounts/$ID/ai/run/@cf/cloudflare/clef`. Vision encoder, 64k context. Apache-2.0 weights. |
| `djev` | [Djev](https://djev.dev/) (DiffusionGemma) | yes — one image | `POST /v1/request`, not `/v1/systemone`. Invite-only. Handles 202 receipts. |
| `jevimage` | [jevimage](https://pypi.org/project/jevimage/) | yes — image | One dual-encoder pass, then a matmul per question. Not a VLM. |
| `omni_jev` | [tinnel123/OmniJev](https://huggingface.co/tinnel123/OmniJev) v1.1 | yes — images as a numbered panel | 4B Qwen3.5 LoRA plus decision heads. Local weights. **Not** OmniJev/OneJev-27B. |
| `openai_decisions` | [OpenAI Decisions API](https://the-decoder.com/openai-expands-codex-and-its-api-at-devday-with-security-scans-a-decisions-api-and-ultrafast/) | yes — text or images | `POST /v1/decisions`. Limited preview, still gated. |
| `systemone_http` | any `POST /v1/systemone` server | only if you set `send_media: true` | OneJev-style `media: [{type, data: data-uri}]` plus an `<image:1>` placeholder. Use this for [OneJev](https://huggingface.co/OmniJev/OneJev-27B), laya-serve, Kev, Decider. Do not send `media` to the TypeSafe API. |

New in the ecosystem (checked 2026-10-08):

| model | what | sees pixels? | status in this package |
| --- | --- | --- | --- |
| [Liquid d1-3B](https://huggingface.co/LiquidAI/d1-3b) | decision model from the LFM2.5-VL-3B backbone; 48.57 on the Jev Decision Index v0.2.1 (best under 10B); single forward pass, no generated tokens; 8 ms/question on a 4090, 62 ms for a 384px image on an M5 Pro | yes (text+image) | **not yet registered** — best next backend: open weights, runs locally via llama.cpp; hosted path is the TypeSafe schema |
| [Liquid d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) | experimental 600M: text+image or text+audio, encoder-based | yes | not registered; ultra-edge option |
| [Cloudflare Clef / Clef-flash](https://huggingface.co/Cloudflare/clef) | 27B / 9B open-weight decision models, Apache-2.0, frozen Qwen backbone + rank-256 LoRA + routing head, RLCD calibration training; Jev Decision Index leader at release | yes — the Workers AI docs now describe clef-flash as reading text, JSON, **images or video** (registered 2026-10-02 as images-only; re-verify the live schema before sending video) | registered as `clef` (hosted; images-only guard in `models/media.py` — likely stale) |
| [Kev 9B](https://huggingface.co/jaredpalmer/kev-9b) | open decision model | text | not registered; the `systemone_http` backend is the intended path |
| Decider 2B/4B/35B-A3B | decision model family benchmarked alongside d1 | text | not registered; `systemone_http` path |

Cloudflare also launched an **RL fine-tuning service** for Clef (FDE-led, self-serve platform later): AI Gateway request capture → rollouts → RL sandbox → redeploy on Workers AI. That is a hosted alternative to the PACE LoRA tier in `h55_experiment/README.md`.

The shared [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) benchmark now versions a **private vision split** (v0.3) — camera-trap decisions are a candidate contribution this project could make.

Run `python -m system_one_detector.main --list-backends` to see what is registered.

Jev and Laya on these frames are a **text-metadata baseline**. They never see the picture. For a visual experiment, use `clef`, `jev_omni`, `djev`, `omni_jev`, or a multimodal `/v1/systemone` server.

Jev-Omni, Clef, and Djev each hold up better with about 20 choice options or fewer. The grid question has 10. `primary_category` is available only when the COCO category list plus `none`/`other` stays at or under 20; add it to `question_ids` to turn it on.

### Per-backend caveats

- **Clef** takes images only. Its input schema has no audio or video field, so a video frame must be sent as an image. Cloudflare's published Clef numbers are text decision benchmarks, not vision results.
- **Djev** is a diffusion sample, so `seed` matters. Hold state, image bytes, question wording, option order, and `options` fixed when comparing runs. `isolation: independent` re-sends the state per question and multiplies input tokens. The endpoint returns `202` with a receipt more often than a direct answer; the adapter polls until the envelope says `succeeded`, because `HTTP 200` alone does not mean inference worked.
- **jevimage** is an embedding model. Cheap for many questions on one frame, but weak when the decision needs reasoning over the scene rather than similarity. It also has a `head` question type trained on your own labelled images; this harness does not build those.
- **omni_jev** v1.1 carries a noul bias (about 0.053 on the 4B checkpoint) that moves the yes/no boundary. The adapter reads it from `release_manifest.json` unless you set `noul_bias`. A raw 0.5 threshold on the released head is not the serving path.
- **openai_decisions** has no published schema and a standard key gets HTTP 403. The adapter sends the shape the rest of the ecosystem uses and reads whichever answer envelope comes back; expect to edit `_build_payload` and `_read_answers` when the real schema ships. For the same job today on an OpenAI key, use `gpt-6-luna` with reasoning set to none and a strict enum schema.

## Layout

```
system_one_detector/
  main.py            # CLI
  engine.py          # one decide() per frame, then metrics
  hubconf.py         # jev / laya / jev_omni / systemone_http builders
  models/            # backends. New ones register with @register_backend
  datasets/          # split filtering + question bank + COCO labels
  util/              # System One schema, YAML config, accuracy / Brier / ECE
  configs/           # not hlwdetector configs
```

## Run

From the repo root:

```bash
python -m system_one_detector.main system_one_detector/configs/laya_h23.yaml --dry-run --limit 4
```

`--dry-run` uses the mock backend: it lists the split, writes `decisions.json` and `metrics.json`, and does not download weights or call an API.

```bash
# local Laya (pip install laya). Text baseline.
python -m system_one_detector.main system_one_detector/configs/laya_h23.yaml

# TypeSafe Jev. Export TYPESAFE_API_KEY first. Text baseline.
python -m system_one_detector.main system_one_detector/configs/jev_h23.yaml

# Jev-Omni, sees the PNG. Needs the HF requirements and a CUDA GPU.
python -m system_one_detector.main system_one_detector/configs/jev_omni_h23.yaml

# OneJev, or any other /v1/systemone server, after `qev serve`.
python -m system_one_detector.main system_one_detector/configs/onejev_h23.yaml

# Cloudflare Clef. Set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_AUTH_TOKEN.
python -m system_one_detector.main system_one_detector/configs/clef_h23.yaml

# Djev hosted preview. Set DJEV_API_KEY (invite-only).
python -m system_one_detector.main system_one_detector/configs/djev_h23.yaml

# jevimage, encoder in this process.
pip install 'jevimage[local]' && python -m system_one_detector.main system_one_detector/configs/jevimage_h23.yaml

# OmniJev v1.1, after the clone in its config comment.
python -m system_one_detector.main system_one_detector/configs/omni_jev_h23.yaml
```

Outputs land in `outputs/<config_name>_<timestamp>/`.

`--dry-run` works with any of these. It swaps in the mock backend, so it checks split filtering, question building, and metric scoring without an API key, weights, or a GPU.

From a notebook, the same way `wz_run_detr_model.ipynb` imports `detr_detector`:

```python
from system_one_detector.hubconf import clef
from system_one_detector.datasets.questions import build_questions

backend = clef("clef-flash")
result = backend.decide(
    {"source": "freeman camera trap", "frame_name": "IMG_0158_frame_000218.png"},
    build_questions(["bird_present"]),
    media_path="data/h23/images/IMG_0158_frame_000218.png",
    modality="image",
)
print(result.answers["bird_present"].noul)
```

## Adding a backend

Subclass `DecisionBackend`, implement `decide` and `from_config`, and decorate with `@register_backend("name")`. Import the module from `models/__init__.py`. Return `DecisionResult` filled with `Answer` objects so `engine.py` can score them without knowing the server.

Check the input shape before credentials. A caller that sent audio to an image-only endpoint should hear that, not a missing API key. `post_json_raw` in `models/http.py` gives you the status and headers for APIs that answer with a receipt instead of a result, and `unwrap_result` handles provider envelopes around the System One body.

`python -m system_one_detector.test_scaffold` checks labels, the Jev response parser, the Jev-Omni option translation, the Clef and Djev envelope shapes, media-type guards, the OmniJev noul bias, the registry, and a mock eval. It does not hit the network.

## Local MLX proof of concept

`jev_omni_mlx_poc.py` runs the community 4-bit MLX conversion of Jev-Omni
([Ruiruiz30/Jev-Omni-MLX-4bit](https://huggingface.co/Ruiruiz30/Jev-Omni-MLX-4bit))
over a directory of frames and serves a browser UI showing each frame beside its
probabilities. One file, no framework, stdlib HTTP server.

It needs its own venv. The checkpoint requires `gemma4_unified`, which only
exists in `mlx-vlm` 0.7.x, and the repo's pinned `mlx-vlm 0.6.4` plus a
`transformers`/TensorFlow/`pandas` numpy ABI clash both break the shared env.

```bash
python -m venv .venv-jevomni
.venv-jevomni/bin/pip install "mlx-vlm==0.7.4" pillow
hf download Ruiruiz30/Jev-Omni-MLX-4bit --local-dir models/Jev-Omni-MLX-4bit

.venv-jevomni/bin/python system_one_detector/jev_omni_mlx_poc.py --images train_images --open
```

Flags: `--image-tokens {10,20,35,70,140,280}`, `--questions`, `--limit`,
`--calibration off`, `--json out.json`, `--port`, `--no-browser`.

This bypasses the `jev_omni` backend in `models/`, which targets the author's
CUDA loader. The MLX conversion is an independent third-party quantization with
its own classifier and its own benchmark; its reported 85.90% is on public
JevBench, not camera-trap frames.

Results are in `poc_results/`. There are no ground-truth labels for
`train_images`, so those files record model behaviour and confidence spread, not
accuracy.

## H55 experiment: video sample → frames → System One → COCO

`h55_experiment/README.md` documents a full pass over a seeded random sample of
5 videos from `data/H55/`: reproducible sampling
(`tools/sample_videos.py`), stride-based frame extraction
(`tools/video_to_frames.py`), Jev-Omni MLX decisions, and a per-frame COCO
mapping (`tools/coco_from_decisions.py`). Start there to rerun or extend it.

## Local multimodal decision-model evaluation

The frame harness sends the full question map for a frame in one request.
The H55 question set covers presence, count, coarse 3x3 location, and an
exploratory behavior label. Behavior predictions are recorded, but they are
not scored unless behavior annotations are provided. A still frame may not
contain enough evidence to determine an action; the `unclear` option is
intentional.

The local HTTP adapter supports Ollama's `/v1/systemone` image request format.
This lets Ollama-hosted decision models share the same dataset loader,
questions, output schema, and metrics. Start with the local Clef build:

```bash
ollama pull clef
python -m system_one_detector.main \
  system_one_detector/configs/h55_ollama_clef.yaml --limit 20
```

To compare models supported by the same Ollama endpoint, pull them and run a
matched sweep. Every model receives the same frame subset and ordered question
map; each run writes a separate output directory.

```bash
ollama pull clef-flash
python -m system_one_detector.sweep \
  system_one_detector/configs/h55_ollama_clef.yaml \
  --models clef clef-flash --limit 20
```

`media_style: ollama` sends the image as raw base64
in `images`; `media_style: onejev` retains the data-URI `media` envelope for
OneJev-compatible servers.

This is one client/server request per frame with multiple named questions. It
does not guarantee that every backend computes all answers with one shared
vision pass: Jev-Omni's current adapter invokes its classifier once per
question. The current pipeline evaluates frames independently;
temporal aggregation and decision reuse require a separate video-level
evaluation because carrying a probability forward can miss a bird's departure.
