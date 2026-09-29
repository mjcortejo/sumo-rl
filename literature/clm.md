# Contrastive Language Models (CLM)

Research notes — collated Sept 27, 2026

Announced Sept 23, 2026 Stanford / Hazy Research / NVIDIA Apache 2.0

## What it is

A **Contrastive Language Model (CLM)** is a new class of "System One" model — one that *judges* rather than *writes*. Instead of generating tokens, it scores candidate actions against a current state using two encoders, and picks the best match. Announced by Jacky Kwok, joint work with Hazy Research (Stanford) and NVIDIA Research.

The first released model is **CLM-8B**. Despite the name, the trained artifact is small — two lightweight projection heads (roughly 20M trainable parameters each, \~40M total) sitting on a **frozen Qwen3-8B** backbone. Most of the "8B" is the frozen encoder, not something newly trained.

## Official details (verified 2026-09-29 against the HF model card and GitHub README)

- **Citation** (from the repo; a Notion blog post, not an arXiv paper yet): Jacky Kwok, Hangoo Kang, Tarun Suresh,
  Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini. *Contrastive Language Models: A System One Model
  for Fast and Generalizable Decision-Making.* 2026. https://contrastive-lm.notion.site (BibTeX key
  `kwok2026contrastivelanguagemodels`)
- **Dates:** HF repo `Contrastive-LM/CLM-v0.1-8B` created 2026-09-21, last modified 2026-09-24; public announcement
  2026-09-23. GitHub repo `Contrastive-LM/CLM` initial commit 2026-09-24.
- **Serving:** official server is `clm-serve` (PyPI package `contrastive-lm`, published 2026-09-24). Endpoints:
  `POST /v1/rank` (what `sumo_rl/agents/clm_agent.py` uses), `POST /v1/systemone` (typed questions: `noul`,
  `choice`, `score`), `GET /v1/models`, `GET /health`. The README does not mention Rapid-MLX, which we use on Apple
  Silicon; that is a third-party route.
- **Context limit:** states longer than **2048 tokens are truncated** by default. Raise with vLLM
  `--max-model-len 8192` plus `clm-serve --max-tokens 8192` (needs more GPU memory).
- **Fine-tuned heads:** `Contrastive-LM/deepswe-clm-heads-8k` (created 2026-09-22) is the heads checkpoint behind the
  DeepSWE result, initialized from `CLM_v0.1-8B.pt`, trained on 59 tasks (7 validation), batch 512, up to 20
  epochs, patience 5. Held-out: 31/38 = 81.6% (pass@1 73.7%, oracle 89.5%). A reference for our own specialization
  recipe.
- **Fine-tuning:** `python train/finetune.py --task clm --init-ckpt "$(clm-download)" ...`.
- **CLM-35B:** the HF model card itself says "A multimodal CLM-35B, trained with more data, compute and parameters for
  stronger generalization, is coming in early October." A mixture-of-experts "CLM-35B-A3B" name appears only in
  secondary sources (unconfirmed).

## How it works

### Training

Trained with a bidirectional **InfoNCE contrastive loss** that pulls correct state–action pairs together in embedding space while pushing incorrect ones apart.

- Pre-trained on 60M Nemotron Q&A pairs
- Mid-trained on 30M synthetic hard negatives
- Post-trained on 1M agentic trajectories

### Inference

State and actions are embedded **independently** by separate heads, so action embeddings can be cached and reused across calls. Scoring is a dot product between state and action embeddings — no generation involved.

## CLM-8B vs. Jev

**Jev** is a rival "System One" model from a company called **TypeSafe**, which introduced the category earlier in September 2026. CLM is positioned as a faster, more flexible alternative.

#### Jev (TypeSafe)

See `jev.md` for details from TypeSafe's own announcement.

- Supports **state caching only** (per CLM's authors; TypeSafe has not published architecture details, noted
  2026-09-29)
- Must re-read state + all candidates on every call
- No public scaling-law data published

#### CLM-8B

- Dual-encoder: caches **state and action** embeddings independently
- Up to **9× faster** zero-shot inference — best case; varies by task: T-Rex \~9×, Super Mario \~4×, WikiRacing \~2.8×, tool-calling \~1.6×
- With \~1,000 candidates, **13× faster** than Jev (under optimal caching conditions)
- Published scaling laws (power-law loss vs. compute/size/data)

### Benchmark results (fine-tuned as a verifier)

| Benchmark | CLM-8B | Jev | Speed vs. Jev |
| --- | --- | --- | --- |
| DeepSWE (38 tasks) | 81.6% | 71.1% | 4.1–5.7× faster |
| Terminal-Bench 2.1 (30 tasks) | 87.6% | 83.1% |  |

Setup: a larger model (Opus 5 for DeepSWE, Fable 5 for Terminal-Bench) generates candidate solutions; CLM or Jev acts as verifier, ranking and picking the best one.

Note: these results come from **fine-tuned heads, not the released checkpoint** — don't expect them from CLM-8B out of the box.

## Key caveats

- **Numbers are self-reported** and only days old — not yet independently verified.
- The backbone is **frozen and fixed** to Qwen3-8B last-token-pooled embeddings — it can't be swapped for another encoder.
- A contrastive model **cannot generate** — it only ranks candidates you supply. It needs a separate generator model upstream (hence the DeepSWE/Terminal-Bench setup above).
- The "8B" framing is arguably misleading: the actual trained checkpoint is reported at \~75MB, since the encoder is frozen.
- Jev's internals are comparatively opaque — TypeSafe hasn't published equivalent scaling-law data.
- **Caching only pays off with a stable candidate set.** If candidates are regenerated from scratch on every call, there's nothing to cache — likely why the verifier speedups (DeepSWE 4.1–5.7×) sit below the headline numbers.
- Scores are **not calibrated probabilities** — useful for ranking, but not as confidence values or fixed thresholds without separate calibration.
- Best suited to a **bounded, stable, enumerable set of candidate actions** (finite action spaces, structured content).

## What's next

The team describes CLM-8B as one rung of a broader **scaling ladder**. A larger, multimodal **CLM-35B** is reportedly already in training with more data, compute, and parameters, with release announced for early October 2026 (per Sanity glossary). Code, model weights, and a playground are public on GitHub and Hugging Face under Apache 2.0.

## Open questions for further research

1. How does CLM-8B perform on tasks *outside* computer-use / gaming / tool-calling / coding-verification — is the "System One" framing general-purpose or narrow?
2. What does independent (non-author) benchmarking show, once third parties reproduce the DeepSWE / Terminal-Bench numbers?
3. How does the frozen-Qwen3-8B design compare against a from-scratch contrastive encoder — is freezing the backbone a genuine architectural insight or a cost-saving shortcut?
4. What exactly is Jev's architecture? TypeSafe's "System One" announcement predates CLM and seems to have prompted it — worth tracing that lineage.
5. Does the scaling law (loss as power-law in compute/size/data) hold up as CLM-35B trains, or is it an artifact of the small released model?
6. Practical angle: how would CLM-8B integrate into an existing agent/verifier pipeline (e.g. alongside Claude), given it only ranks and doesn't generate?

## Sources

- Jacky Kwok, CLM announcement (X/Twitter), Sept 2026x.com/jackyk02/status/2102905335925424285
- "What is a Contrastive Language Model (CLM)?" — Sanity glossarysanity.io/glossary/contrastive-language-model-clm
- Contrastive-LM/CLM-v0.1-8B model card — Hugging Facehuggingface.co/Contrastive-LM/CLM-v0.1-8B
- Contrastive-LM/CLM — GitHub repogithub.com/Contrastive-LM/CLM
- "Stanford and Nvidia's open CLM-8B..." — VentureBeatventurebeat.com/technology/stanford-and-nvidias-open-clm-8b-caches-reusable-agent-actions-and-runs-up-to-9x-faster-than-jev-in-tests
- "CLM-8B vs Jev: 9x Faster, 40M Trained Parameters" — analysis piecevirtualuncle.com/clm-8b-jev-comparison
- CLM-8B listing — There's An AI For Thattheresanaiforthat.com/model/clm-8b

Compiled from web search results, Sept 27, 2026. Treat benchmark figures as preliminary/author-reported pending independent verification.