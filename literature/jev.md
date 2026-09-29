# Jev (TypeSafe AI)

Notes collated 2026-09-29. Primary source: TypeSafe AI blog, "Introducing System One Models & Jev", 2026-09-15,
https://typesafe.ai/blog/introducing-system-one-models-and-jev

## What it is

- TypeSafe AI's first "System One model" (the term is TypeSafe's, after Kahneman's System 1 / System 2). TypeSafe
  introduced the category a week before CLM-8B was announced.
- "Unstructured state in, typed probabilistic decisions out": it takes program state as text and returns typed
  values with calibrated probabilities instead of generating free text.
- Uses parallel sampling that computes all outputs in one pass, not autoregressive decoding.
- Trained with "Reinforcement Learning for Calibrated Decisions" (RLCD), aimed at honest probabilities.
- Claims: 40-200x faster than frontier LLMs (70-500 ms end to end), no hallucinations or type errors.
- Pricing: $0.042 per million input tokens, output free.
- **Availability:** early access via TypeSafe's platform (waitlist); closed weights; architecture unpublished.
- Company: came out of stealth 2026-09-15 with a reported $40M seed (secondary sources only; unconfirmed).

## Relevance to our research

- Jev is the closest rival to CLM, and its outputs are *calibrated* probabilities, whereas CLM scores are not
  calibrated.
- It is closed and API-only, so reproducible experiments with it would be hard. At most it could be an optional
  extra controller if we get early access. Mention it in related work as the other System One model.
- CLM's reported Jev comparisons are self-reported by CLM's authors; see `clm.md`.
