# Literature updates

Running log for `/literature-watch`, newest entry first. Run it about weekly.

Last checked: 2026-09-29

## Log

### 2026-09-29 — First check

**CLM itself**
- The official citation is a Notion blog post with a full author list (Kwok, Kang, Suresh, Saad-Falcon, Pavone, Ré,
  Mirhoseini). There is no arXiv paper yet. Recorded in `clm.md`. Sources:
  https://github.com/Contrastive-LM/CLM, https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- The base checkpoint has not changed since 2026-09-24 (HF last modified). Our pinned `clm-v0.1-mlx/` is still
  current.
- A new public heads checkpoint, `Contrastive-LM/deepswe-clm-heads-8k`, contains the fine-tuned heads behind the
  DeepSWE 81.6% result, and its model card documents the fine-tuning settings. It is a useful reference for our
  traffic-specialization recipe. https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k
- The repo was published to PyPI as `contrastive-lm` (2026-09-24) and serves with `clm-serve`. `/v1/rank` still
  exists; a new typed-question endpoint `/v1/systemone` was added.
- **States over 2048 tokens are truncated by default.** Raise the limit with `--max-model-len` / `--max-tokens`.

**New CLM models**
- CLM-35B: the HF model card (primary source) confirms "a multimodal CLM-35B ... coming in early October". It is not
  released yet. The "CLM-35B-A3B" mixture-of-experts name is unconfirmed (secondary sources only).

**Similar models**
- Jev (TypeSafe AI), announced 2026-09-15: closed, early access, typed calibrated decisions, architecture
  unpublished. New note: `jev.md`.
- ThinkFlowLab/system1-omni: an open project serving several System One models through one `/v1/systemone`
  interface. It already supports a model called **LAYA**; an RFC (2026-09-27) proposes adding CLM. LAYA's origin,
  weights and architecture could not be confirmed, so treat it as unconfirmed.
  https://github.com/ThinkFlowLab/system1-omni/issues/9
- Unverified lead for next time: "CWM: Contrastive World Models for Action Feasibility Learning in Embodied Agent
  Pipelines" (arXiv:2602.22452).

**Language models for traffic signal control**
- New note `llm-traffic-control.md` with 8 verified works. The newest is ProcessLight (arXiv:2609.22746,
  2026-09-19, EMNLP 2026). All use generative LLMs; **none uses a contrastive / System One model**, so the novelty
  claim holds.

**Independent evaluations**
- None found. All CLM-8B vs Jev numbers are still self-reported by CLM's authors. Secondary coverage cites
  author-reported cases where Jev wins: BFCL v4 tool-calling 99.2% vs CLM 95.2%, and WikiRacing 30/30 vs 26/30.

**Candidate citations** (for `/research-assistant` to verify and add)
- Kwok et al., "Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making", 2026,
  Notion blog, https://contrastive-lm.notion.site. Replaces the `clm2026` placeholder in `references.bib`.
- Lai et al., "LLMLight: Large Language Models as Traffic Signal Control Agents", KDD 2025,
  doi:10.1145/3690624.3709379
- Yuan, Lai, Liu, "CoLLMLight", arXiv:2503.11739
- Zhao, Zhou, Wang, Shi, Rao, "ProcessLight", EMNLP 2026, arXiv:2609.22746
- Ding, Mukande, Liu, "HiLLTS", arXiv:2607.22691
- Pang et al., "ReasonLight", arXiv:2605.29425
- Zhang et al., "LATS", arXiv:2603.24361
- Shi, "LLM-Augmented Traffic Signal Control ...", arXiv:2604.23902
- TypeSafe AI, "Introducing System One Models & Jev", blog, 2026-09-15

**Implications (need a decision)**
1. **2048-token truncation:** check that our text state descriptions (`describe_state`) fit, especially for
   big-intersection and the 4x4 grid. Truncation would silently remove lanes from the state. Suggest adding a
   RESEARCH_TODO item: measure description token counts per scenario and raise the limit if needed.
2. **CLM-35B, due early October:** decide whether to add it as a third CLM controller. It would strengthen RQ1/RQ2
   (scaling) at a higher compute cost. Suggest adding it as an optional item in Phase 2.
3. **LLMLight/LightGPT as a baseline:** it is the closest generative analogue to our base-vs-specialized setup.
   Consider whether to run it as a generative-LLM baseline, or only discuss it.
4. **Serving stack:** we use Rapid-MLX; the official stack is `clm-serve`. For reproducibility, check that both give
   the same rankings on a sample of states, or document which one produced the results.

### 2026-09-27 — Baseline

Starting point: CLM-8B (`Contrastive-LM/CLM-v0.1-8B`, announced 2026-09-23) as described in `clm.md`. Pinned local
checkpoint for the base model: `clm-v0.1-mlx/`. Known related models: Jev (TypeSafe). Announced but not released:
CLM-35B (reportedly early October 2026).
