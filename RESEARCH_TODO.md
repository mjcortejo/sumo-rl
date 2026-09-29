# Research TODO — CLM for Traffic Signal Control

Goal: a paper on whether Contrastive Language Models (CLM-8B) are useful for traffic signal control, compared against
fixed-time, max-pressure and Q-learning (plus deep RL) baselines in sumo-rl.

Drafted 2026-09-28.

## Current state

- `sumo_rl/agents/clm_agent.py`: CLM agent (text description of the intersection, ranks phases via `/v1/rank`)
- `sumo_rl/agents/max_pressure_agent.py`: max-pressure baseline
- `experiments/clm_2way-single-intersection.py`: runner with `clm`, `maxpressure`, `fixed` and `ql` controllers
- `experiments/clm_collect_labels.py`: collects lookahead labels for CLM fine-tuning
- `experiments/analyze_runs.ipynb`: analysis notebook
- `outputs/2way-single-intersection/`: one run per controller

Gaps: only one seed, only one scenario, only zero-shot CLM and no statistical tests yet.

## Phase 0 — Frame the research

- [ ] Write the research questions:
  - **RQ1**: Can zero-shot CLM beat fixed-time and match max-pressure?
  - **RQ2**: Does a traffic-specialized CLM-8B beat the base CLM-8B, and does it match or beat Q-learning or DQN?
  - **RQ3**: How much data does each method need? Compare labelled lookahead samples for CLM with environment steps
    for RL.
  - **RQ4**: Does CLM generalize to unseen intersections and demand patterns?
  - **RQ5**: Is it fast enough for real-time control?
- [ ] Literature review: grow `literature/clm.md` into related work
  - Classic control: Webster, max-pressure
  - RL for signal control: sumo-rl, RESCO, PressLight, CoLight
  - LLM-based signal control: LLMLight, LA-Light, iLLM-TSC, and the others in `literature/llm-traffic-control.md`
  - Other System One models: Jev (TypeSafe) is discussed in related work only, not run as a controller, because it is
    paid, closed and early-access (decided 2026-09-29; see `literature/jev.md`)
  - Novelty angle: CLM ranks phases instead of generating text (cheaper, deterministic)
- [ ] Weekly: run `/literature-watch` to check for CLM updates, similar models and related papers (log in
  `literature/updates.md`)
- [ ] Choose the target venue (e.g. IEEE ITSC, T-ITS, TRB). It sets the page limit and how much evaluation reviewers
  will expect.

## Phase 1 — Make the setup sound

- [x] Commit the current work, including the `env.py` `encode` fix (branch `clm-research`)
- [ ] Pin SUMO, Python and CLM checkpoint versions
- [ ] Fixed-time baseline: use a proper Webster-timed plan, not just the network's default program (a naive plan
  looks like a strawman)
- [ ] Q-learning baseline: set a training budget (episodes), plot convergence and report the final greedy policy
- [ ] Add DQN and PPO baselines (the `dqn_*` and `sb3_*` scripts exist)
- [ ] **Generative LLM baseline**: LLMLight / LightGPT (KDD 2025), the closest generative counterpart to our base vs
  specialized CLM comparison (see `literature/llm-traffic-control.md`)
  - [ ] Check that the LLMLight code and LightGPT weights are available and run with SUMO (LLMLight was built on
    CityFlow), or pick another open generative LLM controller
  - [ ] Give it the same state information and phase set as the CLM agent, so the comparison is ranking vs
    generation, not different inputs
  - [ ] Measure its decision latency on the same hardware (feeds RQ5)
- [ ] Same constraints for every controller: min/max green, yellow time, decision interval (`delta_time`)
- [ ] Same tuning budget for every method, or explain why the budgets differ
- [ ] Fix the metrics: average waiting time, travel time, queue length, throughput, delay at entry, CO2 (optional), CLM
  decision latency

### CLM serving

- [x] Token limit: `CLMClient` counts context and answer tokens with the Qwen3-8B tokenizer and raises
  `ContextTooLongError` above `max_context_tokens` (default 2048, `-maxtokens` in the runner) instead of letting the
  server truncate silently; `signals.csv` logs `context_tokens` per decision
- [x] Measure the token budget per scenario: `experiments/clm_token_budget.py` (results in `outputs/token_budget/`)
- [ ] Check that Rapid-MLX (what we run) and the official `clm-serve` rank phases the same way:
  `experiments/clm_compare_servers.py` on the contexts in `outputs/clm_contexts.jsonl`. `clm-serve` needs vLLM on a
  CUDA GPU (24 GB fits the 2048-token default), so this needs access to a GPU machine
  - [ ] If they agree, say so in the paper and run everything on Rapid-MLX; if not, decide which server is the
    reference and record the server in every run's `meta.json`

### Train / validation / test environments

Any method that learns from simulation (Q-learning, DQN, PPO, the traffic-specialized CLM) must be scored on traffic
it never saw. Non-learning controllers (fixed-time, max-pressure, base CLM-8B) are only run on the test split, so every
method is compared on the same test episodes.

- [ ] Define the splits per scenario:
  - **Train**: SUMO seeds and route files used for RL training and for collecting CLM fine-tuning labels
  - **Validation**: separate seeds for hyperparameter tuning and checkpoint selection (e.g. Q-table episode, CLM
    fine-tuning epoch)
  - **Test**: held-out seeds, touched only for final results
  - **Out-of-distribution test**: unseen demand levels/patterns and unseen intersections (feeds RQ4)
- [ ] Generate route files per split (different random trips/flows, not just different SUMO seeds) so test traffic
  genuinely differs from train traffic
- [ ] Store the split definitions in a config file (e.g. `experiments/splits.yaml`) that all runners and
  `clm_collect_labels.py` read, so nothing can accidentally train on test data
- [ ] Freeze learned policies before testing: greedy Q-table / no exploration, no online updates during test episodes
- [ ] Log which split and checkpoint produced each output folder in `meta.json`

## Phase 2 — Experiments

- [ ] Multiple seeds: at least 10 per controller and scenario
- [ ] Scenarios, in order of increasing difficulty:
  - [ ] single-intersection
  - [ ] 2way-single-intersection
  - [ ] big-intersection (its network files are missing from `sumo_rl/nets/`; find them or drop it)
  - [ ] 4x4 grid
  - [ ] RESCO benchmarks (Cologne, Ingolstadt)
- [ ] Demand variations: low, medium and high volume, unbalanced flows, a rush-hour spike
- [ ] CLM variants (both base and specialized are reported side by side in every results table):
  - [ ] **Base CLM-8B** (the current `clm-v0.1-mlx` checkpoint, zero-shot, text prompt only). Keep this checkpoint
    pinned and unchanged as the reference.
  - [ ] Prompt ablation on base CLM-8B: other questions, state descriptions and phase descriptions (prompt
    sensitivity)
  - [ ] **Traffic-specialized CLM-8B**: fine-tune the projection heads on lookahead labels from
    `clm_collect_labels.py`, collected on the train split only
    - [ ] Follow the recipe of the official DeepSWE heads (`Contrastive-LM/deepswe-clm-heads-8k`): initialize from
      `CLM_v0.1-8B.pt`, `train/finetune.py --task clm`, batch 512, up to 20 epochs with early stopping (patience 5)
      on the validation split, held-out tasks for testing (details in `literature/clm.md`)
    - [ ] Decide the label source/teacher (lookahead reward, max-pressure, or a trained RL policy) and document it
    - [ ] Vary lookahead horizon and dataset size (feeds RQ3)
    - [ ] Save the specialized heads as a separate, versioned checkpoint (e.g. `clm-v0.1-traffic-mlx/`)
    - [ ] Check that the base prompt format is identical for both models, so the only difference is the weights
  - [ ] Generalization: specialize on scenario A, test base vs specialized on B
  - [ ] **CLM-35B** (optional; multimodal, announced for early October 2026): once released, run it zero-shot as a
    third CLM controller for a scaling comparison. Check its token limit, hardware needs and serving support first
- [ ] Latency and compute: time per decision, cache hit rate (`CLMClient` LRU cache), hardware; compare with Q-table
  lookup and DQN inference
- [ ] A single sweep script that runs everything and saves to a consistent `outputs/` layout

## Phase 3 — Analysis

- [ ] Extend `analyze_runs.ipynb` to combine results across seeds and report means with 95% CIs
- [ ] Significance tests (Wilcoxon or Mann-Whitney, corrected for multiple comparisons)
- [ ] Figures:
  - [ ] Queue and waiting time over time
  - [ ] Box plots per controller
  - [ ] Learning curve for Q-learning and DQN against the flat CLM line
  - [ ] Performance against the amount of fine-tuning data (base CLM-8B as the zero-data point)
  - [ ] Base vs specialized CLM-8B, in-distribution and out-of-distribution
  - [ ] Latency
- [ ] Qualitative analysis: situations where CLM phase choices differ from max-pressure, using the CLM scores in
  `signals.csv` (interpretability)
- [ ] Failure analysis: where CLM does badly (e.g. starving a phase, reacting to spillback) and why
- [ ] Discussion: ranking (CLM) vs generation (LLMLight): performance, latency, determinism, cost, and how
  specialization compares (fine-tuned heads vs a fine-tuned LLM such as LightGPT)

## Phase 4 — Paper

The paper is written alongside the research in `paper/` (LaTeX, IEEE conference format for now). Use
`/research-assistant [task]` to draft sections, generate result tables/figures from `outputs/`, manage citations and
build the PDF (`cd paper && latexmk main.tex` → `paper/build/main.pdf`).

- [x] Set up the LaTeX skeleton and the `/research-assistant` skill
- [ ] Write scripts in `paper/scripts/` that generate every results table and figure from `outputs/`

- [ ] Outline: Intro, Related Work, Method (text descriptions of state and phases, CLM ranking, fine-tuning labels),
  Experimental Setup, Results, Discussion and Limitations, Conclusion
- [ ] Write the Method section early
- [ ] Limitations: simulation only, sim-to-real gap, reliance on an 8B encoder, how new CLM is
- [ ] Reproducibility: release code and configs, document seeds, share model weights or the fine-tuning recipe
- [ ] Internal review, revise, submit

## Suggested immediate next steps

1. Commit the current state.
2. Define the train/validation/test splits for 2way-single-intersection.
3. Rerun all four controllers on 2way-single-intersection with 10 test seeds. This quickly shows whether zero-shot CLM is
   competitive at all, which decides how the paper is framed.
4. Build the Webster fixed-time baseline.
