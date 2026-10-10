# Research TODO — CLM for Traffic Signal Control

Goal: a paper on whether Contrastive Language Models (CLM-8B) are useful for traffic signal control, compared against
fixed-time, max-pressure and Q-learning (plus deep RL) baselines in sumo-rl.

Drafted 2026-09-28. Methodology review 2026-09-29 (items marked *(review)* were added or changed then). Second review
2026-09-29 against the code (items marked *(review 2)*).

## Current state

- `sumo_rl/agents/clm_agent.py`: CLM agent (text description of the intersection, ranks phases via `/v1/rank`)
- `sumo_rl/agents/max_pressure_agent.py`: max-pressure baseline
- `experiments/clm_2way-single-intersection.py`: runner with `clm`, `maxpressure`, `fixed` and `ql` controllers
- `experiments/clm_collect_labels.py`: collects lookahead labels for CLM fine-tuning
- `experiments/analyze_runs.ipynb`: analysis notebook
- `experiments/clm_token_budget.py`, `experiments/clm_compare_servers.py`: token budget per scenario, Rapid-MLX vs
  `clm-serve` comparison
- `outputs/2way-single-intersection/`: one run per controller (3000 s, seed 42). **Pilot only**, see below.

Gaps: only one seed, only one scenario, only zero-shot CLM and no statistical tests yet.

**Known problem with the pilot Q-learning runs** *(review)*: `ql` trained for 30 and 50 runs that all used SUMO seed 42
and the same route file, and the last run, played greedily, was scored on that same traffic. The Q-learning result
therefore overstates its quality and must not be compared with the other controllers. Treat every run in `outputs/`
as an exploratory pilot: use it to debug, never to tune on or to report.

## Scope *(review)*

The list below is large. To keep the paper finishable, work in this order and treat later tiers as optional:

- **P0, the core claim**: single-intersection and 2way-single-intersection, plus one multi-intersection network
  (RESCO grid4x4 or 4x4). Controllers: fixed-time (Webster), SUMO actuated, max-pressure, Q-learning, DQN, base CLM-8B,
  traffic-specialized CLM-8B, and the sanity controls. Train/validation/test splits, 10+ test seeds, statistics.
- **P1, strengthens it**: the CLM grounding ablations, prompt ablation, out-of-distribution tests, PPO, the DAgger
  relabeling round, Rapid-MLX vs `clm-serve` check.
- **P2, stretch**: LLMLight/LightGPT baseline, CLM-35B, remaining RESCO networks, decision-latency ablation.

If CLM is not the best controller, the paper is still publishable as a trade-off study (performance vs latency vs
training data vs compute, shown as Pareto plots). Plan the figures that way from the start.

## Phase 0 — Frame the research

- [ ] Write the research questions:
  - **RQ1**: Can zero-shot CLM beat fixed-time and match max-pressure?
  - **RQ2**: Does a traffic-specialized CLM-8B beat the base CLM-8B, and does it match or beat Q-learning or DQN?
  - **RQ3**: How much data does each method need? Compare labelled lookahead samples for CLM with environment steps
    for RL.
  - **RQ4**: Does CLM generalize to unseen intersections and demand patterns?
  - **RQ5**: Is it fast enough for real-time control?
- [ ] **Pre-register the analysis before any test run** *(review)*: write down, in `experiments/PROTOCOL.md`:
  - the primary metric: mean total delay per vehicle (`total_delay` in `vehicles.csv`, waiting time plus departure
    delay), with waiting time, queue length, throughput, CO2 and latency as secondary metrics
  - one hypothesis per RQ with its expected direction, and the statistical test for each
  - the unit of analysis: one episode (one SUMO seed), never one vehicle, because vehicles within an episode are
    correlated
  - what counts as exploratory (pilot runs, anything tuned on validation) and what as confirmatory (test split)
  - *(review 2)* **"Match" needs an equivalence test.** RQ1 and RQ2 ask whether CLM *matches* max-pressure or RL. A
    non-significant difference does not show that. Fix an equivalence margin in advance (e.g. within 5% of the
    reference's mean delay) and use a paired equivalence test (TOST), or phrase the RQs as "beats / is no worse than"
  - *(review 2)* **Number of test seeds from a power analysis**, not a fixed 10: use the between-seed variance of the
    exploratory pilot runs (on non-test seeds) to pick the number of seeds that can detect the chosen margin
- [ ] Choose one **shared objective** *(review)*: the RL reward, the lookahead label reward in `clm_collect_labels.py`
  and the primary metric should be aligned (or the mismatch documented), otherwise the RL baselines are graded on
  something they were not trained for. *(review 2)* The mismatch is concrete today: RL and the lookahead labels both
  use sumo-rl's default `diff-waiting-time` reward, which only sees vehicles already on the incoming lanes. The
  primary metric `total_delay` also includes departure delay (vehicles queued to enter the network), which neither
  the reward nor the CLM's state description can see. Either use waiting time as the primary metric, or add the
  entry backlog to the reward, the labels and the state description
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
- [ ] Q-learning baseline: set a training budget (episodes), plot convergence and report the final greedy policy.
  *(review)* **Fix the runner first**: use a different SUMO seed (and route file, see splits) for each training run,
  evaluate the frozen policy on the test seeds only, and train several independent Q-learners (at least 5 training
  seeds) so the spread across training seeds is reported, not just one lucky Q-table. *(review 2)* The runner also
  keeps calling `learn()` during the final greedy run, so the "evaluation" episode still updates the Q-table online.
  Evaluation episodes must not call `learn()`
- [ ] *(review 2)* **Enforce max green for every adaptive controller.** sumo-rl's `enforce_max_green` defaults to False
  and neither the runner nor `clm_collect_labels.py` sets it, so the `-maxgreen 50` flag currently does nothing. CLM,
  max-pressure and RL can hold one green indefinitely and starve the other movements, which no real controller is
  allowed to do. Set `enforce_max_green=True` everywhere (and mention the limit in the CLM prompt), or document that
  max green is not a constraint. Either way, report the longest red time any movement saw
- [ ] *(review 2)* **Document what each controller observes, and control for it.** The inputs differ today:
  - Q-learning and DQN: current phase, whether min green has passed, and per-lane density and queue (discretized
    for Q-learning)
  - max-pressure: vehicle counts on incoming *and outgoing* lanes, so it can see downstream spillback
  - CLM: stopped vehicles, vehicle counts and accumulated waiting time per movement, and the elapsed green. It has no
    downstream information, but it gets waiting time, which the RL agents do not
  Add a table of inputs to the paper. Add an ablation that gives the CLM the same information as the RL observation
  (and optionally the downstream counts max-pressure uses), so a CLM win or loss is not just a difference in inputs
- [ ] *(review)* SUMO **actuated** signal control as a baseline. It is what many real intersections run, and reviewers
  will ask for it next to fixed-time and max-pressure
- [ ] *(review)* **Sanity controls**, cheap and revealing: random phase, round-robin (cycle through the phases), and
  longest-queue-first. If zero-shot CLM cannot beat these, its scores carry little information
- [ ] *(review)* **Lookahead oracle** as an upper reference: the controller that picks the best phase by simulating each
  one ahead (the labelling procedure itself). Too slow to deploy, but it shows how much room there is above the
  teacher, and tells us whether the fine-tuned CLM can approach it
- [ ] Add DQN and PPO baselines (the `dqn_*` and `sb3_*` scripts exist)
- [ ] **Generative LLM baseline**: LLMLight / LightGPT (KDD 2025), the closest generative counterpart to our base vs
  specialized CLM comparison (see `literature/llm-traffic-control.md`)
  - [ ] Check that the LLMLight code and LightGPT weights are available and run with SUMO (LLMLight was built on
    CityFlow), or pick another open generative LLM controller
  - [ ] Give it the same state information and phase set as the CLM agent, so the comparison is ranking vs
    generation, not different inputs
  - [ ] Measure its decision latency on the same hardware (feeds RQ5)
- [ ] Same constraints for every controller: min/max green, yellow time, decision interval (`delta_time`). Note that
  the `fixed` controller runs the network's own program (its own cycle and timings), so it does not follow these
  constraints unless the Webster plan is built to
- [ ] *(review)* Fix the episode protocol once for every experiment: length (the pilot runs use 3000 s, the runner
  default is 20000 s), warm-up period, and how vehicles that never enter or never finish are counted (they are
  logged in `vehicles.csv`; report them, don't drop them)
- [ ] *(review 2)* Fix the signal timing parameters once, and make them realistic: sumo-rl's default yellow is 2 s,
  with no all-red clearance and 5 s minimum green (the runner uses 10 s). Choose yellow (typically 3-4 s), all-red and
  minimum green values, and use the same ones in every controller, in the Webster plan's lost time, and in the label
  lookahead
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
- [ ] *(review 2)* **Check the converted fine-tuned heads the same way.** Fine-tuning produces PyTorch heads, which must
  be converted with `rapid-mlx-convert-clm-head` to run on Rapid-MLX. Run `clm_compare_servers.py` on the original
  and converted specialized heads before any test run, so a conversion error cannot pass for a weak model
- [ ] *(review 2)* **Plan the fine-tuning compute.** `train/finetune.py` needs a CUDA GPU (the same machine as the
  `clm-serve` check). Budget the GPU hours for every fine-tuning run (dataset-size sweep, horizons, DAgger round) and
  report them in RQ3

- [ ] *(review)* **Freeze the base CLM's prompt on the validation split.** The base CLM does not learn, but choosing
  its question and description wording is tuning. Do all prompt work on validation seeds only, then freeze the prompt
  (and record its hash in `meta.json`) before the first test run; the specialized model must use the same prompt

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
- [ ] *(review)* Give each split its **own disjoint seed range** (e.g. train 1000+, validation 2000+, test 3000+).
  The defaults today collide: `clm_collect_labels.py` starts at seed 42 and increments, and the runner tests on
  seed 42, so the first label-collection episode is the same traffic as the test episode
- [ ] *(review)* The RESCO networks ship a single route file each, and two of them (arterial4x4, grid4x4) are licensed
  CC BY-NC-ND (see the license PDFs next to them). Check whether generating modified route files is allowed. If not,
  vary the traffic for those networks with SUMO seeds and `--scale` (demand multiplier) instead of new route files
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
    sensitivity), run on validation seeds
  - [ ] *(review)* **Grounding ablations** on base CLM-8B, to show the scores reflect the traffic and not just prompt
    wording:
    - permute the order of the candidate phases and check the choice follows the phase, not the position
    - remove the vehicle counts from the state (structure only) and see how much performance drops
    - swap the phase descriptions between phases (a deliberately wrong mapping) and expect it to get much worse
    - rank with an off-the-shelf text-embedding model using the same prompts, to see what the contrastive CLM
      training adds over a generic embedder
  - [ ] **Traffic-specialized CLM-8B**: fine-tune the projection heads on lookahead labels from
    `clm_collect_labels.py`, collected on the train split only
    - [ ] Follow the recipe of the official DeepSWE heads (`Contrastive-LM/deepswe-clm-heads-8k`): initialize from
      `CLM_v0.1-8B.pt`, `train/finetune.py --task clm`, batch 512, up to 20 epochs with early stopping (patience 5)
      on the validation split, held-out tasks for testing (details in `literature/clm.md`)
    - [ ] Decide the label source/teacher (lookahead reward, max-pressure, or a trained RL policy) and document it.
      *(review)* If the teacher is max-pressure, the student is capped near max-pressure by construction; the
      lookahead-reward teacher is the one that can exceed it
    - [ ] *(review)* **Split the labels by whole episode** (`task_id`), never by row: neighbouring decisions in one
      episode are nearly identical, so a random row split leaks validation into training. Matches the held-out-task
      protocol of the DeepSWE heads
    - [ ] *(review)* **Distribution shift in the labels**: `clm_collect_labels.py` drives the simulation with
      max-pressure (plus 10% random actions), but the fine-tuned CLM will visit different states when it is in
      control. Add one DAgger-style round: run the fine-tuned CLM, relabel the states it visited with the lookahead,
      retrain, and compare with the single-round model
    - [ ] Vary lookahead horizon and dataset size (feeds RQ3). *(review)* Count the cost honestly: every label needs
      one shadow simulation per phase (`phases x horizon` extra simulated seconds) plus the fine-tuning GPU time.
      Put that on the x-axis of the RQ3 comparison next to the RL agents' environment steps
    - [ ] Save the specialized heads as a separate, versioned checkpoint (e.g. `clm-v0.1-traffic-mlx/`)
    - [ ] Check that the base prompt format is identical for both models, so the only difference is the weights
  - [ ] Generalization: specialize on scenario A, test base vs specialized on B
  - [ ] **CLM-35B** (optional; multimodal, announced for early October 2026): once released, run it zero-shot as a
    third CLM controller for a scaling comparison. Check its token limit, hardware needs and serving support first
- [ ] Latency and compute: time per decision, cache hit rate (`CLMClient` LRU cache), hardware; compare with Q-table
  lookup and DQN inference. *(review)* Report p50/p95/p99, cold vs cached, against the decision interval. Note the
  simulator pauses while the CLM thinks, so slow inference never hurts the traffic in simulation; RQ5 is answered by
  the latency numbers, not by delay. Optional: a decision-latency ablation that applies each decision after its
  measured delay
- [ ] A single sweep script that runs everything and saves to a consistent `outputs/` layout

## Phase 3 — Analysis

- [ ] Extend `analyze_runs.ipynb` to combine results across seeds and report means with 95% CIs
- [ ] Significance tests. *(review)* Controllers are run on the same seeds, so use **paired** tests (Wilcoxon
  signed-rank) on per-episode results, with a Holm correction across comparisons, and report the paired difference
  with a bootstrap CI as the effect size (not only p-values). For learned methods, average over test seeds within each
  training seed first, then compare across training seeds
- [ ] Figures:
  - [ ] Queue and waiting time over time
  - [ ] Box plots per controller
  - [ ] Learning curve for Q-learning and DQN against the flat CLM line
  - [ ] Performance against the amount of fine-tuning data (base CLM-8B as the zero-data point)
  - [ ] Base vs specialized CLM-8B, in-distribution and out-of-distribution
  - [ ] Latency
- [ ] Qualitative analysis: situations where CLM phase choices differ from max-pressure, using the CLM scores in
  `signals.csv` (interpretability)
- [ ] *(review)* Multi-intersection: the CLM decides for each signal independently, with no coordination. Compare it
  with independent-learner RL baselines, list this as a limitation, and optionally try adding neighbour information
  to the prompt
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

1. Write `experiments/PROTOCOL.md` (primary metric, hypotheses, equivalence margins, analysis plan), the shared
   objective and the signal timing parameters. Decide on enforcing max green.
2. Define the train/validation/test splits with disjoint seed ranges for 2way-single-intersection, and make the
   runner read them.
3. Fix the runner: enforce max green, a new seed per Q-learning training run, and no `learn()` during evaluation
   episodes (frozen evaluation on test seeds).
4. Build the Webster fixed-time, actuated and sanity-control baselines.
5. Run every non-learning controller, including base CLM-8B, on 10 test seeds. This quickly shows whether zero-shot
   CLM is competitive at all, which decides how the paper is framed.
