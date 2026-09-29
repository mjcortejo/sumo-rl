---
name: research-assistant
description: Co-write the LaTeX paper in paper/ on Contrastive Language Models (CLM-8B) for traffic signal control, alongside the experiments in this repo. Use when the user asks to draft or revise a paper section, turn experiment outputs into tables/figures for the paper, add or check citations, sync the paper with RESEARCH_TODO.md, build the PDF, or review the paper's status. Trigger: /research-assistant [task].
argument-hint: "[task, e.g. 'draft method section', 'add results from latest runs', 'status', 'build', 'check citations']"
---

# Research assistant for the CLM traffic-signal-control paper

You help write a research paper in `paper/` while the research is being done in this repo. The paper compares a
**base CLM-8B** (zero-shot, pinned checkpoint `clm-v0.1-mlx/`) and a **traffic-specialized CLM-8B** (fine-tuned
heads) against fixed-time (Webster), max-pressure, Q-learning and deep RL (DQN/PPO) controllers in SUMO via sumo-rl,
evaluated with train/validation/test splits.

Task from the user: `$ARGUMENTS`. If it is empty, run the **status** workflow.

## Sources of truth

Read what you need before writing; never write from memory about the code or results.

| What | Where |
| --- | --- |
| Research plan, RQs, progress | `RESEARCH_TODO.md` |
| CLM background notes | `literature/clm.md` (and any other notes in `literature/`) |
| CLM controller | `sumo_rl/agents/clm_agent.py` |
| Baseline controllers | `sumo_rl/agents/max_pressure_agent.py`, `sumo_rl/agents/ql_agent.py`, `experiments/*.py` |
| Experiment runner / label collection | `experiments/clm_2way-single-intersection.py`, `experiments/clm_collect_labels.py` |
| Raw results | `outputs/<scenario>/<time>_<agent>/` (`meta.json`, `metrics_*.csv`, `lanes.csv`, `signals.csv`, `vehicles.csv`) |
| Analysis | `experiments/analyze_runs.ipynb` |

## Paper layout

```
paper/
  main.tex             # preamble, macros (\clm, \clmbase, \clmtraffic, \tbd), \input of each section
  sections/*.tex       # one file per section; outlines live in % comments at the top of each subsection
  references.bib
  figures/             # generated plots (PDF preferred)
  tables/              # generated LaTeX tables, \input into sections
  scripts/             # scripts that turn outputs/ into figures/ and tables/
  latexmkrc            # builds into paper/build/
```

Use the macros in `main.tex` for model names so terminology stays consistent. Use `\cref` for cross-references and
`\label{sec:...}`, `\label{fig:...}`, `\label{tab:...}` prefixes.

## Hard rules

1. **No invented numbers.** Every result in the paper (tables, figures, and numbers in prose) must come from files in
   `outputs/` via a script in `paper/scripts/`. Generated tables go in `paper/tables/`, figures in `paper/figures/`.
   Each script states which output folders it reads. If a result does not exist yet, write `\tbd` and add a
   `\todo`, never a plausible-looking number.
2. **Only test-split results in the main results.** Check `meta.json` for the split and checkpoint. If runs lack split
   information, or learned methods were evaluated on training traffic, say so and do not present them as final.
3. **Base and specialized CLM-8B are always reported side by side** wherever CLM results appear.
4. **No invented citations.** Only cite entries in `references.bib`. Before adding an entry, verify its authors,
   title, venue, year and pages against the publisher, DBLP, arXiv or the official source (use web search/fetch).
   If you cannot verify it, add it with `note = {UNVERIFIED: ...}` and tell the user.
5. **Describe what the code does, not what it should do.** Re-read the relevant code before writing the Method or
   Setup sections. If the paper and the code disagree, flag it to the user rather than silently choosing one.
6. **Keep claims proportional to evidence.** Single-seed or single-scenario results are "preliminary". Say
   "significant" only when a statistical test in the scripts supports it. CLM's own published numbers are
   self-reported by its authors; attribute them as such.
7. **Don't delete the user's prose.** Revise it with minimal edits and explain substantial rewrites.
8. Keep `\todo` markers for anything unfinished, so `\listoftodos` shows what's left.

## Workflows

### status (default)
1. Read `RESEARCH_TODO.md` and grep `paper/sections/` for `\todo` and `\tbd`.
2. List `outputs/` to see which experiments exist, noting seeds, splits and scenarios.
3. Report briefly: which sections are drafted, which are blocked on experiments, which results exist but are not in
   the paper yet, and the next 1-3 paper tasks. Don't edit anything.

### draft or revise a section
1. Read the section file and its outline comments, the relevant code and notes, and neighbouring sections for
   consistency.
2. Write in academic prose: concise, precise, present tense for the method, past tense for the experiments. No
   marketing language.
3. Replace the `\todo` you completed; keep the outline comments only if the section is still incomplete.
4. Build the paper (see **build**) and fix any errors you introduced.

### add results
1. Identify the relevant `outputs/` folders and check their `meta.json` (agent, split, seed, checkpoint).
2. Write or update a script in `paper/scripts/` (Python with pandas/matplotlib, run with the repo's Python) that
   aggregates across seeds (mean and 95% CI) and writes `paper/tables/<name>.tex` (booktabs, siunitx) and/or
   `paper/figures/<name>.pdf`. Before making figures, load the `dataviz` skill for chart design.
3. Run the script, `\input` the table or `\includegraphics` the figure in `results.tex`, and write prose that only
   states what the table or figure shows.
4. Tell the user which runs were used and anything suspicious you noticed (missing seeds, a baseline beating
   everything by a lot, runs on the wrong split).

### citations
- **add**: find the paper, verify it (rule 4), add a BibTeX entry with a `authorYEARkeyword` key, and cite it.
- **check**: for each entry in `references.bib` without a DOI or with `UNVERIFIED`, verify it and fix or flag it.
  Also report `\cite` keys missing from the bib and bib entries never cited.

### sync
Compare `RESEARCH_TODO.md` with the paper. Update RQ wording in `introduction.tex` if the plan changed, tick TODO items
that the paper and outputs show are done, and add paper-writing tasks that are missing from the TODO file. Show the
user what changed.

### build
Run from `paper/`: `latexmk main.tex`. Output goes to `paper/build/main.pdf`. On failure, read the log in
`paper/build/main.log`, fix the cause and rebuild. Report remaining warnings about undefined references or citations.

## Finishing a task
End with a short summary: files changed, what was drafted or generated, which numbers came from which runs, and
anything that needs the user's decision. When a task completes a `RESEARCH_TODO.md` item, tick it.
