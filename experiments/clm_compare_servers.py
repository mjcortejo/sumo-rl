"""Check that two CLM servers (e.g. Rapid-MLX and the official clm-serve) rank phases the same way.

Our experiments use Rapid-MLX on Apple Silicon; the official stack is clm-serve on vLLM (needs a CUDA GPU). If
their rankings differ, results must say which server produced them. This sends the same (context, answers) pairs to
both servers and reports agreement on the chosen phase, rank correlation, score differences and latency.

Make the pairs with ``experiments/clm_token_budget.py -dump outputs/clm_contexts.jsonl``, then e.g.:

    python experiments/clm_compare_servers.py -pairs outputs/clm_contexts.jsonl \
        -a rapid-mlx=http://127.0.0.1:8700 -b clm-serve=http://gpu-box:8000

Both servers must serve the same checkpoint (CLM-v0.1-8B heads on Qwen3-8B).
"""

import argparse
import json
import os
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd

from sumo_rl.agents.clm_agent import CLMClient


def kendall_tau(x, y):
    """Kendall rank correlation (tau-a) between two score lists."""
    pairs = list(combinations(range(len(x)), 2))
    s = sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in pairs)
    return s / len(pairs) if pairs else 1.0


def softmax(x):
    """Scores as probabilities, so servers returning logits and probabilities can be compared."""
    x = np.asarray(x, dtype=float)
    if np.all(x >= 0) and np.isclose(x.sum(), 1.0, atol=1e-3):
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def server(spec):
    name, _, url = spec.partition("=")
    if not url:
        raise argparse.ArgumentTypeError("use NAME=URL")
    return name, url


def timed_rank(client, context, answers):
    start = time.perf_counter()
    scores = client.rank(context, answers)
    return scores, time.perf_counter() - start


if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    prs.add_argument("-pairs", required=True, help="JSONL of {scenario, context, answers} from clm_token_budget.py.")
    prs.add_argument("-a", type=server, required=True, help="First server as NAME=URL.")
    prs.add_argument("-b", type=server, required=True, help="Second server as NAME=URL.")
    prs.add_argument("-repeats", type=int, default=2, help="Queries per pair and server (checks determinism).")
    prs.add_argument("-limit", type=int, default=None, help="Only use the first N pairs.")
    args = prs.parse_args()

    with open(args.pairs) as f:
        pairs = [json.loads(line) for line in f][: args.limit]
    (name_a, url_a), (name_b, url_b) = args.a, args.b
    # No client-side cache (every query reaches the server); lengths were already checked when the pairs were made
    clients = {
        name_a: CLMClient(base_url=url_a, cache_size=0, max_context_tokens=None),
        name_b: CLMClient(base_url=url_b, cache_size=0, max_context_tokens=None),
    }

    rows = []
    for i, pair in enumerate(pairs):
        row = {"pair": i, "scenario": pair.get("scenario"), "candidates": len(pair["answers"])}
        scores = {}
        for name, client in clients.items():
            runs = [timed_rank(client, pair["context"], pair["answers"]) for _ in range(args.repeats)]
            scores[name] = runs[0][0]
            row[f"{name}_top"] = int(np.argmax(runs[0][0]))
            row[f"{name}_latency_ms"] = 1000 * np.median([t for _, t in runs])
            row[f"{name}_deterministic"] = all(np.allclose(s, runs[0][0], atol=1e-6) for s, _ in runs)
        a, b = scores[name_a], scores[name_b]
        row["same_top"] = row[f"{name_a}_top"] == row[f"{name_b}_top"]
        row["kendall_tau"] = kendall_tau(a, b)
        row["max_prob_diff"] = float(np.max(np.abs(softmax(a) - softmax(b))))
        rows.append(row)
        print(f"{i + 1}/{len(pairs)}", end="\r", flush=True)
    print()

    df = pd.DataFrame(rows)
    out_dir = "outputs/server_compare"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{datetime.now():%Y-%m-%d_%H-%M-%S}_{name_a}_vs_{name_b}.csv"
    df.to_csv(out, index=False)

    by_scenario = df.groupby("scenario").agg(
        pairs=("pair", "count"),
        same_top=("same_top", "mean"),
        kendall_tau=("kendall_tau", "mean"),
        max_prob_diff=("max_prob_diff", "max"),
        **{f"{n}_latency_ms": (f"{n}_latency_ms", "median") for n in clients},
    )
    print(by_scenario.round(3).to_string())
    print(
        f"\noverall: same chosen phase in {df['same_top'].mean():.1%} of {len(df)} pairs, mean Kendall tau "
        f"{df['kendall_tau'].mean():.3f}, max probability difference {df['max_prob_diff'].max():.4f}"
    )
    for name in clients:
        print(f"{name}: deterministic in {df[f'{name}_deterministic'].mean():.1%} of pairs")
    print(f"per-pair results in {out}")
