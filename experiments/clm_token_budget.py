"""Measure how many tokens the CLM contexts and phase descriptions use in each scenario.

CLM servers truncate inputs longer than their limit (2048 tokens by default), which would silently drop lanes from
the state description. This drives each scenario with max-pressure control (no CLM server needed), builds the context
the CLM agent would send at every decision, and counts its tokens with the Qwen3-8B tokenizer.

    python experiments/clm_token_budget.py                       # all scenarios
    python experiments/clm_token_budget.py -scenarios 2way cologne8 -dump outputs/clm_contexts.jsonl

``-dump`` writes a sample of (context, answers) pairs, one JSON object per line, for
``experiments/clm_compare_servers.py``.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import CLMAgent, MaxPressureAgent
from sumo_rl.agents.clm_agent import DEFAULT_MAX_CONTEXT_TOKENS, qwen_token_counter


NETS = "sumo_rl/nets"
RESCO = f"{NETS}/RESCO"
SCENARIOS = {
    "single": dict(
        net_file=f"{NETS}/single-intersection/single-intersection.net.xml",
        route_file=f"{NETS}/single-intersection/single-intersection.rou.xml",
    ),
    "2way": dict(
        net_file=f"{NETS}/2way-single-intersection/single-intersection.net.xml",
        route_file=f"{NETS}/2way-single-intersection/single-intersection-vhvh.rou.xml",
    ),
    "4x4": dict(net_file=f"{NETS}/4x4-Lucas/4x4.net.xml", route_file=f"{NETS}/4x4-Lucas/4x4c1c2c1c2.rou.xml"),
    "grid4x4": dict(net_file=f"{RESCO}/grid4x4/grid4x4.net.xml", route_file=f"{RESCO}/grid4x4/grid4x4_1.rou.xml"),
    "arterial4x4": dict(
        net_file=f"{RESCO}/arterial4x4/arterial4x4.net.xml", route_file=f"{RESCO}/arterial4x4/arterial4x4_1.rou.xml"
    ),
    # Begin times as in sumo_rl/environment/resco_envs.py
    **{
        f"cologne{n}": dict(
            net_file=f"{RESCO}/cologne{n}/cologne{n}.net.xml",
            route_file=f"{RESCO}/cologne{n}/cologne{n}.rou.xml",
            begin_time=25200,
        )
        for n in (1, 3, 8)
    },
    **{
        f"ingolstadt{n}": dict(
            net_file=f"{RESCO}/ingolstadt{n}/ingolstadt{n}.net.xml",
            route_file=f"{RESCO}/ingolstadt{n}/ingolstadt{n}.rou.xml",
            begin_time=57600,
        )
        for n in (1, 7, 21)
    },
}


def measure(name, seconds, count_tokens, rng, samples):
    """Token counts of every CLM context in one run of a scenario, plus a random sample of (context, answers)."""
    env = SumoEnvironment(**SCENARIOS[name], num_seconds=seconds, sumo_warnings=False, sumo_seed=42)
    env.reset()
    clm = {ts: CLMAgent(env.traffic_signals[ts], client=None) for ts in env.ts_ids}
    controllers = {ts: MaxPressureAgent(env.traffic_signals[ts]) for ts in env.ts_ids}
    answer_tokens = max(count_tokens(p) for agent in clm.values() for p in agent.phase_texts)

    rows, pairs = [], []
    done = {"__all__": False}
    while not done["__all__"]:
        for ts, agent in clm.items():
            context = agent.context()
            rows.append({"scenario": name, "ts": ts, "step": env.sim_step, "tokens": count_tokens(context)})
            pairs.append((context, agent.phase_texts))
        _, _, done, _ = env.step({ts: c.act() for ts, c in controllers.items()})
    env.close()

    picked = rng.choice(len(pairs), size=min(samples, len(pairs)), replace=False) if samples else []
    sample = [{"scenario": name, "context": pairs[i][0], "answers": pairs[i][1]} for i in sorted(picked)]
    return pd.DataFrame(rows), answer_tokens, sample


if __name__ == "__main__":
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    prs.add_argument("-scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS), help="Scenarios.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=3600, help="Simulation seconds per scenario.\n")
    prs.add_argument("-limit", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS, help="Server token limit.\n")
    prs.add_argument("-dump", type=str, default=None, help="Write sampled (context, answers) pairs to this JSONL.\n")
    prs.add_argument("-samples", type=int, default=50, help="Pairs sampled per scenario for -dump.\n")
    args = prs.parse_args()

    count_tokens = qwen_token_counter()
    rng = np.random.default_rng(0)
    summary, dumped = [], []
    for name in args.scenarios:
        tokens, answer_tokens, sample = measure(name, args.seconds, count_tokens, rng, args.samples if args.dump else 0)
        dumped += sample
        t = tokens["tokens"]
        summary.append(
            {
                "scenario": name,
                "signals": tokens["ts"].nunique(),
                "decisions": len(t),
                "context_median": int(t.median()),
                "context_p99": int(np.percentile(t, 99)),
                "context_max": int(t.max()),
                "answer_max": answer_tokens,
                "headroom": args.limit - int(max(t.max(), answer_tokens)),
            }
        )
        print(summary[-1], flush=True)

    summary = pd.DataFrame(summary)
    out_dir = "outputs/token_budget"
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    over = summary[summary["headroom"] < 0]
    print(f"\nlimit {args.limit}: " + (f"OVER in {', '.join(over['scenario'])}" if len(over) else "all scenarios fit"))
    print(f"summary in {out}")

    if args.dump:
        os.makedirs(os.path.dirname(args.dump) or ".", exist_ok=True)
        with open(args.dump, "w") as f:
            for pair in dumped:
                f.write(json.dumps(pair) + "\n")
        print(f"{len(dumped)} (context, answers) pairs in {args.dump}")
