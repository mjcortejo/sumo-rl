"""Collect lookahead-labelled (state, phase) pairs for fine-tuning a CLM on the 2-way single intersection.

At every decision where the signal is allowed to switch, the simulation state is saved and each green phase is
tried for ``-horizon`` seconds in a second (shadow) simulation. The phase with the highest
reward is the positive example. Rows are written as step transitions (``state``, ``action``, ``task_id``,
``step_idx``, ``reward``, ``passed``), one per candidate phase, for ``train/finetune.py --task clm --data``
in the Contrastive-LM/CLM repository.
"""

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import MaxPressureAgent
from sumo_rl.agents.clm_agent import (
    DEFAULT_QUESTION,
    describe_phases,
    describe_state,
    lane_labels,
)


TS_FIELDS = [
    "green_phase",
    "is_yellow",
    "time_since_last_phase_change",
    "next_action_time",
    "last_ts_waiting_time",
    "last_reward",
]
ENV_FIELDS = ["vehicles", "num_arrived_vehicles", "num_departed_vehicles", "num_teleported_vehicles"]


def python_state(env):
    """Environment state that lives in Python rather than in SUMO."""
    return copy.deepcopy(
        {
            "env": {k: getattr(env, k) for k in ENV_FIELDS},
            "ts": {ts_id: {k: getattr(ts, k) for k in TS_FIELDS} for ts_id, ts in env.traffic_signals.items()},
            "tls": {ts_id: env.sumo.trafficlight.getRedYellowGreenState(ts_id) for ts_id in env.ts_ids},
        }
    )


def load(shadow, path, state):
    """Put ``shadow`` into the SUMO state saved at ``path`` plus the matching Python-side ``state``."""
    shadow.sumo.simulation.loadState(path)
    state = copy.deepcopy(state)
    for k, v in state["env"].items():
        setattr(shadow, k, v)
    for ts_id, fields in state["ts"].items():
        for k, v in fields.items():
            setattr(shadow.traffic_signals[ts_id], k, v)
        shadow.sumo.trafficlight.setRedYellowGreenState(ts_id, state["tls"][ts_id])


def lookahead(env, shadow, ts_id, horizon):
    """Reward of each green phase of ``ts_id`` when held for ``horizon`` seconds; other signals keep their phase.

    Rollouts run in ``shadow``, a second environment on the same network. Loading states back into ``env`` itself
    is avoided because repeatedly doing so degrades SUMO's traffic flow (observed with SUMO 1.24).
    """
    path = os.path.join(tempfile.gettempdir(), f"sumo-rl-lookahead-{os.getpid()}.xml")
    env.sumo.simulation.saveState(path)
    state = python_state(env)
    rewards = []
    for phase in range(env.traffic_signals[ts_id].num_green_phases):
        load(shadow, path, state)
        total = 0.0
        end = shadow.sim_step + horizon
        while shadow.sim_step < end:
            actions = {ts: shadow.traffic_signals[ts].green_phase for ts in shadow.ts_ids}
            actions[ts_id] = phase
            shadow._apply_actions(actions)
            shadow._run_steps()
            if shadow.traffic_signals[ts_id].time_to_act:
                total += float(shadow.traffic_signals[ts_id].compute_reward())
        rewards.append(total)
    os.remove(path)
    return rewards


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""CLM lookahead label collection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-out", dest="out", type=str, default="outputs/clm-labels/2way-single-intersection.jsonl")
    prs.add_argument("-horizon", dest="horizon", type=int, default=20, help="Seconds each phase is tried for.\n")
    prs.add_argument(
        "-behavior",
        choices=["oracle", "maxpressure", "random"],
        default="oracle",
        help="Policy that drives the simulation between labels (oracle = the lookahead's best phase).\n",
    )
    prs.add_argument("-eps", dest="eps", type=float, default=0.1, help="Probability of a random action instead.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=20000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-seed", dest="seed", type=int, default=42, help="SUMO seed of the first run.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs.\n")
    args = prs.parse_args()
    rng = np.random.default_rng(args.seed)

    env_kwargs = dict(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_seed=args.seed,
        sumo_warnings=False,
        additional_sumo_cmd="--save-state.rng",
    )
    env = SumoEnvironment(**env_kwargs)
    shadow = SumoEnvironment(**env_kwargs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    num_labels = 0
    with open(args.out, "w") as f:
        for run in range(1, args.runs + 1):
            env.reset(seed=args.seed + run - 1)
            shadow.reset()
            signals = env.traffic_signals
            phase_texts = {ts: describe_phases(signals[ts]) for ts in env.ts_ids}
            labels = {ts: lane_labels(signals[ts]) for ts in env.ts_ids}
            maxpressure = {ts: MaxPressureAgent(signals[ts]) for ts in env.ts_ids}

            done = {"__all__": False}
            step_idx = 0
            while not done["__all__"]:
                actions = {}
                for ts in env.ts_ids:
                    signal = signals[ts]
                    if signal.time_since_last_phase_change < signal.yellow_time + signal.min_green:
                        actions[ts] = signal.green_phase  # Cannot switch yet, nothing to label
                        continue

                    rewards = lookahead(env, shadow, ts, args.horizon)
                    best = int(np.argmax(rewards))
                    state = describe_state(signal, phase_texts[ts], labels[ts]) + "\n" + DEFAULT_QUESTION
                    for phase, (text, reward) in enumerate(zip(phase_texts[ts], rewards)):
                        row = {
                            "task_id": f"{ts}-run{run}",
                            "step_idx": step_idx,
                            "state": state,
                            "action": text,
                            "reward": reward,
                            "passed": phase == best,
                        }
                        f.write(json.dumps(row) + "\n")
                    num_labels += 1

                    if rng.random() < args.eps or args.behavior == "random":
                        actions[ts] = int(rng.integers(signal.num_green_phases))
                    elif args.behavior == "maxpressure":
                        actions[ts] = maxpressure[ts].act()
                    else:
                        actions[ts] = best
                _, _, done, _ = env.step(action=actions)
                step_idx += 1

            waiting = np.mean([m["system_mean_waiting_time"] for m in env.metrics])
            print(f"run {run} [{args.behavior}, eps {args.eps}] mean waiting time {waiting:.2f} s, labels so far {num_labels}")
    env.close()
    shadow.close()
    print(f"wrote {num_labels} labelled states to {args.out}")
