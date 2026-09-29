"""Control the 2-way single intersection with a Contrastive Language Model (CLM) instead of an RL agent.

Needs a CLM ranking server, e.g. CLM-8B on Apple Silicon via Rapid-MLX:

    pip install 'rapid-mlx[system-one]'
    rapid-mlx-convert-clm-head "$(hf download Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt)" ./clm-v0.1-mlx
    rapid-mlx system-one clm-latest --backend clm --encoder Qwen/Qwen3-8B --head ./clm-v0.1-mlx --port 8700

Run with ``-agent maxpressure``, ``-agent fixed`` or ``-agent ql`` for baselines that need no server. Q-learning keeps
its Q-table across ``-runs`` and the last run is played greedily (no exploration), so compare that run.

Each invocation writes a folder ``outputs/2way-single-intersection/<time>_<agent>/`` with raw logs for
``experiments/analyze_runs.ipynb``:

- ``meta.json``: arguments, plus lane and phase labels of each traffic signal
- ``metrics_conn*_ep<run>.csv``: the environment's per-step system metrics
- ``lanes.csv``: per step and incoming lane: stopped vehicles, vehicles, waiting time, mean speed
- ``signals.csv``: per step and traffic signal: green phase shown, phase requested, CLM scores and context length in tokens
- ``vehicles.csv``: per vehicle: lane, waiting time at the intersection, delay before entering the network
- ``tripinfo_run<run>.xml``: SUMO's raw per-vehicle trip info
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import CLMAgent, CLMClient, MaxPressureAgent, QLAgent
from sumo_rl.agents.clm_agent import DEFAULT_MAX_CONTEXT_TOKENS, lane_movements, phase_movements
from sumo_rl.exploration import EpsilonGreedy


PERCENTILES = [50, 75, 90, 95, 99]
SHORT_TURNS = {"going straight": "straight", "turning right": "right", "turning left": "left", "making a U-turn": "U-turn"}


def short_label(approaches, turns):
    """E.g. ``N+S straight/right``."""
    return "+".join(a[0].upper() for a in approaches) + " " + "/".join(SHORT_TURNS.get(t, t) for t in turns)


class RunLogger:
    """Collects per-lane, per-signal and per-vehicle logs of one run."""

    def __init__(self, env, run):
        """Record the network layout; call right after ``env.reset()``."""
        self.env, self.run = env, run
        self.lane_rows, self.signal_rows = [], []
        self.vehicle_lane = {}  # vehicle -> last incoming lane it was seen on
        self.vehicle_route = {}  # vehicle -> (first edge, last edge) of its route
        self.lanes, self.phase_states, self.route_lane = {}, {}, {}
        for ts_id, ts in env.traffic_signals.items():
            self.lanes[ts_id] = lane_movements(ts)
            if env.fixed_ts:  # Green phases of the net's own program, same filter as TrafficSignal._build_phases
                phases = env.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
                self.phase_states[ts_id] = [
                    p.state for p in phases if "y" not in p.state and p.state.count("r") + p.state.count("s") != len(p.state)
                ]
            else:
                self.phase_states[ts_id] = [p.state for p in ts.green_phases]
            for lane in ts.lanes:
                for link in env.sumo.lane.getLinks(lane, True):
                    key = (env.sumo.lane.getEdgeID(lane), env.sumo.lane.getEdgeID(link[0]))
                    self.route_lane.setdefault(key, lane)

    def meta(self):
        """Lane and phase labels of each traffic signal."""
        meta = {}
        for ts_id, ts in self.env.traffic_signals.items():
            phases = []
            for movements in phase_movements(ts, self.phase_states[ts_id]):
                approaches = list(dict.fromkeys(m.approach for m in movements))
                turns = list(dict.fromkeys(m.turn for m in movements))
                phases.append(short_label(approaches, turns))
            meta[ts_id] = {
                "lanes": {lane: short_label([a], turns) for lane, (a, turns) in self.lanes[ts_id].items()},
                "phases": phases,
            }
        return meta

    def record(self, actions, agents):
        """Log the state after an environment step."""
        sumo, step = self.env.sumo, self.env.sim_step
        for ts_id, lanes in self.lanes.items():
            state = sumo.trafficlight.getRedYellowGreenState(ts_id)
            row = {"run": self.run, "step": step, "ts": ts_id, "state": state, "requested": actions.get(ts_id)}
            states = self.phase_states[ts_id]
            row["green_phase"] = states.index(state) if state in states else None  # None while yellow
            scores = getattr(agents.get(ts_id), "last_scores", None)
            row["context_tokens"] = getattr(agents.get(ts_id), "last_context_tokens", None)
            for i in range(len(states)):
                row[f"score_{i}"] = scores[i] if scores is not None else None
            self.signal_rows.append(row)

            for lane in lanes:
                vehicles = sumo.lane.getLastStepVehicleIDs(lane)
                for v in vehicles:
                    self.vehicle_lane[v] = lane
                self.lane_rows.append(
                    {
                        "run": self.run,
                        "step": step,
                        "ts": ts_id,
                        "lane": lane,
                        "stopped": sumo.lane.getLastStepHaltingNumber(lane),
                        "vehicles": len(vehicles),
                        "waiting_time": sumo.lane.getWaitingTime(lane),
                        "mean_speed": sumo.lane.getLastStepMeanSpeed(lane),
                    }
                )
        for v in sumo.simulation.getPendingVehicles():
            if v not in self.vehicle_route:
                route = sumo.vehicle.getRoute(v)
                self.vehicle_route[v] = (route[0], route[-1])

    def vehicles(self, tripinfo_file):
        """Per-vehicle table from SUMO's tripinfo, with the incoming lane each vehicle used or would have used."""
        rows = []
        for t in ET.parse(tripinfo_file).getroot().iter("tripinfo"):
            v = t.get("id")
            lane = self.vehicle_lane.get(v)
            if lane is None and t.get("departLane") in self.route_lane.values():
                lane = t.get("departLane")
            if lane is None and v in self.vehicle_route:
                lane = self.route_lane.get(self.vehicle_route[v])
            waiting, depart_delay = float(t.get("waitingTime")), float(t.get("departDelay"))
            rows.append(
                {
                    "run": self.run,
                    "id": v,
                    "lane": lane,
                    "depart": float(t.get("depart")),
                    "arrival": float(t.get("arrival")),
                    "entered": float(t.get("depart")) >= 0,
                    "finished": float(t.get("arrival")) >= 0,
                    "waiting_time": waiting,
                    "depart_delay": depart_delay,
                    "total_delay": waiting + depart_delay,
                    "time_loss": float(t.get("timeLoss")),
                    "waiting_count": int(t.get("waitingCount")),
                }
            )
        return pd.DataFrame(rows)


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


if __name__ == "__main__":
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="""CLM 2way-Single-Intersection"""
    )
    prs.add_argument(
        "-route",
        dest="route",
        type=str,
        default="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        help="Route definition xml file.\n",
    )
    prs.add_argument("-agent", choices=["clm", "maxpressure", "fixed", "ql"], default="clm", help="Controller to run.\n")
    prs.add_argument("-url", dest="url", type=str, default=None, help="CLM server URL (default $CLM_BASE_URL or :8700).\n")
    prs.add_argument(
        "-maxtokens",
        dest="max_tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="CLM server token limit; longer contexts stop the run instead of being truncated. 0 disables the check.\n",
    )
    prs.add_argument("-mingreen", dest="min_green", type=int, default=10, required=False, help="Minimum green time.\n")
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n")
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=20000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-seed", dest="seed", type=int, default=42, help="SUMO seed, so controllers see the same traffic.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print every CLM decision.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=1, help="Number of runs (Q-learning trains across them).\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Q-learning alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.99, required=False, help="Q-learning gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Q-learning epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Q-learning min epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1.0, required=False, help="Q-learning epsilon decay.\n")
    args = prs.parse_args()
    # No spaces (additional_sumo_cmd is split on whitespace) and no colons (SUMO reads "host:port" as a socket)
    run_dir = f"outputs/2way-single-intersection/{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.agent}"
    os.makedirs(run_dir, exist_ok=True)
    tripinfo = f"{run_dir}/tripinfo.xml"

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file=args.route,
        out_csv_name=f"{run_dir}/metrics",
        use_gui=args.gui,
        num_seconds=args.seconds,
        min_green=args.min_green,
        max_green=args.max_green,
        sumo_seed=args.seed,
        fixed_ts=args.agent == "fixed",
        additional_sumo_cmd=(
            f"--tripinfo-output {tripinfo} --tripinfo-output.write-unfinished --tripinfo-output.write-undeparted"
        ),
    )
    client = CLMClient(base_url=args.url, max_context_tokens=args.max_tokens or None) if args.agent == "clm" else None
    qlearners = {}  # Kept across runs so the Q-tables keep learning

    for run in range(1, args.runs + 1):
        observations = env.reset()
        if args.agent == "ql":
            for ts in env.ts_ids:
                state = env.encode(observations[ts], ts)
                if ts not in qlearners:
                    qlearners[ts] = QLAgent(
                        starting_state=state,
                        state_space=env.observation_spaces(ts),
                        action_space=env.action_spaces(ts),
                        alpha=args.alpha,
                        gamma=args.gamma,
                        exploration_strategy=EpsilonGreedy(
                            initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay
                        ),
                    )
                agent = qlearners[ts]
                agent.state, agent.acc_reward = state, 0
                agent.q_table.setdefault(state, [0 for _ in range(agent.action_space.n)])
                if run == args.runs and args.runs > 1:  # Evaluate the learned policy without exploration
                    agent.exploration.epsilon = agent.exploration.min_epsilon = 0.0
            agents = qlearners
        elif args.agent == "clm":
            agents = {ts: CLMAgent(env.traffic_signals[ts], client) for ts in env.ts_ids}
        elif args.agent == "maxpressure":
            agents = {ts: MaxPressureAgent(env.traffic_signals[ts]) for ts in env.ts_ids}
        else:
            agents = {}
        logger = RunLogger(env, run)
        if run == 1:
            with open(f"{run_dir}/meta.json", "w") as f:
                json.dump({"args": vars(args), "signals": logger.meta()}, f, indent=2)

        start = time.time()
        done = {"__all__": False}
        while not done["__all__"]:
            actions = {ts: agent.act() for ts, agent in agents.items()}
            if args.v and args.agent == "clm":
                for ts, agent in agents.items():
                    if agent.last_scores is not None:
                        print(env.sim_step, ts, actions[ts], np.round(agent.last_scores, 3))
            observations, rewards, done, _ = env.step(action=actions)
            logger.record(actions, agents)
            if args.agent == "ql":
                for ts in observations:
                    agents[ts].learn(next_state=env.encode(observations[ts], ts), reward=rewards[ts])

        env.save_csv(env.out_csv_name, run)
        env.close()  # SUMO writes the tripinfo file on close
        run_tripinfo = f"{run_dir}/tripinfo_run{run}.xml"
        os.replace(tripinfo, run_tripinfo)
        vehicles = logger.vehicles(run_tripinfo)
        append_csv(vehicles, f"{run_dir}/vehicles.csv")
        append_csv(pd.DataFrame(logger.lane_rows), f"{run_dir}/lanes.csv")
        append_csv(pd.DataFrame(logger.signal_rows), f"{run_dir}/signals.csv")

        delay = vehicles["total_delay"]
        stats = ", ".join(f"p{p} {v:.1f}" for p, v in zip(PERCENTILES, np.percentile(delay, PERCENTILES)))
        print(
            f"run {run} [{args.agent}] {len(vehicles)} vehicles ({(~vehicles['entered']).sum()} never entered, "
            f"{(vehicles['entered'] & ~vehicles['finished']).sum()} still on the road), delay per vehicle (s): "
            f"mean {delay.mean():.1f}, {stats}, max {delay.max():.1f}; wall time {time.time() - start:.0f} s"
            + (f", CLM requests {client.num_requests}" if client else "")
            + (f", Q-table states {sum(len(a.q_table) for a in qlearners.values())}" if qlearners else "")
        )
    print(f"logs in {run_dir}")
