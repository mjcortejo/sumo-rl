import json
import runpy
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sumo_rl import SumoEnvironment
from sumo_rl.agents import MaxPressureAgent
from sumo_rl.agents.clm_agent import (
    CLMAgent,
    CLMClient,
    ContextTooLongError,
    describe_phases,
    describe_state,
    parse_rank_response,
)


ANSWERS = ["green north-south", "green east-west"]


def count_words(text):
    """Stand-in token counter, so tests don't need the Qwen tokenizer."""
    return len(text.split())


def test_parse_rank_response_formats():
    ranked = [
        {"rank": 1, "candidate": "green east-west", "prob": 0.9},
        {"rank": 2, "candidate": "green north-south", "prob": 0.1},
    ]
    assert parse_rank_response(ranked, ANSWERS) == [0.1, 0.9]
    assert parse_rank_response({"results": ranked}, ANSWERS) == [0.1, 0.9]
    assert parse_rank_response({"ranking": [{"index": 0, "score": 2.0}, {"index": 1, "score": -1.0}]}, ANSWERS) == [2.0, -1.0]
    with pytest.raises(ValueError):
        parse_rank_response({"error": "bad request"}, ANSWERS)


class _RankHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _RankHandler.requests.append((self.path, body))
        # Prefer whichever answer mentions "left", like a CLM would for a matching state
        ranked = sorted(body["answers"], key=lambda a: "left" not in a)
        payload = [{"rank": i + 1, "candidate": a, "prob": 1.0 / (i + 1)} for i, a in enumerate(ranked)]
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def rank_server():
    _RankHandler.requests = []
    server = HTTPServer(("127.0.0.1", 0), _RankHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_client_request_and_cache(rank_server):
    client = CLMClient(base_url=rank_server, count_tokens=count_words)
    answers = ["go straight", "turn left"]
    assert client.rank("state", answers) == [0.5, 1.0]
    assert client.rank("state", answers) == [0.5, 1.0]
    assert client.num_requests == 1
    assert _RankHandler.requests == [("/v1/rank", {"context": "state", "answers": answers})]
    assert client.last_context_tokens == 1


def test_client_rejects_contexts_over_token_limit(rank_server):
    client = CLMClient(base_url=rank_server, max_context_tokens=3, count_tokens=count_words)
    assert client.rank("three word state", ANSWERS) == [1.0, 0.5]
    assert client.last_context_tokens == 3
    with pytest.raises(ContextTooLongError):
        client.rank("a state of four", ANSWERS)
    with pytest.raises(ContextTooLongError):
        client.rank("state", ["an answer that is too long"])
    assert len(_RankHandler.requests) == 1  # Nothing over the limit reaches the server

    unchecked = CLMClient(base_url=rank_server, max_context_tokens=None, count_tokens=count_words)
    unchecked.rank("a state of four", ANSWERS)
    assert unchecked.last_context_tokens is None


def test_clm_agent_2way_intersection(rank_server):
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        num_seconds=200,
        min_green=10,
        sumo_warnings=False,
    )
    env.reset()
    ts = env.traffic_signals["t"]
    phases = describe_phases(ts)
    assert len(phases) == ts.num_green_phases == 4
    assert "from the north" in phases[0] and "from the south" in phases[0]
    assert "from the east" in describe_state(ts)

    agent = CLMAgent(ts, CLMClient(base_url=rank_server, count_tokens=count_words))
    assert agent.act() == ts.green_phase  # min green not reached yet, no query
    assert _RankHandler.requests == []

    done = {"__all__": False}
    chosen = set()
    while not done["__all__"]:
        chosen.add(agent.act())
        _, _, done, _ = env.step({"t": agent.action})
    env.close()

    assert len(_RankHandler.requests) > 0
    context = _RankHandler.requests[0][1]["context"]
    assert context.startswith("Traffic signal controlling an intersection.") and context.endswith(agent.question)
    assert chosen <= {0, 1}  # phase 1 (first "left" answer) once the server is consulted
    assert 1 in chosen


def test_lookahead_does_not_disturb_simulation():
    collect = runpy.run_path("experiments/clm_collect_labels.py", run_name="clm_collect_labels")
    kwargs = dict(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        num_seconds=300,
        min_green=10,
        sumo_seed=42,
        sumo_warnings=False,
        additional_sumo_cmd="--save-state.rng",
    )

    def run(with_lookahead):
        env, shadow = SumoEnvironment(**kwargs), SumoEnvironment(**kwargs)
        env.reset()
        shadow.reset()
        agent = MaxPressureAgent(env.traffic_signals["t"])
        done = {"__all__": False}
        while not done["__all__"]:
            if with_lookahead and env.sim_step >= 15:
                assert len(collect["lookahead"](env, shadow, "t", 20)) == 4
            _, _, done, _ = env.step({"t": agent.act()})
        metrics = [(m["system_total_departed"], m["system_total_stopped"]) for m in env.metrics]
        env.close()
        shadow.close()
        return metrics

    assert run(with_lookahead=True) == run(with_lookahead=False)
