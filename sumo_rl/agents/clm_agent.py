"""Contrastive Language Model (CLM) agent.

Instead of learning a value function, the agent describes the intersection in text and asks a CLM
(e.g. CLM-8B served by Rapid-MLX or clm-serve) to rank text descriptions of the green phases against it.
"""

import json
import os
import urllib.request
from collections import OrderedDict
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np


DEFAULT_QUESTION = "Which movements should get the green light next to minimise how long vehicles wait at this intersection?"

# CLM servers truncate longer inputs by default (vLLM --max-model-len / clm-serve --max-tokens), see literature/clm.md
DEFAULT_MAX_CONTEXT_TOKENS = 2048
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"  # CLM-v0.1-8B's frozen encoder

MOVEMENT_NAMES = {
    "s": "going straight",
    "r": "turning right",
    "R": "turning right",
    "l": "turning left",
    "L": "turning left",
    "t": "making a U-turn",
}


class Movement(NamedTuple):
    """A signalised movement from an incoming lane to an outgoing lane."""

    in_lane: str
    out_lane: str
    approach: str
    turn: str


class ContextTooLongError(ValueError):
    """A context or answer is longer than the server's token limit, so the server would silently truncate it."""


def qwen_token_counter(tokenizer_name: str = DEFAULT_TOKENIZER) -> Callable[[str], int]:
    """Token counter using the CLM encoder's tokenizer (needs ``pip install tokenizers``)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer.encode(text).ids)


class CLMClient:
    """Minimal HTTP client for a CLM ``/v1/rank`` endpoint (Rapid-MLX ``system-one`` or ``clm-serve``)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        cache_size: int = 4096,
        max_context_tokens: Optional[int] = DEFAULT_MAX_CONTEXT_TOKENS,
        count_tokens: Optional[Callable[[str], int]] = None,
    ):
        """Initialize the client.

        Args:
            base_url (str): Server URL. Defaults to $CLM_BASE_URL or http://127.0.0.1:8700.
            api_key (str): Bearer token. Defaults to $RAPID_MLX_API_KEY or $CLM_API_KEY, if set.
            timeout (float): Request timeout in seconds.
            cache_size (int): Number of (context, answers) results kept in a client-side LRU cache.
            max_context_tokens (int): The server's token limit. Longer contexts or answers raise
                ``ContextTooLongError`` instead of being truncated by the server. None disables the check.
            count_tokens (callable): Text -> number of tokens. Defaults to the Qwen3-8B tokenizer, loaded on first use.
        """
        self.base_url = (base_url or os.environ.get("CLM_BASE_URL", "http://127.0.0.1:8700")).rstrip("/")
        self.api_key = api_key or os.environ.get("RAPID_MLX_API_KEY") or os.environ.get("CLM_API_KEY")
        self.timeout = timeout
        self.cache_size = cache_size
        self.max_context_tokens = max_context_tokens
        self._count_tokens = count_tokens
        self._cache = OrderedDict()
        self.num_requests = 0
        self.last_context_tokens = None  # Tokens in the last ranked context (None when the check is disabled)

    def count_tokens(self, text: str) -> int:
        if self._count_tokens is None:
            self._count_tokens = qwen_token_counter()
        return self._count_tokens(text)

    def check_length(self, context: str, answers: List[str]) -> Optional[int]:
        """Raise ``ContextTooLongError`` if the context or an answer exceeds the limit; return the context's tokens."""
        if self.max_context_tokens is None:
            return None
        for kind, text in [("context", context)] + [("answer", a) for a in answers]:
            tokens = self.count_tokens(text)
            if tokens > self.max_context_tokens:
                raise ContextTooLongError(
                    f"{kind} has {tokens} tokens, more than the server limit of {self.max_context_tokens}: raise the "
                    f"server's limit (and max_context_tokens) or shorten the description. Starts: {text[:200]!r}"
                )
            if kind == "context":
                context_tokens = tokens
        return context_tokens

    def rank(self, context: str, answers: List[str]) -> List[float]:
        """Return one score per answer (same order as ``answers``); higher means a better fit to ``context``."""
        key = (context, tuple(answers))
        if key in self._cache:
            self._cache.move_to_end(key)
            scores, self.last_context_tokens = self._cache[key]
            return scores

        context_tokens = self.check_length(context, answers)

        request = urllib.request.Request(
            self.base_url + "/v1/rank",
            data=json.dumps({"context": context, "answers": answers}).encode(),
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read())
        self.num_requests += 1

        scores = parse_rank_response(body, answers)
        self.last_context_tokens = context_tokens
        self._cache[key] = (scores, context_tokens)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return scores


def parse_rank_response(body, answers: List[str]) -> List[float]:
    """Map a ``/v1/rank`` response back onto ``answers``.

    Accepts a list of ranked items, or a dict wrapping one. Each item identifies its answer by
    ``index`` or by text (``candidate``/``answer``/``text``) and carries ``prob``/``probability``/``score``.
    """
    items = body
    if isinstance(body, dict):
        items = next(
            (body[k] for k in ("results", "ranking", "rankings", "ranked", "answers", "candidates", "data") if k in body),
            None,
        )
    if not isinstance(items, list):
        raise ValueError(f"Unrecognised /v1/rank response: {str(body)[:500]}")

    scores = [-np.inf] * len(answers)
    for item in items:
        if not isinstance(item, dict):
            continue
        score = next((item[k] for k in ("prob", "probability", "score", "logit") if k in item), None)
        index = item.get("index")
        if index is None:
            text = next((item[k] for k in ("candidate", "answer", "text") if k in item), None)
            index = answers.index(text) if text in answers else None
        if score is not None and index is not None:
            scores[int(index)] = float(score)

    if all(s == -np.inf for s in scores):
        raise ValueError(f"Could not match any answer in /v1/rank response: {str(body)[:500]}")
    return scores


def _approach(ts, lane: str) -> str:
    """Compass side a lane enters the intersection from, based on the direction of its last segment."""
    (x0, y0), (x1, y1) = ts.sumo.lane.getShape(lane)[-2:]
    dx, dy = x1 - x0, y1 - y0
    if abs(dy) >= abs(dx):
        return "north" if dy < 0 else "south"
    return "west" if dx > 0 else "east"


def _turn(ts, in_lane: str, out_lane: str) -> str:
    for link in ts.sumo.lane.getLinks(in_lane, True):
        if link[0] == out_lane:
            return MOVEMENT_NAMES.get(link[6], "going through")
    return "going through"


def phase_movements(ts, states: Optional[List[str]] = None) -> List[List[Movement]]:
    """For each green phase of a TrafficSignal, the movements that get green.

    ``states`` are the phases' signal strings (e.g. ``"GGrrrrGGrrrr"``), by default those of ``ts.green_phases``.
    """
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    approaches = {lane: _approach(ts, lane) for lane in ts.lanes}
    phases = []
    for phase_state in states if states is not None else [p.state for p in ts.green_phases]:
        movements = []
        for signal, state in enumerate(phase_state):
            if state not in "Gg" or not links[signal]:
                continue
            in_lane, out_lane, _ = links[signal][0]
            movement = Movement(in_lane, out_lane, approaches.get(in_lane, "unknown"), _turn(ts, in_lane, out_lane))
            if movement not in movements:
                movements.append(movement)
        phases.append(movements)
    return phases


def _group_by_approach(movements: List[Movement]) -> "OrderedDict[str, List[str]]":
    grouped = OrderedDict()
    for m in movements:
        turns = grouped.setdefault(m.approach, [])
        if m.turn not in turns:
            turns.append(m.turn)
    return grouped


def describe_phases(ts) -> List[str]:
    """Text description of each green phase. These are the candidate actions ranked by the CLM."""
    descriptions = []
    for i, movements in enumerate(phase_movements(ts)):
        grouped = _group_by_approach(movements)
        parts = [f"traffic from the {approach} {' or '.join(turns)}" for approach, turns in grouped.items()]
        text = "Give green to " + "; ".join(parts) + ". All other movements stop." if parts else "All movements stop."
        if text in descriptions:  # Candidates must be distinguishable
            text += f" (phase {i})"
        descriptions.append(text)
    return descriptions


def lane_movements(ts) -> "OrderedDict[str, Tuple[str, List[str]]]":
    """Approach and turns (e.g. ``("north", ["turning right", "going straight"])``) of each incoming lane."""
    movements = OrderedDict()
    for lane in ts.lanes:
        turns = []
        for link in ts.sumo.lane.getLinks(lane, True):
            turn = MOVEMENT_NAMES.get(link[6], "going through")
            if turn not in turns:
                turns.append(turn)
        movements[lane] = (_approach(ts, lane), turns or ["going through"])
    return movements


def lane_labels(ts) -> "OrderedDict[str, str]":
    """Label for each incoming lane of a TrafficSignal, e.g. ``from the north, turning right or going straight``."""
    return OrderedDict(
        (lane, f"from the {approach}, {' or '.join(turns)}") for lane, (approach, turns) in lane_movements(ts).items()
    )


def describe_state(ts, phase_texts: Optional[List[str]] = None, labels: Optional["OrderedDict[str, str]"] = None) -> str:
    """Text description of the intersection as seen by a TrafficSignal.

    ``phase_texts`` and ``labels`` only depend on the network, so callers can compute them once and pass them in.
    """
    phase_texts = phase_texts or describe_phases(ts)
    labels = labels or lane_labels(ts)
    elapsed = int(ts.time_since_last_phase_change)
    can_switch = elapsed >= ts.yellow_time + ts.min_green
    current = phase_texts[ts.green_phase].replace("Give green to", "Green for", 1)

    groups = OrderedDict()  # label -> [stopped, vehicles, waiting seconds], summed over lanes sharing a label
    for lane, label in labels.items():
        totals = groups.setdefault(label, [0, 0, 0.0])
        totals[0] += ts.sumo.lane.getLastStepHaltingNumber(lane)
        totals[1] += ts.sumo.lane.getLastStepVehicleNumber(lane)
        totals[2] += ts.sumo.lane.getWaitingTime(lane)

    lines = [
        f"Traffic signal controlling an intersection. {current}",
        f"The current green has lasted {elapsed} s. "
        + (
            f"Switching to a different green costs {ts.yellow_time} s of yellow."
            if can_switch
            else f"It must stay green for at least {ts.min_green} s before switching."
        ),
        "Traffic on the approaches:",
    ]
    for label, (stopped, vehicles, waiting) in groups.items():
        lines.append(f"- {label}: {stopped} stopped of {vehicles} vehicles, {int(waiting)} s total waiting")
    return "\n".join(lines)


class CLMAgent:
    """Traffic signal agent that picks the green phase ranked highest by a CLM. It does not learn from reward."""

    def __init__(self, ts, client: CLMClient, question: str = DEFAULT_QUESTION):
        """Initialize the agent.

        Args:
            ts (TrafficSignal): The traffic signal controlled by this agent (``env.traffic_signals[ts_id]``).
            client (CLMClient): Client for the CLM ranking server.
            question (str): Appended to the state description to form the ranking context.
        """
        self.ts = ts
        self.client = client
        self.question = question
        self.phase_texts = describe_phases(ts)
        self.labels = lane_labels(ts)
        self.action = None
        self.last_scores = None
        self.last_context_tokens = None

    def act(self) -> int:
        """Choose the next green phase."""
        ts = self.ts
        # The environment ignores the action until min_green has passed, so skip the query
        if ts.time_since_last_phase_change < ts.yellow_time + ts.min_green:
            self.action = ts.green_phase
            self.last_scores = None
            self.last_context_tokens = None
            return self.action

        self.last_scores = self.client.rank(self.context(), self.phase_texts)
        self.last_context_tokens = self.client.last_context_tokens
        self.action = int(np.argmax(self.last_scores))
        return self.action

    def context(self) -> str:
        """The ranking context for the current state: state description followed by the question."""
        return describe_state(self.ts, self.phase_texts, self.labels) + "\n" + self.question
