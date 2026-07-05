"""Tabular Q-learning agent for DiplomaticInterpreter.

The dispute is fixed (a riverside land-use fight); the agent's choice is
WHICH compromise to table. Its action menu is a set of canned compromise
archetypes (`TEMPLATES`) of varying diplomatic quality -- balanced,
one-sided, vague, inflammatory, and light-affirmation. Tabular Q-learning
needs a discrete action set, and the judgment being learned (which
compromise a mediator's LLM will find acceptable to BOTH sides, and when)
is fully expressed by choosing among these archetypes. Free-text drafting
is a documented future path.

State space design note: serialize_state() keys the Q-table on the
polarization bucket alone (resolved / low / medium / high). That is what
makes this genuinely a control problem rather than a bandit: the best
compromise DEPENDS on how polarized the room currently is. A strong DETAILED
compromise that visibly addresses each demand lands best while the dispute
is hot; once it has cooled, a CONCISE good-faith consolidation beats
re-litigating a settled deal, and the same brevity that fits a settled
dispute reads as dismissive while the room is still hot. This is the
diplomatic-tact lesson, and it is one the live LLM mediator actually honors:
the deployed reward prompt folds "fit to the current temperature" into its
single acceptance score (confirmed on studionet -- the concise consolidation
scores ~6 at high polarization and ~8 once resolved). Four buckets keep the
table tiny and convergence fast.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Any

# The fixed dispute the agent is mediating. Sent on every action so the
# contract's LLM mediator has the demands to judge the compromise against.
SIDE_A_DEMAND = "The riverside parcel must become a public park with no development."
SIDE_B_DEMAND = "The riverside parcel must be rezoned for a mixed-use housing development."

# Canned compromise archetypes. The agent is NOT told which are good; it
# learns each one's acceptance from the LLM mediator's rewards. Chosen so a
# real mediator LLM and MockEnv's heuristic broadly agree on their quality.
TEMPLATES: list[str] = [
    # 0: one-sided toward A -- side B will reject
    "The entire riverside parcel will be preserved as public parkland; no "
    "housing will be built there.",
    # 1: one-sided toward B -- side A will reject
    "The entire riverside parcel will be rezoned for mixed-use housing " "development.",
    # 2: vague platitude -- resolves nothing concrete
    "Both communities should come together in a spirit of cooperation and "
    "mutual respect to find a harmonious path forward.",
    # 3: strong DETAILED compromise -- a concrete middle path spelled out in
    # full. Best while the dispute is hot: when polarization is high both sides
    # need to see each demand visibly addressed.
    "The parcel will be split: a public riverfront park along the water, with "
    "mixed-use housing set back from the bank, a shared community green, and "
    "an affordable-housing quota agreed by both communities.",
    # 4: inflammatory -- accusatory, deepens the rift
    "Side B's demands are made in bad faith and should be dismissed outright; "
    "the other community must concede entirely.",
    # 5: CONCISE consolidation -- names the split briefly and commits to it.
    # Best once the dispute has cooled: a settled dispute wants a short
    # good-faith confirmation, not a full renegotiation. That same brevity
    # reads as dismissive of real grievances while the room is still hot, so
    # the mediator scores it low at high polarization.
    "Both sides confirm the split -- riverfront park and set-back housing -- "
    "and commit in good faith to implement it together.",
]

# Action space: table each compromise archetype (the fixed demands ride
# along so the contract can judge it against the dispute).
ACTIONS: list[dict[str, Any]] = [
    {
        "type": "draft",
        "text": text,
        "side_a_demand": SIDE_A_DEMAND,
        "side_b_demand": SIDE_B_DEMAND,
        "_template_id": i,
    }
    for i, text in enumerate(TEMPLATES)
]


def strip_action(action: dict[str, Any]) -> dict[str, Any]:
    """The dict actually sent on-chain: internal keys (leading underscore)
    are stripped so the contract only sees its documented action schema."""
    return {k: v for k, v in action.items() if not k.startswith("_")}


StateKey = tuple


def polarization_bucket(polarization: float) -> str:
    """Bucket the 0.0-1.0 polarization index. The boundaries match MockEnv's
    reward table so the agent's state distinctions line up with where the
    best compromise actually changes."""
    if polarization < 0.25:
        return "resolved"
    if polarization < 0.50:
        return "low"
    if polarization < 0.75:
        return "medium"
    return "high"


def serialize_state(state: dict[str, Any]) -> StateKey:
    return (polarization_bucket(float(state["polarization"])),)


class QLearningAgent:
    def __init__(
        self,
        actions: list[dict[str, Any]] = ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        optimistic_init: float = 0.0,
        seed: int | None = None,
    ):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Optimistic initialization: seeding every action's Q above its
        # realistic value makes any never-tried action look best until it is
        # sampled, forcing the agent to try each compromise at least once in
        # each polarization bucket. Since the best compromise is
        # state-dependent, thorough per-bucket exploration matters. Default
        # 0.0 keeps the base behavior (and the arithmetic tests) unchanged;
        # train.py raises it.
        self.optimistic_init = optimistic_init
        self.q_table: dict[StateKey, list[float]] = {}
        self._rng = random.Random(seed)

    def _ensure_state(self, key: StateKey) -> list[float]:
        if key not in self.q_table:
            self.q_table[key] = [self.optimistic_init] * len(self.actions)
        return self.q_table[key]

    def select_action(self, state: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)

        if self._rng.random() < self.epsilon:
            idx = self._rng.randrange(len(self.actions))
        else:
            best_q = max(q_values)
            best_indices = [i for i, q in enumerate(q_values) if q == best_q]
            idx = self._rng.choice(best_indices)

        return idx, self.actions[idx]

    def update(
        self,
        state: dict[str, Any],
        action_idx: int,
        reward: float,
        next_state: dict[str, Any],
    ) -> None:
        key = serialize_state(state)
        next_key = serialize_state(next_state)
        q_values = self._ensure_state(key)
        next_q_values = self._ensure_state(next_key)

        td_target = reward + self.gamma * max(next_q_values)
        td_error = td_target - q_values[action_idx]
        q_values[action_idx] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def best_action(self, state: dict[str, Any]) -> dict[str, Any]:
        key = serialize_state(state)
        q_values = self._ensure_state(key)
        best_q = max(q_values)
        best_indices = [i for i, q in enumerate(q_values) if q == best_q]
        return self.actions[self._rng.choice(best_indices)]

    def save(self, path: str | Path) -> None:
        payload = {
            "q_table": {repr(key): values for key, values in self.q_table.items()},
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "optimistic_init": self.optimistic_init,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self.q_table = {ast.literal_eval(key): values for key, values in payload["q_table"].items()}
        self.epsilon = payload.get("epsilon", self.epsilon)
        self.alpha = payload.get("alpha", self.alpha)
        self.gamma = payload.get("gamma", self.gamma)
        self.epsilon_min = payload.get("epsilon_min", self.epsilon_min)
        self.epsilon_decay = payload.get("epsilon_decay", self.epsilon_decay)
        self.optimistic_init = payload.get("optimistic_init", self.optimistic_init)
