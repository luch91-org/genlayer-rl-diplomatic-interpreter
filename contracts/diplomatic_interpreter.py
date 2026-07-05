# { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }
"""DiplomaticInterpreter Intelligent Contract.

Cross-community mediation. Two communities hold opposing demands. The agent
drafts a compromise; an LLM mediator estimates how likely BOTH communities
are to accept it (0-10), and the environment's polarization index moves
deterministically from that agreed estimate -- a well-accepted compromise
cools the dispute, an inflammatory or one-sided one does not.

Design, applying the hard lessons from the sibling repos:

  - ONE concrete LLM judgment, single number. The reward is a single
    "mutual acceptance likelihood" 0-10 with explicit scoring anchors
    (addresses both demands + neutral tact + concrete middle path). A
    vaguer multi-facet prompt made validators disagree past tolerance and
    return NO_MAJORITY on a live shared testnet (see the scientific-heretic
    repo); a single anchored number keeps validator scores tightly
    clustered so consensus holds.

  - The state update is DETERMINISTIC. The polarization index is recomputed
    from the leader's already-consensus-agreed acceptance estimate, so every
    validator arrives at the identical post-state. Nature's response to a
    compromise must not be a fresh coin flip, or validators would never
    agree on the new polarization.

  - Self-contained single file (deploy_contract sends exactly one source
    file; no sibling imports). GenVM storage types only (u256 / str /
    DynArray) -- never bare dict/list. No floats on-chain: the polarization
    index (0.0-1.0) is stored as an integer 0-100, and scores are stored
    x100, since floats are neither GenVM-storable nor calldata-encodable.

  - gl.eq_principle.prompt_comparative(fn, principle=...) and
    gl.nondet.exec_prompt(prompt, response_format="json") (returns a parsed
    dict) are the current API forms.
"""

import json

from genlayer import *

# --- pure deterministic helpers (exec'd off-chain by contracts/logic.py) ----

INITIAL_POLARIZATION_100 = 80  # 0.80 on a 0.00-1.00 scale, stored as 0-100

# How hard the polarization index moves toward its new target each round.
# 0.5 == the new index is the midpoint of the old index and the target the
# latest compromise implies. Kept as integer arithmetic (numerator/denom)
# so it is exact and identical on every validator.
POLARIZATION_STEP_NUM = 1
POLARIZATION_STEP_DEN = 2

REWARD_EQUIVALENCE_PRINCIPLE = (
    "The two estimates agree if their acceptance numbers are within 1.5 "
    "points of each other and express the same overall judgment of whether "
    "both communities would accept the compromise. Wording may differ."
)


def clamp_score(value: float) -> float:
    return max(0.0, min(10.0, float(value)))


def clamp_100(value: int) -> int:
    return max(0, min(100, int(value)))


def score_to_x100(score: float) -> int:
    """Floats are not calldata-encodable and not GenVM-storable, so scores
    live on-chain as integers scaled x100 (7.5 -> 750)."""
    return int(round(float(score) * 100))


def build_judge_prompt(text: str, demand_a: str, demand_b: str, polarization_100: int) -> str:
    return (
        "You are mediating a dispute between two communities.\n"
        f"Side A demands: {demand_a}\n"
        f"Side B demands: {demand_b}\n"
        f"Current polarization (0 = fully reconciled, 100 = maximally "
        f"polarized): {polarization_100}\n"
        f"Proposed statement: {text}\n\n"
        "Give ONE number 0-10 for how likely BOTH communities are to accept "
        "this statement RIGHT NOW, at the current polarization. Fold two "
        "judgments into that single number:\n"
        "  (a) Fairness: does it treat both demands with a concrete, neutral "
        "middle path? One-sided, inflammatory, or empty statements are unfair.\n"
        "  (b) Fit to the moment: when polarization is HIGH (above 50) the "
        "sides need a DETAILED concrete plan that visibly addresses each "
        "demand -- a brief or general statement then feels dismissive of real "
        "grievances. When polarization is LOW (below 25) the dispute is "
        "largely settled, so a CONCISE good-faith statement that consolidates "
        "the deal fits best -- re-litigating every settled detail is "
        "unwelcome.\n"
        "Anchors: 9-10 fair AND well-fitted to the moment; 6-8 fair but a poor "
        "fit for the current temperature; 3-5 one-sided or an empty platitude; "
        "0-2 inflammatory or ignores one side.\n"
        'Return ONLY JSON: {"acceptance": <number 0-10>, "reason": "<short reason>"}'
    )


def parse_judge_output(raw) -> tuple:
    """Parse an LLM mediation response into (acceptance, reason).

    Accepts either an already-parsed dict (gl.nondet.exec_prompt output) or a
    JSON string; clamps acceptance to [0, 10]."""
    data = json.loads(raw) if isinstance(raw, str) else raw
    acceptance = clamp_score(data["acceptance"])
    reason = str(data.get("reason", ""))
    return acceptance, reason


def normalize_judge(acceptance: float, reason: str) -> str:
    """Canonical JSON string the leader returns to the equivalence principle,
    so all validators compare the same stable shape."""
    return json.dumps({"acceptance": float(acceptance), "reason": str(reason)}, sort_keys=True)


def updated_polarization_100(old_100: int, acceptance: float) -> int:
    """Deterministic polarization update from the agreed acceptance estimate.

    A compromise both sides are likely to accept (high acceptance) implies a
    low target polarization; an unacceptable one implies a high target. The
    index moves halfway from its current value toward that target. All
    integer arithmetic, so every validator computes the identical result.
    """
    # acceptance 0-10 -> target polarization 100-0 (10 * (10 - acceptance)).
    target_100 = clamp_100(int(round((10.0 - clamp_score(acceptance)) * 10)))
    moved = old_100 + (target_100 - old_100) * POLARIZATION_STEP_NUM // POLARIZATION_STEP_DEN
    return clamp_100(moved)


# --- the contract itself -----------------------------------------------------


class DiplomaticInterpreter(gl.Contract):
    polarization_100: u256
    proposals: DynArray[str]
    proposal_scores_x100: DynArray[u256]
    round: u256
    total_score_x100: u256
    last_reward_x100: u256
    last_acceptance_x100: u256
    last_reason: str

    def __init__(self):
        self.polarization_100 = u256(INITIAL_POLARIZATION_100)
        self.round = u256(0)
        self.total_score_x100 = u256(0)
        self.last_reward_x100 = u256(0)
        self.last_acceptance_x100 = u256(0)
        self.last_reason = ""

    @gl.public.view
    def get_state(self) -> dict:
        return {
            "polarization_100": int(self.polarization_100),
            "proposals": [str(p) for p in self.proposals],
            "proposal_scores_x100": [int(s) for s in self.proposal_scores_x100],
            "num_proposals": len(self.proposals),
            "round": int(self.round),
            "total_score_x100": int(self.total_score_x100),
            "last_reward_x100": int(self.last_reward_x100),
            "last_acceptance_x100": int(self.last_acceptance_x100),
            "last_reason": self.last_reason,
        }

    @gl.public.write
    def take_action(self, action: dict) -> dict:
        self.round = u256(int(self.round) + 1)
        round_after = int(self.round)

        text = str(action.get("text", ""))
        demand_a = str(action.get("side_a_demand", ""))
        demand_b = str(action.get("side_b_demand", ""))

        # Snapshot into LOCALS -- self is NOT accessible in the nondet block.
        text_snap = text
        demand_a_snap = demand_a
        demand_b_snap = demand_b
        polarization_snap = int(self.polarization_100)

        # Leader actually calls the LLM and returns a canonical JSON string
        # (a function that only builds a prompt runs no inference).
        def judge_block() -> str:
            prompt = build_judge_prompt(text_snap, demand_a_snap, demand_b_snap, polarization_snap)
            out = gl.nondet.exec_prompt(prompt, response_format="json")
            acceptance, reason = parse_judge_output(out)
            return normalize_judge(acceptance, reason)

        # Validators agree the leader's estimate is reasonable, not
        # byte-identical -- strict_eq would never pass on a subjective score.
        raw = gl.eq_principle.prompt_comparative(
            judge_block, principle=REWARD_EQUIVALENCE_PRINCIPLE
        )
        acceptance, reason = parse_judge_output(raw)

        # Deterministic state update from the AGREED acceptance estimate.
        reward = acceptance
        reward_x100 = score_to_x100(reward)
        self.polarization_100 = u256(
            updated_polarization_100(int(self.polarization_100), acceptance)
        )
        self.proposals.append(text)
        self.proposal_scores_x100.append(u256(reward_x100))
        self.total_score_x100 = u256(int(self.total_score_x100) + reward_x100)
        self.last_reward_x100 = u256(reward_x100)
        self.last_acceptance_x100 = u256(score_to_x100(acceptance))
        self.last_reason = reason
        return {
            "reward_x100": reward_x100,
            "acceptance_x100": score_to_x100(acceptance),
            "polarization_100": int(self.polarization_100),
            "reason": reason,
            "round": round_after,
        }

    @gl.public.view
    def get_score(self) -> int:
        """Total accumulated reward, scaled x100 (divide by 100 off-chain)."""
        return int(self.total_score_x100)

    @gl.public.view
    def get_polarization_100(self) -> int:
        """Current polarization index on a 0-100 scale (0.00-1.00 x100)."""
        return int(self.polarization_100)
