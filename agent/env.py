"""Environment abstraction so the RL loop in agent/agent.py never has to know
whether it's talking to a local heuristic or a deployed GenLayer contract.

MockEnv is the default everywhere (dev, CI, tuning): it models each
compromise archetype's acceptance as a function of the CURRENT polarization
(so the best compromise genuinely depends on the state), then reuses the
contract's own deterministic polarization update -- imported from
contracts.logic -- so the mock's polarization arithmetic is identical to the
deployed contract's, not a re-derivation that could drift.

GenLayerEnv talks to a deployed contract for the real demo, where every step
is an on-chain LLM-consensus call. It retries with backoff on transient
chain failures (idempotently: a failed take_action does not advance the
round, so it re-reads state before resubmitting).
"""

from __future__ import annotations

import random
import time
from typing import Any, Protocol

from agent.agent import TEMPLATES, polarization_bucket
from contracts.logic import (
    INITIAL_POLARIZATION_100,
    clamp_score,
    score_to_x100,
    updated_polarization_100,
)

DEFAULT_MAX_STEPS = 5

_TEMPLATE_ID_BY_TEXT: dict[str, int] = {text: i for i, text in enumerate(TEMPLATES)}

# MockEnv's stand-in for the LLM mediator: base mutual-acceptance (0-10) for
# each compromise archetype, keyed by the CURRENT polarization bucket. This
# is where the domain's tact lesson lives, and the shape here is directionally
# faithful to what the deployed mediator actually does on studionet (see the
# probe numbers in the tutorial): the strong DETAILED compromise (3) lands
# best while the room is hot, but once the dispute has cooled the CONCISE
# consolidation (5) fits better -- and that same brevity reads as dismissive
# of real grievances while polarization is still high, so 5 scores low at
# "high". One-sided (0, 1) and inflammatory (4) drafts never land; a vague
# platitude (2) is weak everywhere. (Mock magnitudes are a touch more
# optimistic than the chain -- the live mediator is conservative, ~7-8 for a
# well-fitted move -- but the per-bucket argmax matches, so the policy
# transfers.)
_ACCEPTANCE: dict[int, dict[str, float]] = {
    0: {"high": 3.0, "medium": 3.0, "low": 3.0, "resolved": 3.0},
    1: {"high": 3.0, "medium": 3.0, "low": 3.0, "resolved": 3.0},
    2: {"high": 4.0, "medium": 4.0, "low": 5.0, "resolved": 5.0},
    3: {"high": 9.0, "medium": 9.0, "low": 7.0, "resolved": 6.0},
    4: {"high": 1.0, "medium": 1.0, "low": 1.0, "resolved": 1.0},
    5: {"high": 5.0, "medium": 6.0, "low": 8.0, "resolved": 9.0},
}
_UNKNOWN_ACCEPTANCE = 3.0  # off-menu free text the mock can't classify
_NOISE_STD = 0.4


class Env(Protocol):
    def reset(self) -> dict[str, Any]: ...

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        """Returns (reward, reason, next_state)."""
        ...


class MockEnv:
    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        random_start: bool = False,
    ):
        self.max_steps = max_steps
        # random_start seeds each episode from a random polarization instead of
        # always the fixed 0.80. The deployed contract always starts at 0.80,
        # but on-chain the LLM's exact acceptance score can land the running
        # polarization in ANY bucket mid-episode (e.g. a strong compromise
        # scored 8.0 rather than 8.8 lands it on the "medium" boundary). If
        # training only ever walks the one trajectory the fixed start implies,
        # off-path buckets never get learned and the policy picks a garbage
        # action the moment the live chain nudges it off that path. Random
        # starts make every bucket an on-policy start state, so the agent
        # learns the tact-correct move in all of them. Default off so the
        # deterministic MockEnv tests still start highly polarized.
        self.random_start = random_start
        self._rng = random.Random(seed)
        self._reset_fields()
        self.reset()

    def _reset_fields(self) -> None:
        self.polarization_100 = INITIAL_POLARIZATION_100
        self.proposals: list[str] = []
        self.scores_x100: list[int] = []
        self.total_score = 0.0
        self.round = 0
        self.last_reward = 0.0
        self.last_acceptance = 0.0
        self.last_reason = ""

    def reset(self) -> dict[str, Any]:
        self._reset_fields()
        if self.random_start:
            self.polarization_100 = self._rng.randint(0, 100)
        return self._public_state()

    def _public_state(self) -> dict[str, Any]:
        return {
            "polarization": self.polarization_100 / 100.0,
            "polarization_100": self.polarization_100,
            "proposals": list(self.proposals),
            "proposal_scores_x100": list(self.scores_x100),
            "num_proposals": len(self.proposals),
            "round": self.round,
            "total_score": self.total_score,
            "last_reward": self.last_reward,
            "last_acceptance": self.last_acceptance,
            "last_reason": self.last_reason,
        }

    def _acceptance(self, text: str) -> tuple[float, str]:
        tid = _TEMPLATE_ID_BY_TEXT.get(text)
        bucket = polarization_bucket(self.polarization_100 / 100.0)
        if tid is None:
            base = _UNKNOWN_ACCEPTANCE
            label = "off-menu draft"
        else:
            base = _ACCEPTANCE[tid][bucket]
            label = f"compromise #{tid}"
        noisy = clamp_score(base + self._rng.gauss(0.0, _NOISE_STD))
        reason = f"{label}: both-sides acceptance ~{noisy:.1f} at {bucket} polarization"
        return noisy, reason

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        self.round += 1
        text = str(action.get("text", ""))
        acceptance, reason = self._acceptance(text)

        # Reward is the acceptance estimate; polarization updates through the
        # SAME deterministic function the contract uses.
        reward = acceptance
        self.polarization_100 = updated_polarization_100(self.polarization_100, acceptance)
        self.proposals.append(text)
        self.scores_x100.append(score_to_x100(reward))
        self.total_score += reward
        self.last_reward = reward
        self.last_acceptance = acceptance
        self.last_reason = reason
        return reward, reason, self._public_state()

    def is_episode_done(self) -> bool:
        return self.round >= self.max_steps


class GenLayerEnv:
    """Talks to a deployed DiplomaticInterpreter contract via the first-party
    genlayer-py SDK (signatures confirmed against genlayer-py 0.18.0 and live
    studionet runs in sibling repos). genlayer-py requires Python >= 3.12 at
    import time; the import is deferred into __init__ so MockEnv-only
    workflows never need it.
    """

    def __init__(
        self,
        address: str,
        chain: str = "localnet",
        private_key: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_retries: int = 4,
    ):
        from genlayer_py import create_account, create_client
        from genlayer_py.chains import localnet, studionet, testnet_asimov, testnet_bradbury
        from genlayer_py.exceptions import GenLayerError
        from genlayer_py.types import TransactionStatus

        chains = {
            "localnet": localnet,
            "testnet_asimov": testnet_asimov,
            "testnet_bradbury": testnet_bradbury,
            "studionet": studionet,
        }
        if chain not in chains:
            raise ValueError(f"Unknown chain '{chain}'. Choose from: {sorted(chains)}")

        self.address = address
        self.max_steps = max_steps
        self.max_retries = max_retries
        self._TransactionStatus = TransactionStatus
        self._GenLayerError = GenLayerError
        self.account = create_account(private_key) if private_key else create_account()
        self.client = create_client(chain=chains[chain], account=self.account)
        # fund_account only refuses when chain.id != localnet.id -- studionet
        # shares localnet's chain id 61999, so both work.
        if chain in ("localnet", "studionet"):
            try:
                self.client.fund_account(address=self.account.address, amount=10**18)
            except Exception as exc:  # best effort: some setups pre-fund accounts
                print(f"[GenLayerEnv] fund_account skipped: {exc}")
        self._round = 0

    def reset(self) -> dict[str, Any]:
        self._round = 0
        return self._get_state()

    def _get_state(self) -> dict[str, Any]:
        # Reads are idempotent, so retry transient network drops (studionet
        # occasionally aborts the HTTPS connection mid-request). A read blip
        # must not tear down a multi-step live episode.
        raw: Any = None
        for attempt in range(5):
            try:
                raw = self.client.read_contract(
                    address=self.address,
                    function_name="get_state",
                    args=[],
                )
                break
            except Exception as exc:  # noqa: BLE001 -- reads are safe to retry
                if attempt == 4:
                    raise
                print(f"[GenLayerEnv] read_contract retry {attempt + 1}/5 ({str(exc)[:60]})")
                time.sleep(3.0 * (attempt + 1))
        # polarization_100 is an integer 0-100 on-chain; scores are x100.
        # Convert to the float shape MockEnv produces so the agent never sees
        # the difference.
        return {
            "polarization": int(raw["polarization_100"]) / 100.0,
            "polarization_100": int(raw["polarization_100"]),
            "proposals": [str(p) for p in raw["proposals"]],
            "proposal_scores_x100": [int(s) for s in raw["proposal_scores_x100"]],
            "num_proposals": int(raw["num_proposals"]),
            "round": int(raw["round"]),
            "total_score": int(raw["total_score_x100"]) / 100.0,
            "last_reward": int(raw["last_reward_x100"]) / 100.0,
            "last_acceptance": int(raw["last_acceptance_x100"]) / 100.0,
            "last_reason": str(raw.get("last_reason", "")),
        }

    def step(self, action: dict[str, Any]) -> tuple[float, str, dict[str, Any]]:
        # Retry with backoff on transient chain failures (studionet
        # intermittently returns NO_MAJORITY or drops the HTTPS connection).
        #
        # Idempotency is enforced by the on-chain round counter: take_action
        # increments `round`, so a single logical step must advance it by
        # exactly one. The subtle failure this guards against is DOUBLE
        # SUBMISSION: on a slow validator set a take_action can stay PENDING
        # longer than the receipt wait, the wait raises, and a naive retry
        # fires a SECOND identical tx -- then BOTH land and the round jumps by
        # two (observed live: a 5-step episode reached round 9). So once
        # write_contract has returned a tx hash, we NEVER resubmit merely
        # because waiting is slow; we poll the round for a grace window and
        # only resubmit if the round still has not advanced (the submission
        # itself genuinely failed and nothing landed).
        round_before = int(self._get_state()["round"])
        last_exc: Exception | None = None

        def round_advanced() -> bool:
            try:
                return int(self._get_state()["round"]) > round_before
            except Exception:  # a failed read must not be mistaken for "landed"
                return False

        for attempt in range(self.max_retries):
            submitted = False
            try:
                tx_hash = self.client.write_contract(
                    address=self.address,
                    function_name="take_action",
                    account=self.account,
                    args=[action],
                    value=0,
                )
                submitted = True
                # interval*retries = 3s*80 = up to 240s: live studionet steps
                # have been seen to take >100s under LLM-consensus load.
                self.client.wait_for_transaction_receipt(
                    transaction_hash=tx_hash,
                    status=self._TransactionStatus.ACCEPTED,
                    interval=3000,
                    retries=80,
                )
                break
            except Exception as exc:  # noqa: BLE001 -- round guard is the real safety net
                last_exc = exc
                # If the tx was submitted, give it a grace window to land
                # before deciding to resubmit -- resubmitting a still-pending
                # tx is exactly what causes the double-count.
                if submitted:
                    for _ in range(20):  # ~60s grace
                        if round_advanced():
                            break
                        time.sleep(3.0)
                if round_advanced():
                    break  # it landed; do NOT resubmit
                backoff = 3.0 * (attempt + 1)
                print(
                    f"[GenLayerEnv] step attempt {attempt + 1}/{self.max_retries} failed "
                    f"({str(exc)[:80]}); retrying in {backoff:.0f}s"
                )
                time.sleep(backoff)
        else:
            raise RuntimeError(
                f"take_action did not land after {self.max_retries} attempts"
            ) from last_exc

        state = self._get_state()
        self._round += 1
        return float(state["last_reward"]), state.get("last_reason", ""), state

    def is_episode_done(self) -> bool:
        return self._round >= self.max_steps
