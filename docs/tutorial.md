# Tutorial: how DiplomaticInterpreter actually works

## The contract

`contracts/diplomatic_interpreter.py` models one dispute between two
communities with opposing demands, and a single **polarization index**
(0.0 fully reconciled → 1.0 maximally polarized, stored on-chain as an
integer 0-100). Every call to `take_action(action)` tables a compromise:

- **`{"type": "draft", "text": ..., "side_a_demand": ..., "side_b_demand": ...}`**
 - an LLM mediator estimates, on a single 0-10 scale with explicit
  anchors, how likely **both** communities are to accept this compromise.
  That estimate is the reward, and validators reach consensus on it via a
  comparative equivalence principle.

The polarization index then updates **deterministically** from the agreed
acceptance estimate: `updated_polarization_100(old, acceptance)` moves the
index halfway toward the target the compromise implies (high acceptance →
low target → the dispute cools; a rejected or inflammatory draft → high
target → it stays hot or worsens). Every validator computes the identical
new index.

## One concrete LLM number, and why

The mediator returns a **single** integer 0-10 with banded criteria - 9-10
fair *and* well-fitted to the current temperature, 6-8 fair but a poor fit
for the moment, 3-5 one-sided or an empty platitude, 0-2 inflammatory. Two
judgments (is it fair to both sides? does it fit how polarized the room is
right now?) are deliberately folded into that **one** number. This is a
design choice learned the hard way in the sibling `scientific-heretic` repo:
a vaguer, multi-facet prompt that asks for *several separate* ratings ("rate
the tact, the fairness, the likelihood…") makes validators disagree past the
equivalence tolerance and the transaction intermittently returns
**NO_MAJORITY** on the shared testnet, never landing. Collapsing everything
into one anchored acceptance number keeps validator scores tightly clustered
 - a live studionet run of this contract reached consensus on every step with
no `NO_MAJORITY` - and it's still a genuine subjective judgment, exactly the
kind only an LLM-consensus chain can score.

The subjectivity lives in that one number; **everything downstream is
deterministic** (the reward is the number; the polarization update is
integer arithmetic on it). Nature's response to a compromise can't be a
fresh coin flip on-chain, or validators would never agree on the new
polarization.

## GenVM storage and calldata constraints

Inherited from live studionet deploys in the sibling repos:

1. **Storage uses GenVM types only** - `u256`, `str`, `DynArray[str]`,
   `DynArray[u256]` - never bare `dict`/`list` (deploy fails with "class is
   not marked for usage within storage").
2. **No floats on-chain.** The polarization index (0.0-1.0) is stored as an
   integer 0-100; scores are ×100-scaled integers. `GenLayerEnv` converts
   both back to floats for the agent. `calldata.encode(1.5)` raises.

`contracts/logic.py` execs the actual contract source with a stubbed
`genlayer` module, so pytest and MockEnv exercise the deployed code itself;
a regression-guard test asserts the contract has no sibling imports.

## Discretizing a free-text domain - and why this is real RL, not a bandit

Tabular Q-learning needs a discrete action set. The dispute is fixed (a
riverside land-use fight); the agent's menu (`agent/agent.py`'s
`TEMPLATES`) is six compromise archetypes - a strong **detailed** compromise,
two one-sided ones, a vague platitude, an inflammatory one, and a **concise**
consolidation. It is not told which is which; it learns each one's
acceptance from the mediator's rewards.

`serialize_state()` keys the Q-table on the **polarization bucket** alone
(resolved / low / medium / high). That single field is what makes this a
control problem rather than a bandit: **the best compromise depends on how
polarized the room currently is.** While the dispute is hot, the detailed
compromise that spells out how each demand is met (template 3) lands best.
Once it has cooled, the concise good-faith consolidation (template 5) fits
better - a settled dispute wants a short confirmation, not a full
renegotiation - and that same brevity reads as *dismissive of real
grievances* while the room is still hot, so template 5 scores badly at high
polarization. The agent has to learn *diplomatic tact*: match the move to
the temperature of the room.

This is not just a MockEnv fiction. The deployed reward prompt folds
"fit to the current temperature" **into the single acceptance number**, and
a live studionet probe confirms the mediator honors it: the concise
consolidation scored ~6 at high polarization and climbed to ~8 once the
dispute had settled, while the detailed compromise held ~7-8 throughout. The
mock-trained policy (detailed while hot, concise once resolved) therefore
transfers straight to the chain - see the verified live episode in the
README, where the dispute cooled from 0.80 to 0.23.

## Mock vs. live: the tradeoff

- **`MockEnv`** (default) models each archetype's acceptance as a function
  of the current polarization bucket (the table where the tact lesson
  lives), then applies the contract's **own** `updated_polarization_100`
  so its polarization arithmetic is identical to the chain's. Instant and
  free - used for dev, CI, and the 500-episode training curve.
- **`GenLayerEnv`** is the real thing: each step is an on-chain
  LLM-consensus transaction (~25-60 s and gas on studionet), with
  idempotent retry-with-backoff for transient testnet hiccups.

## Tuning the hyperparameters

Standard knobs (`--alpha`, `--gamma`, `--epsilon-decay`, `--epsilon-min`,
`--optimistic-init`, `--max-steps`) in `agent/train.py`. Domain notes:

- `--optimistic-init` (default 50.0) matters most: because the best
  compromise is state-dependent, the agent must try every archetype in
  every polarization bucket. Optimistic Q-init forces that exploration;
  with zero-init it tends to lock onto the first archetype that worked at
  high polarization and never discover the affirmation move once the
  dispute has cooled.
- Episodes default to 5 steps: enough for a strong compromise to cool the
  dispute from "high" into the "resolved" range and for the agent to switch
  to the maintenance move. Below ~3 steps the resolved regime is barely
  sampled.
- `--gamma`: the payoff structure is fairly myopic (each draft's reward
  arrives immediately), so gamma mostly controls how much the agent values
  cooling the dispute quickly versus grabbing the best immediate
  acceptance.

## Running against a real GenVM (optional gltest note)

The default test suite never needs a GenVM. If you want Direct Mode tests
via `genlayer-test`, install it in a **separate virtualenv**: it pins
`genlayer-py==0.3.0`, which conflicts with the `>=0.18.0` this repo's agent
requires.
