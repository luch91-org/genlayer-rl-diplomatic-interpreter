# genlayer-rl-diplomatic-interpreter

Domain-scoped build guide. Inherits the shared engineering spec from the
[GenLayer RL Agent Autonomy](https://github.com/luch91-org)
umbrella CLAUDE.md - read that first. This file only covers what's specific
to this domain.

## Domain

Cross-community mediation. State = a polarization index (0.0 reconciled →
1.0 maximally polarized) plus the log of tabled compromises. Action =
`draft_compromise(text, side_a_demand, side_b_demand)`. An LLM mediator
estimates how likely both communities are to accept the compromise (0-10),
that estimate is the reward, and the polarization index updates
deterministically from it. The agent learns to draft compromises that cool
the dispute - and to match the compromise to the current temperature.

## The two design decisions that matter here

1. **One concrete LLM number; everything else deterministic.** The mediator
   returns a single anchored acceptance score (0-10). The reward is that
   number and the polarization update is integer arithmetic on it
   (`updated_polarization_100`). This is deliberate: the sibling
   `scientific-heretic` repo proved that a vaguer, multi-facet LLM reward
   spreads validator scores past tolerance and returns NO_MAJORITY on live
   studionet. Keep the LLM's job to one grounded number; never move the
   polarization update itself inside a nondet block.
2. **State is the polarization bucket, and the optimal action is
   state-dependent.** This is a genuine control problem, not a bandit: a
   strong *detailed* compromise wins while polarized, a *concise*
   consolidation wins once resolved (and loses while still hot, where its
   brevity reads as dismissive). Crucially this is not just a MockEnv
   fiction - the deployed reward prompt folds "fit to the current
   temperature" into its single acceptance score, and a live studionet probe
   confirmed the mediator honors it. If you extend the action menu, preserve
   at least one archetype whose best-use depends on the bucket, or the RL
   framing collapses to "pick the one good template".

## Where things live

- `contracts/diplomatic_interpreter.py` - the `gl.Contract`, single source
  of truth, fully self-contained (single-file deployment; regression guard
  in tests). Storage: `u256` polarization (0-100), `DynArray[str]`
  proposals, `DynArray[u256]` scores. No floats, no bare dict/list.
- `contracts/logic.py` - no logic of its own; execs the contract source
  with a stubbed `genlayer` module and re-exports the helpers, so pytest
  and MockEnv exercise the deployed code itself.
- `agent/env.py` - MockEnv's `_ACCEPTANCE` table (archetype × polarization
  bucket) is where the tact lesson is encoded; it reuses the contract's
  `updated_polarization_100` so the mock's polarization arithmetic can't
  drift from the chain's. `GenLayerEnv.step` has idempotent
  retry-with-backoff for transient testnet failures.
- `agent/agent.py` - the fixed dispute demands (`SIDE_A_DEMAND`,
  `SIDE_B_DEMAND`) ride along on every action; `strip_action()` removes the
  internal `_template_id` before anything reaches the chain. Optimistic
  Q-init is on in training so every archetype is tried in every bucket.

## Non-negotiable GenLayer rules for this contract

(Full rationale in the umbrella CLAUDE.md.)

- The nondet call lives only inside the inner `judge_block()` passed to
  `gl.eq_principle.prompt_comparative(fn, principle=...)`; never `self`
  inside it - snapshot demands, text, and polarization to locals first.
- Never `strict_eq` for the acceptance score.
- Storage: GenVM types only; no floats on-chain (polarization 0-100
  integer, scores ×100). `GenLayerEnv` converts back for the agent.
- The polarization update must stay deterministic - derive it only from the
  already-agreed acceptance estimate and existing state, never from a fresh
  nondet call or randomness.

## Success bar for this repo

500 mock episodes climb from roughly 4 to 8+ per-step rolling average
(episode rewards sum over `--max-steps`, default 5; optimistic Q-init is on
by default and is what gets it above 8), with the polarization index
visibly falling across a converged episode; `agent/q_table.json`,
`docs/learning_curve.png`, and `logs/training.txt` exist afterward; and a
short live run against a deployed contract shows real mediator-scored drafts
cooling the dispute. The verified live address and log are recorded in the
README.
