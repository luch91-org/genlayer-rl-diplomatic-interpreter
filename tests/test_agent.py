"""Tests for the tabular Q-learning agent, its polarization-bucket
serialization, and a MockEnv behavior/coverage check. Everything here runs
with no network and no GenVM.
"""

from __future__ import annotations

import json

from agent.agent import (
    ACTIONS,
    SIDE_A_DEMAND,
    SIDE_B_DEMAND,
    TEMPLATES,
    QLearningAgent,
    polarization_bucket,
    serialize_state,
    strip_action,
)
from agent.env import MockEnv
from agent.train import run_episode


def state_at(polarization: float) -> dict:
    return {"polarization": polarization}


# --- action space ------------------------------------------------------------


def test_action_space_is_one_draft_per_template():
    assert len(ACTIONS) == len(TEMPLATES)
    assert all(a["type"] == "draft" for a in ACTIONS)


def test_actions_carry_the_fixed_dispute_demands():
    for a in ACTIONS:
        assert a["side_a_demand"] == SIDE_A_DEMAND
        assert a["side_b_demand"] == SIDE_B_DEMAND


def test_strip_action_removes_internal_template_id_but_keeps_schema():
    stripped = strip_action(ACTIONS[3])
    assert "_template_id" not in stripped
    assert set(stripped) == {"type", "text", "side_a_demand", "side_b_demand"}


# --- state serialization -----------------------------------------------------


def test_polarization_buckets_cover_the_range():
    assert polarization_bucket(0.90) == "high"
    assert polarization_bucket(0.60) == "medium"
    assert polarization_bucket(0.40) == "low"
    assert polarization_bucket(0.10) == "resolved"


def test_serialize_state_keys_on_bucket():
    assert serialize_state(state_at(0.80)) == ("high",)
    assert serialize_state(state_at(0.10)) == ("resolved",)


# --- Q-learning mechanics ----------------------------------------------------


def test_epsilon_decays_to_floor():
    agent = QLearningAgent(epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.9)
    for _ in range(200):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05


def test_bellman_update_matches_hand_computed_value():
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    s = state_at(0.80)
    s2 = state_at(0.40)
    agent.update(s, 3, reward=10.0, next_state=s2)
    key = serialize_state(s)
    # td_target = 10 + 0.9*0; new_q = 0 + 0.5*(10-0) = 5.0
    assert agent.q_table[key][3] == 5.0
    assert serialize_state(s2) in agent.q_table


def test_greedy_selection_picks_max_q_action():
    agent = QLearningAgent(epsilon_start=0.0, epsilon_min=0.0)
    key = serialize_state(state_at(0.80))
    agent._ensure_state(key)
    agent.q_table[key][3] = 50.0
    idx, action = agent.select_action(state_at(0.80))
    assert idx == 3
    assert action["_template_id"] == 3


def test_optimistic_init_seeds_unseen_states_high():
    agent = QLearningAgent(optimistic_init=50.0)
    q = agent._ensure_state(serialize_state(state_at(0.80)))
    assert all(v == 50.0 for v in q)


def test_save_and_load_round_trip(tmp_path):
    agent = QLearningAgent(alpha=0.2, gamma=0.8, optimistic_init=50.0)
    key = serialize_state(state_at(0.80))
    agent._ensure_state(key)
    agent.q_table[key][3] = 9.0
    agent.epsilon = 0.33
    path = tmp_path / "q.json"
    agent.save(path)

    loaded = QLearningAgent()
    loaded.load(path)
    assert loaded.q_table == agent.q_table
    assert loaded.epsilon == 0.33
    assert loaded.alpha == 0.2
    assert loaded.optimistic_init == 50.0


def test_saved_q_table_has_string_keys(tmp_path):
    agent = QLearningAgent()
    agent._ensure_state(serialize_state(state_at(0.80)))
    path = tmp_path / "q.json"
    agent.save(path)
    raw = json.loads(path.read_text())
    assert all(isinstance(k, str) for k in raw["q_table"])


# --- MockEnv behavior --------------------------------------------------------


def test_mock_env_starts_highly_polarized():
    env = MockEnv(seed=0)
    state = env.reset()
    assert state["polarization"] > 0.75  # "high" bucket


def test_strong_compromise_lowers_polarization_from_high():
    env = MockEnv(seed=1)
    before = env.reset()["polarization"]
    _r, _reason, after = env.step(strip_action(ACTIONS[3]))  # balanced compromise
    assert after["polarization"] < before


def test_inflammatory_draft_does_not_reduce_polarization():
    env = MockEnv(seed=2)
    before = env.reset()["polarization"]
    _r, _reason, after = env.step(strip_action(ACTIONS[4]))  # inflammatory
    assert after["polarization"] >= before


def test_mock_env_scores_every_action_without_falling_through():
    env = MockEnv(seed=0)
    for a in ACTIONS:
        env.reset()
        reward, reason, _s = env.step(strip_action(a))
        assert 0.0 <= reward <= 10.0
        assert reason  # non-empty


def test_run_episode_accumulates_reward_and_learns():
    env = MockEnv(max_steps=5, seed=7)
    agent = QLearningAgent(alpha=0.5, gamma=0.9, epsilon_start=1.0, optimistic_init=50.0, seed=7)
    episode_reward, last_reason = run_episode(env, agent)
    assert episode_reward > 0.0
    assert isinstance(last_reason, str)
    assert len(agent.q_table) >= 1
