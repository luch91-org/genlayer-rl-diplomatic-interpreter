"""Tests for the deterministic parts of DiplomaticInterpreter.

These exercise the REAL contract source via contracts/logic.py's stubbed
loader -- no GenVM runtime involved. We do NOT assert on exact LLM
acceptance scores anywhere: the LLM mediator is non-deterministic by design,
so instead we test (a) the deterministic polarization update that runs
identically on every validator, and (b) the reward-parsing/normalizing logic.
"""

from __future__ import annotations

import json

import pytest

from contracts.logic import (
    INITIAL_POLARIZATION_100,
    DiplomaticInterpreter,
    build_judge_prompt,
    clamp_100,
    clamp_score,
    normalize_judge,
    parse_judge_output,
    score_to_x100,
    updated_polarization_100,
)

# --- polarization update (the deterministic heart of the domain) -------------


def test_high_acceptance_lowers_polarization():
    # A compromise both sides accept (10) targets polarization 0, so from 80
    # the index moves halfway toward 0 -> 40.
    assert updated_polarization_100(80, 10.0) == 40


def test_zero_acceptance_targets_maximum_polarization():
    # Acceptance 0 targets polarization 100; from 80, halfway toward 100 -> 90.
    assert updated_polarization_100(80, 0.0) == 90


def test_acceptance_holds_polarization_at_its_target():
    # Acceptance 5 targets polarization 50; starting already at 50 stays 50.
    assert updated_polarization_100(50, 5.0) == 50


def test_repeated_high_acceptance_drives_polarization_toward_zero():
    pol = INITIAL_POLARIZATION_100
    seen = [pol]
    for _ in range(6):
        pol = updated_polarization_100(pol, 9.0)
        seen.append(pol)
    # Monotonically non-increasing and ends well into the "resolved" range.
    assert all(b <= a for a, b in zip(seen, seen[1:]))
    assert pol < 25


def test_polarization_update_is_bounded_0_100():
    assert updated_polarization_100(0, 0.0) <= 100
    assert updated_polarization_100(100, 10.0) >= 0
    assert 0 <= updated_polarization_100(100, 0.0) <= 100


def test_polarization_update_clamps_out_of_range_acceptance():
    # Acceptance is clamped to [0,10] before use, so 99 behaves like 10.
    assert updated_polarization_100(80, 99.0) == updated_polarization_100(80, 10.0)


# --- reward parsing / bounds -------------------------------------------------


def test_parse_judge_output_extracts_acceptance_and_reason():
    acceptance, reason = parse_judge_output({"acceptance": 8, "reason": "balanced"})
    assert acceptance == 8.0
    assert reason == "balanced"


def test_parse_judge_output_accepts_json_string_and_clamps():
    acceptance, _r = parse_judge_output(json.dumps({"acceptance": 42, "reason": "x"}))
    assert acceptance == 10.0


def test_parse_judge_output_rejects_missing_acceptance():
    with pytest.raises(KeyError):
        parse_judge_output({"reason": "no acceptance field"})


def test_clamp_and_scale():
    assert clamp_score(-1) == 0.0
    assert clamp_score(11) == 10.0
    assert clamp_100(-5) == 0
    assert clamp_100(150) == 100
    assert score_to_x100(7.5) == 750
    assert score_to_x100(6.666) == 667  # rounds, never truncates


def test_normalize_judge_round_trips_through_parse():
    normalized = normalize_judge(7.0, "ok")
    assert normalized == normalize_judge(7.0, "ok")  # stable
    acceptance, reason = parse_judge_output(normalized)
    assert (acceptance, reason) == (7.0, "ok")


def test_build_judge_prompt_includes_its_key_fields():
    prompt = build_judge_prompt("my compromise", "A wants X", "B wants Y", 80)
    assert "my compromise" in prompt
    assert "A wants X" in prompt
    assert "B wants Y" in prompt
    assert "80" in prompt


# --- contract wiring (real source, stubbed genlayer runtime) -----------------


def test_contract_source_is_self_contained():
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parent.parent / "contracts" / "diplomatic_interpreter.py"
    ).read_text(encoding="utf-8")
    assert "from contracts" not in source
    assert "import contracts" not in source
    assert "import agent" not in source


def test_contract_initializes_at_starting_polarization():
    contract = DiplomaticInterpreter()
    state = contract.get_state()
    assert state["polarization_100"] == INITIAL_POLARIZATION_100
    assert state["proposals"] == []
    assert state["num_proposals"] == 0
    assert state["round"] == 0
    assert state["total_score_x100"] == 0
    assert contract.get_score() == 0
    assert contract.get_polarization_100() == INITIAL_POLARIZATION_100


def test_contract_take_action_cannot_run_off_chain():
    # take_action hits the equivalence principle, which the stub makes raise
    # -- so nothing off-chain can call the LLM mediator.
    contract = DiplomaticInterpreter()
    with pytest.raises(NotImplementedError):
        contract.take_action(
            {"type": "draft", "text": "x", "side_a_demand": "a", "side_b_demand": "b"}
        )
