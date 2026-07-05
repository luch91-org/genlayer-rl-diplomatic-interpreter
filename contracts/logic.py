"""Off-chain access to the REAL contract's pure helpers.

contracts/diplomatic_interpreter.py must be fully self-contained because
deploy_contract(code=...) sends exactly one source file on-chain -- it
cannot import sibling modules. But its deterministic helpers (the
polarization update, reward parsing, prompt building) still need to be
testable with plain pytest, and agent/env.py's MockEnv needs the same
polarization arithmetic.

Rather than maintaining a hand-written mirror that could silently drift
from the deployed code, this module execs the contract's actual source with
a stubbed `genlayer` module (the real one only exists inside the GenVM
runtime; the `genlayer` package on PyPI is a 1.3 KB placeholder). Everything
exported here IS the deployed code, byte for byte.

The stub mimics just enough of the GenVM SDK surface for the contract to be
importable and its deterministic paths runnable off-chain:
  - gl.Contract auto-creates annotated storage fields on instantiation
    (GenVM pre-creates storage slots; plain Python doesn't).
  - DynArray behaves as a list, TreeMap as a dict, u256 as an int.
  - gl.nondet / gl.eq_principle / gl.vm raise on use, so nothing off-chain
    can accidentally "succeed" at calling an LLM.
"""

from __future__ import annotations

import sys
import types
import typing
from pathlib import Path

_CONTRACT_PATH = Path(__file__).resolve().parent / "diplomatic_interpreter.py"


class _StubTreeMap(dict):
    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item if isinstance(item, tuple) else (item,))


class _StubDynArray(list):
    def __class_getitem__(cls, item):
        return types.GenericAlias(cls, item if isinstance(item, tuple) else (item,))


class _StubU256(int):
    pass


def _default_for_annotation(annotation):
    origin = typing.get_origin(annotation)
    if origin is _StubTreeMap or annotation is _StubTreeMap:
        return _StubTreeMap()
    if origin is _StubDynArray or annotation is _StubDynArray:
        return _StubDynArray()
    if annotation is _StubU256:
        return _StubU256(0)
    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is bool:
        return False
    return None


class _StubContract:
    def __new__(cls):
        instance = super().__new__(cls)
        for name, annotation in getattr(cls, "__annotations__", {}).items():
            object.__setattr__(instance, name, _default_for_annotation(annotation))
        return instance


def _make_genlayer_stub() -> types.ModuleType:
    def _gl_runtime_only(*_args, **_kwargs):
        raise NotImplementedError(
            "gl.nondet / gl.eq_principle / gl.vm only exist inside the GenVM "
            "runtime. Off-chain code may only use the contract's deterministic "
            "helpers."
        )

    class _Public:
        @staticmethod
        def view(fn):
            return fn

        @staticmethod
        def write(fn):
            return fn

    gl = types.SimpleNamespace(
        Contract=_StubContract,
        public=_Public,
        nondet=types.SimpleNamespace(
            exec_prompt=_gl_runtime_only,
            web=types.SimpleNamespace(get=_gl_runtime_only, render=_gl_runtime_only),
        ),
        eq_principle=types.SimpleNamespace(
            prompt_comparative=_gl_runtime_only,
            prompt_non_comparative=_gl_runtime_only,
            strict_eq=_gl_runtime_only,
        ),
        vm=types.SimpleNamespace(run_nondet=_gl_runtime_only),
    )

    module = types.ModuleType("genlayer")
    module.gl = gl  # type: ignore[attr-defined]
    module.TreeMap = _StubTreeMap  # type: ignore[attr-defined]
    module.DynArray = _StubDynArray  # type: ignore[attr-defined]
    module.u256 = _StubU256  # type: ignore[attr-defined]
    module.__all__ = ["gl", "TreeMap", "DynArray", "u256"]  # type: ignore[attr-defined]
    return module


def _load_contract_namespace() -> dict:
    source = _CONTRACT_PATH.read_text(encoding="utf-8")
    original = sys.modules.get("genlayer")
    sys.modules["genlayer"] = _make_genlayer_stub()
    try:
        namespace: dict = {"__name__": "contracts._diplomatic_interpreter_offchain"}
        # dont_inherit=True: without it, compile() inherits THIS module's
        # `from __future__ import annotations`, which would stringify the
        # contract's class annotations and break the stub's storage-field
        # auto-creation (it inspects the evaluated annotation objects).
        exec(compile(source, str(_CONTRACT_PATH), "exec", dont_inherit=True), namespace)
        return namespace
    finally:
        if original is not None:
            sys.modules["genlayer"] = original
        else:
            del sys.modules["genlayer"]


_ns = _load_contract_namespace()

INITIAL_POLARIZATION_100 = _ns["INITIAL_POLARIZATION_100"]
POLARIZATION_STEP_NUM = _ns["POLARIZATION_STEP_NUM"]
POLARIZATION_STEP_DEN = _ns["POLARIZATION_STEP_DEN"]
REWARD_EQUIVALENCE_PRINCIPLE = _ns["REWARD_EQUIVALENCE_PRINCIPLE"]
clamp_score = _ns["clamp_score"]
clamp_100 = _ns["clamp_100"]
score_to_x100 = _ns["score_to_x100"]
build_judge_prompt = _ns["build_judge_prompt"]
parse_judge_output = _ns["parse_judge_output"]
normalize_judge = _ns["normalize_judge"]
updated_polarization_100 = _ns["updated_polarization_100"]
# The contract class itself, with gl stubbed: deterministic methods
# (__init__, get_state, get_score, get_polarization_100) work off-chain;
# take_action raises NotImplementedError the moment it reaches the
# equivalence principle.
DiplomaticInterpreter = _ns["DiplomaticInterpreter"]

__all__ = [
    "INITIAL_POLARIZATION_100",
    "POLARIZATION_STEP_NUM",
    "POLARIZATION_STEP_DEN",
    "REWARD_EQUIVALENCE_PRINCIPLE",
    "clamp_score",
    "clamp_100",
    "score_to_x100",
    "build_judge_prompt",
    "parse_judge_output",
    "normalize_judge",
    "updated_polarization_100",
    "DiplomaticInterpreter",
]
