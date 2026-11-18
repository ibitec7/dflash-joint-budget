"""The sglang backend must read both return shapes (dict and object) without losing text.

sglang's offline Engine.generate returns dicts with keys text/output_ids/meta_info and a
meta_info carrying spec_verify_ct + spec_accept_length (tokens emitted per round, bonus
column included). Our first on-box run lost every generation because the adapter only read
attributes, so this is pinned by tests.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from serve import EngineConfig, _SGLangBackend


class _FakeEngine:
    def __init__(self, outs):
        self._outs = outs
        self.calls = []

    def generate(self, prompts, sampling_params=None):
        self.calls.append((list(prompts), sampling_params))
        return self._outs


def _backend(outs):
    b = object.__new__(_SGLangBackend)
    b._engine = _FakeEngine(outs)
    b._cfg = EngineConfig(drafter="dry-run/placeholder", draft_width=16, verify_columns=17)
    return b


def test_dict_output_keeps_text_and_rounds():
    outs = [{"text": "answer", "output_ids": [1, 2],
             "meta_info": {"completion_tokens": 128, "spec_verify_ct": 39,
                           "spec_accept_length": 3.282051282051282}}]
    texts, metas = _backend(outs).generate(["p"])
    assert texts == ["answer"]
    assert metas[0]["rounds"] == 39
    assert metas[0]["accepted"] == pytest.approx(128.0, rel=1e-9)   # 3.282 * 39
    assert metas[0]["completion_tokens"] == 128


def test_object_output_still_works():
    obj = SimpleNamespace(text="obj", meta_info={"spec_verify_ct": 4, "spec_accepted_tokens": 10})
    texts, metas = _backend([obj]).generate(["p"])
    assert texts == ["obj"]
    assert metas[0]["accepted"] == 10            # explicit field wins over the product


def test_missing_meta_is_not_a_crash():
    texts, metas = _backend([{"text": ""}]).generate(["p"])
    assert texts == [""]
    assert metas[0]["rounds"] is None and metas[0]["accepted"] is None


def test_prompts_are_passed_through_unchanged():
    b = _backend([{"text": "x", "meta_info": {}}])
    b.generate(["a", "b"])
    assert b._engine.calls[0][0] == ["a", "b"]
