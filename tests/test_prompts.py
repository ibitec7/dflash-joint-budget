"""The frozen prompt set is the unit of comparison, so its invariants are tested as code.

Every test runs offline: an autouse fixture makes the Hugging Face loader raise unless a test
installs its own, so the suite never touches the network and never depends on a live dataset.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prompts

DATA = ROOT / "data"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Default source is the fixture; a test that wants 'hf' patches `_load_hf` itself."""
    prompts.clear_cache()

    def boom(benchmark):
        raise ConnectionError("offline test")

    monkeypatch.setattr(prompts, "_load_hf", boom)
    yield
    prompts.clear_cache()


def fake_pool(benchmark: str = "math500", size: int = 300) -> list[dict]:
    return [{"id": f"{benchmark}-{i:03d}", "prompt": f"question {i}"} for i in range(size)]


# ------------------------------------------------------------------ determinism

def test_build_is_identical_across_two_calls():
    a = prompts.build("math500", 200, 0)
    b = prompts.build("math500", 200, 0)
    assert a == b
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]


def test_build_returns_two_hundred_distinct_prompts():
    items = prompts.build("math500", 200, 0)
    assert len(items) == 200
    assert len({it["id"] for it in items}) == 200
    assert all(it["max_new_tokens"] == 256 for it in items)
    assert all(set(it) >= {"id", "prompt", "max_new_tokens", "source"} for it in items)


def test_ordering_is_stable_and_seeded():
    base = [it["id"] for it in prompts.build("math500", 200, 0)]
    again = [it["id"] for it in prompts.build("math500", 200, 0)]
    other = [it["id"] for it in prompts.build("math500", 200, 1)]
    assert base == again, "same seed must give the same order"
    assert base != other, "a different seed must give a different order"


def test_a_seed_permutes_without_changing_which_prompts_exist():
    # With n == pool size the whole set is used, so a seed may only reorder it.
    a = [it["id"] for it in prompts.build("mtbench", 80, 0)]
    b = [it["id"] for it in prompts.build("mtbench", 80, 1)]
    assert len(set(a)) == 80
    assert a != b
    assert sorted(a) == sorted(b)


def test_prefix_is_stable_when_n_grows():
    short = prompts.build("math500", 50, 0)
    long = prompts.build("math500", 200, 0)
    assert [it["id"] for it in short] == [it["id"] for it in long[:50]]


# ------------------------------------------------------------------ source labelling

def test_fixture_fallback_is_declared():
    items = prompts.build("math500", 200, 0)
    assert {it["source"] for it in items} == {"fixture"}


def test_hf_source_is_declared(monkeypatch):
    monkeypatch.setattr(prompts, "_load_hf", lambda b: fake_pool("math500", 300))
    prompts.clear_cache()
    items = prompts.build("math500", 200, 0)
    assert {it["source"] for it in items} == {"hf"}
    assert len(items) == 200
    assert {it["prompt"] for it in items} <= {f"question {i}" for i in range(300)}


def test_a_crashing_loader_never_propagates(monkeypatch):
    def explode(benchmark):
        raise RuntimeError("hub down")

    monkeypatch.setattr(prompts, "_load_hf", explode)
    prompts.clear_cache()
    assert len(prompts.build("mtbench", 200, 0)) == 200


def test_offline_env_var_skips_the_hub(monkeypatch):
    monkeypatch.setenv(prompts.OFFLINE_ENV, "1")

    def never_called(benchmark):
        raise AssertionError("_load_hf must not run when DFLASH_PROMPTS_OFFLINE=1")

    monkeypatch.setattr(prompts, "_load_hf", never_called)
    prompts.clear_cache()
    assert {it["source"] for it in prompts.build("math500", 10, 0)} == {"fixture"}


# ------------------------------------------------------------------ mtbench repetition

def test_mtbench_repeats_the_curated_set_with_distinct_ids():
    items = prompts.build("mtbench", 200, 0)
    assert len(items) == 200
    assert len({it["id"] for it in items}) == 200, "repeat ids must be distinct"
    assert all("-r" in it["id"] for it in items[80:]), "second pass onward is suffixed"
    assert len({it["prompt"] for it in items}) == 80


def test_mtbench_fixture_records_the_repetition():
    doc = json.loads((DATA / "frozen_mtbench.json").read_text())
    assert doc["pool_size"] == 80
    assert doc["repeated"] is True
    assert doc["repeats"] == 3


# ------------------------------------------------------------------ freeze / load

def test_freeze_load_roundtrip_is_byte_identical(tmp_path):
    path = prompts.freeze("math500", 20, 0, tmp_path / "f.json")
    got = prompts.load_frozen(path)
    on_disk = json.loads(path.read_text(encoding="utf-8"))["prompts"]
    assert got == on_disk
    assert json.dumps(got, sort_keys=True) == json.dumps(on_disk, sort_keys=True)
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_freeze_twice_is_byte_identical(tmp_path):
    a = prompts.freeze("math500", 20, 0, tmp_path / "a.json")
    first = a.read_bytes()
    again = prompts.freeze("math500", 20, 0, a)
    assert again == a
    assert a.read_bytes() == first, "re-freezing the same set must not rewrite the file"


def test_freeze_is_deterministic_across_paths(tmp_path):
    a = prompts.freeze("math500", 20, 0, tmp_path / "a.json")
    b = prompts.freeze("math500", 20, 0, tmp_path / "b.json")
    assert a.read_bytes() == b.read_bytes()


def test_freeze_refuses_to_replace_a_frozen_set(tmp_path):
    path = prompts.freeze("math500", 20, 0, tmp_path / "f.json")
    with pytest.raises(FileExistsError, match="different frozen prompt set"):
        prompts.freeze("math500", 20, 1, path)
    assert [it["id"] for it in prompts.load_frozen(path)] == [
        it["id"] for it in prompts.build("math500", 20, 0)
    ]
    forced = prompts.freeze("math500", 20, 1, path, force=True)
    assert [it["id"] for it in prompts.load_frozen(forced)] == [
        it["id"] for it in prompts.build("math500", 20, 1)
    ]


def test_inspecting_never_rewrites_the_fixture(tmp_path, monkeypatch, capsys):
    path = prompts.freeze("math500", 20, 0, tmp_path / "frozen_math500.json")
    monkeypatch.setattr(prompts, "DATA_DIR", tmp_path)
    before = path.read_bytes()
    assert prompts.main(["--benchmark", "math500"]) == 0
    assert path.read_bytes() == before
    assert "20 prompts" in capsys.readouterr().out


def test_freeze_preserves_the_pool_for_offline_rebuilds(tmp_path):
    path = prompts.freeze("mtbench", 200, 0, tmp_path / "m.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["pool_size"] == 80
    assert len(doc["pool"]) == 80
    assert [it["id"] for it in doc["prompts"]] == [it["id"] for it in prompts.load_frozen(path)]


def test_committed_fixtures_match_what_build_reproduces():
    for benchmark in prompts.BENCHMARKS:
        path = DATA / f"frozen_{benchmark}.json"
        assert path.is_file(), f"missing committed fixture {path}"
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = prompts.load_frozen(path)
        assert len(items) == doc["n"] == 200
        assert len({it["id"] for it in items}) == 200
        if doc["seed"] == 0 and doc["n"] == 200:
            rebuilt = prompts.build(benchmark, 200, 0)
            assert [it["id"] for it in rebuilt] == [it["id"] for it in items], (
                f"{benchmark}: offline build must reproduce the frozen order"
            )


# ------------------------------------------------------------------ guards

def test_unknown_benchmark_is_rejected():
    with pytest.raises(ValueError, match="unknown benchmark"):
        prompts.build("gsm8k", 10, 0)


def test_load_frozen_rejects_incomplete_entries(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"prompts": [{"id": "x", "prompt": "y"}]}))
    with pytest.raises(ValueError, match="missing"):
        prompts.load_frozen(bad)
