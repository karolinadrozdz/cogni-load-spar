"""The two conventions that keep runs comparable: pinned aliases, frozen prompts.

`test_thinking_assertion_catches_the_silent_failure` is the important one — it
renders the real Qwen3 chat template, so it fails if an upstream template change
ever makes the internal condition externalize its state.
"""

import json
from pathlib import Path

import pytest
import yaml

from cogniload import bands, prompts, registry

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3_chat_template.jinja"


# --- registry -----------------------------------------------------------------


@pytest.mark.parametrize("alias", ["dev", "prod"])
def test_aliases_resolve_and_are_pinned(alias):
    spec = registry.resolve(alias)
    assert len(spec.hf_revision) == 40, "model ref must be a full commit sha"
    assert spec.enable_thinking is False
    assert spec.n_lens_matrices == spec.n_layers - 1


def test_dev_and_prod_differ_only_in_scale():
    """A dev->prod discrepancy must implicate scale, not provenance."""
    dev, prod = registry.resolve("dev"), registry.resolve("prod")
    assert (dev.lens_repo, dev.lens_revision) == (prod.lens_repo, prod.lens_revision)
    assert dev.hf_id.split("-")[0] == prod.hf_id.split("-")[0] or "Qwen" in prod.hf_id
    assert dev.enable_thinking == prod.enable_thinking
    assert dev.hf_id != prod.hf_id


def test_unpinned_alias_is_refused(tmp_path):
    doc = yaml.safe_load(registry.REGISTRY.read_text())
    doc["aliases"]["loose"] = dict(doc["aliases"]["dev"], hf_revision="main")
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="unpinned"):
        registry.resolve("loose", path=p)


def test_thinking_flag_is_required_never_defaulted(tmp_path):
    doc = yaml.safe_load(registry.REGISTRY.read_text())
    del doc["aliases"]["dev"]["enable_thinking"]
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(KeyError, match="enable_thinking"):
        registry.resolve("dev", path=p)


def test_unknown_alias_lists_the_known_ones():
    with pytest.raises(KeyError, match="dev, prod"):
        registry.resolve("nope")


def test_band_must_be_discovered_before_use():
    with pytest.raises(ValueError, match="Phase 0 R1"):
        registry.resolve("dev").band_layers()


def test_experiment_config_names_an_alias_not_a_model():
    cfg = yaml.safe_load((registry.REGISTRY.parent / "exp1.yaml").read_text())
    assert cfg["model"] in yaml.safe_load(registry.REGISTRY.read_text())["aliases"]
    assert "/" not in cfg["model"], "exp1.yaml must not name an HF model directly"


# --- prompts ------------------------------------------------------------------


def test_prompt_sha_changes_with_content():
    a = prompts.Prompt("x", 1, "hello {y}")
    assert a.sha == prompts.Prompt("x", 1, "hello {y}").sha
    assert a.sha != prompts.Prompt("x", 2, "hello {y}").sha  # version bump
    assert a.sha != prompts.Prompt("x", 1, "hello {z}").sha  # text edit


def test_prompt_set_sha_covers_every_prompt(monkeypatch):
    before = prompts.prompt_set_sha()
    monkeypatch.setitem(prompts.ALL, "new", prompts.Prompt("new", 1, "hi"))
    assert prompts.prompt_set_sha() != before


def test_keep_track_prompt_renders():
    out = prompts.KEEP_TRACK.render(tracked="animal, fruit", stream="cat, pear")
    assert "animal, fruit" in out and "cat, pear" in out


# --- the silent-failure guard -------------------------------------------------


class _Tokenizer:
    """Minimal stand-in that renders the real Qwen3 template."""

    def __init__(self, template: str):
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
        self._t = env.from_string(template)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kw):
        return self._t.render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kw
        )


@pytest.fixture
def tok():
    pytest.importorskip("jinja2")
    if not TEMPLATE.exists():
        pytest.skip(f"fixture missing: {TEMPLATE}")
    return _Tokenizer(TEMPLATE.read_text())


MSGS = [{"role": "user", "content": "Track these categories: animal, fruit."}]


def test_thinking_off_renders_the_empty_block_and_prefill_follows_it(tok):
    out = prompts.render_chat(tok, MSGS, enable_thinking=False, prefill="Answer:")
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")


def test_thinking_on_is_accepted_when_requested(tok):
    out = prompts.render_chat(tok, MSGS, enable_thinking=True)
    assert out.endswith("<think>\n")


def test_thinking_assertion_catches_the_silent_failure(tok):
    """A template that ignores enable_thinking=False must raise, not pass.

    This is the failure that would turn the internal condition into the
    externalized one with no error. See DECISIONS.md D4.
    """

    class Ignores(_Tokenizer):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **{**kw, "enable_thinking": True})

    bad = Ignores(TEMPLATE.read_text())
    with pytest.raises(AssertionError, match="reasoning ON"):
        prompts.render_chat(bad, MSGS, enable_thinking=False)


# --- band lives with the results, not the registry ----------------------------


def test_registry_file_is_never_rewritten(tmp_path):
    """A YAML round-trip would drop the comment marking enable_thinking
    mandatory. Nothing in the codebase may write this file. See DECISIONS.md D8.
    """
    assert not hasattr(registry, "pin_band")
    before = registry.REGISTRY.read_text()
    bands.save("dev", (10, 20), tmp_path)
    assert registry.REGISTRY.read_text() == before
    assert "MANDATORY" in before


def test_band_round_trips_through_the_results_tree(tmp_path):
    assert bands.load("dev", tmp_path) is None
    bands.save("dev", (12, 19), tmp_path, found_by="r1", legible_layers=[12, 15, 18])
    assert bands.load("dev", tmp_path) == (12, 19)

    spec = registry.resolve("dev", results_dir=tmp_path)
    assert spec.band == (12, 19)
    assert spec.band_layers() == list(range(12, 19))
    assert spec.provenance()["band"] == [12, 19]


def test_band_is_required_before_use(tmp_path):
    with pytest.raises(FileNotFoundError, match="Phase 0 R1"):
        bands.require("dev", tmp_path)
    with pytest.raises(ValueError, match="valid half-open range"):
        bands.save("dev", (20, 10), tmp_path)


def test_resolve_without_results_dir_has_no_band():
    assert registry.resolve("dev").band is None


# --- turn structure (DECISIONS.md D9) -----------------------------------------

WORDS = ["cat", "pear", "red", "dog", "plum"]
TRACKED = ("animal", "fruit")


def test_q1_is_one_user_turn_ending_in_the_prefill(tok):
    out = prompts.build_q1(tok, WORDS, TRACKED, "animal", enable_thinking=False)
    assert out.count("<|im_start|>user") == 1
    assert out.endswith("Answer:")
    assert "cat, pear, red" in out and "most recent animal" in out


def test_q2_is_a_fresh_render_with_no_q1_in_context(tok):
    """If Q2 followed Q1, one category would already be resolved in context."""
    out = prompts.build_q2(tok, WORDS, TRACKED, enable_thinking=False)
    assert out.count("<|im_start|>user") == 1
    assert "<|im_start|>assistant\n<think>" in out  # only the generation prompt
    assert "most recent animal" not in out
    assert out.endswith("<think>\n\n</think>\n\n")  # no prefill


@pytest.mark.parametrize(
    "build", [
        lambda t: prompts.build_q1(t, WORDS, TRACKED, "animal", enable_thinking=False),
        lambda t: prompts.build_q2(t, WORDS, TRACKED, enable_thinking=False),
    ],
)
def test_every_builder_renders_thinking_off(tok, build):
    assert "<think>\n\n</think>" in build(tok)


# --- R2 prompt (DECISIONS.md D13) ---------------------------------------------


def test_r2_uses_the_same_prefill_and_regime_as_q1(tok):
    out = prompts.build_r2(tok, "animal", enable_thinking=False)
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")
    assert prompts.R2_PREFILL == prompts.Q1_PREFILL
    assert "Think of a animal. Answer in one word." in out


def test_r2_prompt_is_in_the_manifest_hash():
    assert "phase0.r2_think_of" in prompts.ALL


# --- exemplar cache invalidation ----------------------------------------------


def test_exemplar_cache_rebuilds_when_pools_change(tmp_path, monkeypatch):
    """Editing POOLS must not leave a stale cache in place for whoever forgets
    --force; two collaborators would silently run different stimuli."""
    from cogniload import exemplars

    monkeypatch.setattr(exemplars, "CACHE_DIR", tmp_path)

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [0]  # every word is one token

    first = exemplars.build(Tok(), "fake", n_per_category=3)
    sha_before = json.loads((tmp_path / "fake.json").read_text())["pools_sha"]

    pools = {k: list(v) for k, v in exemplars.POOLS.items()}
    pools["animal"] = ["zebra"] + pools["animal"]
    monkeypatch.setattr(exemplars, "POOLS", pools)

    second = exemplars.build(Tok(), "fake", n_per_category=3)
    sha_after = json.loads((tmp_path / "fake.json").read_text())["pools_sha"]

    assert sha_after != sha_before
    assert second["animal"][0] == "zebra"
    assert first["animal"] != second["animal"]


def test_exemplar_cache_is_reused_when_pools_are_unchanged(tmp_path, monkeypatch):
    from cogniload import exemplars

    monkeypatch.setattr(exemplars, "CACHE_DIR", tmp_path)
    calls = []

    class Tok:
        def encode(self, text, add_special_tokens=False):
            calls.append(text)
            return [0]

    exemplars.build(Tok(), "fake", n_per_category=3)
    n = len(calls)
    exemplars.build(Tok(), "fake", n_per_category=3)
    assert len(calls) == n, "cache was not reused"


def test_changing_n_per_category_also_invalidates(tmp_path, monkeypatch):
    from cogniload import exemplars

    monkeypatch.setattr(exemplars, "CACHE_DIR", tmp_path)

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [0]

    assert len(exemplars.build(Tok(), "fake", n_per_category=3)["animal"]) == 3
    assert len(exemplars.build(Tok(), "fake", n_per_category=5)["animal"]) == 5


def test_band_may_not_reach_the_unfitted_final_layer(tmp_path):
    """The lens fits layers 0..n-2; a band touching n-1 must fail here, not
    deep inside lens.apply."""
    spec = registry.resolve("dev")
    bands.save("dev", (spec.n_layers - 3, spec.n_layers), tmp_path)
    with pytest.raises(ValueError, match="only fits layers"):
        registry.resolve("dev", results_dir=tmp_path).band_layers()
