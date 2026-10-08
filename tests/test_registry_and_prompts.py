"""Pinned aliases, where the band lives, and prompt rendering.

`test_thinking_assertion_catches_the_silent_failure` matters most: it renders
the real Qwen3 template, so it fails if a template change ever turns thinking on.
"""

import pytest
import yaml
from conftest import CONFIG, Tokenizer

from cogniload import bands, experiment, prompts, registry


@pytest.mark.parametrize("alias", ["dev", "prod"])
def test_aliases_resolve_and_are_pinned(alias):
    spec = registry.resolve(alias)
    assert len(spec["hf_revision"]) == len(spec["lens_revision"]) == 40
    assert spec["enable_thinking"] is False


def test_dev_and_prod_differ_only_in_scale():
    dev, prod = registry.resolve("dev"), registry.resolve("prod")
    assert all(dev[k] == prod[k] for k in ("lens_repo", "lens_revision", "enable_thinking"))
    assert dev["hf_id"] != prod["hf_id"]


def _registry(tmp_path, edit):
    doc = yaml.safe_load(registry.REGISTRY.read_text())
    edit(doc)
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


@pytest.mark.parametrize("edit", [
    lambda d: d["aliases"]["dev"].update(hf_revision="main"),
    lambda d: d["defaults"].update(lens_revision="qwen-n1000"),
])
def test_unpinned_model_or_lens_is_refused(tmp_path, edit):
    with pytest.raises(ValueError, match="unpinned"):
        registry.resolve("dev", path=_registry(tmp_path, edit))


def test_thinking_flag_is_required_never_defaulted(tmp_path):
    p = _registry(tmp_path, lambda d: d["aliases"]["dev"].pop("enable_thinking"))
    with pytest.raises(KeyError, match="enable_thinking"):
        registry.resolve("dev", path=p)


def test_unknown_alias_lists_the_known_ones():
    with pytest.raises(KeyError, match="dev, prod"):
        registry.resolve("nope")


def test_config_names_an_alias_not_a_model():
    assert CONFIG["model"] in yaml.safe_load(registry.REGISTRY.read_text())["aliases"]


def test_band_round_trips_through_the_results_tree(tmp_path):
    assert registry.resolve("dev", results_dir=tmp_path)["band"] is None
    bands.save("dev", (12, 19), tmp_path, criterion="test")
    assert registry.resolve("dev", results_dir=tmp_path)["band"] == (12, 19)
    with pytest.raises(ValueError, match="valid half-open range"):
        bands.save("dev", (20, 10), tmp_path)


def test_band_must_exist_and_sit_inside_the_recorded_layers(tmp_path):
    with pytest.raises(FileNotFoundError, match="find_band"):
        experiment.recorded_layers(registry.resolve("dev", results_dir=tmp_path), CONFIG)
    n = registry.resolve("dev")["n_layers"]
    bands.save("dev", (n - 3, n), tmp_path)  # the lens has no final-layer matrix
    with pytest.raises(ValueError, match="not inside"):
        experiment.recorded_layers(registry.resolve("dev", results_dir=tmp_path), CONFIG)


# --- prompts --------------------------------------------------------------------

USER = "What is the capital of France? Answer in one word."


def test_thinking_off_renders_the_empty_block_and_prefill_follows_it(tok):
    out = prompts.render_chat(tok, USER, enable_thinking=False, prefill="Answer:")
    assert out.endswith("<think>\n\n</think>\n\nAnswer:")


def test_thinking_on_is_accepted_when_requested(tok):
    assert prompts.render_chat(tok, USER, enable_thinking=True).endswith("<think>\n")


def test_thinking_assertion_catches_the_silent_failure(tok):
    """A template that ignores enable_thinking=False must raise, not pass."""

    class Ignores(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **{**kw, "enable_thinking": True})

    with pytest.raises(AssertionError, match="reasoning ON"):
        prompts.render_chat(Ignores(), USER, enable_thinking=False)
