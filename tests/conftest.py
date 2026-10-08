"""Fakes for running an experiment without a GPU: the real Qwen3 chat template,
one token per whitespace-separated word, and random lens logits."""

import re
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

from cogniload import registry

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3_chat_template.jinja"
CONFIG = yaml.safe_load((registry.REGISTRY.parent / "experiments.yaml").read_text())
VOCAB = 2000


class Tokenizer:
    eos_token_id = 0

    def __init__(self):
        import jinja2

        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
        self._t = env.from_string(TEMPLATE.read_text())
        self.vocab = {}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kw):
        return self._t.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)

    def __call__(self, text, **kw):
        ms = list(re.finditer(r"\S+", text))
        return {"input_ids": [self.vocab.setdefault(m.group(), len(self.vocab) + 1) for m in ms],
                "offset_mapping": [m.span() for m in ms]}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids, **kw):
        inverse = {v: k for k, v in self.vocab.items()}
        return " " + " ".join(inverse.get(int(i), "?") for i in ids)


class Model:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self._hf_model = types.SimpleNamespace(generate=lambda ids, max_new_tokens, **kw: torch.cat(
            [ids, torch.ones(1, max_new_tokens, dtype=torch.long)], dim=1))

    def encode(self, text):
        return torch.tensor([self.tokenizer.encode(text)])


class Lens:
    def apply(self, model, prompt, *, layers, positions, use_jacobian):
        n = len(positions)
        return ({l: torch.randn(n, VOCAB) for l in layers}, torch.randn(n, VOCAB),
                model.encode(prompt))


@pytest.fixture
def tok():
    pytest.importorskip("jinja2")
    return Tokenizer()


@pytest.fixture
def ctx(monkeypatch, tok):
    """What an experiment's runner works with, with the model and lens faked."""
    vis = types.ModuleType("jlens.vis")
    vis._meaningful_token_mask = lambda tokenizer, v, device: torch.ones(v, dtype=torch.bool)
    monkeypatch.setitem(sys.modules, "jlens", types.ModuleType("jlens"))
    monkeypatch.setitem(sys.modules, "jlens.vis", vis)
    torch.manual_seed(0)
    return types.SimpleNamespace(model=Model(), lens=Lens(), spec=registry.resolve("dev"),
                                 config=CONFIG, layers=[10, 11, 12], band=(11, 13))
