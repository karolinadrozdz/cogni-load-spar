# Attribution

This directory adapts the experimental code and data released with:

> Kumaran, D., Daw, N., Osindero, S., Veličković, P., & Patraucean, V. (2026).
> *Causal Evidence that Language Models use Confidence to Drive Behavior.*
> Nature Machine Intelligence. arXiv:2603.22161. Code and data: OSF project `resw4`.

- **Code** (`abstention_modal.py` prompts, option-shuffling, choice-token logprob readout,
  verbal-confidence classes and parsing; the CSV schema): derived from
  `Confidence_Drives_Abstention_Notebook.ipynb`, released under the MIT License.
  Modifications: vLLM batching on Modal, reasoning-trace separation, thinking on/off switch,
  parse-method and truncation flags.
- **Data** (`data/phase0_questions.csv`, `data/phase1_questions.csv`): the paper's SimpleQA
  4-way multiple-choice sets with generated foils, released under CC-BY-4.0; derived from
  SimpleQA (Wei et al., 2024, OpenAI). `data/README-osf.md` is the authors' README, unmodified.

`analyze.py` and `handoff.sh` are ours; the statistics they compute follow the paper's Methods.

`steering_modal.py` and `build_vector.py` follow the paper's steering recipe (confidence direction from high- vs low-margin
correct trials, added at 3% of the mean residual norm at the steered layer) and use its Phase 2 prompt. The
reasoning-mode vector uses the same difference-of-means construction with labels from k-sample consistency.
