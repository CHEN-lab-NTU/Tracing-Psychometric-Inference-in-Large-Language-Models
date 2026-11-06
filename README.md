# Tracing Psychometric Inference in Large Language Models

Code, administered questionnaires, de-identified human item responses, and main-experiment model responses for the NeurIPS 2026 paper *Tracing Psychometric Inference in Large Language Models*.

The study has 272 participants, seven Traditional Chinese questionnaires, 114 administered items, and 16 scored subscales. In the cross-persona experiment, a model sees one participant's answers to one questionnaire and predicts answers to the other six. The seven source-questionnaire rotations are analyzed separately.

## Repository contents

| Path | Contents |
| --- | --- |
| `data/human/` | Item responses for 272 participants, with one CSV per questionnaire. |
| `data/questionnaires/` | Administered Chinese items, response options, and questionnaire instructions. |
| `data/llm_behavior/` | Item responses for all 14 primary model variants and seven persona-scale rotations. |
| `src/psychometric_inference/` | Scoring, prompts, model registry, and mechanistic experiment code. |
| `scripts/behavior/` | Model response generation and behavioral controls. |
| `scripts/analysis/` | Statistical analyses and robustness checks. |
| `scripts/figures/` | Paper figure scripts. |
| `environment.yml` | Software dependencies. |

The `ID` column (`sub-001` through `sub-272`) links each participant's answers across the seven questionnaires. Item columns are `Q1`, `Q2`, and so on. The human CSVs contain already reverse-coded item values. [`scoring.py`](src/psychometric_inference/scoring.py) restores the original response choices when constructing persona prompts and applies the scoring rules to model responses.

The model-response CSVs use the same participant IDs in their `Subject_ID` column. Each `persona_*` directory also has a `*_persona.csv` file with the source-scale responses supplied in the prompt. Activations, generated matrices, figures, and model weights are not included. The model IDs and output directory names are defined in [`model_registry.py`](src/psychometric_inference/model_registry.py). See [REPRODUCE.md](REPRODUCE.md) for analysis commands.

## Start with the human-data check

Run from the repository root:

```bash
conda env create -f environment.yml
conda activate llm_psychometric
PYTHONPATH=src python scripts/analysis/compute_human_bootstrap.py
```

The 2,000-resample human-matrix check yields mean stability `r = 0.9744` with 95% interval `[0.9572, 0.9851]` from the released CSVs, matching the paper's rounded values. The computation uses no model weights.

For model runs, analysis order, and the experiment-to-code map, see [REPRODUCE.md](REPRODUCE.md).
