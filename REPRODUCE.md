# Reproducing the paper analyses

Run commands from the repository root after creating the environment in [README.md](README.md). The 14 primary models' cross-persona response CSVs are bundled; activations are not. A new model run needs access to the listed checkpoints and sufficient GPU memory. Set `MPLBACKEND=Agg` for runs without a graphical display.

## Run order

1. **Check the reported human bootstrap.** This is the 2,000-resample analysis reported in the paper.

   ```bash
   PYTHONPATH=src python scripts/analysis/compute_human_bootstrap.py
   ```

2. **Analyze the included cross-persona responses.** The CSVs under `data/llm_behavior/` cover all seven persona-scale rotations for each of the 14 primary model variants in [`model_registry.py`](src/psychometric_inference/model_registry.py). For example:

   ```bash
   PYTHONPATH=src python scripts/analysis/compute_behavior_structure.py \
     --llm_root data/llm_behavior/llama8b_instruct \
     --output_dir outputs/behavior/llama8b_instruct
   PYTHONPATH=src python scripts/analysis/compute_scale_structure.py \
     --results_dir outputs/behavior/llama8b_instruct
   ```

   Run both analyses for each model. The second command produces `per_scale_alignment.csv`, which the cross-model analysis uses. Analysis outputs are written under `outputs/behavior/`.

   To regenerate responses from model weights, use a separate checkout so the included CSVs are not overwritten. Each command runs one model through all seven rotations; omit `--model` to run every variant of that type. Use `--max_subjects` for a small trial.

   ```bash
   PYTHONPATH=src python scripts/behavior/generate_base_behavior.py --model qwen05b_base --device cuda
   PYTHONPATH=src python scripts/behavior/generate_instruct_behavior.py --model qwen05b_instruct --device cuda
   ```

   After the matrices for all 14 variants are complete, run the seven-scale attenuation correction and the subscale reliability-cut analysis used in the paper:

   ```bash
   PYTHONPATH=src python scripts/analysis/compute_attenuation_correction.py
   MPLBACKEND=Agg PYTHONPATH=src python scripts/analysis/compute_subscale_attenuation.py
   ```

   These scripts use the raw-text instruct runs in the model registry. Check that `attenuation_correction.csv` has 14 model rows and `attenuation_correction_subscale_sensitivity.csv` has 56 rows (14 models at four reliability cuts) before comparing the reported ranges.

   The prompt and readout controls use the same 50 participants for each Qwen 14B Instruct condition:

   ```bash
   PYTHONPATH=src python scripts/behavior/run_prompt_sensitivity.py \
     --models qwen14b_instruct --variants standard_prompt no_scale_name chat_template \
     --readouts argmax expected_value --max_subjects 50 --seed 42
   ```

3. **Run contrastive geometry and its model-level analysis.** The pipeline's `--models` option selects a subset. Omit `--skip_steering` if steering is required in the same run.

   ```bash
   PYTHONPATH=src python -m psychometric_inference.mechanisms.pipeline --skip_steering
   PYTHONPATH=src python scripts/figures/plot_mechanistic_scaling.py
   PYTHONPATH=src python -m psychometric_inference.mechanisms.representation_behavior
   ```

   The first command writes model results under `outputs/mechanistic/results_{model}/`. The next two require completed results for the reported 14-model comparison. Review `outputs/mechanistic/scaling_results_full.csv` and `structure_vs_behavior.csv` for one row per variant before using cross-model statistics.

   The Llama 8B Instruct layer-16 steering analyses use two direction sources. After extracting directions and subject activations, run regression-direction and reliability-filtered contrastive-direction steering separately:

   ```bash
   PYTHONPATH=src python -m psychometric_inference.mechanisms.steering --results_dir outputs/mechanistic/results_llama8b_instruct --extract_only --layer 16
   PYTHONPATH=src python -m psychometric_inference.mechanisms.reliability --results_dir outputs/mechanistic/results_llama8b_instruct --layer 16
   PYTHONPATH=src python -m psychometric_inference.mechanisms.steering --results_dir outputs/mechanistic/results_llama8b_instruct --model_id meta-llama/Llama-3.1-8B-Instruct --steering --layer 16
   PYTHONPATH=src python -m psychometric_inference.mechanisms.steering --results_dir outputs/mechanistic/results_llama8b_instruct --model_id meta-llama/Llama-3.1-8B-Instruct --direction_source contrastive --min_source_reliability 0.3 --steering --layer 16
   ```

4. **Extract decision-point activations and run the item-level probe.** Repeat extraction for the nine models and their selected layers in [`compute_itemlevel_comparison.py`](scripts/analysis/compute_itemlevel_comparison.py); the example uses Llama 8B Instruct.

   ```bash
   PYTHONPATH=src python -m psychometric_inference.mechanisms.cross_persona --model_id meta-llama/Llama-3.1-8B-Instruct --llm_root data/llm_behavior/llama8b_instruct
   PYTHONPATH=src python scripts/analysis/compute_itemlevel_comparison.py --models llama8b_instruct
   ```

   After all nine extraction sets are complete, use `compute_itemlevel_comparison.py --all`. The item-level script requires the first two items per target subscale for all seven rotations and all 272 subjects; it stops instead of saving a partial matrix. Figure 4 likewise requires its item-level matrix and does not substitute the direct subscale-probe matrix. The direct subscale-target probe in `cross_persona.py` and this item-level probe are separate analyses. The cross-persona command also runs the direct probe and its 100 subject-label permutations; `--analyze_only --n_perms 100` repeats these analyses from saved activations.

   The appendix's directed activation–behavior comparison uses the direct subscale probe at the registry's default layer, not the pooled item-level Figure 4 matrix. Once both the cross-persona activations and the model's `scale_llm_implicit.csv` exist, run:

   ```bash
   PYTHONPATH=src python scripts/analysis/compute_directional_probe_alignment.py --model llama8b_instruct
   PYTHONPATH=src python scripts/analysis/compute_directional_probe_alignment.py --model qwen14b_base
   ```

   Each run requires all seven rotations and 272 subjects. It saves the directed activation matrices, the 21 scale pairs, and Pearson `r`/`p` under `outputs/statistics/directional_probe_alignment/`. The direction-difference diagnostic printed by `cross_persona.py` measures target-prediction accuracy, a different quantity.

5. **Run robustness analyses and render figures.** `A1_adj` and Figure 4 read the nine item-level correlation matrices and behavioral alignment summaries generated in earlier steps. The paper's item-level cultural and context controls are in [`run_targeted_checks.py`](scripts/analysis/run_targeted_checks.py).

   ```bash
   PYTHONPATH=src python scripts/analysis/run_targeted_checks.py
   PYTHONPATH=src python scripts/analysis/compute_robustness_suite.py --only A1_minimal A1_full
   PYTHONPATH=src python scripts/analysis/compute_robustness_suite.py --only A1_adj
   PYTHONPATH=src python scripts/figures/make_main_figures.py --outdir reports/figures
   ```

6. **Compute uncertainty intervals after the required matrices exist.**

   ```bash
   PYTHONPATH=src python scripts/analysis/compute_uncertainty_intervals.py
   PYTHONPATH=src python scripts/analysis/compute_rotation_behavior_uncertainty.py --models qwen14b_instruct
   ```

   The first script produces matrix-pair intervals and 14-variant association intervals. The second bootstraps the seven-rotation Qwen 14B Instruct behavioral statistics reported in the paper. It requires complete participant-level responses and the matching `outputs/behavior/qwen14b_instruct/per_scale_alignment.csv` from step 2. Omit `--models qwen14b_instruct` to run this participant bootstrap for all 14 variants. The two scripts estimate different quantities.

## Experiment-to-code map

| Paper analysis | Entry point |
| --- | --- |
| Human 16-subscale bootstrap | [`compute_human_bootstrap.py`](scripts/analysis/compute_human_bootstrap.py). |
| Fourteen-variant behavior and Figure 2 | [`generate_base_behavior.py`](scripts/behavior/generate_base_behavior.py), [`generate_instruct_behavior.py`](scripts/behavior/generate_instruct_behavior.py), [`compute_behavior_structure.py`](scripts/analysis/compute_behavior_structure.py), [`compute_scale_structure.py`](scripts/analysis/compute_scale_structure.py). |
| Attenuation correction | [`compute_attenuation_correction.py`](scripts/analysis/compute_attenuation_correction.py), [`compute_subscale_attenuation.py`](scripts/analysis/compute_subscale_attenuation.py). |
| Prompt/readout variants | [`run_prompt_sensitivity.py`](scripts/behavior/run_prompt_sensitivity.py); geometry prompt variants: [`prompt_geometry.py`](src/psychometric_inference/mechanisms/prompt_geometry.py). |
| Contrastive directions, geometry, reliability, Figure 3 | [`pipeline.py`](src/psychometric_inference/mechanisms/pipeline.py), [`directions.py`](src/psychometric_inference/mechanisms/directions.py), [`geometry.py`](src/psychometric_inference/mechanisms/geometry.py), [`reliability.py`](src/psychometric_inference/mechanisms/reliability.py), [`plot_mechanistic_scaling.py`](scripts/figures/plot_mechanistic_scaling.py). |
| Factorial Big Five geometry | [`factorial_geometry.py`](src/psychometric_inference/mechanisms/factorial_geometry.py). |
| Regression and contrastive steering | [`steering.py`](src/psychometric_inference/mechanisms/steering.py); `--direction_source contrastive --min_source_reliability 0.3` selects contrastive sources meeting the reliability threshold. |
| Direct subscale-target activation probe | [`cross_persona.py`](src/psychometric_inference/mechanisms/cross_persona.py). |
| Directed activation–behavior scale comparison | [`compute_directional_probe_alignment.py`](scripts/analysis/compute_directional_probe_alignment.py); requires cross-persona activations and the behavioral `scale_llm_implicit.csv`. |
| Nine-model item-level activation probe and Figure 4 | [`cross_persona.py`](src/psychometric_inference/mechanisms/cross_persona.py) for activations; [`compute_itemlevel_comparison.py`](scripts/analysis/compute_itemlevel_comparison.py) for item-level predictions. |
| Layer, ridge, shuffle, and rounded-response checks | [`compute_additional_controls.py`](scripts/analysis/compute_additional_controls.py), [`compute_robustness_suite.py`](scripts/analysis/compute_robustness_suite.py) (`B2`), [`ridge_baseline.py`](src/psychometric_inference/mechanisms/ridge_baseline.py). |
| Fourteen-model geometry–behavior association and Figure 6 | [`representation_behavior.py`](src/psychometric_inference/mechanisms/representation_behavior.py); [`compute_robustness_suite.py`](scripts/analysis/compute_robustness_suite.py) (`C1`, `C2`, `F1`, `F1b`, `F2`). |
| Synthetic control and Figure 5 | [`compute_robustness_suite.py`](scripts/analysis/compute_robustness_suite.py) (`A1_minimal`, `A1_full`, `A1_adj`). |
| Item-level cultural and in-prompt context controls | [`run_targeted_checks.py`](scripts/analysis/run_targeted_checks.py) (`D2`, `D2b`, `F3b`). |
| Item-wording semantic controls | [`semantic_controls.py`](src/psychometric_inference/mechanisms/semantic_controls.py). |
| Human split-half and intercept analyses | [`compute_supplementary_behavior.py`](scripts/analysis/compute_supplementary_behavior.py), [`refresh_supplementary_figures.py`](scripts/figures/refresh_supplementary_figures.py). |
| Behavioral, geometry, and association uncertainty | [`compute_uncertainty_intervals.py`](scripts/analysis/compute_uncertainty_intervals.py) for matrix-pair and model-variant intervals; [`compute_rotation_behavior_uncertainty.py`](scripts/analysis/compute_rotation_behavior_uncertainty.py) for the seven-rotation behavioral interval. |

The repository includes the primary model-response CSVs but not activation files or generated result matrices. The direct subscale-target probe uses the first two items per target subscale, yielding 96 cross-validated fits against scored human responses.
