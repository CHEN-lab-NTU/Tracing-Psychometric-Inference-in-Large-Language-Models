#!/usr/bin/env python3
"""Compare directed activation-probe and behavioral scale correlations.

This uses the direct subscale probe, separate from the item-level Figure 3
analysis. Each source scale has its own persona rotation. The comparison uses
the upper 21 entries of the two directed 7-by-7 matrices, in SCALE_NAMES order,
as in compute_behavior_structure.compare_upper_tri.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from psychometric_inference.mechanisms.config import ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE
from psychometric_inference.mechanisms.cross_persona import SCALES, load_human_scores
from psychometric_inference.model_registry import MODEL_BY_MECH_NAME
from psychometric_inference.paths import (
    BEHAVIOR_OUTPUT_DIR, LLM_BEHAVIOR_DIR, MECHANISTIC_OUTPUT_DIR, STATISTICS_OUTPUT_DIR,
)
from psychometric_inference.scoring import SCORING_RULES, compute_subscale_scores


ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


def load_rotation(activation_root: Path, persona: str, layer: int) -> tuple[np.ndarray, pd.DataFrame]:
    rotation_dir = activation_root / f"rotation_{persona}"
    metadata_path = rotation_dir / "meta.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing rotation metadata: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    required = {"subject_id", "target_scale", "item_number"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"{metadata_path} needs columns {sorted(required)}")
    metadata["subject_id"] = metadata["subject_id"].astype(str)

    npz_path = rotation_dir / "activations.npz"
    npy_path = rotation_dir / f"L{layer}.npy"
    if npz_path.is_file():
        with np.load(npz_path) as archive:
            key = f"L{layer}"
            if key not in archive:
                raise ValueError(f"Missing {key} in {npz_path}")
            activations = archive[key]
    elif npy_path.is_file():
        activations = np.load(npy_path, mmap_mode="r")
    else:
        raise FileNotFoundError(f"Missing {npz_path} or {npy_path}")
    if len(activations) != len(metadata):
        raise ValueError(f"Activation and metadata row counts differ in {rotation_dir}")
    return activations, metadata


def check_target_items(metadata: pd.DataFrame, target: str, subject_ids: list[str]) -> pd.DataFrame:
    rows = metadata[metadata["target_scale"] == target]
    expected = sorted({q for items in SCORING_RULES[target]["subscales"].values() for q in items[:2]})
    observed = sorted(rows["item_number"].astype(int).unique().tolist())
    if observed != expected:
        raise ValueError(f"{target}: target items {observed}, expected {expected}")
    for item_num, group in rows.groupby("item_number"):
        ids = group["subject_id"].tolist()
        if len(ids) != len(subject_ids) or set(ids) != set(subject_ids):
            raise ValueError(f"{target} item {item_num}: expected one row for each human subject")
    return rows


def fit_target_subscales(
    activation_root: Path, layer: int, human_scores: pd.DataFrame
) -> dict[str, dict[str, np.ndarray]]:
    subject_ids = list(human_scores.index.astype(str))
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for _, persona in SCALES:
        activations, metadata = load_rotation(activation_root, persona, layer)
        predictions[persona] = {}
        for target in SCALE_NAMES:
            if target == persona:
                continue
            rows = check_target_items(metadata, target, subject_ids)
            by_subject = {sid: group.index.to_numpy() for sid, group in rows.groupby("subject_id", sort=False)}
            x = np.stack([activations[by_subject[sid]].mean(axis=0) for sid in subject_ids])
            for subscale in ALL_SUBSCALES:
                if SUB_TO_SCALE[subscale] != target:
                    continue
                y = human_scores.loc[subject_ids, subscale].to_numpy(dtype=float)
                if not np.isfinite(y).all():
                    raise ValueError(f"Missing target scores for {subscale}")
                ridge = RidgeCV(alphas=ALPHAS, cv=5)
                predictions[persona][subscale] = cross_val_predict(ridge, x, y, cv=5)
        print(f"{persona}: fitted {len(predictions[persona])} target subscales", flush=True)
    n_fits = sum(len(group) for group in predictions.values())
    if n_fits != 96:
        raise ValueError(f"Expected 96 persona-to-target-subscale fits, got {n_fits}")
    return predictions


def load_persona_scores(behavior_root: Path, subject_ids: list[str]) -> pd.DataFrame:
    scores = pd.DataFrame(index=subject_ids, columns=ALL_SUBSCALES, dtype=float)
    for filename, scale in SCALES:
        path = behavior_root / f"persona_{filename}" / f"{filename}_persona.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing persona responses: {path}")
        responses = pd.read_csv(path)
        if "Subject_ID" not in responses:
            raise ValueError(f"Missing Subject_ID in {path}")
        ids = responses["Subject_ID"].astype(str).tolist()
        if len(ids) != len(subject_ids) or set(ids) != set(subject_ids):
            raise ValueError(f"Persona subjects do not match human subjects in {path}")
        item_columns = [c for c in responses if c.startswith("Q")]
        items = responses[item_columns].rename(columns={c: f"{scale}_{c}" for c in item_columns})
        scored = compute_subscale_scores(items, [(scale, c) for c in items.columns])
        scored.index = ids
        for subscale in scored:
            scores[subscale] = scored[subscale].reindex(subject_ids)
    if scores.isna().any().any():
        raise ValueError("Persona source scores are incomplete")
    return scores


def directed_scale_matrix(
    source_scores: pd.DataFrame, predictions: dict[str, dict[str, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subscale_matrix = pd.DataFrame(np.nan, index=ALL_SUBSCALES, columns=ALL_SUBSCALES)
    for source in ALL_SUBSCALES:
        persona = SUB_TO_SCALE[source]
        x = source_scores[source].to_numpy(dtype=float)
        for target, y in predictions[persona].items():
            subscale_matrix.loc[source, target] = pearsonr(x, y).statistic
    scale_matrix = pd.DataFrame(np.nan, index=SCALE_NAMES, columns=SCALE_NAMES)
    for source_scale in SCALE_NAMES:
        source_subs = [s for s in ALL_SUBSCALES if SUB_TO_SCALE[s] == source_scale]
        for target_scale in SCALE_NAMES:
            if source_scale == target_scale:
                continue
            target_subs = [s for s in ALL_SUBSCALES if SUB_TO_SCALE[s] == target_scale]
            scale_matrix.loc[source_scale, target_scale] = np.nanmean(
                subscale_matrix.loc[source_subs, target_subs].to_numpy()
            )
    return subscale_matrix, scale_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_BY_MECH_NAME))
    parser.add_argument("--layer", type=int, help="Default: the model registry's cross-persona layer")
    parser.add_argument("--activation-root", type=Path)
    parser.add_argument("--behavior-root", type=Path)
    parser.add_argument("--behavior-matrix", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model = MODEL_BY_MECH_NAME[args.model]
    layer = args.layer if args.layer is not None else model.default_layer
    activation_root = args.activation_root or MECHANISTIC_OUTPUT_DIR / f"results_{model.mech_name}" / "cross_persona"
    behavior_root = args.behavior_root or LLM_BEHAVIOR_DIR / model.behavior_dir
    behavior_path = args.behavior_matrix or BEHAVIOR_OUTPUT_DIR / model.behavior_dir / "scale_llm_implicit.csv"
    output_dir = args.output_dir or STATISTICS_OUTPUT_DIR / "directional_probe_alignment" / model.mech_name

    if not behavior_path.is_file():
        raise FileNotFoundError(
            f"Missing behavioral scale matrix: {behavior_path}. "
            "Run scripts/analysis/compute_behavior_structure.py for this model first."
        )
    human_scores = load_human_scores()
    if len(human_scores) != 272 or len(set(human_scores.index.astype(str))) != 272:
        raise ValueError("Expected 272 distinct human subjects")
    subject_ids = list(human_scores.index.astype(str))
    predictions = fit_target_subscales(activation_root, layer, human_scores)
    persona_scores = load_persona_scores(behavior_root, subject_ids)
    subscale_matrix, activation_matrix = directed_scale_matrix(persona_scores, predictions)
    behavior_matrix = pd.read_csv(behavior_path, index_col=0).loc[SCALE_NAMES, SCALE_NAMES]

    rows = []
    for i, source_scale in enumerate(SCALE_NAMES):
        for target_scale in SCALE_NAMES[i + 1:]:
            rows.append({
                "source_scale": source_scale,
                "target_scale": target_scale,
                "activation_r": activation_matrix.loc[source_scale, target_scale],
                "behavior_r": behavior_matrix.loc[source_scale, target_scale],
            })
    pairs = pd.DataFrame(rows)
    if len(pairs) != 21 or not np.isfinite(pairs[["activation_r", "behavior_r"]].to_numpy()).all():
        raise ValueError("Expected 21 finite directed scale pairs")
    result = pearsonr(pairs["activation_r"], pairs["behavior_r"])

    output_dir.mkdir(parents=True, exist_ok=True)
    subscale_matrix.to_csv(output_dir / "activation_subscale_directed.csv")
    activation_matrix.to_csv(output_dir / "activation_scale_directed.csv")
    pairs.to_csv(output_dir / "directional_pairs.csv", index=False)
    summary = {
        "model": model.mech_name,
        "layer": layer,
        "n_subjects": len(subject_ids),
        "n_probe_fits": 96,
        "n_scale_pairs": len(pairs),
        "pearson_r": float(result.statistic),
        "pearson_p": float(result.pvalue),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
