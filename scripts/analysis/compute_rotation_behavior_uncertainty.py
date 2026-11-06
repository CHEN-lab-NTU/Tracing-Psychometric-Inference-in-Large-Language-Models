#!/usr/bin/env python3
"""Bootstrap the seven-rotation behavioral statistics reported in the paper.

The headline statistic is the unweighted mean of seven *between-scale*,
per-persona-scale OLS slopes (or Pearson alignments). Resample participants,
retaining repeated draws, and apply the same draw to human and model answers.
This differs from bootstrapping pairs of entries in the implicit matrix.

Examples:
    PYTHONPATH=src python scripts/analysis/compute_rotation_behavior_uncertainty.py
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from psychometric_inference.model_registry import PRIMARY_MODELS
from psychometric_inference.paths import (
    BEHAVIOR_OUTPUT_DIR,
    HUMAN_DATA_DIR,
    LLM_BEHAVIOR_DIR,
    STATISTICS_OUTPUT_DIR,
)
from psychometric_inference.scoring import (
    ALL_SUBSCALES,
    FILENAME_TO_SCALE,
    SCORING_RULES,
    compute_subscale_scores,
    un_reverse_df,
)


SCALES = tuple(FILENAME_TO_SCALE.items())
SCALE_NAMES = tuple(short for _, short in SCALES)
SUBSCALES = tuple(ALL_SUBSCALES)
SUBSCALE_INDEX = {name: i for i, name in enumerate(SUBSCALES)}
SUB_TO_SCALE = {
    f"{scale}_{subscale}": scale
    for scale, rules in SCORING_RULES.items()
    for subscale in rules["subscales"]
}


@dataclass(frozen=True)
class Rotation:
    scale: str
    persona: np.ndarray
    answers: np.ndarray
    source_indices: tuple[int, ...]
    other_indices: tuple[int, ...]
    answer_indices: tuple[int, ...]


def _id_column(frame: pd.DataFrame) -> str:
    for candidate in ("Subject_ID", "ID", "Scan_ID", "id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError("No participant ID column found")


def _scored_scale(path: Path, scale_file: str, scale: str, human: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    id_col = _id_column(frame)
    item_cols = [column for column in frame if column.startswith("Q")]
    if not item_cols:
        raise ValueError(f"No item columns in {path}")
    items = frame[item_cols].copy()
    if human:
        items = un_reverse_df(items, scale_file)
    items.rename(columns={column: f"{scale}_{column}" for column in item_cols}, inplace=True)
    labels = [(scale, column) for column in items.columns]
    scores = compute_subscale_scores(items, labels)
    scores.insert(0, "Subject_ID", frame[id_col].astype(str).to_numpy())
    if scores["Subject_ID"].duplicated().any():
        raise ValueError(f"Duplicate participant IDs in {path}")
    return scores


def _human_scores(human_root: Path) -> pd.DataFrame:
    by_scale = []
    for scale_file, scale in SCALES:
        by_scale.append(_scored_scale(human_root / f"{scale_file}.csv", scale_file, scale, True))
    merged = by_scale[0]
    for frame in by_scale[1:]:
        merged = merged.merge(frame, on="Subject_ID", how="inner", validate="one_to_one")
    if set(merged.columns) != {"Subject_ID", *SUBSCALES}:
        raise ValueError("The human data do not produce all 16 expected subscales")
    return merged


def _rotation(
    llm_root: Path,
    scale_file: str,
    scale: str,
    subject_ids: pd.Index,
) -> Rotation:
    directory = llm_root / f"persona_{scale_file}"
    persona_path = directory / f"{scale_file}_persona.csv"
    if not persona_path.exists():
        persona_path = directory / f"{scale_file}.csv"
    persona = _scored_scale(persona_path, scale_file, scale, False)

    answer_frames = [
        _scored_scale(directory / f"{target_file}.csv", target_file, target_scale, False)
        for target_file, target_scale in SCALES
    ]
    answers = answer_frames[0]
    for frame in answer_frames[1:]:
        answers = answers.merge(frame, on="Subject_ID", how="inner", validate="one_to_one")

    persona = persona.set_index("Subject_ID").reindex(subject_ids)
    answers = answers.set_index("Subject_ID").reindex(subject_ids)
    if persona.isna().any().any() or answers.isna().any().any():
        raise ValueError(f"Missing scored answers in {directory}; full paired bootstrap needs complete data")

    source_names = [name for name in SUBSCALES if SUB_TO_SCALE[name] == scale]
    answer_names = [name for name in SUBSCALES if name in answers.columns]
    other_names = [name for name in SUBSCALES if SUB_TO_SCALE[name] != scale]
    if not set(source_names).issubset(persona.columns) or not set(other_names).issubset(answer_names):
        raise ValueError(f"Missing subscales in {directory}")
    return Rotation(
        scale=scale,
        persona=persona[source_names].to_numpy(float),
        answers=answers[answer_names].to_numpy(float),
        source_indices=tuple(SUBSCALE_INDEX[name] for name in source_names),
        other_indices=tuple(SUBSCALE_INDEX[name] for name in other_names),
        answer_indices=tuple(answer_names.index(name) for name in other_names),
    )


def _slope_and_alignment(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan, np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0]), float(np.corrcoef(x[valid], y[valid])[0, 1])


def _per_rotation(
    human: np.ndarray,
    rotations: list[Rotation],
    indices: np.ndarray,
    human_corr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        if human_corr is None:
            human_corr = np.corrcoef(human[indices], rowvar=False)
        # np.corrcoef can return spurious near-zero numbers for a constant
        # floating column; the original scipy Pearson analysis returns NaN.
        human_constant = np.ptp(human[indices], axis=0) == 0
        if human_constant.any():
            human_corr = human_corr.copy()
            human_corr[human_constant, :] = np.nan
            human_corr[:, human_constant] = np.nan
        slopes = []
        alignments = []
        for rotation in rotations:
            p = rotation.persona[indices]
            a = rotation.answers[indices]
            joined = np.column_stack((p, a))
            corr = np.corrcoef(joined, rowvar=False)
            constant = np.ptp(joined, axis=0) == 0
            if constant.any():
                corr[constant, :] = np.nan
                corr[:, constant] = np.nan
            observed = corr[: p.shape[1], p.shape[1] :][:, rotation.answer_indices]
            reference = human_corr[np.ix_(rotation.source_indices, rotation.other_indices)]
            slope, alignment = _slope_and_alignment(reference.ravel(), observed.ravel())
            slopes.append(slope)
            alignments.append(alignment)
    return np.asarray(slopes), np.asarray(alignments)


def _verify_saved_points(
    reference_path: Path,
    rotations: list[Rotation],
    point_slopes: np.ndarray,
    point_alignments: np.ndarray,
) -> None:
    if not reference_path.is_file():
        raise FileNotFoundError(f"Point-estimate verification needs {reference_path}")
    reference = pd.read_csv(reference_path).set_index("scale")
    for i, rotation in enumerate(rotations):
        expected_slope = reference.loc[rotation.scale, "subscale_slope"]
        expected_alignment = reference.loc[rotation.scale, "subscale_r"]
        if not np.isclose(point_slopes[i], expected_slope, equal_nan=True, atol=1e-10):
            raise ValueError(f"{rotation.scale} slope differs from {reference_path}")
        if not np.isclose(point_alignments[i], expected_alignment, equal_nan=True, atol=1e-10):
            raise ValueError(f"{rotation.scale} alignment differs from {reference_path}")


def _row(
    model: str,
    behavior_dir: str,
    statistic: str,
    point_by_rotation: np.ndarray,
    draws: np.ndarray,
    loo_subject: np.ndarray,
    n_subjects: int,
    seed: int,
) -> dict:
    draws = draws[np.isfinite(draws)]
    loo_subject = loo_subject[np.isfinite(loo_subject)]
    if len(draws) == 0 or len(loo_subject) == 0:
        raise ValueError(f"No valid bootstrap or leave-one-subject-out estimates for {model} {statistic}")
    point = float(np.nanmean(point_by_rotation))
    loo_rotation = np.asarray(
        [np.nanmean(np.delete(point_by_rotation, i)) for i in range(len(point_by_rotation))]
    )
    return {
        "model": model,
        "behavior_dir": behavior_dir,
        "statistic": statistic,
        "point": point,
        "bootstrap_mean": float(np.mean(draws)),
        "bootstrap_ci_lo": float(np.percentile(draws, 2.5)),
        "bootstrap_ci_hi": float(np.percentile(draws, 97.5)),
        "bootstrap_n": len(draws),
        "loo_subject_min": float(np.min(loo_subject)),
        "loo_subject_max": float(np.max(loo_subject)),
        "loo_rotation_min": float(np.nanmin(loo_rotation)),
        "loo_rotation_max": float(np.nanmax(loo_rotation)),
        "n_subjects": n_subjects,
        "n_rotations": len(point_by_rotation),
        "seed": seed,
        "bootstrap_unit": "participant",
        "estimator": "mean of seven between-scale per-rotation statistics",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STATISTICS_OUTPUT_DIR / "rotation_behavior_uncertainty.csv")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", help="Optional mechanistic model names; default is all 14 primary variants")
    args = parser.parse_args()
    if args.n_boot <= 0:
        parser.error("--n-boot must be positive")

    human_root = HUMAN_DATA_DIR
    llm_root = LLM_BEHAVIOR_DIR
    reference_root = BEHAVIOR_OUTPUT_DIR

    models = [m for m in PRIMARY_MODELS if args.models is None or m.mech_name in args.models]
    if not models or (args.models and {m.mech_name for m in models} != set(args.models)):
        parser.error("--models must name primary variants from the model registry")

    human_frame = _human_scores(human_root)
    subject_ids = pd.Index(human_frame["Subject_ID"].astype(str), name="Subject_ID")
    human = human_frame[list(SUBSCALES)].to_numpy(float)
    if not np.isfinite(human).all():
        raise ValueError("The human subscale scores contain missing or nonfinite values")
    n_subjects = len(subject_ids)
    if n_subjects < 20 or not subject_ids.is_unique:
        raise ValueError("Insufficient or repeated participant IDs in human data")

    rng = np.random.default_rng(args.seed)
    sample_indices = rng.integers(0, n_subjects, size=(args.n_boot, n_subjects))
    with np.errstate(divide="ignore", invalid="ignore"):
        human_corrs = [np.corrcoef(human[idx], rowvar=False) for idx in sample_indices]
    full_indices = np.arange(n_subjects)
    rows = []

    for model in models:
        model_root = llm_root / model.behavior_dir
        rotations = [_rotation(model_root, file_name, short, subject_ids) for file_name, short in SCALES]
        point_slopes, point_alignments = _per_rotation(human, rotations, full_indices)
        _verify_saved_points(
            reference_root / model.behavior_dir / "per_scale_alignment.csv",
            rotations,
            point_slopes,
            point_alignments,
        )

        boot_slopes = np.empty(args.n_boot)
        boot_alignments = np.empty(args.n_boot)
        for i, idx in enumerate(sample_indices):
            slopes, alignments = _per_rotation(human, rotations, idx, human_corrs[i])
            if not np.isfinite(slopes).all() or not np.isfinite(alignments).all():
                raise ValueError(f"{model.mech_name}: undefined seven-rotation statistic in bootstrap draw {i}")
            boot_slopes[i] = np.nanmean(slopes)
            boot_alignments[i] = np.nanmean(alignments)

        loo_slopes = np.empty(n_subjects)
        loo_alignments = np.empty(n_subjects)
        for i in range(n_subjects):
            idx = np.delete(full_indices, i)
            slopes, alignments = _per_rotation(human, rotations, idx)
            if not np.isfinite(slopes).all() or not np.isfinite(alignments).all():
                raise ValueError(f"{model.mech_name}: undefined seven-rotation statistic after omitting participant {i}")
            loo_slopes[i] = np.nanmean(slopes)
            loo_alignments[i] = np.nanmean(alignments)

        rows.append(_row(model.mech_name, model.behavior_dir, "subscale_between_slope", point_slopes, boot_slopes, loo_slopes, n_subjects, args.seed))
        rows.append(_row(model.mech_name, model.behavior_dir, "subscale_between_alignment_r", point_alignments, boot_alignments, loo_alignments, n_subjects, args.seed))
        print(f"{model.mech_name}: slope={rows[-2]['point']:.6f}, alignment={rows[-1]['point']:.6f}")

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} rows to {args.output}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        main()
