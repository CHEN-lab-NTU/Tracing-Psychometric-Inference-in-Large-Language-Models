"""Read item-level probe and behavioral alignment outputs for the nine models."""

from pathlib import Path

import numpy as np
import pandas as pd

from psychometric_inference.mechanisms.config import ALL_SUBSCALES, SCALE_NAMES


# Layers selected by scripts/analysis/compute_itemlevel_comparison.py.
ITEMLEVEL_LAYERS = (
    ("qwen05b_instruct", 8),
    ("llama1b_instruct", 8),
    ("qwen3b_instruct", 14),
    ("llama3b_instruct", 14),
    ("qwen7b_instruct", 18),
    ("llama8b_instruct", 16),
    ("llama8b_base", 14),
    ("qwen14b_instruct", 32),
    ("qwen14b_base", 28),
)


def load_itemlevel_pairs(mechanistic_root: Path, model_name: str,
                         human_corr: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return human and predicted upper-triangle correlations in canonical order."""
    path = (mechanistic_root / f"results_{model_name}" / "cross_persona"
            / "itemlevel_pred_corr_matrix.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing item-level probe matrix: {path}. "
            "Run scripts/analysis/compute_itemlevel_comparison.py for this model."
        )

    predicted = pd.read_csv(path, index_col=0)
    labels = ALL_SUBSCALES
    n = len(labels)
    if (predicted.shape != (n, n)
            or set(predicted.index) != set(labels)
            or set(predicted.columns) != set(labels)):
        raise ValueError(f"Expected a complete {n}-by-{n} item-level matrix: {path}")
    if (human_corr.shape != (n, n)
            or set(human_corr.index) != set(labels)
            or set(human_corr.columns) != set(labels)):
        raise ValueError("Expected a complete human subscale correlation matrix")

    try:
        pred_values = predicted.loc[labels, labels].to_numpy(dtype=float)
        human_values = human_corr.loc[labels, labels].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric correlation values in {path} or human matrix") from exc
    if not np.isfinite(pred_values).all() or not np.isfinite(human_values).all():
        raise ValueError(f"Missing or non-finite correlations in {path} or human matrix")

    upper = np.triu_indices(n, k=1)
    return human_values[upper], pred_values[upper]


def load_itemlevel_slope(mechanistic_root: Path, model_name: str,
                         human_corr: pd.DataFrame) -> float:
    """OLS slope of predicted versus human subscale correlations."""
    human, predicted = load_itemlevel_pairs(mechanistic_root, model_name, human_corr)
    return float(np.polyfit(human, predicted, 1)[0])


def load_behavioral_slope(behavior_root: Path, behavior_dir: str) -> float:
    """Mean of the seven per-scale subscale slopes used in Figure 1."""
    path = behavior_root / behavior_dir / "per_scale_alignment.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing behavioral alignment summary: {path}")
    summary = pd.read_csv(path)
    if ("scale" not in summary or "subscale_slope" not in summary
            or len(summary) != len(SCALE_NAMES)
            or set(summary["scale"]) != set(SCALE_NAMES)):
        raise ValueError(f"Expected one subscale slope per scale in {path}")
    slopes = pd.to_numeric(summary["subscale_slope"], errors="coerce").to_numpy()
    if not np.isfinite(slopes).all():
        raise ValueError(f"Missing or non-finite subscale slopes in {path}")
    return float(slopes.mean())
