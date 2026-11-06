#!/usr/bin/env python3
"""
Apples-to-apples comparison: predict item-level responses from activations,
then average to subscale scores, mirroring behavioral pipeline exactly.

Behavioral pipeline: argmax(logprob) per item → mean = subscale score
This pipeline: ridge_predict per item (continuous) → mean = subscale score

Usage:
    PYTHONPATH=src python scripts/analysis/compute_itemlevel_comparison.py --all
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

from psychometric_inference.paths import HUMAN_DATA_DIR, MECHANISTIC_OUTPUT_DIR
MECH_ROOT = MECHANISTIC_OUTPUT_DIR

from psychometric_inference.scoring import SCORING_RULES, un_reverse_df
from psychometric_inference.mechanisms.config import ALL_SUBSCALES, SCALE_NAMES, SUB_TO_SCALE
from psychometric_inference.mechanisms.cross_persona import load_human_scores, SCALES
from psychometric_inference.mechanisms.geometry import compute_human_correlation_matrix, mantel_test


def main():
    print("="*60)
    print("  ITEM-LEVEL RIDGE → SUBSCALE SCORE (Option 1)")
    print("="*60)

    scores_df = load_human_scores()
    subject_ids = list(scores_df.index)
    human_corr = compute_human_correlation_matrix("subscale")

    # Parse command line for model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama8b_instruct"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    all_models = {
        "llama1b_instruct": 8,
        "llama3b_instruct": 14,
        "llama8b_instruct": 16,
        "llama8b_base": 14,
        "qwen05b_instruct": 8,
        "qwen3b_instruct": 14,
        "qwen7b_instruct": 18,
        "qwen14b_instruct": 32,
        "qwen14b_base": 28,
    }

    if args.all:
        models_to_run = all_models
    else:
        unknown = set(args.models).difference(all_models)
        if unknown:
            parser.error(f"Unknown model(s): {', '.join(sorted(unknown))}")
        models_to_run = {m: all_models[m] for m in args.models}

    for model_name, default_layer in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"  {model_name} (layer {default_layer})")
        print(f"{'='*60}")

        cp_dir = MECH_ROOT / f"results_{model_name}" / "cross_persona"
        if not cp_dir.exists():
            raise FileNotFoundError(f"Missing cross-persona activations for {model_name}: {cp_dir}")

        layer = default_layer
        run_itemlevel(cp_dir, layer, scores_df, subject_ids, human_corr)


def run_itemlevel(cp_dir, layer, scores_df, subject_ids, human_corr):

    # Load raw human item-level data for targets
    human_items = {}  # {scale_file: DataFrame with Subject_ID and Q columns}
    for scale_file, scale_short in SCALES:
        csv_path = HUMAN_DATA_DIR / f"{scale_file}.csv"
        if csv_path.exists():
            combined = un_reverse_df(pd.read_csv(csv_path), scale_file)
            id_col = next((c for c in ["Subject_ID", "ID", "Scan_ID"] if c in combined.columns), None)
            if id_col:
                combined = combined.rename(columns={id_col: "Subject_ID"})
            else:
                combined["Subject_ID"] = [f"S{i:04d}" for i in range(len(combined))]
            combined["Subject_ID"] = combined["Subject_ID"].astype(str)
            human_items[scale_file] = combined

    # For each rotation, predict each TARGET ITEM (not subscale) from activation
    # Then average item predictions within subscale → subscale score
    # Then compute correlation matrix of these subscale scores

    # We need item-level human responses aligned with our subjects
    # First build {subject_id: {scale_file: {item_num: response}}}
    human_item_responses = {}
    for scale_file, scale_short in SCALES:
        if scale_file not in human_items:
            continue
        df = human_items[scale_file]
        for _, row in df.iterrows():
            sid = str(row["Subject_ID"])
            if sid not in human_item_responses:
                human_item_responses[sid] = {}
            if scale_file not in human_item_responses[sid]:
                human_item_responses[sid][scale_file] = {}
            for col in df.columns:
                if col.startswith("Q"):
                    try:
                        item_num = int(col[1:])
                        human_item_responses[sid][scale_file][item_num] = row[col]
                    except ValueError:
                        pass

    # For each rotation x target item, predict item response from activation
    item_predictions = {}  # {(persona_short, scale_file, item_num): {sid: predicted_value}}

    for persona_file, persona_short in SCALES:
        rotation_dir = cp_dir / f"rotation_{persona_short}"
        act_path = rotation_dir / "activations.npz"
        meta_path = rotation_dir / "meta.csv"
        if not act_path.exists():
            continue

        acts_data = np.load(act_path)
        meta_df = pd.read_csv(meta_path)
        meta_df["subject_id"] = meta_df["subject_id"].astype(str)
        layer_key = f"L{layer}"
        if layer_key not in acts_data:
            continue
        acts = acts_data[layer_key]

        # Group by (target_scale, item_number)
        for (target_scale, item_num), group in meta_df.groupby(["target_scale", "item_number"]):
            # Map target_scale back to scale_file
            target_file = None
            for sf, ss in SCALES:
                if ss == target_scale:
                    target_file = sf
                    break
            if target_file is None:
                continue

            # Get activations and true item responses for these subjects
            indices = group.index.values
            sids = group["subject_id"].values

            X_list, y_list, sid_list = [], [], []
            for idx_pos, sid in zip(indices, sids):
                if idx_pos >= len(acts):
                    continue
                if sid not in human_item_responses:
                    continue
                if target_file not in human_item_responses[sid]:
                    continue
                if item_num not in human_item_responses[sid][target_file]:
                    continue

                true_val = human_item_responses[sid][target_file][item_num]
                if pd.isna(true_val):
                    continue

                X_list.append(acts[idx_pos])
                y_list.append(float(true_val))
                sid_list.append(sid)

            if len(X_list) < 50:
                continue

            X = np.array(X_list)
            y = np.array(y_list)

            # Cross-validated ridge prediction
            ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
            try:
                y_pred = cross_val_predict(ridge, X, y, cv=5)
            except Exception:
                continue

            key = (persona_short, target_file, item_num)
            item_predictions[key] = {sid: yp for sid, yp in zip(sid_list, y_pred)}

    print(f"  Predicted {len(item_predictions)} (rotation, scale, item) combinations")

    # A partial extraction must not be saved as the paper's full item-level matrix.
    expected_keys = set()
    for _, persona_short in SCALES:
        for target_file, target_short in SCALES:
            if target_short == persona_short:
                continue
            selected_items = {q for items in SCORING_RULES[target_short]["subscales"].values()
                              for q in items[:2]}
            expected_keys.update((persona_short, target_file, q) for q in selected_items)
    missing = expected_keys.difference(item_predictions)
    unexpected = set(item_predictions).difference(expected_keys)
    incomplete = [key for key, scores in item_predictions.items()
                  if key in expected_keys and set(scores) != set(subject_ids)]
    if missing or unexpected or incomplete:
        raise ValueError(
            f"Incomplete item-level extraction at layer {layer}: "
            f"{len(missing)} missing and {len(unexpected)} unexpected rotation/item fits; "
            f"{len(incomplete)} fits without all {len(subject_ids)} subjects. "
            "Use the first two items per target subscale for all seven rotations."
        )

    # Now aggregate: for each subject, for each target subscale,
    # average the item-level predictions → subscale score
    # Mirror the behavioral pipeline exactly
    subscale_predictions = {}  # {subscale_name: {sid: mean_predicted_score}}

    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        rev_items = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)

        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            sid_scores = {}

            for sid in subject_ids:
                item_preds = []
                for item_num in item_nums:
                    # Find predictions for this item across rotations
                    preds_for_item = []
                    for persona_file, persona_short in SCALES:
                        if persona_short == scale_short:
                            continue  # skip same-scale
                        key = (persona_short, scale_file, item_num)
                        if key in item_predictions and sid in item_predictions[key]:
                            preds_for_item.append(item_predictions[key][sid])

                    if preds_for_item:
                        # Average across rotations for this item
                        pred_val = np.mean(preds_for_item)

                        # Apply reverse coding (same as behavioral pipeline)
                        if item_num in rev_items:
                            pred_val = (max_val + 1) - pred_val

                        item_preds.append(pred_val)

                if item_preds:
                    sid_scores[sid] = np.mean(item_preds)

            if sid_scores:
                subscale_predictions[sub_full] = sid_scores

    # Build predicted subscale score matrix (subjects x subscales)
    available = [s for s in ALL_SUBSCALES if s in subscale_predictions]
    print(f"  Available subscales: {len(available)}")

    pred_df = pd.DataFrame({sub: [subscale_predictions[sub].get(sid, np.nan)
                                    for sid in subject_ids]
                             for sub in available}, index=subject_ids)

    # Compute correlation matrix of item-averaged predicted scores
    n = len(available)
    pred_corr = np.zeros((n, n))
    for i, si in enumerate(available):
        for j, sj in enumerate(available):
            vi = pred_df[si].values
            vj = pred_df[sj].values
            v = ~(np.isnan(vi) | np.isnan(vj))
            if v.sum() > 10:
                pred_corr[i, j], _ = pearsonr(vi[v], vj[v])
            else:
                pred_corr[i, j] = np.nan

    pred_corr_df = pd.DataFrame(pred_corr, index=available, columns=available)

    # Save predicted correlation matrix
    out_csv = cp_dir / "itemlevel_pred_corr_matrix.csv"
    pred_corr_df.to_csv(out_csv)
    print(f"  Saved predicted correlation matrix to {out_csv}")

    # Compare with human
    common = [s for s in available if s in human_corr.index]
    nc = len(common)
    idx = np.triu_indices(nc, k=1)
    pv = pred_corr_df.loc[common, common].values[idx]
    hv = human_corr.loc[common, common].values[idx]
    valid = ~(np.isnan(pv) | np.isnan(hv))

    r_val, p_val = pearsonr(hv[valid], pv[valid])
    slope, intercept = np.polyfit(hv[valid], pv[valid], 1)
    mr, mp = mantel_test(pred_corr_df.loc[common, common].values,
                          human_corr.loc[common, common].values)

    print(f"\n  Item-level prediction → subscale mean (Option 1):")
    print(f"    Pearson r:  {r_val:.3f}")
    print(f"    Slope:      {slope:.3f}")
    print(f"    Mantel r:   {mr:.3f} (p = {mp:.4f})")

    # Also do Option 2: round item predictions to integer then average
    subscale_predictions_discrete = {}
    for scale_file, scale_short in SCALES:
        rules = SCORING_RULES.get(scale_short, {})
        rev_items = set(rules.get("reverse_items", []))
        max_val = rules.get("max_val", 5)
        lo = 1

        for sub_name, item_nums in rules.get("subscales", {}).items():
            sub_full = f"{scale_short}_{sub_name}"
            sid_scores = {}

            for sid in subject_ids:
                item_preds = []
                for item_num in item_nums:
                    preds_for_item = []
                    for persona_file, persona_short in SCALES:
                        if persona_short == scale_short:
                            continue
                        key = (persona_short, scale_file, item_num)
                        if key in item_predictions and sid in item_predictions[key]:
                            preds_for_item.append(item_predictions[key][sid])

                    if preds_for_item:
                        pred_val = np.mean(preds_for_item)
                        # Round to integer and clip (like argmax would produce)
                        pred_val = int(np.clip(np.round(pred_val), lo, max_val))
                        # Apply reverse coding
                        if item_num in rev_items:
                            pred_val = (max_val + 1) - pred_val
                        item_preds.append(pred_val)

                if item_preds:
                    sid_scores[sid] = np.mean(item_preds)

            if sid_scores:
                subscale_predictions_discrete[sub_full] = sid_scores

    available_d = [s for s in ALL_SUBSCALES if s in subscale_predictions_discrete]
    pred_df_d = pd.DataFrame({sub: [subscale_predictions_discrete[sub].get(sid, np.nan)
                                      for sid in subject_ids]
                               for sub in available_d}, index=subject_ids)

    n_d = len(available_d)
    pred_corr_d = np.zeros((n_d, n_d))
    for i, si in enumerate(available_d):
        for j, sj in enumerate(available_d):
            vi = pred_df_d[si].values
            vj = pred_df_d[sj].values
            v = ~(np.isnan(vi) | np.isnan(vj))
            if v.sum() > 10:
                pred_corr_d[i, j], _ = pearsonr(vi[v], vj[v])
            else:
                pred_corr_d[i, j] = np.nan

    pred_corr_d_df = pd.DataFrame(pred_corr_d, index=available_d, columns=available_d)
    common_d = [s for s in available_d if s in human_corr.index]
    nc_d = len(common_d)
    idx_d = np.triu_indices(nc_d, k=1)
    pv_d = pred_corr_d_df.loc[common_d, common_d].values[idx_d]
    hv_d = human_corr.loc[common_d, common_d].values[idx_d]
    valid_d = ~(np.isnan(pv_d) | np.isnan(hv_d))

    r_d, p_d = pearsonr(hv_d[valid_d], pv_d[valid_d])
    slope_d = np.polyfit(hv_d[valid_d], pv_d[valid_d], 1)[0]
    mr_d, mp_d = mantel_test(pred_corr_d_df.loc[common_d, common_d].values,
                              human_corr.loc[common_d, common_d].values)

    print(f"\n  Item-level prediction → round → subscale mean (Option 2):")
    print(f"    Pearson r:  {r_d:.3f}")
    print(f"    Slope:      {slope_d:.3f}")
    print(f"    Mantel r:   {mr_d:.3f} (p = {mp_d:.4f})")

    # Summary comparison
    print(f"\n  {'='*50}")
    print(f"  SUMMARY COMPARISON")
    print(f"  {'='*50}")
    print(f"  {'Pipeline':<45} {'Slope':>8} {'Mantel r':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Option 1 (ridge → item cont → mean):':<45} {slope:>8.3f} {mr:>10.3f}")
    print(f"  {'Option 2 (ridge → item → round → mean):':<45} {slope_d:>8.3f} {mr_d:>10.3f}")
    print(f"  {'Human (reference):':<45} {'1.000':>8} {'1.000':>10}")


if __name__ == "__main__":
    main()
