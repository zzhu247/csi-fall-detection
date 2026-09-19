"""
rebuild_splits.py

Rebuilds CSI-Bench train/val/test split files to eliminate window-level data
leakage.

Default strategy: group-level split.
    Each unique combination of GROUP_COLS is treated as an indivisible unit
    and assigned entirely to one split (train / val / test). No group is ever
    split across splits, so a window-level crossing cannot occur by
    construction. The split is balanced greedily by window count so the
    realized ratios land close to the requested 70/15/15.

Legacy strategy: buffered trial-block split.
    Windows are first segmented into contiguous "trial blocks" using gaps in
    the per-group `session` sequence, and whole trial blocks are assigned to
    each split with a buffer of skipped blocks at each boundary. This is more
    aggressive but discards a lot of data when groups contain few trial
    blocks -- for CSI-Bench HID it dropped ~38% of windows into train_UNSAFE
    and ~8% into unassigned buffers. Only use if you have a specific reason
    to split *within* a group.

Command-line usage:
    python rebuild_splits.py \
        --task_dir /home/zhuzih19/data/csi-bench-dataset/Multitask/HumanIdentification \
        --label_col user

    # (optional) legacy buffered path:
    python rebuild_splits.py \
        --task_dir /home/zhuzih19/data/csi-bench-dataset/Multitask/HumanIdentification \
        --label_col user --strategy buffered --gap_thresh 5 --buffer_blocks 1

Notebook usage:
    from rebuild_splits import load_metadata, main
    meta = load_metadata(f"{TASK_DIR}/metadata/sample_metadata.csv")
    meta_final, train_pool = main(TASK_DIR, label_col='user')
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_COLS = ['user', 'activity', 'environment', 'device', 'num_sub']
ORDER_COL = 'session'
ID_COL = 'id'


# ── Step 0: backup original split files ─────────────────────────────

def backup_original_splits(splits_dir):
    splits_dir = Path(splits_dir)
    backup_dir = splits_dir / "backup_original"
    backup_dir.mkdir(exist_ok=True, parents=True)
    for f in splits_dir.glob("*.json"):
        if f.parent == splits_dir:  # skip files already inside backup_original
            shutil.copy(f, backup_dir / f.name)
    print(f"Backed up original split files to {backup_dir}")


# ── Step 1: metadata + gap inspection ───────────────────────────────

def load_metadata(meta_path):
    return pd.read_csv(meta_path)


def compute_gap_stats(meta, group_cols=GROUP_COLS, order_col=ORDER_COL):
    """Diagnostic only. Useful for choosing GAP_THRESH in the legacy
    buffered strategy; irrelevant for the default group-level strategy."""
    gaps = []
    for _, g in meta.groupby(group_cols):
        s = g.sort_values(order_col)[order_col].values
        if len(s) > 1:
            gaps.extend(np.diff(s))
    gaps = np.array(gaps)
    stats = {
        "n_gaps": len(gaps),
        "fraction_gap_eq_1": float(np.mean(gaps == 1)) if len(gaps) else float("nan"),
        "percentiles_50_75_90_95_99": (
            np.percentile(gaps, [50, 75, 90, 95, 99]).tolist() if len(gaps) else []
        ),
    }
    print("Gap stats:", stats)
    return gaps, stats


# ── Step 2: (legacy) segment each group into contiguous trial blocks ─

def build_trial_blocks(meta, group_cols=GROUP_COLS, gap_thresh=5, order_col=ORDER_COL):
    """Vectorized segmentation. Avoids groupby().apply() because pandas 3.x
    drops the grouping columns from the returned frame."""
    meta = meta.copy()

    # capture group identity BEFORE sorting so it stays aligned
    group_key = meta[group_cols].astype(str).agg('_'.join, axis=1)

    meta = meta.assign(_grp=group_key).sort_values(
        group_cols + [order_col]
    ).reset_index(drop=True)
    group_key = meta['_grp']

    diff = meta.groupby(group_key, sort=False)[order_col].diff()
    new_trial = (diff > gap_thresh) | diff.isna()

    meta['trial_idx'] = new_trial.groupby(group_key, sort=False).cumsum()
    meta['trial_key'] = group_key + '_trial' + meta['trial_idx'].astype(str)
    meta = meta.drop(columns=['_grp'])

    print(f"num trial blocks: {meta['trial_key'].nunique()}")
    return meta


# ── Step 3a: group-level split (default) ────────────────────────────

def group_level_split(meta, group_cols=GROUP_COLS,
                      ratios=(('train', 0.70), ('val', 0.15), ('test', 0.15))):
    """Assign each unique tuple of group_cols entirely to one split, greedily
    balancing window counts to match the requested ratios.

    Every window ends up in exactly one of `names`; there is no UNSAFE or
    unassigned bucket and no possibility of a split crossing a group."""
    names, _ = zip(*ratios)
    assert abs(sum(r for _, r in ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    sizes = meta.groupby(group_cols, dropna=False).size().sort_values(ascending=False)
    total = int(sizes.sum())
    targets = {n: total * r for n, r in ratios}
    got = {n: 0 for n in names}

    assignment = {}
    for grp, size in sizes.items():
        # send this group to whichever split is furthest below its target share
        best = min(names, key=lambda n: got[n] / max(targets[n], 1e-9))
        assignment[grp] = best
        got[best] += size

    meta = meta.copy()
    keys = list(zip(*[meta[c] for c in group_cols]))
    meta['split'] = [assignment[k] for k in keys]

    print(f"group_level_split: {len(sizes)} groups, {total} windows")
    for n in names:
        print(f"  {n}: {got[n]} windows ({100 * got[n] / total:.1f}%)")
    return meta


# ── Step 3b: (legacy) buffered trial-block split ────────────────────

def buffered_group_split(meta, group_cols=GROUP_COLS,
                          split_ratios=(('train', 0.70), ('val', 0.15), ('test', 0.15)),
                          buffer_blocks=1, order_col=ORDER_COL):
    names, ratios = zip(*split_ratios)
    assert abs(sum(ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
    meta = meta.copy()
    meta['split'] = 'unassigned'
    min_blocks_needed = len(names) + buffer_blocks * (len(names) - 1)

    for _, g in meta.groupby(group_cols):
        block_order = g.groupby('trial_key')[order_col].min().sort_values().index.tolist()
        n = len(block_order)
        if n < min_blocks_needed:
            meta.loc[meta['trial_key'].isin(block_order), 'split'] = names[0] + '_UNSAFE'
            continue

        counts = [max(1, round(n * r)) for r in ratios]
        cursor = 0
        for i, (name, cnt) in enumerate(zip(names, counts)):
            blocks = block_order[cursor:cursor + cnt]
            meta.loc[meta['trial_key'].isin(blocks), 'split'] = name
            cursor += cnt
            if i < len(names) - 1:
                cursor += buffer_blocks
    return meta


# ── Step 5: validation ───────────────────────────────────────────────

def crossing_check(meta, group_cols=GROUP_COLS, order_col=ORDER_COL, valid_splits=None):
    """Fraction of adjacent-window pairs (same recording, session diff==1)
    that land on opposite sides of a split boundary. Should be 0.00% for a
    correct group-level or buffered split."""
    total, crossing = 0, 0
    sub = meta if valid_splits is None else meta[meta['split'].isin(valid_splits)]
    for _, g in sub.groupby(group_cols):
        g = g.sort_values(order_col)
        if len(g) < 2:
            continue
        diffs = np.diff(g[order_col].values)
        same_recording = diffs == 1
        crosses = g['split'].values[:-1] != g['split'].values[1:]
        total += same_recording.sum()
        crossing += (same_recording & crosses).sum()
    pct = 100 * crossing / max(total, 1)
    print(f"adjacent pairs: {total}, crossing split boundary: {crossing} ({pct:.2f}%)")
    return total, crossing, pct


def coverage_check(meta, label_col, splits=('train', 'val', 'test')):
    """Per-split label counts, to catch labels missing from val/test."""
    sub = meta[meta['split'].isin(splits)]
    coverage = sub.groupby(['split', label_col]).size().unstack(fill_value=0)
    print(coverage)
    for s in splits:
        if s not in coverage.index:
            print(f"  WARNING: split '{s}' is empty")
            continue
        missing = coverage.columns[coverage.loc[s] == 0].tolist()
        if missing:
            print(f"  WARNING: labels missing from split '{s}': {missing}")
    return coverage


# ── Step 7: write split JSONs ─────────────────────────────────────────

def write_split_json(id_series, name, out_dir):
    out_path = Path(out_dir) / f"{name}.json"
    with open(out_path, 'w') as f:
        json.dump(id_series.tolist(), f)
    print(f"{name}: {len(id_series)} ids -> {out_path}")
    return out_path


# ── End-to-end pipeline ────────────────────────────────────────────────

def main(task_dir,
         # kept for backward compat with the notebook; only used when
         # strategy == 'buffered'
         gap_thresh=None,
         buffer_blocks=1,
         final_ratios=(('train', 0.70), ('val', 0.15), ('test', 0.15)),
         hp_ratios=(('train_hp', 0.85), ('val_hp', 0.15)),
         meta_rel="metadata/sample_metadata.csv",
         splits_subdir="splits",
         label_col='user',
         group_cols=GROUP_COLS,
         order_col=ORDER_COL,
         id_col=ID_COL,
         strategy='group'):
    task_dir = Path(task_dir)
    meta_path = task_dir / meta_rel
    splits_dir = task_dir / splits_subdir

    backup_original_splits(splits_dir)

    meta = load_metadata(meta_path)
    compute_gap_stats(meta, group_cols, order_col)

    # ── Final split ─────────────────────────────────────────────────
    if strategy == 'group':
        meta_final = group_level_split(meta, group_cols, final_ratios)
    elif strategy == 'buffered':
        assert gap_thresh is not None, "gap_thresh is required for strategy='buffered'"
        meta = build_trial_blocks(meta, group_cols, gap_thresh, order_col)
        meta_final = buffered_group_split(meta, group_cols, final_ratios,
                                          buffer_blocks, order_col)
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")

    print("\n--- Final split value counts ---")
    print(meta_final['split'].value_counts())

    if strategy == 'buffered':
        unsafe = meta_final[meta_final['split'].str.contains('UNSAFE', na=False)]
        if len(unsafe):
            n_unsafe_groups = unsafe[group_cols].drop_duplicates().shape[0]
            print(f"\nWARNING: {n_unsafe_groups} group(s) flagged UNSAFE "
                  f"(too few trial blocks to buffer safely). Inspect manually:")
            print(unsafe[group_cols].drop_duplicates())

    final_names = [n for n, _ in final_ratios]
    print("\n--- Crossing check (final split) ---")
    crossing_check(meta_final, group_cols, order_col, valid_splits=final_names)

    print("\n--- Coverage check (final split) ---")
    coverage_check(meta_final, label_col, splits=final_names)

    # ── Nested HP split, carved out of the train pool only ──────────
    train_pool = meta_final[meta_final['split'] == 'train'].copy()
    if strategy == 'group':
        train_pool = group_level_split(train_pool, group_cols, hp_ratios)
    else:
        train_pool = buffered_group_split(train_pool, group_cols, hp_ratios,
                                          buffer_blocks, order_col)

    hp_names = [n for n, _ in hp_ratios]
    print("\n--- HP split value counts (within train pool) ---")
    print(train_pool['split'].value_counts())

    print("\n--- Crossing check (HP split) ---")
    crossing_check(train_pool, group_cols, order_col, valid_splits=hp_names)

    print("\n--- Coverage check (HP split) ---")
    coverage_check(train_pool, label_col, splits=hp_names)

    # ── Write out split JSONs (new names -- originals are never overwritten)
    print("\n--- Writing split files ---")
    for name in final_names:
        write_split_json(meta_final.loc[meta_final['split'] == name, id_col],
                          f"{name}_final", splits_dir)
    for name in hp_names:
        write_split_json(train_pool.loc[train_pool['split'] == name, id_col],
                          name, splits_dir)

    return meta_final, train_pool


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild CSI-Bench splits to remove window-level leakage."
    )
    parser.add_argument("--task_dir", required=True,
                         help="e.g. /home/zhuzih19/data/csi-bench-dataset/Multitask/HumanIdentification")
    parser.add_argument("--label_col", default="user")
    parser.add_argument("--strategy", choices=["group", "buffered"], default="group")
    parser.add_argument("--gap_thresh", type=int, default=None,
                         help="required only when --strategy=buffered")
    parser.add_argument("--buffer_blocks", type=int, default=1,
                         help="only used when --strategy=buffered")
    args = parser.parse_args()

    main(args.task_dir,
         gap_thresh=args.gap_thresh,
         buffer_blocks=args.buffer_blocks,
         label_col=args.label_col,
         strategy=args.strategy)