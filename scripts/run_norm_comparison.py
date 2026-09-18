"""
run_norm_comparison.py
-------------------------
Compares MAEDownstreamHead's normalization layer (LayerNorm / BatchNorm / no
normalization) on a fixed checkpoint, full backbone unfreeze, multi-seed. Same
design pattern as run_l2sp_sweep.py -- multi-seed aggregation, mean +/- std, plot.

Usage:
    python run_norm_comparison.py \
        --checkpoint checkpoints/mae_har/<exp_name>_best.pt \
        --result_json results/mae_har/<exp_name>.json \
        --layer 12 \
        --norm_types layernorm,batchnorm,none \
        --seeds 42,43,44

Activation ablation:
    --activation now also accepts 'gelu' (and silu/leaky_relu/elu/mish) in addition to
    the original 'relu' / 'none'. This is passed straight through to finetune_eval ->
    MAEDownstreamHead -> train_mae_har.make_activation(). Useful in combination with
    --norm_types none: without LayerNorm stabilizing hidden-unit scale, a smooth
    non-zero-gradient-on-negative-half activation like gelu is usually more robust
    against dead ReLU units.

    IMPORTANT: the output JSON/PNG filenames now include an activation tag
    (norm_comparison_<exp>_layer<L>_act<activation>.{json,png}). Previously a
    layernorm/relu run and a layernorm/gelu run would both write to the same
    ..._layer12.json and silently overwrite each other.

Note on BatchNorm: nn.BatchNorm1d requires batch_size > 1 in train() mode (raises
otherwise) -- if your train_loader's last batch of an epoch has exactly 1 sample
(possible if len(dataset) % batch_size == 1), the batchnorm run will crash mid-epoch.
This script does not change your DataLoader's drop_last setting -- if you hit this,
rebuild train_loader with drop_last=True before calling finetune_eval, or pick a
--batch_size that evenly divides your train_id split size.
"""
import argparse, json, sys
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

from train.train_mae_har import (
    compute_padded_size, pad_csi, finetune_eval, ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W,
)

DATA_ROOT = config.DATA_ROOT
META_PATH = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
OOD_SPLITS = ['test_id', 'test_cross_device', 'test_cross_env', 'test_cross_user']
SPLIT_LABELS = {
    'test_id': 'ID (in-distribution)', 'test_cross_device': 'Cross-Device',
    'test_cross_env': 'Cross-Env', 'test_cross_user': 'Cross-User',
}
SPLIT_COLORS = {
    'test_id': '#64748B', 'test_cross_device': '#0891B2',
    'test_cross_env': '#7C3AED', 'test_cross_user': '#D97706',
}


def build_model(train_args, padded_h, padded_w, device):
    common = dict(
        in_channels=1, img_h=padded_h, img_w=padded_w,
        patch_h=train_args['patch_h'], patch_w=train_args['patch_w'],
        encoder_dim=train_args['encoder_dim'], encoder_ff_dim=train_args['encoder_dim'] * 4,
        encoder_heads=ENCODER_HEADS, encoder_depth=train_args['encoder_depth'],
        decoder_dim=train_args['decoder_dim'], decoder_heads=2, decoder_depth=2,
        mask_ratio=train_args['mask_ratio'],
    )
    if train_args['mask_strategy'] == 'random':
        return MAE(**common).to(device)
    return MAEv2(**common, mask_strategy=train_args['mask_strategy']).to(device)


def plot_comparison(sweep_results, out_path, monitor_metric, activation):
    """
    sweep_results: list of {'norm_type': str, 'n_seeds': int, split: {'acc_mean':..., 'acc_std':...}, ...}
    activation is only used in the title here -- the per-bar labels are the norm_types.
    """
    norm_types = [r['norm_type'] for r in sweep_results]
    n_seeds = sweep_results[0]['n_seeds'] if sweep_results else 1
    x_pos = list(range(len(norm_types)))

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.2
    for i, split in enumerate(OOD_SPLITS):
        y_mean = [r[split]['acc_mean'] for r in sweep_results]
        y_std = [r[split]['acc_std'] for r in sweep_results]
        offsets = [x + (i - 1.5) * width for x in x_pos]
        ax.bar(offsets, y_mean, width=width, yerr=y_std, label=SPLIT_LABELS[split],
               color=SPLIT_COLORS[split], capsize=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(norm_types)
    ax.set_xlabel('norm_type (full backbone unfreeze)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'MAEDownstreamHead Normalization Comparison '
                 f'(activation={activation}, mean \u00b1 std, n={n_seeds} seeds)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--norm_types', default='layernorm,batchnorm,none',
                        help='Comma-separated list of norm_type values to compare')
    parser.add_argument('--seeds', default='42,43,44',
                        help='Comma-separated list of seeds to run at EACH norm_type. Reseeds torch '
                             'before each run (head init + train_loader shuffle order), results '
                             'reported as mean +/- std across seeds.')
    parser.add_argument('--unfreeze_last_n_layers', default='None',
                        help="'None' (full unfreeze, default), '0' (frozen), or an int k (last k layers)")

    # ── activation choices extended to match train_mae_har.make_activation() ──
    parser.add_argument('--activation', default='relu',
                        choices=['relu', 'gelu', 'silu', 'leaky_relu', 'elu', 'mish', 'none'],
                        help="Activation used inside the downstream head (between the two Linear "
                             "layers of mlp_head). Routed through train_mae_har.make_activation(). "
                             "'relu' (default) preserves prior behavior; 'gelu'/'silu'/'leaky_relu'/"
                             "'elu'/'mish' are usually more robust against dead units when combined "
                             "with --norm_types none (no LayerNorm stabilizing hidden scale). "
                             "'none' is the ablation control: with no non-linearity, the two Linear "
                             "layers collapse to a single effective Linear at eval time.")

    parser.add_argument('--l2sp_lambda', type=float, default=0.0,
                        help='Optional: apply L2-SP at a fixed lambda while comparing norm types '
                             '(0.0 = off, matching finetune_eval default)')
    parser.add_argument('--finetune_epochs', type=int, default=25)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--monitor_metric', default='loss', choices=['loss', 'acc'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', default='figs/norm_comparison')
    args = parser.parse_args()

    norm_types = args.norm_types.split(',')
    seeds = [int(x) for x in args.seeds.split(',')]
    unfreeze = None if args.unfreeze_last_n_layers == 'None' else int(args.unfreeze_last_n_layers)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.result_json) as f:
        result = json.load(f)
    train_args = result['args']
    print(f"Config: {result['exp']}")
    print(f"  norm_types to compare: {norm_types}")
    print(f"  activation:            {args.activation}")
    print(f"  seeds per norm_type:   {seeds}  (n={len(seeds)})")
    print(f"  unfreeze_last_n_layers: {unfreeze}   l2sp_lambda: {args.l2sp_lambda}\n")

    padded_h = compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = compute_padded_size(RAW_IMG_W, train_args['patch_w'])

    model = build_model(train_args, padded_h, padded_w, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, loss={ckpt.get('loss'):.4f}\n")

    meta = pd.read_csv(META_PATH)
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(json.load(f))
    train_df = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    # NOTE: drop_last=True -- avoids the BatchNorm batch_size=1 crash described in this
    # script's docstring, at the (usually negligible) cost of dropping up to batch_size-1
    # training samples per epoch. Remove if you specifically need every sample seen.
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4, drop_last=True)

    eval_loaders = {}
    for split in OOD_SPLITS:
        with open(f'{SPLITS_DIR}/{split}.json') as f:
            ids = set(json.load(f))
        df = meta[meta['id'].isin(ids)].reset_index(drop=True)
        ds = MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)
        eval_loaders[split] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=4)
        print(f"  {split}: {len(ds)} samples")
    print()

    common_kwargs = dict(
        epochs=args.finetune_epochs, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
        eval_every=args.eval_every, early_stop_patience=args.early_stop_patience,
        early_stop_split='ood_avg', monitor_metric=args.monitor_metric,
        unfreeze_last_n_layers=unfreeze, l2sp_lambda=args.l2sp_lambda,
        activation=args.activation,
    )

    import statistics
    sweep_results = []
    for norm_type in norm_types:
        print(f"\n{'='*70}")
        print(f"norm_type = {norm_type}  activation = {args.activation}  "
              f"({len(seeds)} seed(s): {seeds})")
        print("=" * 70)

        per_seed_results = []
        for seed in seeds:
            print(f"\n  --- seed={seed} ---")
            torch.manual_seed(seed)
            results, history = finetune_eval(model, train_loader, eval_loaders, num_classes, args.layer,
                                             device, padded_h, padded_w, norm_type=norm_type,
                                             verbose=True, **common_kwargs)
            per_seed_results.append(results)
            for split in OOD_SPLITS:
                print(f"    {split:<20} acc={results[split]['acc']:.4f}  loss={results[split]['loss']:.4f}")

        entry = {'norm_type': norm_type, 'activation': args.activation, 'n_seeds': len(seeds)}
        for split in OOD_SPLITS:
            accs = [r[split]['acc'] for r in per_seed_results]
            f1s = [r[split]['f1'] for r in per_seed_results]
            losses = [r[split]['loss'] for r in per_seed_results]
            entry[split] = {
                'acc_mean': sum(accs) / len(accs),
                'acc_std': statistics.stdev(accs) if len(accs) > 1 else 0.0,
                'f1_mean': sum(f1s) / len(f1s),
                'f1_std': statistics.stdev(f1s) if len(f1s) > 1 else 0.0,
                'loss_mean': sum(losses) / len(losses),
                'loss_std': statistics.stdev(losses) if len(losses) > 1 else 0.0,
                'acc_per_seed': accs,
                'f1_per_seed': f1s,
            }
        sweep_results.append(entry)
        print(f"\n  Aggregated (n={len(seeds)}):")
        for split in OOD_SPLITS:
            print(f"    {split:<20} acc={entry[split]['acc_mean']:.4f} \u00b1 {entry[split]['acc_std']:.4f}"
                  f"   f1={entry[split]['f1_mean']:.4f} \u00b1 {entry[split]['f1_std']:.4f}")

    # ── Filenames include the activation tag so relu/gelu runs don't overwrite ──
    norm_tag = args.norm_types.replace(',', '-')
    out_stem = f"norm_comparison_{result['exp']}_layer{args.layer}_norm{norm_tag}_act{args.activation}"
    out_json = out_dir / f"{out_stem}.json"
    with open(out_json, 'w') as f:
        json.dump({'exp': result['exp'], 'layer': args.layer,
                   'activation': args.activation,
                   'sweep': sweep_results,
                   'monitor_metric': args.monitor_metric, 'l2sp_lambda': args.l2sp_lambda,
                   'unfreeze_last_n_layers': args.unfreeze_last_n_layers}, f, indent=2)
    print(f"\nSaved raw results: {out_json}")

    out_png = out_dir / f"{out_stem}.png"
    plot_comparison(sweep_results, out_png, args.monitor_metric, args.activation)
    print(f"Saved comparison plot: {out_png}")

    col_w = 24
    print(f"\n--- Accuracy (activation={args.activation}) ---")
    print(f"{'norm_type':>12}" + "".join(f"{SPLIT_LABELS[s]:>{col_w}}" for s in OOD_SPLITS))
    for r in sweep_results:
        row = f"{r['norm_type']:>12}"
        for s in OOD_SPLITS:
            cell = f"{r[s]['acc_mean']:.4f} \u00b1 {r[s]['acc_std']:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print(f"\n--- F1 (weighted, activation={args.activation}) ---")
    print(f"{'norm_type':>12}" + "".join(f"{SPLIT_LABELS[s]:>{col_w}}" for s in OOD_SPLITS))
    for r in sweep_results:
        row = f"{r['norm_type']:>12}"
        for s in OOD_SPLITS:
            cell = f"{r[s]['f1_mean']:.4f} \u00b1 {r[s]['f1_std']:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)


if __name__ == '__main__':
    main()