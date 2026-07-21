"""
run_l2sp_sweep.py
-------------------
Sweeps l2sp_lambda across a list of values (full backbone unfreeze each time), records
test_id + all OOD split accuracy at each value, and plots the trade-off curve: as
l2sp_lambda increases, how much test_id gain from fine-tuning is given up, and how much
OOD stability is recovered.

Also runs the frozen baseline (unfreeze_last_n_layers=0) once for reference -- plotted as
horizontal dashed lines, since a frozen backbone can't drift at all regardless of
l2sp_lambda, so it's the natural "OOD stability ceiling" / "test_id fine-tuning-gain floor"
to compare every lambda against.

Usage:
    python run_l2sp_sweep.py \
        --checkpoint checkpoints/mae_har/<exp_name>_best.pt \
        --result_json results/mae_har/<exp_name>.json \
        --layer 12 \
        --lambdas 0,0.1,0.5,1.0,5.0,10.0 \
        --finetune_epochs 25 \
        --out_dir figs/l2sp_sweep
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
    'test_id': 'ID (in-distribution)',
    'test_cross_device': 'Cross-Device',
    'test_cross_env': 'Cross-Env',
    'test_cross_user': 'Cross-User',
}
SPLIT_COLORS = {
    'test_id': '#64748B',
    'test_cross_device': '#0891B2',
    'test_cross_env': '#7C3AED',
    'test_cross_user': '#D97706',
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


def plot_tradeoff(sweep_results, frozen_baseline, out_path, monitor_metric):
    """sweep_results: list of {'lambda': float, 'n_seeds': int, split: {'acc_mean':..., 'acc_std':..., ...}, ...}
    frozen_baseline: {split: {'acc':..., 'loss':...}} from unfreeze_last_n_layers=0 (single run --
    the backbone can't drift regardless of seed, so this is kept single-seed as a fixed reference,
    unlike the lambda sweep itself which is now multi-seed).
    """
    lambdas = [r['lambda'] for r in sweep_results]
    n_seeds = sweep_results[0]['n_seeds'] if sweep_results else 1
    # x-axis: lambda=0 breaks a log scale, so use ordinal positions with the real lambda
    # values as tick labels instead of a true numeric/log axis.
    x_pos = list(range(len(lambdas)))

    fig, ax = plt.subplots(figsize=(9, 6))
    for split in OOD_SPLITS:
        y_mean = [r[split]['acc_mean'] for r in sweep_results]
        y_std = [r[split]['acc_std'] for r in sweep_results]
        ax.errorbar(x_pos, y_mean, yerr=y_std, fmt='o-', label=SPLIT_LABELS[split],
                    color=SPLIT_COLORS[split], linewidth=2, markersize=6, capsize=4)
        # frozen-baseline reference line for this split (single-seed, see docstring)
        ax.axhline(frozen_baseline[split]['acc'], color=SPLIT_COLORS[split], linestyle=':', alpha=0.4, linewidth=1.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel('l2sp_lambda (full backbone unfreeze)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'L2-SP Trade-off: test_id gain vs. OOD stability (mean \u00b1 std, n={n_seeds} seeds)\n'
                  '(dotted lines = frozen-backbone reference, i.e. lambda=∞, single-seed)')
    ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--result_json', required=True)
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--lambdas', default='0,0.1,0.5,1.0,5.0,10.0',
                        help='Comma-separated list of l2sp_lambda values to sweep')
    parser.add_argument('--seeds', default='42',
                        help='Comma-separated list of seeds to run at EACH lambda value (default: single '
                             'seed=42, matching prior behavior). Each seed reseeds torch before that run, '
                             'which changes both the downstream head\'s random initialization and the '
                             'training DataLoader\'s shuffle order -- results are reported as mean +/- std '
                             'across seeds. Use e.g. --seeds 42,43,44 to check whether a given lambda\'s '
                             'result is stable or just noise from a single run.')
    parser.add_argument('--finetune_epochs', type=int, default=25)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--head_lr', type=float, default=1e-3)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--monitor_metric', default='loss', choices=['loss', 'acc'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--out_dir', default='figs/l2sp_sweep')
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.result_json) as f:
        result = json.load(f)
    train_args = result['args']
    print(f"Config: {result['exp']}")
    print(f"  patch={train_args['patch_h']}x{train_args['patch_w']}  "
          f"encoder_depth={train_args['encoder_depth']}  mask_strategy={train_args['mask_strategy']}")
    print(f"  lambdas to sweep: {lambdas}")
    print(f"  seeds per lambda: {seeds}  (n={len(seeds)})\n")

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
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    eval_loaders = {}
    for split in OOD_SPLITS:
        with open(f'{SPLITS_DIR}/{split}.json') as f:
            ids = set(json.load(f))
        df = meta[meta['id'].isin(ids)].reset_index(drop=True)
        ds = MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)
        eval_loaders[split] = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f"  {split}: {len(ds)} samples")
    print()

    common_kwargs = dict(
        epochs=args.finetune_epochs, backbone_lr=args.backbone_lr, head_lr=args.head_lr,
        eval_every=args.eval_every, early_stop_patience=args.early_stop_patience,
        early_stop_split='ood_avg', monitor_metric=args.monitor_metric,
    )

    print("=" * 70)
    print("Frozen baseline (unfreeze=0, reference for the trade-off plot)")
    print("=" * 70)
    frozen_baseline, _ = finetune_eval(model, train_loader, eval_loaders, num_classes, args.layer,
                                       device, padded_h, padded_w, unfreeze_last_n_layers=0,
                                       verbose=False, **common_kwargs)
    for split in OOD_SPLITS:
        print(f"  {split:<20} acc={frozen_baseline[split]['acc']:.4f}  loss={frozen_baseline[split]['loss']:.4f}")

    import statistics

    sweep_results = []
    for lam in lambdas:
        print(f"\n{'='*70}")
        print(f"l2sp_lambda = {lam}  (full backbone unfreeze, {len(seeds)} seed(s): {seeds})")
        print("=" * 70)

        per_seed_results = []
        for seed in seeds:
            print(f"\n  --- seed={seed} ---")
            torch.manual_seed(seed)  # reseeds BOTH the downstream head's init AND the
                                     # train_loader's shuffle order (DataLoader(shuffle=True)
                                     # draws from the global torch RNG unless given its own
                                     # generator) -- so this one call makes the whole run
                                     # reproducible per-seed, not just the head weights.
            results, history = finetune_eval(model, train_loader, eval_loaders, num_classes, args.layer,
                                             device, padded_h, padded_w, unfreeze_last_n_layers=None,
                                             l2sp_lambda=lam, verbose=True, **common_kwargs)
            per_seed_results.append(results)
            for split in OOD_SPLITS:
                print(f"    {split:<20} acc={results[split]['acc']:.4f}  loss={results[split]['loss']:.4f}")

        # Aggregate across seeds: mean +/- std per split, per metric
        entry = {'lambda': lam, 'n_seeds': len(seeds)}
        for split in OOD_SPLITS:
            accs = [r[split]['acc'] for r in per_seed_results]
            losses = [r[split]['loss'] for r in per_seed_results]
            entry[split] = {
                'acc_mean': sum(accs) / len(accs),
                'acc_std': statistics.stdev(accs) if len(accs) > 1 else 0.0,
                'loss_mean': sum(losses) / len(losses),
                'loss_std': statistics.stdev(losses) if len(losses) > 1 else 0.0,
                'acc_per_seed': accs,  # keep raw per-seed values too, for later inspection
            }
        sweep_results.append(entry)
        print(f"\n  Aggregated (n={len(seeds)}):")
        for split in OOD_SPLITS:
            print(f"    {split:<20} acc={entry[split]['acc_mean']:.4f} \u00b1 {entry[split]['acc_std']:.4f}")

    # Save raw numbers
    out_json = out_dir / f"l2sp_sweep_{result['exp']}_layer{args.layer}.json"
    with open(out_json, 'w') as f:
        json.dump({'exp': result['exp'], 'layer': args.layer, 'frozen_baseline': frozen_baseline,
                   'sweep': sweep_results, 'monitor_metric': args.monitor_metric}, f, indent=2)
    print(f"\nSaved raw sweep results: {out_json}")

    # Plot
    out_png = out_dir / f"l2sp_tradeoff_{result['exp']}_layer{args.layer}.png"
    plot_tradeoff(sweep_results, frozen_baseline, out_png, args.monitor_metric)
    print(f"Saved trade-off plot: {out_png}")

    # Summary table (mean +/- std across seeds; frozen baseline stays single-run, see docstring)
    col_w = 24
    print(f"\n{'lambda':>8}" + "".join(f"{SPLIT_LABELS[s]:>{col_w}}" for s in OOD_SPLITS))
    print(f"{'frozen':>8}" + "".join(f"{frozen_baseline[s]['acc']:>{col_w}.4f}" for s in OOD_SPLITS))
    for r in sweep_results:
        row = f"{r['lambda']:>8}"
        for s in OOD_SPLITS:
            cell = f"{r[s]['acc_mean']:.4f} \u00b1 {r[s]['acc_std']:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)


if __name__ == '__main__':
    main()