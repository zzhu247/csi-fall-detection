"""
train_mae_har.py
MAE pretraining on HAR train_id, followed by:
  - KNN eval on test_id + OOD splits
  - Linear probe eval on test_id + OOD splits

Usage:
    python train_mae_har.py --epochs 300 --mask_ratio 0.75 --encoder_depth 6
"""
import os, sys, json, argparse, random, torch, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE
from models.mae_v2 import MAEv2

DATA_ROOT  = config.DATA_ROOT
META_PATH  = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
RESULTS_DIR = '/home/zhuzih19/csi-project/csi-fall-detection/results/mae_har'
CKPT_DIR    = '/home/zhuzih19/csi-project/csi-fall-detection/checkpoints/mae_har'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,    exist_ok=True)

OOD_SPLITS = ['test_id', 'test_cross_device', 'test_cross_env', 'test_cross_user']

# ── Data loading ──────────────────────────────────────────────────────────────
def load_split(name, meta, label_map):
    import json as _json
    with open(f'{SPLITS_DIR}/{name}.json') as f:
        ids = set(_json.load(f))
    df = meta[meta['id'].isin(ids)].reset_index(drop=True)
    return MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)

# ── Evaluation helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def get_features(model, loader, layer, device):
    model.eval()
    feats, labels = [], []
    for csi, y in loader:
        emb = model.extract_layer_embeddings(csi.to(device), [layer])
        feats.append(emb[layer].cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def knn_eval(train_feats, train_labels, eval_feats, eval_labels, k=10):
    # Normalize
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std
    # Cosine similarity
    tf_n = tf / (tf.norm(dim=1, keepdim=True) + 1e-8)
    ef_n = ef / (ef.norm(dim=1, keepdim=True) + 1e-8)
    sim  = ef_n @ tf_n.T  # [N_eval, N_train]
    topk = sim.topk(k, dim=1).indices
    preds = train_labels[topk].mode(dim=1).values
    acc = (preds == eval_labels).float().mean().item()
    f1  = f1_score(eval_labels.numpy(), preds.numpy(),
                   average='weighted', zero_division=0)
    return acc, f1

def linear_probe_eval(train_feats, train_labels, eval_feats, eval_labels,
                      num_classes, device, epochs=50):
    mu  = train_feats.mean(0, keepdim=True)
    std = train_feats.std(0,  keepdim=True) + 1e-8
    tf = (train_feats - mu) / std
    ef = (eval_feats  - mu) / std

    head  = nn.Linear(tf.shape[1], num_classes).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    crit  = nn.CrossEntropyLoss()

    ds  = torch.utils.data.TensorDataset(tf, train_labels)
    ldr = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

    best_acc, best_f1 = 0.0, 0.0
    for _ in range(epochs):
        head.train()
        for xb, yb in ldr:
            loss = crit(head(xb.to(device)), yb.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        preds = head(ef.to(device)).argmax(1).cpu()
    acc = (preds == eval_labels).float().mean().item()
    f1  = f1_score(eval_labels.numpy(), preds.numpy(),
                   average='weighted', zero_division=0)
    return acc, f1

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--mask_ratio',    type=float, default=0.75)
    parser.add_argument('--encoder_depth', type=int,   default=6)
    parser.add_argument('--encoder_dim',   type=int,   default=128)
    parser.add_argument('--decoder_dim',   type=int,   default=64)
    parser.add_argument('--batch_size',    type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1.5e-4)
    parser.add_argument('--eval_layers',   type=str,   default='1,3,6')
    parser.add_argument('--eval_every',    type=int,   default=50)
    parser.add_argument('--mask_strategy', type=str, default='random', choices=['random','time','freq','mixed','2d'])
    parser.add_argument('--patch_h',       type=int,   default=29)
    parser.add_argument('--patch_w',       type=int,   default=25)
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    # Seed control for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    eval_layers = [int(x) for x in args.eval_layers.split(',')]
    exp_name = (f"mae_har_ep{args.epochs}_mask{args.mask_ratio}_strategy{args.mask_strategy}_ph{args.patch_h}pw{args.patch_w}_seed{args.seed}"
                f"_enc{args.encoder_depth}_dim{args.encoder_dim}_bs{args.batch_size}")
    print(f"\nExperiment: {exp_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    meta = pd.read_csv(META_PATH)
    import json as _json
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(_json.load(f))
    train_df  = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)
    print(f"label_map: {label_map}  num_classes: {num_classes}")

    train_ds = MultiTaskDataset(train_df, DATA_ROOT, 'Multitask', label_map=label_map)
    pretrain_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4, pin_memory=True)
    # For feature extraction (no shuffle)
    train_feat_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4)

    # OOD loaders
    ood_loaders = {}
    for sname in OOD_SPLITS:
        ds = load_split(sname, meta, label_map)
        ood_loaders[sname] = DataLoader(ds, batch_size=args.batch_size,
                                        shuffle=False, num_workers=4)
        print(f"  {sname}: {len(ds)} samples")

    # Model
    if args.mask_strategy == 'random':
        model = MAE(
            in_channels=1, img_h=232, img_w=500,
            patch_h=args.patch_h, patch_w=args.patch_w,
            encoder_dim=args.encoder_dim,
            encoder_ff_dim=args.encoder_dim * 4,
            encoder_heads=4, encoder_depth=args.encoder_depth,
            decoder_dim=args.decoder_dim,
            decoder_heads=2, decoder_depth=2,
            mask_ratio=args.mask_ratio
        ).to(device)
    else:
        model = MAEv2(
            in_channels=1, img_h=232, img_w=500,
            patch_h=args.patch_h, patch_w=args.patch_w,
            encoder_dim=args.encoder_dim,
            encoder_ff_dim=args.encoder_dim * 4,
            encoder_heads=4, encoder_depth=args.encoder_depth,
            decoder_dim=args.decoder_dim,
            decoder_heads=2, decoder_depth=2,
            mask_ratio=args.mask_ratio,
            mask_strategy=args.mask_strategy
        ).to(device)
    print(f"MAE params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    results = {'exp': exp_name, 'args': vars(args), 'loss_log': [], 'evals': {}}
    best_loss = float('inf')

    # ── Pretraining loop ──────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for csi, _ in pretrain_loader:
            csi = csi.to(device)
            optimizer.zero_grad()
            out = model(csi); loss = out[0]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(pretrain_loader)
        results['loss_log'].append({'epoch': epoch, 'loss': avg_loss})

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({'epoch': epoch, 'loss': avg_loss,
                        'model_state': model.state_dict()},
                       f'{CKPT_DIR}/{exp_name}_best.pt')

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}/{args.epochs} | "
                  f"loss={avg_loss:.4f} | best={best_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            sys.stdout.flush()

        # ── Periodic eval ─────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            print(f"\n--- Eval at epoch {epoch} ---")
            epoch_results = {}

            # Extract train features once per eval layer
            for layer in eval_layers:
                print(f"  Layer {layer}:")
                train_feats, train_labels = get_features(
                    model, train_feat_loader, layer, device)

                layer_results = {}
                for sname, ldr in ood_loaders.items():
                    eval_feats, eval_labels = get_features(model, ldr, layer, device)

                    knn_acc, knn_f1 = knn_eval(
                        train_feats, train_labels, eval_feats, eval_labels, k=10)
                    lp_acc, lp_f1 = linear_probe_eval(
                        train_feats, train_labels, eval_feats, eval_labels,
                        num_classes, device, epochs=50)

                    tag = '(in-dist)' if sname == 'test_id' else '(OOD)    '
                    print(f"    {sname:25s} {tag} "
                          f"KNN={knn_acc*100:.1f}% LP={lp_acc*100:.1f}%")
                    layer_results[sname] = {
                        'knn_acc': knn_acc, 'knn_f1': knn_f1,
                        'lp_acc':  lp_acc,  'lp_f1':  lp_f1,
                    }
                epoch_results[f'layer_{layer}'] = layer_results

            results['evals'][f'epoch_{epoch}'] = epoch_results
            # Save intermediate results
            with open(f'{RESULTS_DIR}/{exp_name}.json', 'w') as f:
                json.dump(results, f, indent=2)
            print()

    print(f"\nDone. Best pretrain loss: {best_loss:.4f}")
    print(f"Results: {RESULTS_DIR}/{exp_name}.json")
    print(f"Checkpoint: {CKPT_DIR}/{exp_name}_best.pt")

if __name__ == '__main__':
    main()
