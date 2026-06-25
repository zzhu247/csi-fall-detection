"""
finetune_mae_har.py
Fine-tune MAE pretrained encoder on HAR, eval on OOD splits.
"""
import os, sys, json, argparse, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
sys.path.insert(0, '/home/zhuzih19/csi-project/csi-fall-detection')
import config
from data.dataset import MultiTaskDataset
from models.mae import MAE

DATA_ROOT  = config.DATA_ROOT
META_PATH  = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/metadata/sample_metadata.csv'
SPLITS_DIR = f'{DATA_ROOT}/Multitask/HumanActivityRecognition/splits'
RESULTS_DIR = '/home/zhuzih19/csi-project/csi-fall-detection/results/mae_har'
os.makedirs(RESULTS_DIR, exist_ok=True)

OOD_SPLITS = ['test_id','test_cross_device','test_cross_env','test_cross_user']

def load_split(name, meta, label_map):
    import json as _j
    with open(f'{SPLITS_DIR}/{name}.json') as f:
        ids = set(_j.load(f))
    df = meta[meta['id'].isin(ids)].reset_index(drop=True)
    return MultiTaskDataset(df, DATA_ROOT, 'Multitask', label_map=label_map)

def encode(model, x):
    """Forward through encoder only, with gradient support."""
    tokens = model.patch_embedding(x) + model.encoder_pos_embed
    h = model.encoder_blocks(tokens)
    h = model.encoder_norm(h)
    return h.mean(dim=1)  # global average pool -> [B, D]

@torch.no_grad()
def evaluate(model, head, loader, device):
    model.eval(); head.eval()
    preds, labels = [], []
    for csi, y in loader:
        feats = encode(model, csi.to(device))
        preds.append(head(feats).argmax(1).cpu())
        labels.append(y)
    preds  = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return float((preds==labels).mean()), float(f1_score(labels, preds, average='weighted', zero_division=0))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--encoder_depth', type=int, default=6)
    parser.add_argument('--encoder_dim',   type=int, default=128)
    parser.add_argument('--epochs',        type=int, default=50)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--batch_size',    type=int, default=128)
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Epochs to train only head before unfreezing encoder')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta   = pd.read_csv(META_PATH)
    import json as _j
    with open(f'{SPLITS_DIR}/train_id.json') as f:
        train_ids = set(_j.load(f))
    train_df  = meta[meta['id'].isin(train_ids)].reset_index(drop=True)
    label_map = {l: i for i, l in enumerate(sorted(train_df['label'].unique(), key=str))}
    num_classes = len(label_map)

    train_ds = load_split('train_id', meta, label_map)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    ood_loaders = {s: DataLoader(load_split(s, meta, label_map),
                                 batch_size=args.batch_size, shuffle=False, num_workers=4)
                   for s in OOD_SPLITS}

    # Load MAE encoder
    mae = MAE(
        in_channels=1, img_h=232, img_w=500, patch_h=29, patch_w=25,
        encoder_dim=args.encoder_dim, encoder_ff_dim=args.encoder_dim*4,
        encoder_heads=4, encoder_depth=args.encoder_depth,
        decoder_dim=64, decoder_heads=2, decoder_depth=2, mask_ratio=0.75
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    mae.load_state_dict(ck['model_state'])
    mae.encoder_depth = args.encoder_depth
    print(f"Loaded: {args.ckpt}")

    # Classification head
    head = nn.Linear(args.encoder_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    exp_name = f"finetune_{os.path.basename(args.ckpt).replace('_best.pt','')}_ep{args.epochs}"
    results  = {'exp': exp_name, 'evals': {}}

    # Phase 1: freeze encoder, train head only
    for p in mae.parameters(): p.requires_grad_(False)
    optim = torch.optim.Adam(head.parameters(), lr=args.lr)

    print(f"\nPhase 1: training head only ({args.freeze_epochs} epochs)...")
    for epoch in range(1, args.freeze_epochs + 1):
        mae.eval(); head.train()
        for csi, y in train_loader:
            with torch.no_grad():
                feats = encode(mae, csi.to(device))
            loss = criterion(head(feats), y.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{args.freeze_epochs}")

    # Phase 2: unfreeze encoder, fine-tune end-to-end
    for p in mae.parameters(): p.requires_grad_(True)
    optim = torch.optim.AdamW(
        [{'params': mae.parameters(), 'lr': args.lr * 0.1},
         {'params': head.parameters(), 'lr': args.lr}],
        weight_decay=0.05
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.epochs - args.freeze_epochs)

    print(f"\nPhase 2: end-to-end fine-tune ({args.epochs - args.freeze_epochs} epochs)...")
    best_val_acc = 0
    for epoch in range(1, args.epochs - args.freeze_epochs + 1):
        mae.train(); head.train()
        for csi, y in train_loader:
            feats = encode(mae, csi.to(device))
            loss  = criterion(head(feats), y.to(device))
            optim.zero_grad(); loss.backward(); optim.step()
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs - args.freeze_epochs:
            print(f"\n  Epoch {epoch} eval:")
            epoch_res = {}
            for sn, ldr in ood_loaders.items():
                acc, f1 = evaluate(mae, head, ldr, device)
                tag = '(in-dist)' if sn == 'test_id' else '(OOD)    '
                print(f"    {sn:25s} {tag} acc={acc*100:.1f}% f1={f1*100:.1f}%")
                epoch_res[sn] = {'acc': acc, 'f1': f1}
            results['evals'][f'epoch_{epoch}'] = epoch_res

    with open(f'{RESULTS_DIR}/{exp_name}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. Results: {RESULTS_DIR}/{exp_name}.json")

if __name__ == '__main__':
    main()
