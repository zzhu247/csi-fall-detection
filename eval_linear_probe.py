# eval_linear_probe.py
import os, json, torch, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from models.mae import MAE
from data.dataset import CSIFallDataset
import config

config.N_LAYERS = 12
device = torch.device("cuda")

def get_feats(model, loader, layer, device):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.extract_layer_embeddings(x, [layer])
            feats.append(emb[layer].cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def run_linear_probe(model, train_loader, eval_loader, layer, device, epochs=30):
    train_feats, train_labels = get_feats(model, train_loader, layer, device)
    eval_feats,  eval_labels  = get_feats(model, eval_loader,  layer, device)

    head = nn.Linear(train_feats.shape[1], 2).to(device)
    optim = torch.optim.Adam(head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_feats  = train_feats.to(device)
    train_labels = train_labels.to(device)

    for _ in range(epochs):
        head.train()
        loss = criterion(head(train_feats), train_labels)
        optim.zero_grad(); loss.backward(); optim.step()

    head.eval()
    with torch.no_grad():
        preds = head(eval_feats.to(device)).argmax(dim=1).cpu()
    return (preds == eval_labels).float().mean().item()

# Load data
train_df     = pd.read_csv("/home/zhuzih19/data/train.csv")
val_df       = pd.read_csv("/home/zhuzih19/data/val.csv")
test_easy_df = pd.read_csv("/home/zhuzih19/data/test_easy.csv")
test_hard_df = pd.read_csv("/home/zhuzih19/data/test_hard.csv")

train_loader     = DataLoader(CSIFallDataset(train_df,     config.DATA_ROOT),
                              batch_size=64, num_workers=4, pin_memory=True)
val_loader       = DataLoader(CSIFallDataset(val_df,       config.DATA_ROOT),
                              batch_size=64, num_workers=4, pin_memory=True)
test_easy_loader = DataLoader(CSIFallDataset(test_easy_df, config.DATA_ROOT),
                              batch_size=64, num_workers=4, pin_memory=True)
test_hard_loader = DataLoader(CSIFallDataset(test_hard_df, config.DATA_ROOT),
                              batch_size=64, num_workers=4, pin_memory=True)

# Evaluate each checkpoint
checkpoints = {
    "mae_200": ("checkpoints/mae_ep200_mask0.75_dec64_bs32_best.pth",  64),
    "mae_300": ("checkpoints/mae_ep300_mask0.75_dec64_bs32_best.pth",  64),
    "mae_500": ("checkpoints/mae_ep500_mask0.75_dec128_bs32_best.pth", 128),
}

all_results = {}

for exp_name, (ckpt_path, dec_dim) in checkpoints.items():
    print(f"\n{'='*50}")
    print(f"Evaluating: {exp_name}")

    model = MAE(
        in_channels=1, img_h=232, img_w=500, patch_h=8, patch_w=25,
        encoder_dim=128, encoder_ff_dim=512, encoder_heads=4, encoder_depth=12,
        decoder_dim=dec_dim, decoder_heads=2, decoder_depth=2,
        mask_ratio=0.75
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded: {ckpt_path}")

    exp_results = {}
    for eval_name, eval_loader in [
        ("val",       val_loader),
        ("test_easy", test_easy_loader),
        ("test_hard", test_hard_loader),
    ]:
        exp_results[eval_name] = {}
        print(f"\n  -- {eval_name} --")
        for layer in [1, 4, 8, 12]:
            acc = run_linear_probe(model, train_loader, eval_loader,
                                   layer, device, epochs=30)
            exp_results[eval_name][f"layer_{layer}"] = round(acc, 4)
            print(f"    Layer {layer:2d}  →  {acc:.3f}")

    all_results[exp_name] = exp_results

# Save and print summary
with open("results/linear_probe_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n\n" + "="*60)
print("LINEAR PROBE SUMMARY")
print("="*60)
print(f"{'Experiment':<12} {'Split':<12} {'L1':>6} {'L4':>6} {'L8':>6} {'L12':>6} {'Best':>6}")
print("-"*60)
for exp, splits in all_results.items():
    for split, layers in splits.items():
        vals = [layers[f'layer_{l}'] for l in [1,4,8,12]]
        best = max(vals)
        print(f"{exp:<12} {split:<12} "
              f"{vals[0]:>6.3f} {vals[1]:>6.3f} {vals[2]:>6.3f} {vals[3]:>6.3f} {best:>6.3f}")
