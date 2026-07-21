# train_ablation.py
#
# Three-way ablation study:
#   A) MAE baseline     — random masking, MSE only
#   B) MAE + block mask — block masking (mixed time/freq), MSE only
#   C) MAE + block mask + physics loss — block masking + spectral/temporal diff loss
#
# Each variant shares identical encoder/decoder architecture and hyperparameters.
# Only the masking strategy and loss function differ.
#
# Usage:
#   python train_ablation.py --variant A   # or B, C
#   python train_ablation.py --all         # run all three sequentially

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── adjust these imports to match your project layout ──
import config
from data.dataset import get_pretrain_dataloader_all
from models.mae_v2 import MAEv2


# ── Shared hyperparameters ────────────────────────────────────────────────────

ARCH = dict(
    in_channels    = 1,
    img_h          = 232,
    img_w          = 500,
    patch_h        = 8,
    patch_w        = 25,
    encoder_dim    = 128,
    encoder_ff_dim = 512,
    encoder_heads  = 4,
    encoder_depth  = 12,
    decoder_dim    = 64,
    decoder_heads  = 4,
    decoder_depth  = 4,
    mask_ratio     = 0.75,
)

TRAIN = dict(
    epochs    = 100,
    lr        = 1.5e-4,
    weight_decay = 0.05,
    warmup_epochs = 10,
    save_every    = 20,
    device    = "cuda" if torch.cuda.is_available() else "cpu",
)

# Output directory — checkpoints saved here
CKPT_DIR = "checkpoints/ablation"
os.makedirs(CKPT_DIR, exist_ok=True)


# ── Variant definitions ───────────────────────────────────────────────────────

VARIANTS = {
    "A": dict(
        name             = "MAE-Random",
        mask_strategy    = "random",
        use_physics_loss = False,
        lambda_spec      = 0.0,
        lambda_temp      = 0.0,
    ),
    "B": dict(
        name             = "MAE-Block",
        mask_strategy    = "mixed",   # randomly chooses time or freq block per sample
        use_physics_loss = False,
        lambda_spec      = 0.0,
        lambda_temp      = 0.0,
    ),
    "C": dict(
        name             = "MAE-Block-Physics",
        mask_strategy    = "mixed",
        use_physics_loss = True,
        lambda_spec      = 0.1,
        lambda_temp      = 0.1,
    ),
}


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(epoch, base_lr, warmup_epochs, total_epochs):
    """Linear warmup + cosine decay."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_variant(variant_key, loader):
    cfg     = VARIANTS[variant_key]
    device  = TRAIN["device"]

    print(f"\n{'='*60}")
    print(f"  Variant {variant_key}: {cfg['name']}")
    print(f"  mask_strategy    = {cfg['mask_strategy']}")
    print(f"  use_physics_loss = {cfg['use_physics_loss']}")
    if cfg["use_physics_loss"]:
        print(f"  lambda_spec={cfg['lambda_spec']}  lambda_temp={cfg['lambda_temp']}")
    print(f"{'='*60}\n")

    model = MAEv2(
        **ARCH,
        mask_strategy    = cfg["mask_strategy"],
        use_physics_loss = cfg["use_physics_loss"],
        lambda_spec      = cfg["lambda_spec"],
        lambda_temp      = cfg["lambda_temp"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = TRAIN["lr"],
        weight_decay = TRAIN["weight_decay"],
    )

    log_path = os.path.join(CKPT_DIR, f"variant_{variant_key}_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,loss,loss_mse,loss_spec,loss_temp\n")

    for epoch in range(TRAIN["epochs"]):
        # Update learning rate
        lr = get_lr(epoch, TRAIN["lr"], TRAIN["warmup_epochs"], TRAIN["epochs"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        epoch_loss      = 0.0
        epoch_loss_mse  = 0.0
        epoch_loss_spec = 0.0
        epoch_loss_temp = 0.0
        n_batches       = 0

        for csi in loader:
            csi = csi.to(device)
            optimizer.zero_grad()

            loss, pred, mask, loss_dict = model(csi)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss      += loss_dict.get("loss",      0.0)
            epoch_loss_mse  += loss_dict.get("loss_mse",  loss_dict.get("loss", 0.0))
            epoch_loss_spec += loss_dict.get("loss_spec", 0.0)
            epoch_loss_temp += loss_dict.get("loss_temp", 0.0)
            n_batches       += 1

        avg_loss      = epoch_loss      / n_batches
        avg_loss_mse  = epoch_loss_mse  / n_batches
        avg_loss_spec = epoch_loss_spec / n_batches
        avg_loss_temp = epoch_loss_temp / n_batches

        print(
            f"[{cfg['name']}] Epoch {epoch+1:03d}/{TRAIN['epochs']} | "
            f"lr={lr:.2e} | loss={avg_loss:.4f} | "
            f"mse={avg_loss_mse:.4f} spec={avg_loss_spec:.4f} temp={avg_loss_temp:.4f}"
        )

        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.6f},{avg_loss_mse:.6f},"
                    f"{avg_loss_spec:.6f},{avg_loss_temp:.6f}\n")

        # Save checkpoint
        if (epoch + 1) % TRAIN["save_every"] == 0 or (epoch + 1) == TRAIN["epochs"]:
            ckpt_path = os.path.join(
                CKPT_DIR, f"variant_{variant_key}_ep{epoch+1:03d}.pt"
            )
            torch.save({
                "epoch":       epoch + 1,
                "variant":     variant_key,
                "config":      cfg,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "loss":        avg_loss,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    print(f"\nVariant {variant_key} done. Log: {log_path}\n")
    return model


# ── Evaluation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_representations(model, loader, device, layer=-1):
    """
    Extract encoder representations for downstream KNN / linear probe.
    layer=-1 uses final encoder layer (mean-pooled).
    Returns features [N, D] and (dummy) labels [N].
    """
    model.eval()
    all_feats = []

    for csi in loader:
        csi = csi.to(device)
        # Use all patches, no masking
        layer_embs = model.extract_layer_embeddings(csi)
        last_layer = max(layer_embs.keys()) if layer == -1 else layer
        all_feats.append(layer_embs[last_layer].cpu())

    return torch.cat(all_feats, dim=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["A", "B", "C"],
                        help="Which variant to train")
    parser.add_argument("--all",     action="store_true",
                        help="Train all three variants sequentially")
    args = parser.parse_args()

    if not args.all and args.variant is None:
        parser.error("Specify --variant A/B/C or --all")

    print("Loading pretrain dataloader...")
    loader = get_pretrain_dataloader_all(
        data_root   = config.DATA_ROOT,
        sample_frac = 1.0,     # use full 341K dataset
    )

    variants_to_run = ["A", "B", "C"] if args.all else [args.variant]

    for v in variants_to_run:
        train_variant(v, loader)

    print("\nAll done. Checkpoints saved to:", CKPT_DIR)
    print("Next step: run eval_finetune.py or eval_cross_task.py on each checkpoint.")


if __name__ == "__main__":
    main()
