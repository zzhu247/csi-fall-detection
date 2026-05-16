# train.py

import torch
import torch.nn as nn

import config
from models.vit import ViT
from data.dataset import get_dataloaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for csi, label in loader:
        csi, label = csi.to(device), label.to(device)
        out  = model(csi)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(dim=1) == label).sum().item()
        total      += label.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for csi, label in loader:
            csi, label = csi.to(device), label.to(device)
            out  = model(csi)
            loss = criterion(out, label)
            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == label).sum().item()
            total      += label.size(0)
    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders()

    model = ViT(
        in_channels = config.IN_CHANNELS,
        img_h       = config.IMG_H,
        img_w       = config.IMG_W,
        patch_h     = config.PATCH_H,
        patch_w     = config.PATCH_W,
        d_model     = config.D_MODEL,
        d_ff        = config.D_FF,
        h           = config.N_HEADS,
        N           = config.N_LAYERS,
        num_classes = config.NUM_CLASSES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device)
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()