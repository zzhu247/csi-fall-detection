from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd

import config
from data.split_utils import build_session_disjoint_splits
from diagnostics.leakage_check import run_leakage_diagnostic
from models.mae import MAE
from models.mae_v2 import MAEv2
from train.train_mae_har import (
    compute_padded_size, get_features, ENCODER_HEADS, RAW_IMG_H, RAW_IMG_W,
    META_PATH, load_split
)

def main():
    base_dir = Path("/home/zhuzih19/csi-project/csi-fall-detection")
    results_json = base_dir / "results" / "mae_har" / "mae_har_ep300_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128.json"
    checkpoint_pt = base_dir / "checkpoints" / "mae_har" / "mae_har_ep300_mask0.75_strategyrandom_ph29pw25_seed42_enc12_dim128_bs128_best.pt"
    fig_path = base_dir / "figs" / "leakage_diagnostic_result.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run Session-Disjoint Re-split
    print("--- 1. Building Session-Disjoint Split ---")
    train_set, test_id_set = build_session_disjoint_splits(META_PATH, train_ratio=0.7, buffer_windows=10)
    print(f"Clean Train IDs: {len(train_set)}, Clean Test_ID IDs: {len(test_id_set)}")

    # 2. Load Checkpoint and Extract Features
    print("--- 2. Extracting Features for NN Diagnostic ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(results_json) as f:
        train_args = json.load(f)['args']
        
    padded_h = train_args.get('padded_h') or compute_padded_size(RAW_IMG_H, train_args['patch_h'])
    padded_w = train_args.get('padded_w') or compute_padded_size(RAW_IMG_W, train_args['patch_w'])
    
    model = MAE(
        in_channels=1, img_h=padded_h, img_w=padded_w,
        patch_h=train_args['patch_h'], patch_w=train_args['patch_w'],
        encoder_dim=train_args['encoder_dim'], encoder_ff_dim=train_args['encoder_dim'] * 4,
        encoder_heads=ENCODER_HEADS, encoder_depth=train_args['encoder_depth'],
        decoder_dim=train_args['decoder_dim'], decoder_heads=2, decoder_depth=2,
        mask_ratio=train_args['mask_ratio']
    ).to(device)
    
    ckpt = torch.load(checkpoint_pt, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    meta_df = pd.read_csv(META_PATH)
    label_map = {l: i for i, l in enumerate(sorted(meta_df['label'].unique(), key=str))}

    train_loader = torch.utils.data.DataLoader(
        load_split('train_id', meta_df, label_map), batch_size=128, shuffle=False, num_workers=4
    )
    test_id_loader = torch.utils.data.DataLoader(
        load_split('test_id', meta_df, label_map), batch_size=128, shuffle=False, num_workers=4
    )
    test_cross_env_loader = torch.utils.data.DataLoader(
        load_split('test_cross_env', meta_df, label_map), batch_size=128, shuffle=False, num_workers=4
    )

    X_train, _ = get_features(model, train_loader, 12, device, padded_h, padded_w)
    X_test_id, _ = get_features(model, test_id_loader, 12, device, padded_h, padded_w)
    X_test_cross_env, _ = get_features(model, test_cross_env_loader, 12, device, padded_h, padded_w)

    # 3. Run Diagnostic Engine
    print("--- 3. Running Nearest Neighbor Cosine Distance Diagnostic ---")
    splits_dict = {
        'test_id': X_test_id.numpy(),
        'test_cross_env': X_test_cross_env.numpy()
    }
    
    run_leakage_diagnostic(
        X_train=X_train.numpy(),
        X_splits=splits_dict,
        threshold=0.05,
        save_path=fig_path
    )
    print(f"Diagnostic plot saved to: {fig_path}")

if __name__ == "__main__":
    main()